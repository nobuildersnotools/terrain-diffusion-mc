package com.github.xandergos.terraindiffusionmc.pipeline;

import com.github.xandergos.terraindiffusionmc.infinitetensor.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Java port of terrain_diffusion/inference/world_pipeline.py WorldPipeline.
 *
 * <p>Three stages: coarse (20-step DPM-Solver++), latent (2 flow-matching steps),
 * decoder (1 flow-matching step).  All pixel coordinates are native-resolution space.
 */
public final class WorldPipeline implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(WorldPipeline.class);

    static final int LATENT_COMPRESSION = WorldPipelineModelConfig.latentCompression();
    static final float SIGMA_DATA = EDMScheduler.SIGMA_DATA;

    static final int COARSE_TILE_SIZE   = 64;
    static final int COARSE_TILE_STRIDE = 48;
    static final int LATENT_TILE_SIZE   = 64;
    static final int LATENT_TILE_STRIDE = 32;
    static final int DECODER_TILE_SIZE  = 256;
    static final int DECODER_TILE_STRIDE = 192;

    static final float[] MODEL_MEANS = WorldPipelineModelConfig.coarseMeans();
    static final float[] MODEL_STDS = WorldPipelineModelConfig.coarseStds();

    static final float[] COND_MEANS = {14.99f, 11.65f, 15.87f, 619.26f, 833.12f, 69.40f, 0.66f};
    static final float[] COND_STDS  = {21.72f, 21.78f, 10.40f, 452.29f, 738.09f, 34.59f, 0.47f};

    static final float LOWFREQ_MEAN  = -31.4f;
    static final float LOWFREQ_STD   = 38.6f;
    static final float RESIDUAL_MEAN = WorldPipelineModelConfig.residualMean();
    static final float RESIDUAL_STD  = WorldPipelineModelConfig.residualStd();

    static final float[] COND_SNR  = WorldPipelineModelConfig.conditioningSnr();
    static final int COARSE_POOLING = WorldPipelineModelConfig.coarsePooling();
    static final float[] COND_VALS;  // log(COND_SNR / 8)
    static {
        if (COARSE_POOLING != 1) {
            throw new IllegalStateException("coarse_pooling=" + COARSE_POOLING + " is not supported in the Java pipeline yet");
        }
        COND_VALS = new float[COND_SNR.length];
        for (int i = 0; i < COND_SNR.length; i++) COND_VALS[i] = (float) Math.log(COND_SNR[i] / 8.0);
    }

    // mp_concat scales for 6 tensors of sizes [16, 16, 4, 16, 5, 1] → 58 total
    static final int[] COND_DIMS = {16, 16, 4, 16, 5, 1};
    static final float[] MP_CONCAT_SCALES;
    static final float[] HISTOGRAM_RAW;
    static {
        int sumN = 0; for (int n : COND_DIMS) sumN += n;
        int k = COND_DIMS.length;
        float C = (float) Math.sqrt((double) sumN * k);
        MP_CONCAT_SCALES = new float[k];
        for (int i = 0; i < k; i++) MP_CONCAT_SCALES[i] = C / (float) Math.sqrt(COND_DIMS[i]) / k;
        float[] configuredHistogramRaw = WorldPipelineModelConfig.histogramRaw();
        HISTOGRAM_RAW = configuredHistogramRaw != null ? configuredHistogramRaw : new float[]{0f, 0f, 0f, 0f, 0f};
    }

    private final OnnxModel coarseModel;
    private final OnnxModel baseModel;
    private final OnnxModel decoderModel;
    private final boolean ownModels;
    private volatile SyntheticMapFactory syntheticMapFactory;
    private volatile long seed;

    private final MemoryTileStore tileStore;
    private final long cacheLimitBytes = 100L * 1024 * 1024;

    final InfiniteTensor coarse;
    final InfiniteTensor latents;
    final InfiniteTensor residual;

    /** Uses shared models from PipelineModels (e.g. from mod init). Does not close models on close(). Seed is 64-bit (Python: seed & 0xFFFFFFFFFFFFFFFF). */
    public WorldPipeline(long seed, PipelineModels models) {
        this.seed = seed & 0xFFFFFFFFFFFFFFFFL;
        this.coarseModel = models.getCoarseModel();
        this.baseModel = models.getBaseModel();
        this.decoderModel = models.getDecoderModel();
        this.ownModels = false;
        this.syntheticMapFactory = new SyntheticMapFactory(this.seed);
        this.tileStore = new MemoryTileStore();
        this.coarse = buildCoarseStage();
        this.latents = buildLatentStage();
        this.residual = buildDecoderStage();
    }

    /** Loads its own models (e.g. for tests). Caller must close. */
    public WorldPipeline(long seed) {
        this.seed = seed & 0xFFFFFFFFFFFFFFFFL;
        ModelAssetManager.ensureAssetsReady();
        this.coarseModel = new OnnxModel(ModelAssetManager.resolveAssetPath("coarse_model.onnx"), "coarse");
        this.baseModel = new OnnxModel(ModelAssetManager.resolveAssetPath("base_model.onnx"), "base");
        this.decoderModel = new OnnxModel(ModelAssetManager.resolveAssetPath("decoder_model.onnx"), "decoder");
        this.ownModels = true;
        this.syntheticMapFactory = new SyntheticMapFactory(this.seed);
        this.tileStore = new MemoryTileStore();
        this.coarse = buildCoarseStage();
        this.latents = buildLatentStage();
        this.residual = buildDecoderStage();
    }

    /** Lightweight seed change (Python change_seed): update seed and synthetic map, clear tile caches. Models stay loaded. */
    public void setSeed(long newSeed) {
        long s = newSeed & 0xFFFFFFFFFFFFFFFFL;
        if (s == this.seed) return;
        this.seed = s;
        this.syntheticMapFactory = new SyntheticMapFactory(s);
        tileStore.clearAllCaches();
    }

    // =========================================================================
    // Coarse Stage
    // =========================================================================

    private InfiniteTensor buildCoarseStage() {
        int S = COARSE_TILE_SIZE, ST = COARSE_TILE_STRIDE;
        float[] ww = linearWeightWindow(S);
        TensorWindow outWin = new TensorWindow(new int[]{7, S, S}, new int[]{7, ST, ST});
        return tileStore.getOrCreateBatched("base_coarse_map", new Integer[]{7, null, null},
                (wis, args) -> coarseBatch(wis, ww), outWin,
                new InfiniteTensor[]{}, new TensorWindow[]{}, cacheLimitBytes, 2);
    }

    private List<FloatTensor> coarseBatch(List<int[]> wis, float[] ww) {
        int S = COARSE_TILE_SIZE, ST = COARSE_TILE_STRIDE;
        int pixels = S * S;
        int batch = wis.size();
        float[] condMixedBatch = new float[batch * 5 * pixels];
        float[] sampleBatch = new float[batch * 6 * pixels];
        EDMScheduler sched = new EDMScheduler(20);

        for (int b = 0; b < batch; b++) {
            int[] wi = wis.get(b);
            int i1 = wi[1] * ST, j1 = wi[2] * ST;

            // Synthetic map conditioning: channels [elev_sqrt, temp, tempStd, precip, precipStd]
            // Python call: synthetic_map_factory(j1, i1, j2, i2); coordinates are intentionally swapped
            float[][][] syn = syntheticMapFactory.sample(j1, i1, j1 + S, i1 + S);

            // Modify temp channel (index 1): where <= 20, scale toward 20
            for (int r = 0; r < S; r++) VectorMath.adjustAtMostInPlace(syn[1][r], 20.0f, 1.25f);

            // Normalize with MODEL_MEANS/STDS indices [0,2,3,4,5]
            int[] meanIdx = {0, 2, 3, 4, 5};
            float[] condImg = new float[5 * pixels];
            for (int ch = 0; ch < 5; ch++) {
                float mean = MODEL_MEANS[meanIdx[ch]], std = MODEL_STDS[meanIdx[ch]];
                for (int r = 0; r < S; r++) {
                    VectorMath.normalize(syn[ch][r], 0, mean, std,
                            condImg, ch * pixels + r * S, S);
                }
            }

            // cond_img_mixed = cos(t_cond) * normalized + sin(t_cond) * noise
            float[] condNoise = flatten3D(GaussianNoisePatch.generate(seed, i1, j1, S, S, 5, S, S));
            int condOffset = b * 5 * pixels;
            for (int ch = 0; ch < 5; ch++) {
                float cosT = (float) Math.cos(Math.atan(COND_SNR[ch]));
                float sinT = (float) Math.sin(Math.atan(COND_SNR[ch]));
                int channelOffset = ch * pixels;
                VectorMath.linearCombination(condImg, channelOffset, cosT,
                        condNoise, channelOffset, sinT,
                        condMixedBatch, condOffset + channelOffset, pixels);
            }

            // Initial sample: (6, S, S) noise * sigma_max
            float[] sample = flatten3D(GaussianNoisePatch.generate(seed + 1, i1, j1, S, S, 6, S, S));
            int sampleOffset = b * 6 * pixels;
            VectorMath.scale(sample, 0, sched.sigmas[0], sampleBatch, sampleOffset, sample.length);
        }

        // 20-step DPM-Solver++
        float[][] condInputs = new float[5][batch];
        long[][] condShapes  = new long[5][1];
        for (int ci = 0; ci < 5; ci++) {
            for (int b = 0; b < batch; b++) condInputs[ci][b] = COND_VALS[ci];
            condShapes[ci] = new long[]{batch};
        }

        String chunkList = wis.stream().map(w -> "(" + w[1] + "," + w[2] + ")").collect(Collectors.joining(", "));
        LOG.debug("Coarse model called for {} chunks: {} (20 steps)", batch, chunkList);
        for (int step = 0; step < 20; step++) {
            float sigma  = sched.sigmas[step];
            float cnoise = EDMScheduler.trigflowPreconditionNoise(sigma);
            float cIn = 1.0f / (float) Math.sqrt(sigma * sigma + SIGMA_DATA * SIGMA_DATA);
            float[] xIn = new float[batch * 11 * pixels];
            for (int b = 0; b < batch; b++) {
                int sampleOffset = b * 6 * pixels;
                int inputOffset = b * 11 * pixels;
                VectorMath.scale(sampleBatch, sampleOffset, cIn, xIn, inputOffset, 6 * pixels);
                System.arraycopy(condMixedBatch, b * 5 * pixels, xIn, inputOffset + 6 * pixels, 5 * pixels);
            }

            float[] noiseLabels = new float[batch];
            for (int b = 0; b < batch; b++) noiseLabels[b] = cnoise;
            float[] modelOut = coarseModel.runModel(
                    xIn, new long[]{batch, 11, S, S}, noiseLabels, condInputs, condShapes);
            sampleBatch = sched.step(modelOut, sampleBatch);
        }

        List<FloatTensor> results = new ArrayList<>(batch);
        for (int b = 0; b < batch; b++) {
            int sampleOffset = b * 6 * pixels;
            float[] out = new float[6 * pixels];
            for (int ch = 0; ch < 6; ch++) {
                VectorMath.affineAfterDivide(sampleBatch, sampleOffset + ch * pixels, SIGMA_DATA,
                        MODEL_STDS[ch], MODEL_MEANS[ch], out, ch * pixels, pixels);
            }
            VectorMath.subtract(out, 0, out, pixels, out, pixels, pixels);

            FloatTensor result = new FloatTensor(new int[]{7, S, S});
            for (int ch = 0; ch < 6; ch++) {
                VectorMath.multiply(out, ch * pixels, ww, 0, result.data, ch * pixels, pixels);
            }
            System.arraycopy(ww, 0, result.data, 6 * pixels, pixels);
            results.add(result);
        }
        return results;
    }

    // =========================================================================
    // Latent Stage
    // =========================================================================

    private InfiniteTensor buildLatentStage() {
        int S = LATENT_TILE_SIZE, ST = LATENT_TILE_STRIDE;
        TensorWindow outWin = new TensorWindow(new int[]{6, S, S}, new int[]{6, ST, ST});
        TensorWindow coarseWin = new TensorWindow(new int[]{7, 4, 4}, new int[]{7, 1, 1}, new int[]{0, -1, -1});
        float[] ww = linearWeightWindow(S);
        float tInit = (float) Math.atan(EDMScheduler.SIGMA_MAX / SIGMA_DATA);

        InfiniteTensor initLatent = tileStore.getOrCreateBatched(
                "init_latent_map", new Integer[]{6, null, null},
                (wis, args) -> latentBatch(wis, null, args.get(0), tInit, 5819, ww),
                outWin, new InfiniteTensor[]{coarse}, new TensorWindow[]{coarseWin},
                cacheLimitBytes, 4);

        float interT = (float) Math.atan(0.35f / SIGMA_DATA);
        return tileStore.getOrCreateBatched(
                "step_latent_map_0", new Integer[]{6, null, null},
                (wis, args) -> latentBatch(wis, args.get(0), args.get(1), interT, 5820, ww),
                outWin, new InfiniteTensor[]{initLatent, coarse}, new TensorWindow[]{outWin, coarseWin},
                cacheLimitBytes, 4);
    }

    private List<FloatTensor> latentBatch(List<int[]> wis, List<FloatTensor> prevSamples,
                                           List<FloatTensor> coarseSlices, float t,
                                           int seedOffset, float[] ww) {
        int S = LATENT_TILE_SIZE, ST = LATENT_TILE_STRIDE;
        int batch = wis.size();
        float cosT = (float) Math.cos(t), sinT = (float) Math.sin(t);

        // Intermediate storage: xT per batch element (needed for output step)
        float[][] xTArr = new float[batch][5 * S * S];

        float[] modelInBatch   = new float[batch * 5 * S * S];
        float[] condInputBatch = new float[batch * 58];

        for (int b = 0; b < batch; b++) {
            int[] ctx = wis.get(b);
            int i1 = ctx[1] * ST, j1 = ctx[2] * ST;

            // Build conditioning from coarse slice (7, 4, 4)
            float[] cond58 = buildLatentConditioning(coarseSlices.get(b));
            System.arraycopy(cond58, 0, condInputBatch, b * 58, 58);

            // Build sample (unnormalized prev output or zeros)
            float[] sample = new float[5 * S * S];
            if (prevSamples != null) {
                FloatTensor ps = prevSamples.get(b);
                for (int ch = 0; ch < 5; ch++) {
                    VectorMath.divideMultiplyWhereGreater(ps.data, ch * S * S,
                            ps.data, 5 * S * S, 1e-6f, SIGMA_DATA,
                            sample, ch * S * S, S * S);
                }
            }

            // z = noise * sigma_data; x_t = cos(t)*sample + sin(t)*z
            float[] noise = flatten3D(GaussianNoisePatch.generate(seed + seedOffset, i1, j1, S, S, 5, S, S));
            float[] xT = new float[5 * S * S];
            VectorMath.noiseMix(sample, noise, cosT, sinT, SIGMA_DATA, xT);
            xTArr[b] = xT;

            // model_in = xT / sigma_data
            VectorMath.divide(xT, 0, SIGMA_DATA, modelInBatch, b * 5 * S * S, 5 * S * S);
        }

        String chunkList = wis.stream().map(w -> "(" + w[1] + "," + w[2] + ")").collect(Collectors.joining(", "));
        LOG.debug("Base model called for {} chunks: {}", batch, chunkList);

        float[] noiseLabels = new float[batch];
        for (int b = 0; b < batch; b++) noiseLabels[b] = t;

        float[] predBatch = baseModel.runModel(
                modelInBatch, new long[]{batch, 5, S, S},
                noiseLabels, new float[][]{condInputBatch}, new long[][]{{batch, 58}});

        // Build outputs: pred = -raw_model_out; sample = cos(t)*xT - sin(t)*sigma_data*pred
        List<FloatTensor> results = new ArrayList<>(batch);
        for (int b = 0; b < batch; b++) {
            float[] xT = xTArr[b];
            float[] newSample = new float[5 * S * S];
            VectorMath.flowDenoise(xT, 0, predBatch, b * 5 * S * S,
                    cosT, sinT * SIGMA_DATA, SIGMA_DATA, newSample, 0, 5 * S * S);

            FloatTensor out = new FloatTensor(new int[]{6, S, S});
            for (int ch = 0; ch < 5; ch++) {
                VectorMath.multiply(newSample, ch * S * S, ww, 0,
                        out.data, ch * S * S, S * S);
            }
            System.arraycopy(ww, 0, out.data, 5 * S * S, S * S);
            results.add(out);
        }
        return results;
    }

    /** Build 58-dim conditioning vector from a (7,4,4) coarse tile slice. */
    private float[] buildLatentConditioning(FloatTensor coarseSlice) {
        int N = 4 * 4;
        // Unnormalize: cond[:-1] / cond[-1] for each pixel
        float[] condFlat = new float[6 * N];
        for (int ch = 0; ch < 6; ch++) {
            VectorMath.divideMultiplyWhereGreater(coarseSlice.data, ch * N,
                    coarseSlice.data, 6 * N, 1e-6f, 1.0f,
                    condFlat, ch * N, N);
        }

        // Append mask channel (all ones = (1 - mean) / std normalized)
        float[] condImg7 = new float[7 * N];
        System.arraycopy(condFlat, 0, condImg7, 0, 6 * N);
        float maskNorm = (1.0f - COND_MEANS[6]) / COND_STDS[6];
        for (int px = 0; px < N; px++) condImg7[6 * N + px] = maskNorm;

        // Normalize all 7 channels
        for (int ch = 0; ch < 6; ch++) {
            VectorMath.normalizeNaNToZero(condFlat, ch * N, COND_MEANS[ch], COND_STDS[ch],
                    condImg7, ch * N, N);
        }

        // Extract components
        float[] meansCrop    = new float[16]; System.arraycopy(condImg7, 0,      meansCrop, 0, 16);
        float[] p5Crop       = new float[16]; System.arraycopy(condImg7, 16,     p5Crop,    0, 16);
        float[] maskCrop     = new float[16]; System.arraycopy(condImg7, 6 * 16, maskCrop,  0, 16);
        float[] climateMeans = new float[4];
        for (int ch = 0; ch < 4; ch++) {
            float sum = 0;
            for (int r = 1; r < 3; r++) for (int c = 1; c < 3; c++)
                sum += condImg7[(2 + ch) * 16 + r * 4 + c];
            climateMeans[ch] = sum / 4f;
            if (Float.isNaN(climateMeans[ch])) climateMeans[ch] = 0f;
        }

        float noiseLevelNorm = (0f - 0.5f) * (float) Math.sqrt(12.0);
        float[] histRaw = HISTOGRAM_RAW;

        // mp_concat
        float[] out = new float[58];
        int off = 0;
        off = appendScaled(out, off, meansCrop,    MP_CONCAT_SCALES[0]);
        off = appendScaled(out, off, p5Crop,       MP_CONCAT_SCALES[1]);
        off = appendScaled(out, off, climateMeans, MP_CONCAT_SCALES[2]);
        off = appendScaled(out, off, maskCrop,     MP_CONCAT_SCALES[3]);
        off = appendScaled(out, off, histRaw,      MP_CONCAT_SCALES[4]);
        out[off] = noiseLevelNorm * MP_CONCAT_SCALES[5];
        return out;
    }

    // =========================================================================
    // Decoder Stage
    // =========================================================================

    private InfiniteTensor buildDecoderStage() {
        int S = DECODER_TILE_SIZE, ST = DECODER_TILE_STRIDE, lc = LATENT_COMPRESSION;
        TensorWindow outWin  = new TensorWindow(new int[]{2, S, S},  new int[]{2, ST, ST});
        TensorWindow inpWin  = new TensorWindow(new int[]{6, S/lc, S/lc}, new int[]{6, ST/lc, ST/lc});
        float[] ww = linearWeightWindow(S);
        float t = (float) Math.atan(EDMScheduler.SIGMA_MAX / SIGMA_DATA);

        return tileStore.getOrCreateBatched("init_residual_map", new Integer[]{2, null, null},
                (wis, args) -> decoderBatch(wis, args.get(0), t, ww),
                outWin, new InfiniteTensor[]{latents}, new TensorWindow[]{inpWin}, cacheLimitBytes, 2);
    }

    private List<FloatTensor> decoderBatch(List<int[]> wis, List<FloatTensor> latentSlices,
                                           float t, float[] ww) {
        int S = DECODER_TILE_SIZE, ST = DECODER_TILE_STRIDE, lc = LATENT_COMPRESSION;
        int Slc = S / lc;
        int batch = wis.size();
        float cosT = (float) Math.cos(t), sinT = (float) Math.sin(t);
        float[][] xTArr = new float[batch][];
        float[] modelInBatch = new float[batch * 5 * S * S];

        for (int b = 0; b < batch; b++) {
            int[] wi = wis.get(b);
            FloatTensor latentSlice = latentSlices.get(b);
            int i1 = wi[1] * ST, j1 = wi[2] * ST;

            // Unnormalize latents channels 0..3 (4 channels)
            float[] latFlat = new float[4 * Slc * Slc];
            for (int ch = 0; ch < 4; ch++) {
                VectorMath.divideMultiplyWhereGreater(latentSlice.data, ch * Slc * Slc,
                        latentSlice.data, 5 * Slc * Slc, 1e-6f, 1.0f,
                        latFlat, ch * Slc * Slc, Slc * Slc);
            }

            // Nearest-neighbor upsample (4, Slc, Slc) → (4, S, S)
            float[] upsampled = nearestUpsample(latFlat, 4, Slc, Slc, S, S);

            // One flow-matching step (sample starts at zero)
            float[] noise = flatten3D(GaussianNoisePatch.generate(seed + 5819, i1, j1, S, S, 1, S, S));
            float[] xT = new float[S * S];
            int modelOffset = b * 5 * S * S;
            VectorMath.scaleTwice(noise, 0, sinT, SIGMA_DATA, xT, 0, S * S);
            VectorMath.divide(xT, 0, SIGMA_DATA, modelInBatch, modelOffset, S * S);
            xTArr[b] = xT;
            System.arraycopy(upsampled, 0, modelInBatch, modelOffset + S * S, 4 * S * S);
        }

        String chunkList = wis.stream().map(w -> "(" + w[1] + "," + w[2] + ")").collect(Collectors.joining(", "));
        LOG.debug("Decoder model called for {} chunks: {}", batch, chunkList);
        float[] noiseLabels = new float[batch];
        for (int b = 0; b < batch; b++) noiseLabels[b] = t;
        float[] rawPred = decoderModel.runModel(
                modelInBatch, new long[]{batch, 5, S, S}, noiseLabels, null, null);

        // sample = cos(t)*xT - sin(t)*sigma_data*(-rawPred); then / sigma_data
        List<FloatTensor> results = new ArrayList<>(batch);
        for (int b = 0; b < batch; b++) {
            float[] xT = xTArr[b];
            FloatTensor result = new FloatTensor(new int[]{2, S, S});
            float[] newSample = new float[S * S];
            VectorMath.flowDenoise(xT, 0, rawPred, b * S * S,
                    cosT, sinT * SIGMA_DATA, SIGMA_DATA, newSample, 0, S * S);
            VectorMath.multiply(newSample, 0, ww, 0, result.data, 0, S * S);
            System.arraycopy(ww, 0, result.data, S * S, S * S);
            results.add(result);
        }
        return results;
    }

    // =========================================================================
    // Public API
    // =========================================================================

    /** Returns the current world seed. */
    public long getSeed() {
        return seed;
    }

    /** Returns the total count of newly computed tensor windows since startup. */
    public long getTotalComputedWindowCount() {
        return tileStore.getTotalComputedWindowCount();
    }

    /**
     * Returns a coarse tensor slice with shape [7, ci1-ci0, cj1-cj0].
     * Coordinates are in coarse index units (1 unit = 256 native pixels).
     * Channel 6 is the blend weight; channels 0–5 are weighted sums.
     */
    public FloatTensor getCoarseSlice(int ci0, int cj0, int ci1, int cj1) {
        return coarse.getSlice(new int[]{0, ci0, cj0}, new int[]{7, ci1, cj1});
    }

    /**
     * Get elevation and climate for a bounding box.
     *
     * @return float[2]: [0] = elev (H*W flat), [1] = climate (5*H*W flat, or null)
     */
    public float[][] get(int i1, int j1, int i2, int j2, boolean withClimate) {
        float[] elevFlat = computeElev(i1, j1, i2, j2);
        int H = i2 - i1, W = j2 - j1;
        float[] climate = withClimate ? computeClimate(i1, j1, i2, j2, elevFlat, H, W) : null;
        return new float[][]{elevFlat, climate};
    }

    // =========================================================================
    // Elevation
    // =========================================================================

    private float[] computeElev(int i1, int j1, int i2, int j2) {
        int lc = LATENT_COMPRESSION;
        float sigma = 5.0f;
        int ks = ((int) (sigma * 2) / 2) * 2 + 1;
        int padLr = ks / 2 + 1;
        int padHr = padLr * lc;

        // Align padding to lc-pixel grid
        int pi1 = Math.floorDiv(i1 - padHr, lc) * lc;
        int pj1 = Math.floorDiv(j1 - padHr, lc) * lc;
        int pi2 = -Math.floorDiv(-(i2 + padHr), lc) * lc;
        int pj2 = -Math.floorDiv(-(j2 + padHr), lc) * lc;
        int pH = pi2 - pi1, pW = pj2 - pj1;

        // Residual slice (2, pH, pW)
        FloatTensor resSlice = residual.getSlice(new int[]{0, pi1, pj1}, new int[]{2, pi2, pj2});
        float[][] residualP = new float[pH][pW];
        for (int r = 0; r < pH; r++) {
            VectorMath.divideAffineWhereGreater(resSlice.data, r * pW,
                    resSlice.data, pH * pW + r * pW, 1e-6f,
                    RESIDUAL_STD, RESIDUAL_MEAN, residualP[r], 0, pW);
        }

        // Latent slice (6, lH, lW)
        int lH = pH / lc, lW = pW / lc;
        FloatTensor latSlice = latents.getSlice(
                new int[]{0, pi1 / lc, pj1 / lc}, new int[]{6, pi2 / lc, pj2 / lc});
        float[][] lowfreqP = new float[lH][lW];
        for (int r = 0; r < lH; r++) {
            VectorMath.divideAffineWhereGreater(latSlice.data, 4 * lH * lW + r * lW,
                    latSlice.data, 5 * lH * lW + r * lW, 1e-6f,
                    LOWFREQ_STD, LOWFREQ_MEAN, lowfreqP[r], 0, lW);
        }

        float[][] newLowres = LaplacianUtils.laplacianDenoise(residualP, lowfreqP, sigma);
        float[][] elevP = LaplacianUtils.laplacianDecode(residualP, newLowres);

        int oi = i1 - pi1, oj = j1 - pj1, H = i2 - i1, W = j2 - j1;
        float[] flat = new float[H * W];
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++) {
                float es = elevP[oi + r][oj + c];
                flat[r * W + c] = (float) (Math.signum(es) * es * es);
            }
        return flat;
    }

    // =========================================================================
    // Climate
    // =========================================================================

    private float[] computeClimate(int i1, int j1, int i2, int j2,
                                    float[] elevFlat, int H, int W) {
        int lc = LATENT_COMPRESSION;
        int S = 32 * lc;  // native pixels per coarse pixel in stride sense

        int ci1 = Math.floorDiv(i1, S);
        int cj1 = Math.floorDiv(j1, S);
        int ci2 = -Math.floorDiv(-i2, S);
        int cj2 = -Math.floorDiv(-j2, S);

        int win = 15, pad = (win - 1) / 2 + 1;

        FloatTensor coarseSlice = coarse.getSlice(
                new int[]{0, ci1 - pad, cj1 - pad}, new int[]{7, ci2 + pad, cj2 + pad});
        int cH = ci2 + pad - (ci1 - pad);
        int cW = cj2 + pad - (cj1 - pad);

        // Unnormalize all 6 coarse channels
        float[][] coarseMap = new float[6][cH * cW];
        for (int ch = 0; ch < 6; ch++) {
            VectorMath.divideMultiplyWhereGreater(coarseSlice.data, ch * cH * cW,
                    coarseSlice.data, 6 * cH * cW, 1e-6f, 1.0f,
                    coarseMap[ch], 0, cH * cW);
        }

        // Coarse elevation (undo sqrt): max(0, v)^2  — ocean pixels clamp to 0, matching Python
        float[] coarseElev = new float[cH * cW];
        for (int px = 0; px < cH * cW; px++) {
            float v = Math.max(0f, coarseMap[0][px]);
            coarseElev[px] = v * v;
        }

        // Windowed lapse-rate regression
        float[][][] lbt = LaplacianUtils.localBaselineTemperature(
                to2D(coarseMap[2], cH, cW), to2D(coarseElev, cH, cW), win, 0.02f);
        int lH = lbt[0].length, lW = lbt[0][0].length;

        // Central coarse (crop pad pixels from each side)
        int cenPad = win / 2;
        int cenH = cH - 2 * cenPad, cenW = cW - 2 * cenPad;
        float[][][] centralCoarse = new float[6][cenH][cenW];
        for (int ch = 0; ch < 6; ch++) {
            float[][] full = to2D(coarseMap[ch], cH, cW);
            centralCoarse[ch] = cropArray(full, cenPad, cenPad, cenH, cenW);
        }

        // Bilinear upsample to native resolution
        float[] climate = new float[5 * H * W];
        for (int r = 0; r < H; r++) {
            // fractional index into lbt/centralCoarse arrays (matches Python's u = (ii+0.5)/S - ci1 + 0.5)
            float gridY    = (i1 + r + 0.5f) / S - ci1 + 0.5f;
            float cenGridY = gridY;
            for (int c = 0; c < W; c++) {
                float gridX    = (j1 + c + 0.5f) / S - cj1 + 0.5f;
                float cenGridX = gridX;

                float tBase = bilinearSample2D(lbt[0], lH, lW, gridY, gridX);
                float beta  = bilinearSample2D(lbt[1], lH, lW, gridY, gridX);
                float tempReal = tBase + beta * Math.max(0f, elevFlat[r * W + c]);

                climate[r * W + c]             = tempReal;
                climate[H * W + r * W + c]     = bilinearSample2D(centralCoarse[3], cenH, cenW, cenGridY, cenGridX);
                climate[2 * H * W + r * W + c] = bilinearSample2D(centralCoarse[4], cenH, cenW, cenGridY, cenGridX);
                climate[3 * H * W + r * W + c] = bilinearSample2D(centralCoarse[5], cenH, cenW, cenGridY, cenGridX);
                climate[4 * H * W + r * W + c] = beta;
            }
        }
        return climate;
    }

    // =========================================================================
    // Static helpers
    // =========================================================================

    static float[] linearWeightWindow(int size) {
        float[] w = new float[size * size];
        float mid = (size - 1) / 2.0f, eps = 1e-3f;
        for (int r = 0; r < size; r++) {
            float wy = 1f - (1f - eps) * Math.min(1f, Math.abs(r - mid) / mid);
            for (int c = 0; c < size; c++) {
                float wx = 1f - (1f - eps) * Math.min(1f, Math.abs(c - mid) / mid);
                w[r * size + c] = wy * wx;
            }
        }
        return w;
    }

    static float[] flatten3D(float[][][] arr) {
        int C = arr.length, H = arr[0].length, W = arr[0][0].length;
        float[] out = new float[C * H * W];
        for (int c = 0; c < C; c++)
            for (int r = 0; r < H; r++)
                System.arraycopy(arr[c][r], 0, out, c * H * W + r * W, W);
        return out;
    }

    static int appendScaled(float[] out, int off, float[] arr, float scale) {
        VectorMath.scale(arr, 0, scale, out, off, arr.length);
        return off + arr.length;
    }

    static float[] nearestUpsample(float[] src, int C, int sH, int sW, int dH, int dW) {
        float[] dst = new float[C * dH * dW];
        for (int c = 0; c < C; c++)
            for (int r = 0; r < dH; r++) {
                int sr = r * sH / dH;
                for (int col = 0; col < dW; col++)
                    dst[c * dH * dW + r * dW + col] = src[c * sH * sW + sr * sW + col * sW / dW];
            }
        return dst;
    }

    static float[][] to2D(float[] flat, int H, int W) {
        float[][] a = new float[H][W];
        for (int r = 0; r < H; r++) System.arraycopy(flat, r * W, a[r], 0, W);
        return a;
    }

    static float[][] cropArray(float[][] src, int r0, int c0, int H, int W) {
        float[][] out = new float[H][W];
        for (int r = 0; r < H; r++) System.arraycopy(src[r + r0], c0, out[r], 0, W);
        return out;
    }

    static float bilinearSample2D(float[][] src, int H, int W, float gy, float gx) {
        float y = Math.max(0f, Math.min(H - 1f, gy));
        float x = Math.max(0f, Math.min(W - 1f, gx));
        int y0 = (int) y, y1 = Math.min(H - 1, y0 + 1);
        int x0 = (int) x, x1 = Math.min(W - 1, x0 + 1);
        float wy = y - y0, wx = x - x0;
        return (1-wy)*(1-wx)*src[y0][x0] + (1-wy)*wx*src[y0][x1]
             + wy*(1-wx)*src[y1][x0] + wy*wx*src[y1][x1];
    }

    @Override
    public void close() {
        if (ownModels) {
            coarseModel.close();
            baseModel.close();
            decoderModel.close();
        }
    }
}
