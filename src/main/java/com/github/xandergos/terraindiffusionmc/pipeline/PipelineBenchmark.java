package com.github.xandergos.terraindiffusionmc.pipeline;

import com.github.xandergos.terraindiffusionmc.config.TerrainDiffusionConfig;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;

/**
 * Command-line benchmark for full terrain-pipeline inference.
 *
 * <p>Each task samples a distinct native-resolution region so the timing includes
 * newly computed model windows instead of returning a cached result. Run through
 * {@code ./gradlew benchmarkPipeline} to inherit the selected ONNX build variant.
 */
public final class PipelineBenchmark {
    private static final long DEFAULT_SEED = -5408366058459925370L;
    private static final int DEFAULT_TASKS = 10;
    private static final int DEFAULT_WARMUP = 1;
    private static final int DEFAULT_REGION_SIZE = 256;
    private static final int REGION_GAP = 512;

    private PipelineBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        Options options = Options.parse(args);
        if (options.help) {
            printUsage();
            return;
        }

        // This must happen before TerrainDiffusionConfig or model classes are initialized.
        if (options.backend != null) {
            System.setProperty("terrain-diffusion.backend", options.backend);
        }

        String requestedBackend = TerrainDiffusionConfig.inferenceBackend();
        System.out.printf(Locale.ROOT,
                "Benchmark: backend=%s tasks=%d warmup=%d region=%dx%d climate=%s seed=%d%n",
                requestedBackend, options.tasks, options.warmup, options.regionSize, options.regionSize,
                options.withClimate, options.seed);

        long loadStarted = System.nanoTime();
        try (WorldPipeline pipeline = new WorldPipeline(options.seed)) {
            double modelLoadMillis = nanosToMillis(System.nanoTime() - loadStarted);

            for (int task = 0; task < options.warmup; task++) {
                runTask(pipeline, task, options, -options.warmup);
            }

            long[] durationsNanos = new long[options.tasks];
            long windowsBefore = pipeline.getTotalComputedWindowCount();
            double checksum = 0.0;
            long measuredStarted = System.nanoTime();
            for (int task = 0; task < options.tasks; task++) {
                long taskStarted = System.nanoTime();
                checksum += runTask(pipeline, task, options, 0);
                durationsNanos[task] = System.nanoTime() - taskStarted;
            }
            long measuredNanos = System.nanoTime() - measuredStarted;
            long newlyComputedWindows = pipeline.getTotalComputedWindowCount() - windowsBefore;

            Report report = new Report(
                    requestedBackend,
                    OnnxModel.getResolvedInferenceProvider(),
                    options,
                    modelLoadMillis,
                    durationsNanos,
                    measuredNanos,
                    newlyComputedWindows,
                    checksum);
            report.print();
            if (options.output != null) {
                report.writeJson(options.output);
                System.out.println("Wrote JSON report: " + options.output.toAbsolutePath());
            }
        }
    }

    private static double runTask(WorldPipeline pipeline, int task, Options options, int taskOffset) {
        long index = (long) task + taskOffset;
        long coordinate = index * (options.regionSize + REGION_GAP);
        int i1 = Math.toIntExact(coordinate);
        int j1 = Math.toIntExact(coordinate + 100_000L);
        float[][] result = pipeline.get(i1, j1, i1 + options.regionSize, j1 + options.regionSize, options.withClimate);
        // Read values so benchmark results cannot be optimized into an unused computation.
        double checksum = result[0][0] + result[0][result[0].length - 1];
        if (result[1] != null) {
            checksum += result[1][0] + result[1][result[1].length - 1];
        }
        return checksum;
    }

    private static void printUsage() {
        System.out.println("Usage: PipelineBenchmark [--backend=cpu|cuda|directml|coreml|gpu|auto]"
                + " [--tasks=N] [--warmup=N] [--region-size=N] [--seed=N]"
                + " [--climate=true|false] [--output=report.json]");
    }

    private static double nanosToMillis(long nanos) {
        return nanos / 1_000_000.0;
    }

    private record Options(String backend, int tasks, int warmup, int regionSize, long seed,
                           boolean withClimate, Path output, boolean help) {
        private static Options parse(String[] args) {
            String backend = null;
            int tasks = DEFAULT_TASKS;
            int warmup = DEFAULT_WARMUP;
            int regionSize = DEFAULT_REGION_SIZE;
            long seed = DEFAULT_SEED;
            boolean withClimate = true;
            Path output = null;
            boolean help = false;

            for (String arg : args) {
                if ("--help".equals(arg) || "-h".equals(arg)) {
                    help = true;
                    continue;
                }
                int equals = arg.indexOf('=');
                if (!arg.startsWith("--") || equals < 3) {
                    throw new IllegalArgumentException("Invalid argument: " + arg);
                }
                String key = arg.substring(2, equals);
                String value = arg.substring(equals + 1);
                switch (key) {
                    case "backend" -> backend = value;
                    case "tasks" -> tasks = parsePositiveInt(key, value);
                    case "warmup" -> warmup = parseNonNegativeInt(key, value);
                    case "region-size" -> regionSize = parsePositiveInt(key, value);
                    case "seed" -> seed = Long.parseLong(value);
                    case "climate" -> withClimate = parseBoolean(key, value);
                    case "output" -> output = Path.of(value);
                    default -> throw new IllegalArgumentException("Unknown argument: --" + key);
                }
            }
            return new Options(backend, tasks, warmup, regionSize, seed, withClimate, output, help);
        }

        private static int parsePositiveInt(String key, String value) {
            int parsed = Integer.parseInt(value);
            if (parsed <= 0) throw new IllegalArgumentException("--" + key + " must be greater than zero");
            return parsed;
        }

        private static int parseNonNegativeInt(String key, String value) {
            int parsed = Integer.parseInt(value);
            if (parsed < 0) throw new IllegalArgumentException("--" + key + " must not be negative");
            return parsed;
        }

        private static boolean parseBoolean(String key, String value) {
            if ("true".equalsIgnoreCase(value) || "false".equalsIgnoreCase(value)) {
                return Boolean.parseBoolean(value);
            }
            throw new IllegalArgumentException("--" + key + " must be true or false");
        }
    }

    private record Report(String requestedBackend, String resolvedProvider, Options options,
                          double modelLoadMillis, long[] durationsNanos, long measuredNanos,
                          long newlyComputedWindows, double checksum) {
        private void print() {
            long[] sorted = durationsNanos.clone();
            Arrays.sort(sorted);
            double totalMillis = nanosToMillis(measuredNanos);
            double meanMillis = totalMillis / durationsNanos.length;
            double tasksPerSecond = durationsNanos.length / (measuredNanos / 1_000_000_000.0);
            double pixelsPerSecond = tasksPerSecond * options.regionSize * (double) options.regionSize;

            System.out.println("=== Terrain Diffusion Benchmark ===");
            System.out.printf(Locale.ROOT, "Requested backend:       %s%n", requestedBackend);
            System.out.printf(Locale.ROOT, "Resolved provider:       %s%n", resolvedProvider);
            System.out.printf(Locale.ROOT, "Model load:              %.2f ms%n", modelLoadMillis);
            System.out.printf(Locale.ROOT, "Measured tasks:          %d%n", durationsNanos.length);
            System.out.printf(Locale.ROOT, "New tensor windows:      %d%n", newlyComputedWindows);
            System.out.printf(Locale.ROOT, "Total inference time:    %.2f ms%n", totalMillis);
            System.out.printf(Locale.ROOT, "Latency (min/mean/p50/p95/max): %.2f / %.2f / %.2f / %.2f / %.2f ms%n",
                    nanosToMillis(sorted[0]), meanMillis, nanosToMillis(percentile(sorted, 0.50)),
                    nanosToMillis(percentile(sorted, 0.95)), nanosToMillis(sorted[sorted.length - 1]));
            System.out.printf(Locale.ROOT, "Throughput:              %.4f tasks/s (%.2f native pixels/s)%n",
                    tasksPerSecond, pixelsPerSecond);
            System.out.printf(Locale.ROOT, "Result checksum:         %.6f%n", checksum);
        }

        private void writeJson(Path output) throws IOException {
            Path parent = output.toAbsolutePath().getParent();
            if (parent != null) Files.createDirectories(parent);
            long[] sorted = durationsNanos.clone();
            Arrays.sort(sorted);
            double totalMillis = nanosToMillis(measuredNanos);
            String json = String.format(Locale.ROOT, """
                    {
                      "requestedBackend": "%s",
                      "resolvedProvider": "%s",
                      "tasks": %d,
                      "warmupTasks": %d,
                      "regionSize": %d,
                      "withClimate": %s,
                      "modelLoadMillis": %.6f,
                      "totalInferenceMillis": %.6f,
                      "minLatencyMillis": %.6f,
                      "meanLatencyMillis": %.6f,
                      "p50LatencyMillis": %.6f,
                      "p95LatencyMillis": %.6f,
                      "maxLatencyMillis": %.6f,
                      "newTensorWindows": %d,
                      "checksum": %.6f
                    }
                    """, requestedBackend, resolvedProvider, durationsNanos.length, options.warmup,
                    options.regionSize, options.withClimate, modelLoadMillis, totalMillis,
                    nanosToMillis(sorted[0]), totalMillis / durationsNanos.length,
                    nanosToMillis(percentile(sorted, 0.50)), nanosToMillis(percentile(sorted, 0.95)),
                    nanosToMillis(sorted[sorted.length - 1]), newlyComputedWindows, checksum);
            Files.writeString(output, json);
        }

        private static long percentile(long[] sorted, double percentile) {
            int index = Math.max(0, (int) Math.ceil(percentile * sorted.length) - 1);
            return sorted[index];
        }
    }
}
