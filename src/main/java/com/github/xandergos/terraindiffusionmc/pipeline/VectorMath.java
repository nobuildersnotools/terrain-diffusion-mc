package com.github.xandergos.terraindiffusionmc.pipeline;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/** SIMD implementations of the pipeline's contiguous element-wise float operations. */
final class VectorMath {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private VectorMath() {
    }

    static void scaleInPlace(float[] values, float scale) {
        int i = 0;
        int upperBound = SPECIES.loopBound(values.length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, values, i)
                    .mul(scale)
                    .intoArray(values, i);
        }
        for (; i < values.length; i++) values[i] *= scale;
    }

    static void scale(float[] source, int sourceOffset, float scale,
                      float[] destination, int destinationOffset, int length) {
        int i = 0;
        int upperBound = SPECIES.loopBound(length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, source, sourceOffset + i)
                    .mul(scale)
                    .intoArray(destination, destinationOffset + i);
        }
        for (; i < length; i++) destination[destinationOffset + i] = source[sourceOffset + i] * scale;
    }

    static void scaleTwice(float[] source, int sourceOffset, float firstScale, float secondScale,
                           float[] destination, int destinationOffset, int length) {
        int i = 0;
        int upperBound = SPECIES.loopBound(length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, source, sourceOffset + i)
                    .mul(firstScale)
                    .mul(secondScale)
                    .intoArray(destination, destinationOffset + i);
        }
        for (; i < length; i++) {
            destination[destinationOffset + i] = source[sourceOffset + i] * firstScale * secondScale;
        }
    }

    static void normalize(float[] source, int sourceOffset, float mean, float standardDeviation,
                          float[] destination, int destinationOffset, int length) {
        int i = 0;
        int upperBound = SPECIES.loopBound(length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, source, sourceOffset + i)
                    .sub(mean)
                    .div(standardDeviation)
                    .intoArray(destination, destinationOffset + i);
        }
        for (; i < length; i++) {
            destination[destinationOffset + i] = (source[sourceOffset + i] - mean) / standardDeviation;
        }
    }

    static void normalizeNaNToZero(float[] source, int sourceOffset, float mean, float standardDeviation,
                                   float[] destination, int destinationOffset, int length) {
        int i = 0;
        int upperBound = SPECIES.loopBound(length);
        FloatVector zero = FloatVector.zero(SPECIES);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector normalized = FloatVector.fromArray(SPECIES, source, sourceOffset + i)
                    .sub(mean)
                    .div(standardDeviation);
            zero.blend(normalized, normalized.compare(VectorOperators.EQ, normalized))
                    .intoArray(destination, destinationOffset + i);
        }
        for (; i < length; i++) {
            float normalized = (source[sourceOffset + i] - mean) / standardDeviation;
            destination[destinationOffset + i] = Float.isNaN(normalized) ? 0.0f : normalized;
        }
    }

    static void divide(float[] source, int sourceOffset, float divisor,
                       float[] destination, int destinationOffset, int length) {
        int i = 0;
        int upperBound = SPECIES.loopBound(length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, source, sourceOffset + i)
                    .div(divisor)
                    .intoArray(destination, destinationOffset + i);
        }
        for (; i < length; i++) destination[destinationOffset + i] = source[sourceOffset + i] / divisor;
    }

    static void multiply(float[] left, int leftOffset, float[] right, int rightOffset,
                         float[] destination, int destinationOffset, int length) {
        int i = 0;
        int upperBound = SPECIES.loopBound(length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, left, leftOffset + i)
                    .mul(FloatVector.fromArray(SPECIES, right, rightOffset + i))
                    .intoArray(destination, destinationOffset + i);
        }
        for (; i < length; i++) {
            destination[destinationOffset + i] = left[leftOffset + i] * right[rightOffset + i];
        }
    }

    static void add(float[] left, float[] right, float[] destination, int length) {
        int i = 0;
        int upperBound = SPECIES.loopBound(length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, left, i)
                    .add(FloatVector.fromArray(SPECIES, right, i))
                    .intoArray(destination, i);
        }
        for (; i < length; i++) destination[i] = left[i] + right[i];
    }

    static void subtract(float[] left, int leftOffset, float[] right, int rightOffset,
                         float[] destination, int destinationOffset, int length) {
        int i = 0;
        int upperBound = SPECIES.loopBound(length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, left, leftOffset + i)
                    .sub(FloatVector.fromArray(SPECIES, right, rightOffset + i))
                    .intoArray(destination, destinationOffset + i);
        }
        for (; i < length; i++) {
            destination[destinationOffset + i] = left[leftOffset + i] - right[rightOffset + i];
        }
    }

    static void linearCombination(float[] left, int leftOffset, float leftScale,
                                  float[] right, int rightOffset, float rightScale,
                                  float[] destination, int destinationOffset, int length) {
        int i = 0;
        int upperBound = SPECIES.loopBound(length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector leftPart = FloatVector.fromArray(SPECIES, left, leftOffset + i).mul(leftScale);
            FloatVector rightPart = FloatVector.fromArray(SPECIES, right, rightOffset + i).mul(rightScale);
            leftPart.add(rightPart).intoArray(destination, destinationOffset + i);
        }
        for (; i < length; i++) {
            destination[destinationOffset + i] = leftScale * left[leftOffset + i]
                    + rightScale * right[rightOffset + i];
        }
    }

    static void scaledDifference(float[] left, int leftOffset, float leftScale,
                                 float[] right, int rightOffset, float rightScale,
                                 float[] destination, int destinationOffset, int length) {
        int i = 0;
        int upperBound = SPECIES.loopBound(length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector leftPart = FloatVector.fromArray(SPECIES, left, leftOffset + i).mul(leftScale);
            FloatVector rightPart = FloatVector.fromArray(SPECIES, right, rightOffset + i).mul(rightScale);
            leftPart.sub(rightPart).intoArray(destination, destinationOffset + i);
        }
        for (; i < length; i++) {
            destination[destinationOffset + i] = leftScale * left[leftOffset + i]
                    - rightScale * right[rightOffset + i];
        }
    }

    static void secondOrder(float[] previousModelOutput, float[] modelOutput, float[] sample,
                            float r0, float sampleScale, float modelScale, float derivativeScale,
                            float[] destination) {
        int i = 0;
        int upperBound = SPECIES.loopBound(sample.length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector previous = FloatVector.fromArray(SPECIES, previousModelOutput, i);
            FloatVector model = FloatVector.fromArray(SPECIES, modelOutput, i);
            FloatVector derivative = model.sub(previous).div(r0);
            FloatVector result = FloatVector.fromArray(SPECIES, sample, i).mul(sampleScale)
                    .add(model.mul(modelScale))
                    .add(derivative.mul(derivativeScale));
            result.intoArray(destination, i);
        }
        for (; i < sample.length; i++) {
            float derivative = (modelOutput[i] - previousModelOutput[i]) / r0;
            destination[i] = sampleScale * sample[i] + modelScale * modelOutput[i]
                    + derivativeScale * derivative;
        }
    }

    static void affineAfterDivide(float[] source, int sourceOffset, float divisor,
                                  float scale, float offset, float[] destination,
                                  int destinationOffset, int length) {
        int i = 0;
        int upperBound = SPECIES.loopBound(length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, source, sourceOffset + i)
                    .div(divisor)
                    .mul(scale)
                    .add(offset)
                    .intoArray(destination, destinationOffset + i);
        }
        for (; i < length; i++) {
            destination[destinationOffset + i] = (source[sourceOffset + i] / divisor) * scale + offset;
        }
    }

    static void divideMultiplyWhereGreater(float[] numerator, int numeratorOffset,
                                           float[] weights, int weightsOffset, float threshold, float scale,
                                           float[] destination, int destinationOffset, int length) {
        int i = 0;
        int upperBound = SPECIES.loopBound(length);
        FloatVector zero = FloatVector.zero(SPECIES);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector weight = FloatVector.fromArray(SPECIES, weights, weightsOffset + i);
            FloatVector value = FloatVector.fromArray(SPECIES, numerator, numeratorOffset + i)
                    .div(weight)
                    .mul(scale);
            zero.blend(value, weight.compare(VectorOperators.GT, threshold))
                    .intoArray(destination, destinationOffset + i);
        }
        for (; i < length; i++) {
            float weight = weights[weightsOffset + i];
            destination[destinationOffset + i] = weight > threshold
                    ? numerator[numeratorOffset + i] / weight * scale
                    : 0.0f;
        }
    }

    static void divideAffineWhereGreater(float[] numerator, int numeratorOffset,
                                         float[] weights, int weightsOffset, float threshold,
                                         float scale, float offset,
                                         float[] destination, int destinationOffset, int length) {
        int i = 0;
        int upperBound = SPECIES.loopBound(length);
        FloatVector zero = FloatVector.zero(SPECIES);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector weight = FloatVector.fromArray(SPECIES, weights, weightsOffset + i);
            FloatVector normalized = FloatVector.fromArray(SPECIES, numerator, numeratorOffset + i).div(weight);
            FloatVector selected = zero.blend(normalized, weight.compare(VectorOperators.GT, threshold));
            selected.mul(scale).add(offset).intoArray(destination, destinationOffset + i);
        }
        for (; i < length; i++) {
            float weight = weights[weightsOffset + i];
            float normalized = weight > threshold ? numerator[numeratorOffset + i] / weight : 0.0f;
            destination[destinationOffset + i] = normalized * scale + offset;
        }
    }

    static void adjustAtMostInPlace(float[] values, float threshold, float scale) {
        int i = 0;
        int upperBound = SPECIES.loopBound(values.length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector value = FloatVector.fromArray(SPECIES, values, i);
            FloatVector adjusted = value.sub(threshold).mul(scale).add(threshold);
            value.blend(adjusted, value.compare(VectorOperators.LE, threshold)).intoArray(values, i);
        }
        for (; i < values.length; i++) {
            float value = values[i];
            if (value <= threshold) values[i] = (value - threshold) * scale + threshold;
        }
    }

    static void noiseMix(float[] sample, float[] noise, float sampleScale, float noiseScale,
                         float noiseMagnitude, float[] destination) {
        int i = 0;
        int upperBound = SPECIES.loopBound(sample.length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector samplePart = FloatVector.fromArray(SPECIES, sample, i).mul(sampleScale);
            FloatVector noisePart = FloatVector.fromArray(SPECIES, noise, i)
                    .mul(noiseMagnitude)
                    .mul(noiseScale);
            samplePart.add(noisePart).intoArray(destination, i);
        }
        for (; i < sample.length; i++) {
            float scaledNoise = noise[i] * noiseMagnitude;
            destination[i] = sampleScale * sample[i] + noiseScale * scaledNoise;
        }
    }

    static void flowDenoise(float[] state, int stateOffset, float[] rawPrediction, int predictionOffset,
                            float stateScale, float predictionScale, float divisor,
                            float[] destination, int destinationOffset, int length) {
        int i = 0;
        int upperBound = SPECIES.loopBound(length);
        for (; i < upperBound; i += SPECIES.length()) {
            FloatVector statePart = FloatVector.fromArray(SPECIES, state, stateOffset + i).mul(stateScale);
            FloatVector prediction = FloatVector.fromArray(SPECIES, rawPrediction, predictionOffset + i).neg();
            statePart.sub(prediction.mul(predictionScale))
                    .div(divisor)
                    .intoArray(destination, destinationOffset + i);
        }
        for (; i < length; i++) {
            float prediction = -rawPrediction[predictionOffset + i];
            destination[destinationOffset + i] = (stateScale * state[stateOffset + i]
                    - predictionScale * prediction) / divisor;
        }
    }

    static void horizontalConvolution(float[] source, float[] kernel, int padding, float[] destination) {
        int width = source.length;
        int interiorStart = padding;
        int interiorEnd = width - padding;

        if (interiorStart >= interiorEnd) {
            for (int c = 0; c < width; c++) {
                float sum = 0.0f;
                for (int ki = 0; ki < kernel.length; ki++) {
                    int sourceColumn = Math.max(0, Math.min(width - 1, c + ki - padding));
                    sum += source[sourceColumn] * kernel[ki];
                }
                destination[c] = sum;
            }
            return;
        }

        int c = 0;
        for (; c < Math.min(interiorStart, width); c++) {
            float sum = 0.0f;
            for (int ki = 0; ki < kernel.length; ki++) {
                int sourceColumn = Math.max(0, Math.min(width - 1, c + ki - padding));
                sum += source[sourceColumn] * kernel[ki];
            }
            destination[c] = sum;
        }

        int vectorLength = SPECIES.loopBound(Math.max(0, interiorEnd - interiorStart));
        int vectorEnd = interiorStart + vectorLength;
        for (; c < vectorEnd; c += SPECIES.length()) {
            FloatVector sum = FloatVector.zero(SPECIES);
            for (int ki = 0; ki < kernel.length; ki++) {
                FloatVector values = FloatVector.fromArray(SPECIES, source, c + ki - padding);
                sum = sum.add(values.mul(kernel[ki]));
            }
            sum.intoArray(destination, c);
        }

        for (; c < width; c++) {
            float sum = 0.0f;
            for (int ki = 0; ki < kernel.length; ki++) {
                int sourceColumn = Math.max(0, Math.min(width - 1, c + ki - padding));
                sum += source[sourceColumn] * kernel[ki];
            }
            destination[c] = sum;
        }
    }

    static void verticalConvolution(float[][] source, float[] kernel, int row, int padding,
                                    float[] destination) {
        int height = source.length;
        int width = source[0].length;
        int c = 0;
        int upperBound = SPECIES.loopBound(width);
        for (; c < upperBound; c += SPECIES.length()) {
            FloatVector sum = FloatVector.zero(SPECIES);
            for (int ki = 0; ki < kernel.length; ki++) {
                int sourceRow = Math.max(0, Math.min(height - 1, row + ki - padding));
                FloatVector values = FloatVector.fromArray(SPECIES, source[sourceRow], c);
                sum = sum.add(values.mul(kernel[ki]));
            }
            sum.intoArray(destination, c);
        }
        for (; c < width; c++) {
            float sum = 0.0f;
            for (int ki = 0; ki < kernel.length; ki++) {
                int sourceRow = Math.max(0, Math.min(height - 1, row + ki - padding));
                sum += source[sourceRow][c] * kernel[ki];
            }
            destination[c] = sum;
        }
    }
}
