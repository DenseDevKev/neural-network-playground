import { describe, expect, it } from 'vitest';
import {
    applyGradientTransformInto,
    binaryCrossEntropyLogitDelta,
    binaryCrossEntropyWithLogits,
    buildObjectiveBreakdown,
    categoricalCrossEntropyLogitDelta,
    categoricalCrossEntropyWithLogits,
    compileObjective,
    computeGradientTransform,
    gradientNorm,
    huberLoss,
    huberLossDelta,
    l1Penalty,
    l1PenaltyGradient,
    l2Penalty,
    l2PenaltyGradient,
    meanSquaredError,
    meanSquaredErrorDelta,
} from '../objective.js';
import type { ObjectiveSpecV2 } from '../types.js';

const finiteDifference = (fn: (value: number) => number, value: number): number => {
    const epsilon = 1e-6;
    return (fn(value + epsilon) - fn(value - epsilon)) / (2 * epsilon);
};

describe('binary cross-entropy from logits', () => {
    it('stays finite and correct for extreme logits', () => {
        expect(binaryCrossEntropyWithLogits(1_000, 0)).toBeCloseTo(1_000, 12);
        expect(binaryCrossEntropyWithLogits(-1_000, 1)).toBeCloseTo(1_000, 12);
        expect(binaryCrossEntropyWithLogits(1_000, 1)).toBe(0);
        expect(binaryCrossEntropyWithLogits(-1_000, 0)).toBe(0);

        expect(binaryCrossEntropyLogitDelta(1_000, 0)).toBeCloseTo(1, 12);
        expect(binaryCrossEntropyLogitDelta(-1_000, 1)).toBeCloseTo(-1, 12);
        expect(Number.isFinite(binaryCrossEntropyLogitDelta(1_000, 0))).toBe(true);
        expect(Number.isFinite(binaryCrossEntropyLogitDelta(-1_000, 1))).toBe(true);
    });

    it('matches a finite-difference derivative at an ordinary logit', () => {
        const logit = 0.37;
        const target = 1;
        const numerical = finiteDifference(
            (candidate) => binaryCrossEntropyWithLogits(candidate, target),
            logit,
        );
        expect(binaryCrossEntropyLogitDelta(logit, target)).toBeCloseTo(numerical, 8);
    });

    it('rejects non-finite logits and non-binary targets', () => {
        expect(() => binaryCrossEntropyWithLogits(Number.NaN, 0)).toThrow(RangeError);
        expect(() => binaryCrossEntropyWithLogits(0, 0.5)).toThrow(RangeError);
        expect(() => binaryCrossEntropyLogitDelta(0, -1)).toThrow(RangeError);
    });
});

describe('categorical cross-entropy from logits', () => {
    it('uses stable log-sum-exp at extreme logits', () => {
        expect(categoricalCrossEntropyWithLogits([1_000, -1_000], [0, 1])).toBeCloseTo(2_000, 12);
        expect(categoricalCrossEntropyWithLogits([1_000, -1_000], [1, 0])).toBe(0);
        expect(categoricalCrossEntropyLogitDelta([1_000, -1_000], [0, 1])).toEqual([1, -1]);
    });

    it('matches finite differences for normalized soft targets', () => {
        const logits = [0.3, -0.2, 1.1];
        const target = [0.2, 0.3, 0.5];
        const analytical = categoricalCrossEntropyLogitDelta(logits, target);

        for (let i = 0; i < logits.length; i++) {
            const numerical = finiteDifference((candidate) => {
                const perturbed = [...logits];
                perturbed[i] = candidate;
                return categoricalCrossEntropyWithLogits(perturbed, target);
            }, logits[i]);
            expect(analytical[i]).toBeCloseTo(numerical, 8);
        }
    });

    it('validates vector shape, finiteness, and target normalization', () => {
        expect(() => categoricalCrossEntropyWithLogits([], [])).toThrow(RangeError);
        expect(() => categoricalCrossEntropyWithLogits([0], [1, 0])).toThrow(RangeError);
        expect(() => categoricalCrossEntropyWithLogits([Number.POSITIVE_INFINITY], [1])).toThrow(RangeError);
        expect(() => categoricalCrossEntropyWithLogits([0, 1], [0.2, 0.2])).toThrow(RangeError);
        expect(() => categoricalCrossEntropyLogitDelta([0, 1], [-1, 2])).toThrow(RangeError);
    });
});

describe('mean squared error', () => {
    it('is true mean squared error with its complete derivative', () => {
        expect(meanSquaredError([2], [0])).toBe(4);
        expect(meanSquaredErrorDelta([2], [0])).toEqual([4]);
        expect(meanSquaredError([2, -1], [0, 1])).toBe(4);
        expect(meanSquaredErrorDelta([2, -1], [0, 1])).toEqual([2, -2]);
    });

    it('matches finite differences for every output', () => {
        const prediction = [0.4, -0.25, 1.2];
        const target = [-0.1, 0.5, 0.7];
        const analytical = meanSquaredErrorDelta(prediction, target);

        for (let i = 0; i < prediction.length; i++) {
            const numerical = finiteDifference((candidate) => {
                const perturbed = [...prediction];
                perturbed[i] = candidate;
                return meanSquaredError(perturbed, target);
            }, prediction[i]);
            expect(analytical[i]).toBeCloseTo(numerical, 9);
        }
    });

    it('rejects empty, mismatched, and non-finite vectors', () => {
        expect(() => meanSquaredError([], [])).toThrow(RangeError);
        expect(() => meanSquaredError([0], [0, 1])).toThrow(RangeError);
        expect(() => meanSquaredErrorDelta([Number.NaN], [0])).toThrow(RangeError);
    });
});

describe('Huber loss', () => {
    it('uses the conventional formula below, at, and above delta', () => {
        expect(huberLoss([0.5], [0], 1)).toBe(0.125);
        expect(huberLoss([1], [0], 1)).toBe(0.5);
        expect(huberLoss([2], [0], 1)).toBe(1.5);

        expect(huberLossDelta([0.5], [0], 1)).toEqual([0.5]);
        expect(huberLossDelta([1], [0], 1)).toEqual([1]);
        expect(huberLossDelta([2], [0], 1)).toEqual([1]);
        expect(huberLossDelta([-2], [0], 1)).toEqual([-1]);
    });

    it('averages values and derivatives across outputs and matches finite differences', () => {
        const prediction = [0.25, 2];
        const target = [0, 0];
        expect(huberLoss(prediction, target, 1)).toBe((0.03125 + 1.5) / 2);
        expect(huberLossDelta(prediction, target, 1)).toEqual([0.125, 0.5]);

        const analytical = huberLossDelta(prediction, target, 1);
        for (let i = 0; i < prediction.length; i++) {
            const numerical = finiteDifference((candidate) => {
                const perturbed = [...prediction];
                perturbed[i] = candidate;
                return huberLoss(perturbed, target, 1);
            }, prediction[i]);
            expect(analytical[i]).toBeCloseTo(numerical, 9);
        }
    });

    it('rejects invalid vectors and deltas', () => {
        expect(() => huberLoss([], [], 1)).toThrow(RangeError);
        expect(() => huberLoss([0], [0, 1], 1)).toThrow(RangeError);
        for (const delta of [0, -1, 1_000_001, Number.POSITIVE_INFINITY, Number.NaN]) {
            expect(() => huberLoss([0], [0], delta)).toThrow(RangeError);
            expect(() => huberLossDelta([0], [0], delta)).toThrow(RangeError);
        }
    });
});

describe('weight-only penalties', () => {
    it('implements the exact L1 value and gradient', () => {
        expect(l1Penalty([-2, 0, 3], 0.5)).toBe(2.5);
        expect(l1PenaltyGradient([-2, 0, 3], 0.5)).toEqual([-0.5, 0, 0.5]);
    });

    it('implements the exact L2 value and gradient', () => {
        expect(l2Penalty([2], 0.5)).toBe(1);
        expect(l2PenaltyGradient([2], 0.5)).toEqual([1]);

        const weight = 0.7;
        const numerical = finiteDifference((candidate) => l2Penalty([candidate], 0.5), weight);
        expect(l2PenaltyGradient([weight], 0.5)[0]).toBeCloseTo(numerical, 9);
    });

    it('validates coefficients and finite weights', () => {
        for (const coefficient of [0, -1, 1.1, Number.NaN, Number.POSITIVE_INFINITY]) {
            expect(() => l1Penalty([1], coefficient)).toThrow(RangeError);
            expect(() => l2PenaltyGradient([1], coefficient)).toThrow(RangeError);
        }
        expect(() => l1Penalty([Number.NaN], 0.5)).toThrow(RangeError);
        expect(() => l2PenaltyGradient([Number.POSITIVE_INFINITY], 0.5)).toThrow(RangeError);
    });
});

describe('compiled objective', () => {
    const mseWithL2: ObjectiveSpecV2 = {
        dataLoss: { kind: 'mean-squared-error' },
        penalty: { kind: 'l2', coefficient: 0.5, applyTo: 'weights' },
        reduction: 'mean-per-sample',
    };

    it('evaluates and seeds every approved data-loss kind', () => {
        const cases: Array<{
            spec: ObjectiveSpecV2;
            network: { outputSize: number; outputActivation: 'sigmoid' | 'softmax' | 'linear' };
            logits: number[];
            outputs: number[];
            target: number[];
            expectedLoss: number;
            expectedDelta: number[];
        }> = [
            {
                spec: {
                    dataLoss: { kind: 'binary-cross-entropy-with-logits' },
                    penalty: { kind: 'none' },
                    reduction: 'mean-per-sample',
                },
                network: { outputSize: 1, outputActivation: 'sigmoid' },
                logits: [1_000],
                outputs: [1],
                target: [0],
                expectedLoss: 1_000,
                expectedDelta: [1],
            },
            {
                spec: {
                    dataLoss: { kind: 'categorical-cross-entropy-with-logits' },
                    penalty: { kind: 'none' },
                    reduction: 'mean-per-sample',
                },
                network: { outputSize: 3, outputActivation: 'softmax' },
                logits: [1_000, -1_000, -1_000],
                outputs: [1, 0, 0],
                target: [0, 1, 0],
                expectedLoss: 2_000,
                expectedDelta: [1, -1, 0],
            },
            {
                spec: mseWithL2,
                network: { outputSize: 1, outputActivation: 'linear' },
                logits: [2],
                outputs: [2],
                target: [0],
                expectedLoss: 4,
                expectedDelta: [4],
            },
            {
                spec: {
                    dataLoss: { kind: 'huber', delta: 1 },
                    penalty: { kind: 'none' },
                    reduction: 'mean-per-sample',
                },
                network: { outputSize: 1, outputActivation: 'linear' },
                logits: [2],
                outputs: [2],
                target: [0],
                expectedLoss: 1.5,
                expectedDelta: [1],
            },
        ];

        for (const testCase of cases) {
            const objective = compileObjective(testCase.spec, testCase.network);
            expect(objective.evaluateDataSample(
                testCase.logits,
                testCase.outputs,
                testCase.target,
            )).toBe(testCase.expectedLoss);
            const destination = new Float64Array(testCase.network.outputSize);
            objective.seedOutputDeltaInto(
                testCase.logits,
                testCase.outputs,
                testCase.target,
                destination,
            );
            expect(Array.from(destination)).toEqual(testCase.expectedDelta);
        }
    });

    it('adds L2 gradients to weights only and returns the penalty-gradient norm', () => {
        const objective = compileObjective(mseWithL2, { outputSize: 1, outputActivation: 'linear' });
        const weights = [new Float64Array([2]), new Float64Array([-3, 0])];
        const weightGradients = [new Float64Array([0]), new Float64Array([1, 0])];
        const biasGradients = [new Float64Array([7]), new Float64Array([-11, 13])];

        expect(objective.regularizationPenalty(weights)).toBe(3.25);
        expect(objective.addPenaltyGradientInto(weights, weightGradients)).toBeCloseTo(
            Math.hypot(1, -1.5, 0),
            12,
        );
        expect(Array.from(weightGradients[0])).toEqual([1]);
        expect(Array.from(weightGradients[1])).toEqual([-0.5, 0]);
        expect(biasGradients.map((values) => Array.from(values))).toEqual([[7], [-11, 13]]);
    });

    it('builds an explicit total-objective breakdown', () => {
        expect(buildObjectiveBreakdown(2.5, 0.75)).toEqual({
            dataLoss: 2.5,
            regularizationPenalty: 0.75,
            totalObjective: 3.25,
        });
    });

    it('rejects incompatible output contracts and malformed objective specs', () => {
        const bce: ObjectiveSpecV2 = {
            dataLoss: { kind: 'binary-cross-entropy-with-logits' },
            penalty: { kind: 'none' },
            reduction: 'mean-per-sample',
        };
        expect(() => compileObjective(bce, { outputSize: 2, outputActivation: 'sigmoid' })).toThrow(RangeError);
        expect(() => compileObjective(bce, { outputSize: 1, outputActivation: 'linear' })).toThrow(RangeError);
        expect(() => compileObjective({ ...bce, reduction: 'sum' as never }, {
            outputSize: 1,
            outputActivation: 'sigmoid',
        })).toThrow(RangeError);
        expect(() => compileObjective({
            dataLoss: { kind: 'huber', delta: 1_000_001 },
            penalty: { kind: 'none' },
            reduction: 'mean-per-sample',
        }, { outputSize: 1, outputActivation: 'linear' })).toThrow(RangeError);
    });
});

describe('complete-gradient transform', () => {
    it('computes global norms and pure clip decisions', () => {
        expect(gradientNorm([new Float64Array([3])], [new Float64Array([4])])).toBe(5);
        expect(computeGradientTransform(5, { kind: 'none' })).toEqual({
            totalGradientNorm: 5,
            clippedGradientNorm: 5,
            clipScale: 1,
        });
        expect(computeGradientTransform(5, {
            kind: 'global-norm',
            maximumNorm: 1,
            scope: 'total-objective-gradient',
        })).toEqual({
            totalGradientNorm: 5,
            clippedGradientNorm: 1,
            clipScale: 0.2,
        });
    });

    it('clips a zero-data-gradient, large-L2 complete objective gradient', () => {
        const objective = compileObjective({
            dataLoss: { kind: 'mean-squared-error' },
            penalty: { kind: 'l2', coefficient: 1, applyTo: 'weights' },
            reduction: 'mean-per-sample',
        }, { outputSize: 1, outputActivation: 'linear' });
        const weights = [new Float64Array([100])];
        const weightGradients = [new Float64Array([0])];
        const biasGradients = [new Float64Array([0])];
        const penaltyGradientNorm = objective.addPenaltyGradientInto(weights, weightGradients);

        const diagnostics = applyGradientTransformInto(
            weightGradients,
            biasGradients,
            0,
            penaltyGradientNorm,
            { kind: 'global-norm', maximumNorm: 0.01, scope: 'total-objective-gradient' },
        );

        expect(diagnostics).toEqual({
            dataGradientNorm: 0,
            penaltyGradientNorm: 100,
            totalGradientNorm: 100,
            clippedGradientNorm: 0.01,
            clipScale: 0.0001,
        });
        expect(weightGradients[0][0]).toBeCloseTo(0.01, 12);
        expect(biasGradients[0][0]).toBe(0);
    });

    it('applies one scale to all weight and bias gradients', () => {
        const weightGradients = [new Float64Array([3])];
        const biasGradients = [new Float64Array([4])];
        const diagnostics = applyGradientTransformInto(
            weightGradients,
            biasGradients,
            5,
            0,
            { kind: 'global-norm', maximumNorm: 1, scope: 'total-objective-gradient' },
        );
        expect(weightGradients[0][0]).toBeCloseTo(0.6, 12);
        expect(biasGradients[0][0]).toBeCloseTo(0.8, 12);
        expect(diagnostics.clippedGradientNorm).toBe(1);
    });

    it('validates diagnostics and clip specifications before mutation', () => {
        const weights = [new Float64Array([3])];
        const biases = [new Float64Array([4])];
        expect(() => computeGradientTransform(Number.NaN, { kind: 'none' })).toThrow(RangeError);
        expect(() => computeGradientTransform(5, {
            kind: 'global-norm',
            maximumNorm: 0,
            scope: 'total-objective-gradient',
        })).toThrow(RangeError);
        expect(() => applyGradientTransformInto(
            weights,
            biases,
            -1,
            0,
            { kind: 'none' },
        )).toThrow(RangeError);
        expect(weights[0][0]).toBe(3);
        expect(biases[0][0]).toBe(4);
    });
});
