import { describe, expect, it } from 'vitest';
import type { NetworkSnapshot } from '@nn-playground/engine';
import {
    captureExperimentRun,
    createSerializedNetworkFromFrameBuffer,
    historyArraysToPoints,
} from './experimentRunCapture.ts';
import {
    resetFrameBuffer,
    updateFrameBuffer,
} from '../worker/frameBuffer.ts';

const snapshot: NetworkSnapshot = {
    step: 4,
    epoch: 1,
    weights: [[[0.5, -0.25]]],
    biases: [[0.1]],
    trainLoss: 0.2,
    testLoss: 0.3,
    trainMetrics: { loss: 0.2, accuracy: 0.9 },
    testMetrics: { loss: 0.3, accuracy: 0.8 },
    outputGrid: [],
    gridSize: 50,
    historyPoint: { step: 4, trainLoss: 0.2, testLoss: 0.3, trainAccuracy: 0.9, testAccuracy: 0.8 },
};

const config = {
    data: { dataset: 'circle' as const, problemType: 'classification' as const, trainTestRatio: 0.5, noise: 0, numSamples: 200, seed: 42 },
    network: { inputSize: 2, hiddenLayers: [], outputSize: 1, activation: 'tanh' as const, outputActivation: 'sigmoid' as const, weightInit: 'xavier' as const, seed: 42 },
    training: { learningRate: 0.03, batchSize: 10, lossType: 'crossEntropy' as const, optimizer: 'sgd' as const, momentum: 0.9, regularization: 'none' as const, regularizationRate: 0, gradientClip: null },
    features: { x: true, y: true, xSquared: false, ySquared: false, xy: false, sinX: false, sinY: false, cosX: false, cosY: false },
    ui: { showTestData: false, discretizeOutput: false },
};

describe('experiment run capture', () => {
    it('converts history typed arrays to serializable history points', () => {
        const points = historyArraysToPoints({
            step: Float64Array.from([1, 2]),
            trainLoss: Float64Array.from([0.5, 0.4]),
            testLoss: Float64Array.from([0.6, 0.45]),
            trainAccuracy: Float64Array.from([0, 0.7]),
            testAccuracy: Float64Array.from([0, 0.65]),
            hasTrainAccuracy: Uint8Array.from([0, 1]),
            hasTestAccuracy: Uint8Array.from([0, 1]),
            count: 2,
        });

        expect(points).toEqual([
            { step: 1, trainLoss: 0.5, testLoss: 0.6 },
            { step: 2, trainLoss: 0.4, testLoss: 0.45, trainAccuracy: 0.7, testAccuracy: 0.65 },
        ]);
    });

    it('captures a bounded serializable run record from current state', () => {
        const record = captureExperimentRun({
            config,
            snapshot,
            history: [{ step: 4, trainLoss: 0.2, testLoss: 0.3 }],
            pauseReason: 'manual',
            status: 'paused',
            now: () => new Date('2026-04-26T00:00:00.000Z'),
            id: () => 'run-fixed',
        });

        expect(record?.id).toBe('run-fixed');
        expect(record?.network?.weights).toEqual(snapshot.weights);
        expect(record?.summary.step).toBe(4);
        expect(record?.history).toHaveLength(1);
    });

    it('returns null when no snapshot is available', () => {
        expect(captureExperimentRun({ config, snapshot: null, history: [] })).toBeNull();
    });

    it('uses current frame-buffer parameters when creating a serialized network', () => {
        resetFrameBuffer();
        updateFrameBuffer({
            weights: Float32Array.from([0.9, -0.4]),
            biases: Float32Array.from([0.2]),
            weightLayout: { layerSizes: [2, 1] },
        });

        const network = createSerializedNetworkFromFrameBuffer(config.network);

        expect(network?.weights[0][0][0]).toBeCloseTo(0.9, 6);
        expect(network?.weights[0][0][1]).toBeCloseTo(-0.4, 6);
        expect(network?.biases[0][0]).toBeCloseTo(0.2, 6);
    });

    it('does not persist empty snapshot parameter arrays as a serialized network', () => {
        const record = captureExperimentRun({
            config,
            snapshot: { ...snapshot, weights: [], biases: [] },
            history: [],
        });

        expect(record?.network).toBeNull();
    });
});
