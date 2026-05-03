import type {
    HistoryPoint,
    NetworkConfig,
    NetworkSnapshot,
    SerializedNetwork,
} from '@nn-playground/engine';
import type {
    AppConfig,
    ExperimentRunRecordV1,
    PauseReason,
    TrainingStatus,
} from '@nn-playground/shared';
import { sanitizeExperimentHistory } from '@nn-playground/shared';
import type { HistoryArrays } from './historyBuffer.ts';
import { getFrameBuffer } from '../worker/frameBuffer.ts';
import {
    unflattenBiases,
    unflattenWeights,
} from '../worker/frameBufferLayout.ts';

interface CaptureArgs {
    config: AppConfig;
    snapshot: NetworkSnapshot | null;
    history: HistoryPoint[];
    status?: TrainingStatus;
    pauseReason?: PauseReason | null;
    network?: SerializedNetwork | null;
    now?: () => Date;
    id?: () => string;
}

export function historyArraysToPoints(history: HistoryArrays): HistoryPoint[] {
    const points: HistoryPoint[] = [];
    for (let i = 0; i < history.count; i++) {
        const point: HistoryPoint = {
            step: history.step[i],
            trainLoss: history.trainLoss[i],
            testLoss: history.testLoss[i],
        };
        if (history.hasTrainAccuracy[i]) point.trainAccuracy = history.trainAccuracy[i];
        if (history.hasTestAccuracy[i]) point.testAccuracy = history.testAccuracy[i];
        points.push(point);
    }
    return points;
}

function defaultId(): string {
    if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
        return crypto.randomUUID();
    }
    return `run-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function cloneNestedParams<T>(value: T): T {
    return JSON.parse(JSON.stringify(value)) as T;
}

function expectedParamCounts(layerSizes: number[]): { weights: number; biases: number } {
    let weights = 0;
    let biases = 0;
    for (let i = 0; i < layerSizes.length - 1; i++) {
        weights += layerSizes[i] * layerSizes[i + 1];
        biases += layerSizes[i + 1];
    }
    return { weights, biases };
}

function networkFromSnapshot(config: NetworkConfig, snapshot: NetworkSnapshot): SerializedNetwork | null {
    if (snapshot.weights.length === 0 || snapshot.biases.length === 0) return null;
    return {
        config: cloneNestedParams(config),
        weights: cloneNestedParams(snapshot.weights),
        biases: cloneNestedParams(snapshot.biases),
    };
}

export function createSerializedNetworkFromFrameBuffer(config: NetworkConfig): SerializedNetwork | null {
    const frame = getFrameBuffer();
    const layerSizes = frame.weightLayout?.layerSizes;
    if (!frame.weights || !frame.biases || !layerSizes || layerSizes.length < 2) return null;

    const expected = expectedParamCounts(layerSizes);
    if (frame.weights.length !== expected.weights || frame.biases.length !== expected.biases) return null;

    return {
        config: cloneNestedParams({ ...config, inputSize: layerSizes[0], outputSize: layerSizes.at(-1)! }),
        weights: unflattenWeights(frame.weights, layerSizes),
        biases: unflattenBiases(frame.biases, layerSizes),
    };
}

export function captureExperimentRun({
    config,
    snapshot,
    history,
    status = 'paused',
    pauseReason = null,
    network,
    now = () => new Date(),
    id = defaultId,
}: CaptureArgs): ExperimentRunRecordV1 | null {
    if (!snapshot) return null;
    const timestamp = now().toISOString();
    const stepLabel = snapshot.step.toLocaleString();
    return {
        schemaVersion: 1,
        id: id(),
        createdAt: timestamp,
        updatedAt: timestamp,
        title: `${config.data.dataset} at step ${stepLabel}`,
        config: cloneNestedParams(config),
        summary: {
            status,
            pauseReason,
            step: snapshot.step,
            epoch: snapshot.epoch,
            trainLoss: snapshot.trainLoss,
            testLoss: snapshot.testLoss,
            trainMetrics: cloneNestedParams(snapshot.trainMetrics),
            testMetrics: cloneNestedParams(snapshot.testMetrics),
        },
        network: network === undefined ? networkFromSnapshot(config.network, snapshot) : network,
        history: sanitizeExperimentHistory(history),
    };
}
