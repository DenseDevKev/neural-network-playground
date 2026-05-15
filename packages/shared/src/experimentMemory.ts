import type {
    HistoryPoint,
    Metrics,
    SerializedNetwork,
} from '@nn-playground/engine';
import type {
    AppConfig,
    PauseReason,
    TrainingStatus,
} from './types.js';
import { isPauseReason } from './types.js';
import { validateImportedConfig } from './serialization.js';

export const EXPERIMENT_MEMORY_SCHEMA_VERSION = 1;
export const EXPERIMENT_MEMORY_MAX_RECORDS = 20;
export const EXPERIMENT_MEMORY_MAX_HISTORY = 512;

export interface ExperimentRunSummary {
    status: TrainingStatus;
    pauseReason: PauseReason | null;
    step: number;
    epoch: number;
    trainLoss: number;
    testLoss: number;
    trainMetrics: Metrics;
    testMetrics: Metrics;
}

export interface ExperimentRunRecordV1 {
    schemaVersion: typeof EXPERIMENT_MEMORY_SCHEMA_VERSION;
    id: string;
    createdAt: string;
    updatedAt: string;
    title?: string;
    config: AppConfig;
    summary: ExperimentRunSummary;
    network: SerializedNetwork | null;
    history: HistoryPoint[];
}

export interface ExperimentMemoryEnvelopeV1 {
    schemaVersion: typeof EXPERIMENT_MEMORY_SCHEMA_VERSION;
    records: ExperimentRunRecordV1[];
}

interface ValidationResult {
    record: ExperimentRunRecordV1 | null;
    error: string | null;
}

export interface ExperimentMemoryValidationOptions {
    allowMulticlass?: boolean;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
    return typeof value === 'number' && Number.isFinite(value);
}

function isIsoDateString(value: unknown): value is string {
    return typeof value === 'string' && value.length > 0 && !Number.isNaN(Date.parse(value));
}

function isTrainingStatus(value: unknown): value is TrainingStatus {
    return value === 'idle' || value === 'running' || value === 'paused';
}

function cloneJson<T>(value: T): T {
    return JSON.parse(JSON.stringify(value)) as T;
}

function validateMetrics(value: unknown): Metrics | null {
    if (!isRecord(value) || !isFiniteNumber(value.loss)) return null;
    const metrics: Metrics = { loss: value.loss };
    if (value.accuracy !== undefined) {
        if (!isFiniteNumber(value.accuracy)) return null;
        metrics.accuracy = value.accuracy;
    }
    if (isRecord(value.confusionMatrix)) {
        const { tp, tn, fp, fn } = value.confusionMatrix;
        if (![tp, tn, fp, fn].every(isFiniteNumber)) return null;
        metrics.confusionMatrix = {
            tp: tp as number,
            tn: tn as number,
            fp: fp as number,
            fn: fn as number,
        };
    }
    return metrics;
}

function validateHistoryPoint(value: unknown): HistoryPoint | null {
    if (!isRecord(value)) return null;
    if (!isFiniteNumber(value.step) || !isFiniteNumber(value.trainLoss) || !isFiniteNumber(value.testLoss)) {
        return null;
    }
    const point: HistoryPoint = {
        step: value.step,
        trainLoss: value.trainLoss,
        testLoss: value.testLoss,
    };
    if (value.trainAccuracy !== undefined) {
        if (!isFiniteNumber(value.trainAccuracy)) return null;
        point.trainAccuracy = value.trainAccuracy;
    }
    if (value.testAccuracy !== undefined) {
        if (!isFiniteNumber(value.testAccuracy)) return null;
        point.testAccuracy = value.testAccuracy;
    }
    return point;
}

function validateSerializedNetworkParams(
    value: { weights: unknown[]; biases: unknown[] },
    layerSizes: number[],
): string | null {
    const layerCount = layerSizes.length - 1;
    if (value.weights.length !== layerCount || value.biases.length !== layerCount) {
        return 'Run record network parameters are invalid.';
    }
    for (let layer = 0; layer < layerCount; layer++) {
        const weightMatrix = value.weights[layer];
        const biasVector = value.biases[layer];
        const previousLayerSize = layerSizes[layer];
        const nextLayerSize = layerSizes[layer + 1];
        if (!Array.isArray(weightMatrix) || weightMatrix.length !== nextLayerSize) {
            return 'Run record network parameters are invalid.';
        }
        for (const row of weightMatrix) {
            if (!Array.isArray(row) || row.length !== previousLayerSize) {
                return 'Run record network parameters are invalid.';
            }
            for (let column = 0; column < previousLayerSize; column++) {
                if (!isFiniteNumber(row[column])) return 'Run record network parameters are invalid.';
            }
        }
        if (!Array.isArray(biasVector) || biasVector.length !== nextLayerSize) {
            return 'Run record network parameters are invalid.';
        }
        for (let unit = 0; unit < nextLayerSize; unit++) {
            if (!isFiniteNumber(biasVector[unit])) return 'Run record network parameters are invalid.';
        }
    }
    return null;
}

function validateSerializedNetwork(
    value: unknown,
    recordConfig: AppConfig,
    options: ExperimentMemoryValidationOptions = {},
): { network: SerializedNetwork | null; error: string | null } {
    if (value === null) return { network: null, error: null };
    if (!isRecord(value)) return { network: null, error: 'Run record network is invalid.' };
    if (!isRecord(value.config) || !Array.isArray(value.weights) || !Array.isArray(value.biases)) {
        return { network: null, error: 'Run record network is invalid.' };
    }
    const serializedNetwork = value as {
        config: Record<string, unknown>;
        weights: unknown[];
        biases: unknown[];
    };
    const embeddedConfig = validateImportedConfig({
        ...recordConfig,
        network: serializedNetwork.config,
    }, {
        allowMulticlass: options.allowMulticlass,
    });
    if (!embeddedConfig.config) {
        return {
            network: null,
            error: `Run record network config is invalid: ${embeddedConfig.error ?? 'Invalid network config.'}`,
        };
    }
    if (serializedNetwork.config.inputSize !== embeddedConfig.config.network.inputSize) {
        return { network: null, error: 'Run record network input size does not match active features.' };
    }
    const layerSizes = [
        embeddedConfig.config.network.inputSize,
        ...embeddedConfig.config.network.hiddenLayers,
        embeddedConfig.config.network.outputSize,
    ];
    const paramError = validateSerializedNetworkParams(serializedNetwork, layerSizes);
    if (paramError) return { network: null, error: paramError };
    return { network: cloneJson(value as unknown as SerializedNetwork), error: null };
}

export function sanitizeExperimentHistory(value: unknown): HistoryPoint[] {
    if (!Array.isArray(value)) return [];
    const valid: HistoryPoint[] = [];
    for (const item of value) {
        const point = validateHistoryPoint(item);
        if (point) valid.push(point);
    }
    return valid.slice(-EXPERIMENT_MEMORY_MAX_HISTORY);
}

function validateSummary(value: unknown): ExperimentRunSummary | null {
    if (!isRecord(value)) return null;
    const trainMetrics = validateMetrics(value.trainMetrics);
    const testMetrics = validateMetrics(value.testMetrics);
    const pauseReason = value.pauseReason === null
        ? null
        : isPauseReason(value.pauseReason)
            ? value.pauseReason
            : undefined;
    if (
        !isTrainingStatus(value.status) ||
        pauseReason === undefined ||
        !isFiniteNumber(value.step) ||
        !isFiniteNumber(value.epoch) ||
        !isFiniteNumber(value.trainLoss) ||
        !isFiniteNumber(value.testLoss) ||
        !trainMetrics ||
        !testMetrics
    ) {
        return null;
    }
    return {
        status: value.status,
        pauseReason,
        step: value.step,
        epoch: value.epoch,
        trainLoss: value.trainLoss,
        testLoss: value.testLoss,
        trainMetrics,
        testMetrics,
    };
}

export function validateExperimentRunRecord(
    value: unknown,
    options: ExperimentMemoryValidationOptions = {},
): ValidationResult {
    if (!isRecord(value)) return { record: null, error: 'Run record must be an object.' };
    if (value.schemaVersion !== EXPERIMENT_MEMORY_SCHEMA_VERSION) {
        return { record: null, error: 'Unsupported experiment-memory record version.' };
    }
    if (typeof value.id !== 'string' || value.id.trim().length === 0) {
        return { record: null, error: 'Run record id is required.' };
    }
    if (!isIsoDateString(value.createdAt) || !isIsoDateString(value.updatedAt)) {
        return { record: null, error: 'Run record timestamps are invalid.' };
    }
    const configResult = validateImportedConfig(value.config, {
        allowMulticlass: options.allowMulticlass,
    });
    if (!configResult.config) {
        return { record: null, error: configResult.error ?? 'Run record config is invalid.' };
    }
    const summary = validateSummary(value.summary);
    if (!summary) return { record: null, error: 'Run record summary is invalid.' };
    const networkResult = validateSerializedNetwork(value.network, configResult.config, options);
    if (networkResult.error) return { record: null, error: networkResult.error };
    const history = sanitizeExperimentHistory(value.history);
    return {
        record: {
            schemaVersion: EXPERIMENT_MEMORY_SCHEMA_VERSION,
            id: value.id,
            createdAt: value.createdAt,
            updatedAt: value.updatedAt,
            title: typeof value.title === 'string' && value.title.trim() ? value.title : undefined,
            config: configResult.config,
            summary,
            network: networkResult.network,
            history,
        },
        error: null,
    };
}

export function createExperimentMemoryEnvelope(
    records: unknown[],
    options: ExperimentMemoryValidationOptions = {},
): ExperimentMemoryEnvelopeV1 {
    const normalized = records
        .map((record) => validateExperimentRunRecord(record, options).record)
        .filter((record): record is ExperimentRunRecordV1 => record !== null)
        .sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt))
        .slice(0, EXPERIMENT_MEMORY_MAX_RECORDS);
    return {
        schemaVersion: EXPERIMENT_MEMORY_SCHEMA_VERSION,
        records: normalized,
    };
}

export function normalizeExperimentMemoryEnvelope(
    value: unknown,
    options: ExperimentMemoryValidationOptions = {},
): ExperimentMemoryEnvelopeV1 {
    if (!isRecord(value) || value.schemaVersion !== EXPERIMENT_MEMORY_SCHEMA_VERSION || !Array.isArray(value.records)) {
        return createExperimentMemoryEnvelope([], options);
    }
    return createExperimentMemoryEnvelope(value.records, options);
}
