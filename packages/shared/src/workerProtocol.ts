// ── Worker Protocol Types ──
// Shared between the main thread and the training web worker.
// Transport / demand types only — no engine internals.

import type {
    HistoryPoint,
    LayerStats,
    ConfusionMatrixData,
} from '@nn-playground/engine';

// ─────────────────────────────────────────────────────────
// Visualization Demand
// ─────────────────────────────────────────────────────────

/**
 * Tells the worker which visual data the UI currently needs.
 * Fields that are `false` let the worker skip expensive computations.
 */
export interface VisualizationDemand {
    /** Whether to compute the decision-boundary heatmap grid. */
    needDecisionBoundary: boolean;
    /** Whether to compute per-neuron activation grids (mini heatmaps). */
    needNeuronGrids: boolean;
    /** Whether to compute per-layer weight/gradient/activation statistics. */
    needLayerStats: boolean;
    /** Whether to evaluate the confusion matrix on the test set. */
    needConfusionMatrix: boolean;
    /** How many training ticks between full test-set evaluations. */
    testEvalInterval: number;
}

/** Sensible defaults — everything visible, test eval every 10 ticks. */
export const DEFAULT_DEMAND: VisualizationDemand = {
    needDecisionBoundary: true,
    needNeuronGrids: true,
    needLayerStats: false,     // InspectionPanel starts collapsed
    needConfusionMatrix: true,
    testEvalInterval: 10,
};

// ─────────────────────────────────────────────────────────
// Worker → Main  (streamed snapshot messages)
// ─────────────────────────────────────────────────────────

/** Lightweight scalars extracted from a snapshot. */
export interface SnapshotScalars {
    step: number;
    epoch: number;
    trainLoss: number;
    testLoss: number;
    trainAccuracy?: number;
    testAccuracy?: number;
    gridSize: number;
    /**
     * True when the test metrics in this snapshot were reused from a previous
     * evaluation (i.e. `testEvalInterval` throttled a fresh run). UIs can dim
     * the test-loss display to hint at the cached value.
     */
    testMetricsStale?: boolean;
}

/** Full snapshot message posted from the worker. */
export interface WorkerSnapshotMessage {
    type: 'snapshot';
    runId: number;
    snapshotId: number;
    scalars: SnapshotScalars;

    // Heavy payloads — presence depends on demand flags
    outputGrid?: Float32Array;
    neuronGrids?: Float32Array;
    neuronGridLayout?: { count: number; gridSize: number };
    weights?: Float32Array;
    biases?: Float32Array;
    weightLayout?: { layerSizes: number[] };
    layerStats?: LayerStats[];

    historyPoint: HistoryPoint;
    confusionMatrix?: ConfusionMatrixData;
}

export interface WorkerStatusMessage {
    type: 'status';
    runId: number;
    status: 'idle' | 'running' | 'paused';
}

export interface WorkerErrorMessage {
    type: 'error';
    runId: number;
    message: string;
}

export type WorkerToMainMessage =
    | WorkerSnapshotMessage
    | WorkerStatusMessage
    | WorkerErrorMessage;

// ─────────────────────────────────────────────────────────
// Main → Worker  (streaming commands via MessageChannel)
// ─────────────────────────────────────────────────────────

export interface StartTrainingCommand {
    type: 'startTraining';
    stepsPerFrame: number;
}

export interface StopTrainingCommand {
    type: 'stopTraining';
}

export interface UpdateDemandCommand {
    type: 'updateDemand';
    demand: VisualizationDemand;
}

export interface UpdateSpeedCommand {
    type: 'updateSpeed';
    stepsPerFrame: number;
}

/**
 * Sent by the main thread after a streamed snapshot has been applied to the
 * frame buffer. The worker uses this to back-pressure snapshot posting —
 * training continues, but new snapshots are only posted once the previous one
 * has been consumed. This keeps the postMessage queue (and its Transferable
 * ArrayBuffers) from growing unbounded under load.
 */
export interface FrameAckCommand {
    type: 'frameAck';
}

export type MainToWorkerCommand =
    | StartTrainingCommand
    | StopTrainingCommand
    | UpdateDemandCommand
    | UpdateSpeedCommand
    | FrameAckCommand;

// ─────────────────────────────────────────────────────────
// Runtime type guards (hand-rolled, shallow — no zod)
// ─────────────────────────────────────────────────────────

/**
 * Shallow runtime guard for messages flowing Worker → Main.
 * Validates the `type` discriminator and required primitive fields only;
 * does not recurse into `layerStats` arrays to avoid per-frame overhead.
 */
export function isWorkerToMainMessage(x: unknown): x is WorkerToMainMessage {
    if (x === null || typeof x !== 'object') return false;
    const m = x as Record<string, unknown>;
    if (typeof m['type'] !== 'string') return false;
    if (typeof m['runId'] !== 'number') return false;
    switch (m['type']) {
        case 'snapshot':
            return (
                typeof m['snapshotId'] === 'number' &&
                m['scalars'] !== null &&
                typeof m['scalars'] === 'object'
            );
        case 'status':
            return (
                m['status'] === 'idle' ||
                m['status'] === 'running' ||
                m['status'] === 'paused'
            );
        case 'error':
            return typeof m['message'] === 'string';
        default:
            return false;
    }
}

/**
 * Shallow runtime guard for commands flowing Main → Worker.
 * Validates the `type` discriminator and required primitive fields only.
 */
export function isMainToWorkerCommand(x: unknown): x is MainToWorkerCommand {
    if (x === null || typeof x !== 'object') return false;
    const m = x as Record<string, unknown>;
    if (typeof m['type'] !== 'string') return false;
    switch (m['type']) {
        case 'startTraining':
            return typeof m['stepsPerFrame'] === 'number';
        case 'stopTraining':
            return true;
        case 'updateDemand':
            return m['demand'] !== null && typeof m['demand'] === 'object';
        case 'updateSpeed':
            return typeof m['stepsPerFrame'] === 'number';
        case 'frameAck':
            return true;
        default:
            return false;
    }
}
