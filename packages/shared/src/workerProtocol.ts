// ── Worker Protocol Types ──
// Shared between the main thread and the training web worker.
// Transport / demand types only — no engine internals.

import type {
    HistoryPoint,
    LayerStats,
    ConfusionMatrixData,
} from '@nn-playground/engine';
import { isPauseReason } from './types.js';
import type { PauseReason } from './types.js';

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
    /** How many snapshots between full test-set evaluations. */
    testEvalInterval: number;
    /** How many snapshots between full train-set evaluations. Between these,
     *  the worker reports a running EMA of the per-step batch loss instead
     *  of the true dataset loss. */
    trainEvalInterval: number;
    /** How many snapshots between decision-boundary / neuron-grid rebuilds.
     *  Between these, the previously-computed grids are reused. */
    gridInterval: number;
}

/**
 * Sensible defaults — everything visible. Test-eval every 10 snapshots,
 * train-eval every 5, grid every 2. Keeps the common case smooth while
 * letting power users (or explicit UI toggles) dial the cadence down.
 */
export const DEFAULT_DEMAND: VisualizationDemand = {
    needDecisionBoundary: true,
    needNeuronGrids: true,
    needLayerStats: false,     // InspectionPanel starts collapsed
    needConfusionMatrix: true,
    testEvalInterval: 10,
    trainEvalInterval: 5,
    gridInterval: 2,
};

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isPositiveInteger(value: unknown): value is number {
    return typeof value === 'number' && Number.isFinite(value) && Number.isInteger(value) && value > 0;
}

function isBoolean(value: unknown): value is boolean {
    return typeof value === 'boolean';
}

export function normalizeVisualizationDemand(value: unknown): VisualizationDemand | null {
    if (!isRecord(value)) return null;

    const {
        needDecisionBoundary,
        needNeuronGrids,
        needLayerStats,
        needConfusionMatrix,
        testEvalInterval,
        trainEvalInterval,
        gridInterval,
    } = value;

    if (
        !isBoolean(needDecisionBoundary) ||
        !isBoolean(needNeuronGrids) ||
        !isBoolean(needLayerStats) ||
        !isBoolean(needConfusionMatrix) ||
        !isPositiveInteger(testEvalInterval) ||
        !isPositiveInteger(trainEvalInterval) ||
        !isPositiveInteger(gridInterval)
    ) {
        return null;
    }

    return {
        needDecisionBoundary,
        needNeuronGrids,
        needLayerStats,
        needConfusionMatrix,
        testEvalInterval,
        trainEvalInterval,
        gridInterval,
    };
}

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
    confusionMatrixVersion?: number;

    /**
     * When the worker is publishing heavy buffers (outputGrid, neuronGrids,
     * weights, biases) via the shared-memory fast path, the corresponding
     * message fields above are omitted and this value is the seqlock counter
     * the main thread should observe after reading the SAB views. The main
     * thread retries the read until the seq it observes at the start matches
     * the seq at the end of the read (standard seqlock). If this field is
     * absent, heavy buffers are either present inline on this message or
     * reused from a previous frame (cadence gating).
     */
    sharedSeq?: number;
}

export interface WorkerStatusMessage {
    type: 'status';
    runId: number;
    status: 'idle' | 'running' | 'paused';
    pauseReason?: PauseReason | null;
}

export interface WorkerErrorMessage {
    type: 'error';
    runId: number;
    message: string;
}

/**
 * One-off handshake message sent by the worker whenever it has (re)allocated
 * its SharedArrayBuffer-backed snapshot buffers — i.e. at init time, after a
 * reset, and after any network shape change. The main thread installs views
 * over these SABs into the frame buffer; subsequent snapshot messages carry
 * only a `sharedSeq` counter, and the main thread reads the latest data
 * directly out of these permanently-installed views.
 *
 * The control buffer layout (Int32Array view over `control`) is:
 *   [0] seqStart  — incremented by the writer before any data write
 *   [1] seqEnd    — stored with the same value after the data write
 *   [2] flags     — bit 0: outputGrid valid, bit 1: neuronGrids valid
 * Readers read seqEnd, then the data, then seqStart; if they differ the
 * read observed a concurrent write and must retry.
 */
export interface WorkerSharedBuffersMessage {
    type: 'sharedBuffers';
    runId: number;
    /** SAB for [seqStart, seqEnd, flags] control words. */
    control: SharedArrayBuffer;
    /** SAB for the decision-boundary grid, Float32Array of gridSize*gridSize. */
    outputGrid: SharedArrayBuffer;
    /** SAB for concatenated per-neuron activation grids, Float32Array. */
    neuronGrids: SharedArrayBuffer;
    gridSize: number;
    neuronGridLayout: { count: number; gridSize: number };
}

export type WorkerToMainMessage =
    | WorkerSnapshotMessage
    | WorkerStatusMessage
    | WorkerErrorMessage
    | WorkerSharedBuffersMessage;

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
    if (!isRecord(x)) return false;
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
                (
                    m['status'] === 'idle' ||
                    m['status'] === 'running' ||
                    m['status'] === 'paused'
                ) &&
                (
                    !('pauseReason' in m) ||
                    m['pauseReason'] === null ||
                    isPauseReason(m['pauseReason'])
                )
            );
        case 'error':
            return typeof m['message'] === 'string';
        case 'sharedBuffers':
            // SharedArrayBuffer is a distinct global constructor; fall back to
            // a truthy-object check in environments that don't expose it
            // (tests, hosts without cross-origin isolation). We never *send*
            // this message from such hosts, so accepting the fallback there
            // only matters for symmetry.
            return (
                typeof m['gridSize'] === 'number' &&
                m['control'] !== null &&
                typeof m['control'] === 'object' &&
                m['outputGrid'] !== null &&
                typeof m['outputGrid'] === 'object' &&
                m['neuronGrids'] !== null &&
                typeof m['neuronGrids'] === 'object'
            );
        default:
            return false;
    }
}

/**
 * Shallow runtime guard for commands flowing Main → Worker.
 * Validates the `type` discriminator and required primitive fields only.
 */
export function isMainToWorkerCommand(x: unknown): x is MainToWorkerCommand {
    if (!isRecord(x)) return false;
    const m = x as Record<string, unknown>;
    if (typeof m['type'] !== 'string') return false;
    switch (m['type']) {
        case 'startTraining':
            return typeof m['stepsPerFrame'] === 'number';
        case 'stopTraining':
            return true;
        case 'updateDemand':
            return normalizeVisualizationDemand(m['demand']) !== null;
        case 'updateSpeed':
            return typeof m['stepsPerFrame'] === 'number';
        case 'frameAck':
            return true;
        default:
            return false;
    }
}
