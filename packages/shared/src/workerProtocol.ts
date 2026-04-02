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

export type MainToWorkerCommand =
    | StartTrainingCommand
    | StopTrainingCommand
    | UpdateDemandCommand
    | UpdateSpeedCommand;
