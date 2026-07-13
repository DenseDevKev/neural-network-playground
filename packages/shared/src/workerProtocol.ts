// ── Worker Protocol Types ──
// Shared between the main thread and the training web worker.
// Transport / demand types only — no engine internals.

import type {
    LayerStats,
    ConfusionMatrixData,
    MulticlassConfusionMatrixData,
    ActivationHistogramLayer,
} from '@nn-playground/engine';
import { canonicalRecipeKey, validateExperimentDocument } from './experimentSchema.js';
import { createStableJsonSnapshot } from './canonicalJson.js';
import {
    parseArtifactProvenance,
    parseLiveTrainingSignal,
    parsePairedEvaluation,
} from './metricProvenance.js';
import type {
    ArtifactProvenance,
    EvaluationTrigger,
    LiveTrainingSignal,
    ModelRevision,
    PairedEvaluation,
} from './metricProvenance.js';
import { isPauseReason } from './types.js';
import type {
    ExperimentDocumentV2,
    PauseReason,
    RecipeFingerprint,
    ValidatedExperimentDocumentV2,
} from './types.js';
import {
    GRID_SIZE,
    MAX_HIDDEN_LAYERS,
    MAX_NEURONS_PER_LAYER,
} from './constants.js';

// ─────────────────────────────────────────────────────────
// Strict protocol version 2
// ─────────────────────────────────────────────────────────

export const WORKER_PROTOCOL_VERSION = 2 as const;

export interface ClaimedExperimentIdentitiesV2 {
    readonly canonicalRecipeKey: string;
    readonly recipeFingerprint: string;
    readonly datasetKey: string;
    readonly objectiveKey: string;
}

export interface WorkerExperimentRequestV2 {
    readonly type: 'initialize-experiment';
    readonly protocolVersion: typeof WORKER_PROTOCOL_VERSION;
    readonly requestId: number;
    readonly document: ExperimentDocumentV2;
    readonly claimedIdentities: ClaimedExperimentIdentitiesV2;
}

export type ForcedEvaluationTriggerV2 = Exclude<EvaluationTrigger, 'initial' | 'cadence'>;

export interface ForceEvaluationRequestV2 {
    readonly type: 'force-evaluation';
    readonly protocolVersion: typeof WORKER_PROTOCOL_VERSION;
    readonly requestId: number;
    readonly trigger: ForcedEvaluationTriggerV2;
}

export interface CaptureRunRequestV2 {
    readonly type: 'capture-run';
    readonly protocolVersion: typeof WORKER_PROTOCOL_VERSION;
    readonly requestId: number;
    readonly id: string;
    readonly createdAt: string;
    readonly updatedAt: string;
    readonly title?: string;
}

/** Scientific metadata supplied by the UI; the worker authors every evidence field. */
export interface CaptureRunArtifactRequestV2 {
    readonly id: string;
    readonly createdAt: string;
    readonly updatedAt: string;
    readonly title?: string;
}

export interface CaptureCheckpointRequestV2 {
    readonly type: 'capture-checkpoint';
    readonly protocolVersion: typeof WORKER_PROTOCOL_VERSION;
    readonly requestId: number;
}

export interface RestoreCheckpointRequestV2 {
    readonly type: 'restore-checkpoint';
    readonly protocolVersion: typeof WORKER_PROTOCOL_VERSION;
    readonly requestId: number;
    readonly checkpointId: number;
}

export type MainToWorkerRequestV2 =
    | WorkerExperimentRequestV2
    | ForceEvaluationRequestV2
    | CaptureRunRequestV2
    | CaptureCheckpointRequestV2
    | RestoreCheckpointRequestV2;

export interface WorkerArtifactProvenanceV2 {
    readonly confusionMatrix?: ArtifactProvenance;
    readonly activationStatistics?: ArtifactProvenance;
    readonly predictionTrace?: ArtifactProvenance;
    readonly decisionBoundary?: ArtifactProvenance;
    readonly lossLandscape?: ArtifactProvenance;
    readonly neuronGrids?: ArtifactProvenance;
    readonly activationHistogram?: ArtifactProvenance;
}

export interface WorkerEvidenceMessageV2 {
    readonly type: 'evidence';
    readonly protocolVersion: typeof WORKER_PROTOCOL_VERSION;
    readonly liveSignal?: LiveTrainingSignal;
    readonly latestEvaluation?: PairedEvaluation;
    readonly artifacts?: WorkerArtifactProvenanceV2;
}

export type WorkerProtocolErrorCodeV2 =
    | 'malformed-request'
    | 'unsupported-protocol-version'
    | 'invalid-experiment'
    | 'identity-mismatch'
    | 'not-initialized'
    | 'stale-request'
    | 'evaluation-failed'
    | 'artifact-failed'
    | 'capture-failed'
    | 'checkpoint-failed'
    | 'runtime-failure';

export type WorkerProtocolErrorSourceV2 =
    | 'protocol'
    | 'preparation'
    | 'training'
    | 'evaluation'
    | 'artifact'
    | 'persistence'
    | 'checkpoint'
    | 'runtime';

export interface WorkerProtocolErrorMessageV2 {
    readonly type: 'worker-error';
    readonly protocolVersion: typeof WORKER_PROTOCOL_VERSION;
    readonly requestId: number | null;
    readonly generationId: number | null;
    readonly code: WorkerProtocolErrorCodeV2;
    readonly path: string;
    readonly message: string;
    readonly source: WorkerProtocolErrorSourceV2;
}

export type WorkerToMainMessageV2 = WorkerEvidenceMessageV2 | WorkerProtocolErrorMessageV2;

const EXACT_DIGEST_LENGTH = 43;
const FORCED_EVALUATION_TRIGGERS = new Set<ForcedEvaluationTriggerV2>([
    'manual-step',
    'pause',
    'checkpoint',
    'save',
    'stop-condition',
    'restore',
]);
const WORKER_ERROR_CODES = new Set<WorkerProtocolErrorCodeV2>([
    'malformed-request',
    'unsupported-protocol-version',
    'invalid-experiment',
    'identity-mismatch',
    'not-initialized',
    'stale-request',
    'evaluation-failed',
    'artifact-failed',
    'capture-failed',
    'checkpoint-failed',
    'runtime-failure',
]);
const WORKER_ERROR_SOURCES = new Set<WorkerProtocolErrorSourceV2>([
    'protocol',
    'preparation',
    'training',
    'evaluation',
    'artifact',
    'persistence',
    'checkpoint',
    'runtime',
]);
const ARTIFACT_BASIS = {
    confusionMatrix: new Set(['full-split']),
    activationStatistics: new Set(['bounded-sample']),
    predictionTrace: new Set(['bounded-sample']),
    decisionBoundary: new Set(['prediction-grid']),
    lossLandscape: new Set(['parameter-grid']),
    neuronGrids: new Set(['prediction-grid']),
    activationHistogram: new Set(['bounded-sample']),
} as const;

function hasExactOwnKeys(
    value: unknown,
    required: readonly string[],
    optional: readonly string[] = [],
): value is Record<string, unknown> {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) return false;
    const object = value as Record<string, unknown>;
    const allowed = new Set([...required, ...optional]);
    const keys = Reflect.ownKeys(object);
    if (keys.some((key) => typeof key !== 'string' || !allowed.has(key))) return false;
    for (const key of keys) {
        const descriptor = Object.getOwnPropertyDescriptor(object, key);
        if (descriptor === undefined || !('value' in descriptor) || !descriptor.enumerable) {
            return false;
        }
    }
    return required.every((key) => Object.prototype.hasOwnProperty.call(object, key));
}

function isRequestId(value: unknown): value is number {
    return Number.isSafeInteger(value) && (value as number) > 0;
}

function isNullableRequestId(value: unknown): value is number | null {
    return value === null || isRequestId(value);
}

function isBoundedString(value: unknown, maximum: number): value is string {
    return typeof value === 'string' && value.length > 0 && value.length <= maximum;
}

function isFingerprint(value: unknown, prefix: 'r2.1.' | 'd2.1.' | 'o2.1.'): value is string {
    if (typeof value !== 'string' || !value.startsWith(prefix)) return false;
    const digest = value.slice(prefix.length);
    return digest.length === EXACT_DIGEST_LENGTH && /^[A-Za-z0-9_-]+$/u.test(digest);
}

function isClaimedExperimentIdentitiesV2(
    value: unknown,
    document: ValidatedExperimentDocumentV2,
): value is ClaimedExperimentIdentitiesV2 {
    if (!hasExactOwnKeys(value, [
        'canonicalRecipeKey',
        'recipeFingerprint',
        'datasetKey',
        'objectiveKey',
    ])) return false;
    return isBoundedString(value['canonicalRecipeKey'], 32_768)
        && value['canonicalRecipeKey'] === canonicalRecipeKey(document.recipe)
        && isFingerprint(value['recipeFingerprint'], 'r2.1.')
        && isFingerprint(value['datasetKey'], 'd2.1.')
        && isFingerprint(value['objectiveKey'], 'o2.1.');
}

function isWorkerExperimentRequestV2Snapshot(value: unknown): boolean {
    if (!hasExactOwnKeys(value, [
        'type',
        'protocolVersion',
        'requestId',
        'document',
        'claimedIdentities',
    ])) return false;
    if (value['type'] !== 'initialize-experiment'
        || value['protocolVersion'] !== WORKER_PROTOCOL_VERSION
        || !isRequestId(value['requestId'])) return false;
    const validated = validateExperimentDocument(value['document']);
    return validated.ok
        && isClaimedExperimentIdentitiesV2(value['claimedIdentities'], validated.value);
}

export function parseWorkerExperimentRequestV2(value: unknown): WorkerExperimentRequestV2 {
    const snapshot = createStableJsonSnapshot(value);
    if (!isWorkerExperimentRequestV2Snapshot(snapshot)) {
        throw new TypeError('worker experiment request: invalid version-2 request');
    }
    return snapshot as WorkerExperimentRequestV2;
}

export function isWorkerExperimentRequestV2(value: unknown): boolean {
    try {
        void parseWorkerExperimentRequestV2(value);
        return true;
    } catch {
        return false;
    }
}

export function isForceEvaluationRequestV2(value: unknown): boolean {
    return hasExactOwnKeys(value, ['type', 'protocolVersion', 'requestId', 'trigger'])
        && value['type'] === 'force-evaluation'
        && value['protocolVersion'] === WORKER_PROTOCOL_VERSION
        && isRequestId(value['requestId'])
        && typeof value['trigger'] === 'string'
        && FORCED_EVALUATION_TRIGGERS.has(value['trigger'] as ForcedEvaluationTriggerV2);
}

function isExactCaptureRequest(
    value: unknown,
    type: CaptureCheckpointRequestV2['type'],
): boolean {
    return hasExactOwnKeys(value, ['type', 'protocolVersion', 'requestId'])
        && value['type'] === type
        && value['protocolVersion'] === WORKER_PROTOCOL_VERSION
        && isRequestId(value['requestId']);
}

export function isCaptureRunRequestV2(value: unknown): boolean {
    return hasExactOwnKeys(
        value,
        ['type', 'protocolVersion', 'requestId', 'id', 'createdAt', 'updatedAt'],
        ['title'],
    )
        && value['type'] === 'capture-run'
        && value['protocolVersion'] === WORKER_PROTOCOL_VERSION
        && isRequestId(value['requestId'])
        && isCaptureRunArtifactRequestV2({
            id: value['id'],
            createdAt: value['createdAt'],
            updatedAt: value['updatedAt'],
            ...(Object.prototype.hasOwnProperty.call(value, 'title')
                ? { title: value['title'] }
                : {}),
        });
}

export function isCaptureCheckpointRequestV2(
    value: unknown,
): boolean {
    return isExactCaptureRequest(value, 'capture-checkpoint');
}

export function isRestoreCheckpointRequestV2(
    value: unknown,
): value is RestoreCheckpointRequestV2 {
    return hasExactOwnKeys(
        value,
        ['type', 'protocolVersion', 'requestId', 'checkpointId'],
    )
        && value['type'] === 'restore-checkpoint'
        && value['protocolVersion'] === WORKER_PROTOCOL_VERSION
        && isRequestId(value['requestId'])
        && isRequestId(value['checkpointId']);
}

const CANONICAL_UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/u;

function isCanonicalTimestamp(value: unknown): value is string {
    if (typeof value !== 'string' || value.length === 0) return false;
    const timestamp = Date.parse(value);
    return Number.isFinite(timestamp) && new Date(timestamp).toISOString() === value;
}

export function isCaptureRunArtifactRequestV2(
    value: unknown,
): value is CaptureRunArtifactRequestV2 {
    if (!hasExactOwnKeys(value, ['id', 'createdAt', 'updatedAt'], ['title'])) return false;
    if (typeof value['id'] !== 'string' || !CANONICAL_UUID.test(value['id'])) return false;
    if (!isCanonicalTimestamp(value['createdAt']) || !isCanonicalTimestamp(value['updatedAt'])) {
        return false;
    }
    if (Date.parse(value['updatedAt']) < Date.parse(value['createdAt'])) return false;
    return !Object.prototype.hasOwnProperty.call(value, 'title')
        || (
            typeof value['title'] === 'string'
            && value['title'].length > 0
            && Array.from(value['title']).length <= 120
        );
}

export function parseCaptureRunArtifactRequestV2(
    value: unknown,
): CaptureRunArtifactRequestV2 {
    const snapshot = createStableJsonSnapshot(value);
    if (!isCaptureRunArtifactRequestV2(snapshot)) {
        throw new TypeError('capture run artifact request: invalid metadata');
    }
    return snapshot;
}

export function parseMainToWorkerRequestV2(value: unknown): MainToWorkerRequestV2 {
    const snapshot = createStableJsonSnapshot(value);
    if (isWorkerExperimentRequestV2Snapshot(snapshot)
        || isForceEvaluationRequestV2(snapshot)
        || isCaptureRunRequestV2(snapshot)
        || isCaptureCheckpointRequestV2(snapshot)
        || isRestoreCheckpointRequestV2(snapshot)) {
        return snapshot as MainToWorkerRequestV2;
    }
    throw new TypeError('main-to-worker request: invalid version-2 request');
}

export function isMainToWorkerRequestV2(value: unknown): boolean {
    try {
        void parseMainToWorkerRequestV2(value);
        return true;
    } catch {
        return false;
    }
}

function isWorkerArtifactProvenanceV2(value: unknown): value is WorkerArtifactProvenanceV2 {
    const keys = Object.keys(ARTIFACT_BASIS);
    if (!hasExactOwnKeys(value, [], keys)) return false;
    const artifacts = value as Record<string, unknown>;
    const present = keys.filter((key) => Object.prototype.hasOwnProperty.call(artifacts, key));
    if (present.length === 0) return false;
    try {
        for (const key of present) {
            const provenance = parseArtifactProvenance(artifacts[key]);
            const allowed = ARTIFACT_BASIS[key as keyof typeof ARTIFACT_BASIS] as ReadonlySet<string>;
            if (!allowed.has(provenance.basis.kind)) return false;
            if (key === 'activationStatistics') {
                if (provenance.basis.kind !== 'bounded-sample'
                    || provenance.basis.split !== 'train'
                    || provenance.basis.sampleCount !== Math.min(
                        128,
                        provenance.dataset.trainCount,
                    )) {
                    return false;
                }
            }
        }
        return true;
    } catch {
        return false;
    }
}

interface EvidenceIdentity {
    readonly model: { readonly generationId: number };
    readonly dataset: {
        readonly generatorVersion: number;
        readonly datasetKey: string;
        readonly trainCount: number;
        readonly testCount: number;
    };
    readonly objectiveKey: string;
}

function hasSameEvidenceIdentity(left: EvidenceIdentity, right: EvidenceIdentity): boolean {
    return left.model.generationId === right.model.generationId
        && left.objectiveKey === right.objectiveKey
        && left.dataset.generatorVersion === right.dataset.generatorVersion
        && left.dataset.datasetKey === right.dataset.datasetKey
        && left.dataset.trainCount === right.dataset.trainCount
        && left.dataset.testCount === right.dataset.testCount;
}

function isWorkerEvidenceMessageV2Snapshot(value: unknown): boolean {
    if (!hasExactOwnKeys(
        value,
        ['type', 'protocolVersion'],
        ['liveSignal', 'latestEvaluation', 'artifacts'],
    )) return false;
    if (value['type'] !== 'evidence' || value['protocolVersion'] !== WORKER_PROTOCOL_VERSION) {
        return false;
    }
    const hasLive = Object.prototype.hasOwnProperty.call(value, 'liveSignal');
    const hasEvaluation = Object.prototype.hasOwnProperty.call(value, 'latestEvaluation');
    const hasArtifacts = Object.prototype.hasOwnProperty.call(value, 'artifacts');
    if (!hasLive && !hasEvaluation && !hasArtifacts) return false;
    try {
        const identities: EvidenceIdentity[] = [];
        if (hasLive) {
            const liveSignal = parseLiveTrainingSignal(value['liveSignal']);
            identities.push(liveSignal);
        }
        if (hasEvaluation) {
            const evaluation = parsePairedEvaluation(value['latestEvaluation']);
            identities.push(evaluation);
        }
        if (hasArtifacts) {
            const artifacts = value['artifacts'];
            if (!isWorkerArtifactProvenanceV2(artifacts)) return false;
            for (const provenance of Object.values(artifacts)) {
                if (provenance !== undefined) identities.push(provenance);
            }
        }
        const [first, ...rest] = identities;
        if (first === undefined || rest.some((entry) => !hasSameEvidenceIdentity(first, entry))) {
            return false;
        }
        return true;
    } catch {
        return false;
    }
}

export function parseWorkerEvidenceMessageV2(value: unknown): WorkerEvidenceMessageV2 {
    const snapshot = createStableJsonSnapshot(value);
    if (!isWorkerEvidenceMessageV2Snapshot(snapshot)) {
        throw new TypeError('worker evidence message: invalid version-2 evidence');
    }
    return snapshot as WorkerEvidenceMessageV2;
}

export function isWorkerEvidenceMessageV2(value: unknown): boolean {
    try {
        void parseWorkerEvidenceMessageV2(value);
        return true;
    } catch {
        return false;
    }
}

function isWorkerProtocolErrorMessageV2Snapshot(
    value: unknown,
): boolean {
    return hasExactOwnKeys(value, [
        'type',
        'protocolVersion',
        'requestId',
        'generationId',
        'code',
        'path',
        'message',
        'source',
    ])
        && value['type'] === 'worker-error'
        && value['protocolVersion'] === WORKER_PROTOCOL_VERSION
        && isNullableRequestId(value['requestId'])
        && isNullableRequestId(value['generationId'])
        && typeof value['code'] === 'string'
        && WORKER_ERROR_CODES.has(value['code'] as WorkerProtocolErrorCodeV2)
        && isBoundedString(value['path'], 1_024)
        && isBoundedString(value['message'], 4_096)
        && typeof value['source'] === 'string'
        && WORKER_ERROR_SOURCES.has(value['source'] as WorkerProtocolErrorSourceV2);
}

export function parseWorkerProtocolErrorMessageV2(
    value: unknown,
): WorkerProtocolErrorMessageV2 {
    const snapshot = createStableJsonSnapshot(value);
    if (!isWorkerProtocolErrorMessageV2Snapshot(snapshot)) {
        throw new TypeError('worker protocol error: invalid version-2 error');
    }
    return snapshot as WorkerProtocolErrorMessageV2;
}

export function isWorkerProtocolErrorMessageV2(value: unknown): boolean {
    try {
        void parseWorkerProtocolErrorMessageV2(value);
        return true;
    } catch {
        return false;
    }
}

export function parseWorkerToMainMessageV2(value: unknown): WorkerToMainMessageV2 {
    const snapshot = createStableJsonSnapshot(value);
    if (isWorkerEvidenceMessageV2Snapshot(snapshot)
        || isWorkerProtocolErrorMessageV2Snapshot(snapshot)) {
        return snapshot as WorkerToMainMessageV2;
    }
    throw new TypeError('worker-to-main message: invalid version-2 message');
}

export function isWorkerToMainMessageV2(value: unknown): boolean {
    try {
        void parseWorkerToMainMessageV2(value);
        return true;
    } catch {
        return false;
    }
}

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
    /** Whether to compute bounded layer-level activation histogram bins. */
    needActivationHistograms: boolean;
    /** Whether to evaluate the confusion matrix on the test set. */
    needConfusionMatrix: boolean;
    /** How many snapshots between decision-boundary / neuron-grid rebuilds.
     *  Between these, the previously-computed grids are reused. */
    gridInterval: number;
    /** How many snapshots between bounded activation histogram rebuilds. */
    activationHistogramInterval: number;
}

/**
 * Sensible defaults — everything visible, with display-only artifact cadence.
 * Scientific evaluation cadence is owned by EvaluationRuntime, not the view.
 */
export const DEFAULT_DEMAND: VisualizationDemand = {
    needDecisionBoundary: true,
    needNeuronGrids: true,
    needLayerStats: false,     // InspectionPanel starts collapsed
    needActivationHistograms: false,
    needConfusionMatrix: true,
    gridInterval: 2,
    activationHistogramInterval: 5,
};

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isPositiveInteger(value: unknown): value is number {
    return typeof value === 'number' && Number.isSafeInteger(value) && value > 0;
}

function isNonNegativeInteger(value: unknown): value is number {
    return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

function isFiniteNumber(value: unknown): value is number {
    return typeof value === 'number' && Number.isFinite(value);
}

function isModelRevision(value: unknown): value is ModelRevision {
    return isRecord(value)
        && isPositiveInteger(value['generationId'])
        && isNonNegativeInteger(value['revision'])
        && isNonNegativeInteger(value['step'])
        && isNonNegativeInteger(value['epoch']);
}

function isPositiveFiniteNumber(value: unknown): value is number {
    return isFiniteNumber(value) && value > 0;
}

function isBoolean(value: unknown): value is boolean {
    return typeof value === 'boolean';
}

function isFloat32Array(value: unknown): value is Float32Array {
    return value instanceof Float32Array;
}

function isUint8Array(value: unknown): value is Uint8Array {
    return value instanceof Uint8Array;
}

function isActivationHistogramLayer(value: unknown): value is ActivationHistogramLayer {
    if (!isRecord(value)) return false;
    return (
        isNonNegativeInteger(value['layerIndex']) &&
        isPositiveInteger(value['binCount']) &&
        isFiniteNumber(value['binStart']) &&
        isPositiveFiniteNumber(value['binWidth']) &&
        isFiniteNumber(value['minActivation']) &&
        isFiniteNumber(value['maxActivation']) &&
        value['minActivation'] <= value['maxActivation'] &&
        isNonNegativeInteger(value['totalCount']) &&
        isNonNegativeInteger(value['nearZeroCount']) &&
        isNonNegativeInteger(value['saturatedCount']) &&
        value['nearZeroCount'] <= value['totalCount'] &&
        value['saturatedCount'] <= value['totalCount']
    );
}

function isActivationHistogramLayout(value: unknown): value is ActivationHistogramLayout {
    if (!isRecord(value)) return false;
    if (!isPositiveInteger(value['binCount'])) return false;
    if (!Array.isArray(value['layers'])) return false;
    if (!value['layers'].every(isActivationHistogramLayer)) return false;
    return value['layers'].every((layer, index) => (
        layer.layerIndex === index && layer.binCount === value['binCount']
    ));
}

function hasMalformedActivationHistogramPayload(m: Record<string, unknown>): boolean {
    const hasBins = m['activationHistogramBins'] !== undefined;
    const hasLayout = m['activationHistogramLayout'] !== undefined;
    const hasVersion = m['activationHistogramVersion'] !== undefined;
    if (!hasBins && !hasLayout && !hasVersion) return false;
    if (!isFloat32Array(m['activationHistogramBins'])) return true;
    if (!isActivationHistogramLayout(m['activationHistogramLayout'])) return true;
    if (!isNonNegativeInteger(m['activationHistogramVersion'])) return true;
    const expectedBins = m['activationHistogramLayout'].layers.reduce(
        (sum, layer) => sum + layer.binCount,
        0,
    );
    if (m['activationHistogramBins'].length !== expectedBins
        || m['activationHistogramBins'].some(
            (value) => !Number.isSafeInteger(value) || value < 0,
        )) return true;
    let offset = 0;
    for (const layer of m['activationHistogramLayout'].layers) {
        let sum = 0;
        for (let index = 0; index < layer.binCount; index++) {
            sum += m['activationHistogramBins'][offset + index];
        }
        if (sum !== layer.totalCount) return true;
        offset += layer.binCount;
    }
    return false;
}

export interface MulticlassBoundaryLayout {
    gridSize: number;
    classCount: 3;
    classLabels: readonly [0, 1, 2];
}

function isMulticlassBoundaryLayout(
    value: unknown,
    expectedGridSize: number,
): value is MulticlassBoundaryLayout {
    if (!isRecord(value)) return false;
    if (value['gridSize'] !== expectedGridSize) return false;
    if (value['classCount'] !== 3) return false;
    const labels = value['classLabels'];
    return (
        Array.isArray(labels) &&
        labels.length === 3 &&
        labels[0] === 0 &&
        labels[1] === 1 &&
        labels[2] === 2
    );
}

function hasMalformedMulticlassBoundaryPayload(m: Record<string, unknown>): boolean {
    const hasClassGrid = m['multiclassClassGrid'] !== undefined;
    const hasConfidenceGrid = m['multiclassConfidenceGrid'] !== undefined;
    const hasLayout = m['multiclassBoundaryLayout'] !== undefined;
    const hasVersion = m['multiclassBoundaryVersion'] !== undefined;
    if (!hasClassGrid && !hasConfidenceGrid && !hasLayout && !hasVersion) return false;
    if (!isUint8Array(m['multiclassClassGrid'])) return true;
    if (!isFloat32Array(m['multiclassConfidenceGrid'])) return true;
    if (!isNonNegativeInteger(m['multiclassBoundaryVersion'])) return true;
    if (!isRecord(m['scalars']) || !isPositiveInteger(m['scalars']['gridSize'])) return true;
    const gridSize = m['scalars']['gridSize'];
    if (!isMulticlassBoundaryLayout(m['multiclassBoundaryLayout'], gridSize)) return true;
    const expectedLength = gridSize * gridSize;
    if (
        m['multiclassClassGrid'].length !== expectedLength ||
        m['multiclassConfidenceGrid'].length !== expectedLength
    ) {
        return true;
    }
    for (let i = 0; i < expectedLength; i++) {
        if (m['multiclassClassGrid'][i] > 2) return true;
        const confidence = m['multiclassConfidenceGrid'][i];
        if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) return true;
    }
    return false;
}

function isMulticlassConfusionMatrixData(value: unknown): value is MulticlassConfusionMatrixData {
    if (!isRecord(value)) return false;
    if (value['classCount'] !== 3) return false;
    const labels = value['classLabels'];
    if (
        !Array.isArray(labels) ||
        labels.length !== 3 ||
        labels[0] !== 0 ||
        labels[1] !== 1 ||
        labels[2] !== 2
    ) {
        return false;
    }
    const counts = value['counts'];
    if (!Array.isArray(counts) || counts.length !== 9) return false;
    for (let i = 0; i < 9; i++) {
        if (!Object.prototype.hasOwnProperty.call(counts, i)) return false;
        if (!isNonNegativeInteger(counts[i])) return false;
    }
    return true;
}

function isBinaryConfusionMatrixData(value: unknown): value is ConfusionMatrixData {
    if (!isRecord(value)
        || Object.keys(value).sort().join(',') !== 'fn,fp,tn,tp') return false;
    return isNonNegativeInteger(value['tp'])
        && isNonNegativeInteger(value['tn'])
        && isNonNegativeInteger(value['fp'])
        && isNonNegativeInteger(value['fn']);
}

function hasMalformedMulticlassConfusionMatrixPayload(m: Record<string, unknown>): boolean {
    const hasMatrix = m['multiclassConfusionMatrix'] !== undefined;
    const hasVersion = m['multiclassConfusionMatrixVersion'] !== undefined;
    if (!hasMatrix && !hasVersion) return false;
    if (!hasMatrix || !hasVersion) return true;
    if (m['confusionMatrix'] !== undefined) return true;
    return (
        !isMulticlassConfusionMatrixData(m['multiclassConfusionMatrix']) ||
        !isNonNegativeInteger(m['multiclassConfusionMatrixVersion'])
    );
}

function isOptionalFiniteNumber(value: unknown): value is number | undefined {
    return value === undefined || isFiniteNumber(value);
}

function isCheckpointSummary(value: unknown): value is CheckpointSummary {
    if (!hasExactOwnKeys(
        value,
        ['id', 'step', 'epoch', 'trainDataLoss', 'testDataLoss', 'label'],
        ['trainAccuracy', 'testAccuracy'],
    )) return false;
    return (
        isPositiveInteger(value['id']) &&
        isNonNegativeInteger(value['step']) &&
        isNonNegativeInteger(value['epoch']) &&
        isFiniteNumber(value['trainDataLoss']) &&
        isFiniteNumber(value['testDataLoss']) &&
        isOptionalFiniteNumber(value['trainAccuracy']) &&
        (value['trainAccuracy'] === undefined || (value['trainAccuracy'] >= 0 && value['trainAccuracy'] <= 1)) &&
        isOptionalFiniteNumber(value['testAccuracy']) &&
        (value['testAccuracy'] === undefined || (value['testAccuracy'] >= 0 && value['testAccuracy'] <= 1)) &&
        typeof value['label'] === 'string' &&
        value['label'].length > 0 &&
        value['label'].length <= 120
    );
}

function isNullablePositiveInteger(value: unknown): value is number | null {
    return value === null || isPositiveInteger(value);
}

function assertCheckpointTimelineV2Snapshot(value: unknown): asserts value is CheckpointTimeline {
    if (!hasExactOwnKeys(value, [
        'checkpoints',
        'maxCheckpoints',
        'evictedCount',
        'liveCheckpointId',
        'restoredCheckpointId',
    ])) {
        throw new TypeError('checkpoint timeline must contain exactly the version-2 metadata fields');
    }
    if (value['maxCheckpoints'] !== 8) {
        throw new TypeError('checkpoint timeline maxCheckpoints must be 8');
    }
    if (!Array.isArray(value['checkpoints']) || value['checkpoints'].length > 8) {
        throw new TypeError('checkpoint timeline checkpoints must be bounded to 8 entries');
    }
    if (!value['checkpoints'].every(isCheckpointSummary)) {
        throw new TypeError('checkpoint timeline contains a malformed checkpoint summary');
    }
    if (!isNonNegativeInteger(value['evictedCount'])) {
        throw new TypeError('checkpoint timeline evictedCount must be a non-negative integer');
    }
    if (!isNullablePositiveInteger(value['liveCheckpointId'])) {
        throw new TypeError('checkpoint timeline liveCheckpointId must be null or positive');
    }
    if (!isNullablePositiveInteger(value['restoredCheckpointId'])) {
        throw new TypeError('checkpoint timeline restoredCheckpointId must be null or positive');
    }
    const ids = value['checkpoints'].map((entry) => entry.id);
    if (new Set(ids).size !== ids.length) {
        throw new TypeError('checkpoint timeline checkpoint IDs must be unique');
    }
    if (value['liveCheckpointId'] !== null && !ids.includes(value['liveCheckpointId'])) {
        throw new TypeError('checkpoint timeline liveCheckpointId must be present in checkpoints');
    }
    if (value['restoredCheckpointId'] !== null
        && !ids.includes(value['restoredCheckpointId'])) {
        throw new TypeError('checkpoint timeline restoredCheckpointId must be present in checkpoints');
    }
}

export function parseCheckpointTimelineV2(value: unknown): CheckpointTimeline {
    const snapshot = createStableJsonSnapshot(value);
    assertCheckpointTimelineV2Snapshot(snapshot);
    return snapshot;
}

export function isCheckpointTimelineV2(value: unknown): value is CheckpointTimeline {
    try {
        assertCheckpointTimelineV2Snapshot(value);
        return true;
    } catch {
        return false;
    }
}

function hasMalformedCheckpointTimelinePayload(m: Record<string, unknown>): boolean {
    return !isCheckpointTimelineV2(m['checkpointTimeline']);
}

function hasValidSnapshotArtifactProvenance(m: Record<string, unknown>): boolean {
    if (!hasExactOwnKeys(m, [
        'type',
        'protocolVersion',
        'runId',
        'snapshotId',
        'model',
        'scalars',
        'weights',
        'biases',
        'weightLayout',
        'recipeFingerprint',
        'checkpointTimeline',
    ], [
        'outputGrid',
        'neuronGrids',
        'neuronGridLayout',
        'layerStats',
        'layerStatsGradientRevision',
        'activationHistogramBins',
        'activationHistogramLayout',
        'activationHistogramVersion',
        'multiclassClassGrid',
        'multiclassConfidenceGrid',
        'multiclassBoundaryLayout',
        'multiclassBoundaryVersion',
        'artifacts',
        'confusionMatrix',
        'confusionMatrixEvaluationId',
        'confusionMatrixVersion',
        'multiclassConfusionMatrix',
        'multiclassConfusionMatrixVersion',
        'sharedSeq',
    ])) return false;
    if (m['type'] !== 'snapshot' || m['protocolVersion'] !== WORKER_PROTOCOL_VERSION) {
        return false;
    }
    if (!isRequestId(m['runId'])) return false;
    if (!isNonNegativeInteger(m['snapshotId'])) return false;

    const scalars = m['scalars'];
    if (!hasExactOwnKeys(scalars, ['step', 'epoch', 'gridSize'])
        || !isNonNegativeInteger(scalars['step'])
        || !isNonNegativeInteger(scalars['epoch'])
        || !isPositiveInteger(scalars['gridSize'])) {
        return false;
    }
    const frameModel = m['model'];
    if (!isModelRevision(frameModel)
        || frameModel.generationId !== m['runId']
        || frameModel.step !== scalars['step']
        || frameModel.epoch !== scalars['epoch']) {
        return false;
    }

    if (m['outputGrid'] !== undefined && !isFloat32Array(m['outputGrid'])) return false;
    if (m['neuronGrids'] !== undefined && !isFloat32Array(m['neuronGrids'])) return false;
    if (m['layerStats'] !== undefined && !Array.isArray(m['layerStats'])) return false;
    if (m['sharedSeq'] !== undefined && !isNonNegativeInteger(m['sharedSeq'])) return false;
    const outputGrid = m['outputGrid'];
    if (isFloat32Array(outputGrid)
        && outputGrid.length > 0
        && (
            outputGrid.length !== scalars['gridSize'] * scalars['gridSize']
            || outputGrid.some((value) => !Number.isFinite(value))
        )) return false;

    const neuronLayout = m['neuronGridLayout'];
    if (neuronLayout !== undefined && (
        !isRecord(neuronLayout)
        || !isPositiveInteger(neuronLayout['count'])
        || neuronLayout['gridSize'] !== scalars['gridSize']
    )) return false;
    const neuronGrids = m['neuronGrids'];
    const neuronCount = isRecord(neuronLayout) && isPositiveInteger(neuronLayout['count'])
        ? neuronLayout['count']
        : undefined;
    if (isFloat32Array(neuronGrids) && neuronGrids.length > 0 && (
        neuronCount === undefined
        || neuronGrids.length
            !== neuronCount * scalars['gridSize'] * scalars['gridSize']
        || neuronGrids.some((value) => !Number.isFinite(value))
    )) return false;
    if (neuronLayout !== undefined
        && !(isFloat32Array(neuronGrids) && neuronGrids.length > 0)
        && m['sharedSeq'] === undefined) return false;

    if (!isFloat32Array(m['weights'])
        || !isFloat32Array(m['biases'])
        || !isRecord(m['weightLayout'])
        || !Array.isArray(m['weightLayout']['layerSizes'])
        || m['weightLayout']['layerSizes'].length < 2
        || !m['weightLayout']['layerSizes'].every(isPositiveInteger)
        || m['weights'].some((value) => !Number.isFinite(value))
        || m['biases'].some((value) => !Number.isFinite(value))) {
        return false;
    }
    if (!isFingerprint(m['recipeFingerprint'], 'r2.1.')) return false;
    const layerSizes = m['weightLayout']['layerSizes'] as number[];
    const expectedWeights = layerSizes.slice(0, -1).reduce(
        (sum, size, index) => sum + size * layerSizes[index + 1],
        0,
    );
    const expectedBiases = layerSizes.slice(1).reduce((sum, size) => sum + size, 0);
    if (m['weights'].length !== expectedWeights || m['biases'].length !== expectedBiases) {
        return false;
    }
    const expectedNeuronCount = layerSizes.slice(1).reduce((sum, size) => sum + size, 0);
    if (isRecord(neuronLayout) && neuronLayout['count'] !== expectedNeuronCount) return false;
    if (isFloat32Array(outputGrid)
        && outputGrid.length > 0
        && m['multiclassClassGrid'] !== undefined) return false;

    const layerStats = m['layerStats'];
    if (layerStats !== undefined && (
        !Array.isArray(layerStats)
        || layerStats.length !== layerSizes.length - 1
        || layerStats.some((stats) => (
            !isRecord(stats)
            || Object.keys(stats).sort().join(',')
                !== 'activationStd,meanAbsGradient,meanAbsWeight,meanActivation'
            || !isFiniteNumber(stats['meanActivation'])
            || !isFiniteNumber(stats['activationStd'])
            || stats['activationStd'] < 0
            || !isFiniteNumber(stats['meanAbsWeight'])
            || stats['meanAbsWeight'] < 0
            || !isFiniteNumber(stats['meanAbsGradient'])
            || stats['meanAbsGradient'] < 0
        ))
    )) return false;
    if (isRecord(m['activationHistogramLayout'])
        && Array.isArray(m['activationHistogramLayout']['layers'])
        && m['activationHistogramLayout']['layers'].length !== layerSizes.length - 1) {
        return false;
    }
    if (isFloat32Array(m['activationHistogramBins'])
        && m['activationHistogramBins'].some((value) => !Number.isFinite(value) || value < 0)) {
        return false;
    }

    const artifacts = m['artifacts'];
    if (artifacts !== undefined && !isWorkerArtifactProvenanceV2(artifacts)) return false;
    const artifactRecord = isRecord(artifacts) ? artifacts : null;
    const hasArtifact = (key: keyof WorkerArtifactProvenanceV2): boolean => (
        artifactRecord !== null && Object.prototype.hasOwnProperty.call(artifactRecord, key)
    );

    const hasSharedPayload = m['sharedSeq'] !== undefined;
    const decisionBoundaryPayload = (
        isFloat32Array(m['outputGrid']) && m['outputGrid'].length > 0
    ) || m['multiclassClassGrid'] !== undefined || hasSharedPayload;
    const neuronGridsPayload = (
        isFloat32Array(m['neuronGrids']) && m['neuronGrids'].length > 0
    ) || (hasSharedPayload && m['neuronGridLayout'] !== undefined);
    const activationStatisticsPayload = m['layerStats'] !== undefined;
    if (activationStatisticsPayload !== (m['layerStatsGradientRevision'] !== undefined)) {
        return false;
    }
    if (m['layerStatsGradientRevision'] !== undefined
        && !isNonNegativeInteger(m['layerStatsGradientRevision'])) {
        return false;
    }
    const presentArtifacts = artifactRecord === null
        ? []
        : Object.values(artifactRecord).filter(
            (value): value is ArtifactProvenance => value !== undefined,
        );
    const [firstArtifact, ...remainingArtifacts] = presentArtifacts;
    if (presentArtifacts.some((provenance) => provenance.model.generationId !== m['runId'])) {
        return false;
    }
    if (firstArtifact !== undefined
        && remainingArtifacts.some(
            (provenance) => !hasSameEvidenceIdentity(firstArtifact, provenance),
        )) {
        return false;
    }
    for (const key of [
        'decisionBoundary',
        'neuronGrids',
        'activationStatistics',
        'activationHistogram',
    ] as const) {
        const provenance = artifactRecord?.[key] as ArtifactProvenance | undefined;
        if (provenance !== undefined && (
            provenance.model.generationId !== frameModel.generationId
            || provenance.model.revision !== frameModel.revision
            || provenance.model.step !== frameModel.step
            || provenance.model.epoch !== frameModel.epoch
        )) return false;
    }
    const confusionProvenance = artifactRecord?.['confusionMatrix'] as
        | ArtifactProvenance
        | undefined;
    if (confusionProvenance !== undefined && (
        confusionProvenance.model.generationId !== frameModel.generationId
        || confusionProvenance.model.revision > frameModel.revision
        || confusionProvenance.model.step > frameModel.step
        || confusionProvenance.model.epoch > frameModel.epoch
    )) return false;
    const gridPointCount = scalars['gridSize'] * scalars['gridSize'];
    for (const key of ['decisionBoundary', 'neuronGrids'] as const) {
        const provenance = artifactRecord?.[key] as ArtifactProvenance | undefined;
        if (provenance !== undefined && (
            provenance.basis.kind !== 'prediction-grid'
            || provenance.basis.pointCount !== gridPointCount
            || provenance.basis.domain.some(
                (value, index) => value !== [-1, 1, -1, 1][index],
            )
        )) return false;
    }
    const activationHistogramProvenance = artifactRecord?.['activationHistogram'] as
        | ArtifactProvenance
        | undefined;
    if (activationHistogramProvenance !== undefined && (
        activationHistogramProvenance.basis.kind !== 'bounded-sample'
        || activationHistogramProvenance.basis.split !== 'train'
        || activationHistogramProvenance.basis.populationCount
            !== activationHistogramProvenance.dataset.trainCount
        || activationHistogramProvenance.basis.sampleCount
            !== Math.min(128, activationHistogramProvenance.dataset.trainCount)
    )) return false;
    const activationHistogramLayout = m['activationHistogramLayout'];
    if (activationHistogramProvenance !== undefined && (
        !isActivationHistogramLayout(activationHistogramLayout)
        || activationHistogramLayout.binCount !== 12
        || activationHistogramLayout.layers.length !== layerSizes.length - 1
        || activationHistogramLayout.layers.some((layer, index) => (
            layer.layerIndex !== index
            || layer.totalCount !== (
                activationHistogramProvenance.basis.kind === 'bounded-sample'
                    ? activationHistogramProvenance.basis.sampleCount * layerSizes[index + 1]
                    : -1
            )
        ))
    )) return false;
    const binaryConfusion = m['confusionMatrix'];
    const multiclassConfusion = m['multiclassConfusionMatrix'];
    const hasConfusionMatrix = binaryConfusion !== undefined || multiclassConfusion !== undefined;
    if (hasConfusionMatrix !== (m['confusionMatrixEvaluationId'] !== undefined)
        || (hasConfusionMatrix && !isPositiveInteger(m['confusionMatrixEvaluationId']))) {
        return false;
    }
    if (binaryConfusion !== undefined && (
        !isBinaryConfusionMatrixData(binaryConfusion)
        || !isNonNegativeInteger(m['confusionMatrixVersion'])
        || multiclassConfusion !== undefined
    )) return false;
    if (confusionProvenance !== undefined && (
        confusionProvenance.basis.kind !== 'full-split'
        || confusionProvenance.basis.split !== 'test'
    )) return false;
    const confusionSampleCount = confusionProvenance?.basis.kind === 'full-split'
        ? confusionProvenance.basis.sampleCount
        : undefined;
    if (confusionProvenance !== undefined && binaryConfusion !== undefined
        && isBinaryConfusionMatrixData(binaryConfusion)
        && binaryConfusion.tp + binaryConfusion.tn + binaryConfusion.fp + binaryConfusion.fn
            !== confusionSampleCount) return false;
    if (confusionProvenance !== undefined && isMulticlassConfusionMatrixData(multiclassConfusion)
        && multiclassConfusion.counts.reduce((sum, count) => sum + count, 0)
            !== confusionSampleCount) return false;
    const activationProvenance = artifactRecord?.['activationStatistics'] as
        | ArtifactProvenance
        | undefined;
    if (activationProvenance !== undefined
        && typeof m['layerStatsGradientRevision'] === 'number'
        && m['layerStatsGradientRevision'] > activationProvenance.model.revision) {
        return false;
    }
    const activationHistogramPayload = m['activationHistogramBins'] !== undefined;
    const confusionMatrixPayload = (
        m['confusionMatrix'] !== undefined || m['multiclassConfusionMatrix'] !== undefined
    );

    return (
        decisionBoundaryPayload === hasArtifact('decisionBoundary')
        && neuronGridsPayload === hasArtifact('neuronGrids')
        && activationStatisticsPayload === hasArtifact('activationStatistics')
        && activationHistogramPayload === hasArtifact('activationHistogram')
        && confusionMatrixPayload === hasArtifact('confusionMatrix')
        && !hasArtifact('predictionTrace')
        && !hasArtifact('lossLandscape')
    );
}

export function normalizeVisualizationDemand(value: unknown): VisualizationDemand | null {
    if (!hasExactOwnKeys(value, [
        'needDecisionBoundary',
        'needNeuronGrids',
        'needLayerStats',
        'needActivationHistograms',
        'needConfusionMatrix',
        'gridInterval',
        'activationHistogramInterval',
    ])) return null;

    const {
        needDecisionBoundary,
        needNeuronGrids,
        needLayerStats,
        needActivationHistograms,
        needConfusionMatrix,
        gridInterval,
        activationHistogramInterval,
    } = value;

    if (
        !isBoolean(needDecisionBoundary) ||
        !isBoolean(needNeuronGrids) ||
        !isBoolean(needLayerStats) ||
        !isBoolean(needActivationHistograms) ||
        !isBoolean(needConfusionMatrix) ||
        !isPositiveInteger(gridInterval) ||
        !isPositiveInteger(activationHistogramInterval)
    ) {
        return null;
    }

    return {
        needDecisionBoundary,
        needNeuronGrids,
        needLayerStats,
        needActivationHistograms,
        needConfusionMatrix,
        gridInterval,
        activationHistogramInterval,
    };
}

// ─────────────────────────────────────────────────────────
// Worker → Main  (streamed snapshot messages)
// ─────────────────────────────────────────────────────────

/** Lightweight scalars extracted from a snapshot. */
export interface SnapshotScalars {
    step: number;
    epoch: number;
    gridSize: number;
}

interface WorkerSnapshotMessageBase {
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
    /** Revision of the most recently applied clipped gradient summarized above. */
    layerStatsGradientRevision?: number;
    activationHistogramBins?: Float32Array;
    activationHistogramLayout?: ActivationHistogramLayout;
    activationHistogramVersion?: number;
    multiclassClassGrid?: Uint8Array;
    multiclassConfidenceGrid?: Float32Array;
    multiclassBoundaryLayout?: MulticlassBoundaryLayout;
    multiclassBoundaryVersion?: number;

    /** Per-artifact identity and basis for every strict heavy artifact payload. */
    artifacts?: WorkerArtifactProvenanceV2;
    confusionMatrix?: ConfusionMatrixData;
    /** Exact paired evaluation that produced the strict confusion payload. */
    confusionMatrixEvaluationId?: number;
    confusionMatrixVersion?: number;
    multiclassConfusionMatrix?: MulticlassConfusionMatrixData;
    multiclassConfusionMatrixVersion?: number;
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

/** Strict scientific-trust frame with mandatory current-model and checkpoint metadata. */
export interface WorkerSnapshotMessage extends WorkerSnapshotMessageBase {
    protocolVersion: typeof WORKER_PROTOCOL_VERSION;
    model: ModelRevision;
    recipeFingerprint: RecipeFingerprint;
    checkpointTimeline: CheckpointTimeline;
}

export interface ActivationHistogramLayout {
    binCount: number;
    layers: ActivationHistogramLayer[];
}

export interface CheckpointSummary {
    id: number;
    step: number;
    epoch: number;
    trainDataLoss: number;
    testDataLoss: number;
    trainAccuracy?: number;
    testAccuracy?: number;
    label: string;
}

export interface CheckpointTimeline {
    checkpoints: CheckpointSummary[];
    maxCheckpoints: number;
    evictedCount: number;
    liveCheckpointId: number | null;
    restoredCheckpointId: number | null;
}

export interface WorkerStatusMessage {
    type: 'status';
    protocolVersion: typeof WORKER_PROTOCOL_VERSION;
    runId: number;
    status: 'idle' | 'running' | 'paused';
    pauseReason?: PauseReason | null;
    checkpointTimeline?: CheckpointTimeline;
}

export interface WorkerErrorMessage {
    type: 'error';
    protocolVersion: typeof WORKER_PROTOCOL_VERSION;
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
    protocolVersion: typeof WORKER_PROTOCOL_VERSION;
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

const SHARED_CONTROL_WORD_COUNT = 8;
const MAX_SHARED_NEURON_COUNT = MAX_HIDDEN_LAYERS * MAX_NEURONS_PER_LAYER + 3;

function exactSharedArrayBufferByteLength(value: unknown): number | null {
    if (typeof SharedArrayBuffer !== 'function') return null;
    const byteLengthGetter = Object.getOwnPropertyDescriptor(
        SharedArrayBuffer.prototype,
        'byteLength',
    )?.get;
    if (byteLengthGetter === undefined) return null;
    try {
        const byteLength = byteLengthGetter.call(value) as unknown;
        return Number.isSafeInteger(byteLength) && (byteLength as number) >= 0
            ? byteLength as number
            : null;
    } catch {
        return null;
    }
}

function safePositiveProduct(...factors: number[]): number | null {
    let product = 1;
    for (const factor of factors) {
        if (!Number.isSafeInteger(factor) || factor <= 0) return null;
        if (product > Number.MAX_SAFE_INTEGER / factor) return null;
        product *= factor;
    }
    return product;
}

function isWorkerSharedBuffersMessage(
    message: Record<string, unknown>,
): boolean {
    if (!hasExactOwnKeys(message, [
        'type',
        'protocolVersion',
        'runId',
        'control',
        'outputGrid',
        'neuronGrids',
        'gridSize',
        'neuronGridLayout',
    ])) return false;
    const gridSize = message['gridSize'];
    const layout = message['neuronGridLayout'];
    if (!Number.isSafeInteger(gridSize)
        || (gridSize as number) <= 0
        || (gridSize as number) > GRID_SIZE
        || !hasExactOwnKeys(layout, ['count', 'gridSize'])) {
        return false;
    }
    const neuronCount = layout['count'];
    if (!Number.isSafeInteger(neuronCount)
        || (neuronCount as number) <= 0
        || (neuronCount as number) > MAX_SHARED_NEURON_COUNT
        || layout['gridSize'] !== gridSize) {
        return false;
    }
    const gridPointCount = safePositiveProduct(gridSize as number, gridSize as number);
    if (gridPointCount === null) return false;
    const outputBytes = safePositiveProduct(gridPointCount, Float32Array.BYTES_PER_ELEMENT);
    const neuronBytes = safePositiveProduct(
        neuronCount as number,
        gridPointCount,
        Float32Array.BYTES_PER_ELEMENT,
    );
    if (outputBytes === null || neuronBytes === null) return false;
    return exactSharedArrayBufferByteLength(message['control'])
            === SHARED_CONTROL_WORD_COUNT * Int32Array.BYTES_PER_ELEMENT
        && exactSharedArrayBufferByteLength(message['outputGrid']) === outputBytes
        && exactSharedArrayBufferByteLength(message['neuronGrids']) === neuronBytes;
}

export type WorkerToMainMessage =
    | WorkerSnapshotMessage
    | WorkerStatusMessage
    | WorkerErrorMessage
    | WorkerSharedBuffersMessage
    | WorkerToMainMessageV2;

// ─────────────────────────────────────────────────────────
// Main → Worker  (streaming commands via MessageChannel)
// ─────────────────────────────────────────────────────────

export interface StartTrainingCommand {
    type: 'startTraining';
    protocolVersion: typeof WORKER_PROTOCOL_VERSION;
    stepsPerFrame: number;
}

export interface StopTrainingCommand {
    type: 'stopTraining';
    protocolVersion: typeof WORKER_PROTOCOL_VERSION;
}

export interface UpdateDemandCommand {
    type: 'updateDemand';
    protocolVersion: typeof WORKER_PROTOCOL_VERSION;
    demand: VisualizationDemand;
}

export interface UpdateSpeedCommand {
    type: 'updateSpeed';
    protocolVersion: typeof WORKER_PROTOCOL_VERSION;
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
    protocolVersion: typeof WORKER_PROTOCOL_VERSION;
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
function isWorkerToMainMessageUnchecked(x: unknown): x is WorkerToMainMessage {
    if (isRecord(x)
        && (x['type'] === 'evidence' || x['type'] === 'worker-error')) {
        return isWorkerToMainMessageV2(x);
    }
    if (!isRecord(x)) return false;
    const m = x as Record<string, unknown>;
    if (typeof m['type'] !== 'string') return false;
    if (m['protocolVersion'] !== WORKER_PROTOCOL_VERSION || !isRequestId(m['runId'])) {
        return false;
    }
    switch (m['type']) {
        case 'snapshot':
            return (
                typeof m['snapshotId'] === 'number' &&
                m['scalars'] !== null &&
                typeof m['scalars'] === 'object' &&
                hasValidSnapshotArtifactProvenance(m) &&
                !hasMalformedActivationHistogramPayload(m) &&
                !hasMalformedMulticlassBoundaryPayload(m) &&
                !hasMalformedMulticlassConfusionMatrixPayload(m) &&
                !hasMalformedCheckpointTimelinePayload(m)
            );
        case 'status':
            return (
                hasExactOwnKeys(
                    m,
                    ['type', 'protocolVersion', 'runId', 'status'],
                    ['pauseReason', 'checkpointTimeline'],
                ) &&
                (
                    m['status'] === 'idle' ||
                    m['status'] === 'running' ||
                    m['status'] === 'paused'
                ) &&
                (
                    !('pauseReason' in m) ||
                    m['pauseReason'] === null ||
                    isPauseReason(m['pauseReason'])
                ) &&
                (
                    !('checkpointTimeline' in m) ||
                    isCheckpointTimelineV2(m['checkpointTimeline'])
                )
            );
        case 'error':
            return hasExactOwnKeys(
                m,
                ['type', 'protocolVersion', 'runId', 'message'],
            ) && isBoundedString(m['message'], 4_096);
        case 'sharedBuffers':
            return isWorkerSharedBuffersMessage(m);
        default:
            return false;
    }
}

/** Total Worker -> Main guard: adversarial accessors and proxies fail closed. */
export function isWorkerToMainMessage(x: unknown): x is WorkerToMainMessage {
    try {
        return isWorkerToMainMessageUnchecked(x);
    } catch {
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
    if (typeof m['type'] !== 'string' || m['protocolVersion'] !== WORKER_PROTOCOL_VERSION) {
        return false;
    }
    switch (m['type']) {
        case 'startTraining':
            return hasExactOwnKeys(
                m,
                ['type', 'protocolVersion', 'stepsPerFrame'],
            ) && isPositiveInteger(m['stepsPerFrame']);
        case 'stopTraining':
            return hasExactOwnKeys(m, ['type', 'protocolVersion']);
        case 'updateDemand':
            return hasExactOwnKeys(
                m,
                ['type', 'protocolVersion', 'demand'],
            ) && normalizeVisualizationDemand(m['demand']) !== null;
        case 'updateSpeed':
            return hasExactOwnKeys(
                m,
                ['type', 'protocolVersion', 'stepsPerFrame'],
            ) && isPositiveInteger(m['stepsPerFrame']);
        case 'frameAck':
            return hasExactOwnKeys(m, ['type', 'protocolVersion']);
        default:
            return false;
    }
}
