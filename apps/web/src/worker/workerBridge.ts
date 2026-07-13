// ── Worker Bridge ──
// Manages Communication with the training web worker.
// - Comlink proxy for request/response RPC (initialize, updateConfig, reset, step, etc.)
// - MessageChannel for high-frequency streamed snapshots during training
// - rAF-gated rendering loop that applies at most one snapshot per animation frame

import * as Comlink from 'comlink';
import type { TrainingWorkerApi } from './training.worker.ts';
import type {
    WorkerToMainMessage,
    WorkerSharedBuffersMessage,
    MainToWorkerCommand,
    WorkerSnapshotMessage,
    WorkerEvidenceMessageV2,
    WorkerArtifactProvenanceV2,
    ArtifactProvenance,
} from '@nn-playground/shared';
import {
    isWorkerToMainMessage,
    parseArtifactProvenance,
    parseWorkerToMainMessageV2,
    WORKER_PROTOCOL_VERSION,
} from '@nn-playground/shared';
import {
    updateFrameBuffer,
    resetFrameBuffer,
    type FrameBufferPatch,
} from './frameBuffer.ts';
import {
    attachSharedSnapshotViews,
    FLAG_NEURON_GRIDS,
    FLAG_OUTPUT_GRID,
    readSharedSnapshot,
    type SharedSnapshotViews,
} from './sharedSnapshot.ts';

// ── Singleton state ──
let _worker: Worker | null = null;
let _comlinkApi: Comlink.Remote<TrainingWorkerApi> | null = null;
let _streamPort: MessagePort | null = null;
let _currentRunId = 0;
let _latestSnapshotId = -1;
let _minimumSnapshotRevision = 0;
let _rafId: number | null = null;
let _pendingSnapshot: WorkerToMainMessage | null = null;

// ── Shared-snapshot transport (AS-3) ──────────────────────────────────────
// When the worker successfully allocates SharedArrayBuffers, it sends a
// `sharedBuffers` handshake; we attach views here and use them to read
// grid payloads without ever receiving them through postMessage. The
// `_sharedReadBuffers` are private per-frame read destinations so the
// renderer never observes torn state even if the worker publishes again
// while React is mid-paint.
let _sharedViews: SharedSnapshotViews | null = null;
let _sharedViewsRunId: number | null = null;
/** Staging destinations are never exposed until a complete frame commit. */
let _sharedOutputReadBuf: Float32Array | null = null;
let _sharedNeuronReadBuf: Float32Array | null = null;
/** Previously committed buffers; swapped back to staging after the next commit. */
let _sharedOutputPublishedBuf: Float32Array | null = null;
let _sharedNeuronPublishedBuf: Float32Array | null = null;
let _sharedNeuronGridLayout: { count: number; gridSize: number } | null = null;

function installSharedBuffers(msg: WorkerSharedBuffersMessage): void {
    const nextViews = attachSharedSnapshotViews({
        control: msg.control,
        outputGrid: msg.outputGrid,
        neuronGrids: msg.neuronGrids,
        gridSize: msg.gridSize,
        neuronCount: msg.neuronGridLayout.count,
    });
    // Allocate reader-side destination arrays sized to the new shape.
    // These are regular (non-shared) Float32Arrays so downstream renderers
    // work on a stable copy; the cost is a single memcpy per snapshot.
    const nextOutputReadBuf = new Float32Array(msg.gridSize * msg.gridSize);
    const nextNeuronReadBuf = new Float32Array(
        Math.max(1, msg.neuronGridLayout.count * msg.gridSize * msg.gridSize),
    );
    const nextOutputPublishedBuf = new Float32Array(msg.gridSize * msg.gridSize);
    const nextNeuronPublishedBuf = new Float32Array(
        Math.max(1, msg.neuronGridLayout.count * msg.gridSize * msg.gridSize),
    );
    _sharedViews = nextViews;
    _sharedViewsRunId = msg.runId;
    _sharedOutputReadBuf = nextOutputReadBuf;
    _sharedNeuronReadBuf = nextNeuronReadBuf;
    _sharedOutputPublishedBuf = nextOutputPublishedBuf;
    _sharedNeuronPublishedBuf = nextNeuronPublishedBuf;
    _sharedNeuronGridLayout = msg.neuronGridLayout;
}

function tearDownSharedBuffers(): void {
    _sharedViews = null;
    _sharedViewsRunId = null;
    _sharedOutputReadBuf = null;
    _sharedNeuronReadBuf = null;
    _sharedOutputPublishedBuf = null;
    _sharedNeuronPublishedBuf = null;
    _sharedNeuronGridLayout = null;
}

function clearSharedBuffersIfRunMismatch(): void {
    if (_sharedViewsRunId !== null && _sharedViewsRunId !== _currentRunId) {
        tearDownSharedBuffers();
    }
}

// Callback for when a new snapshot is ready to be applied (called from rAF loop)
type SnapshotCallback = (msg: WorkerToMainMessage) => void;
let _onSnapshot: SnapshotCallback | null = null;

// Synthesize a WorkerErrorMessage and dispatch it through _onSnapshot so that
// bridge-level failures (onerror, onmessageerror) surface through the same
// path as worker-emitted errors.
function emitWorkerError(message: string): void {
    if (_onSnapshot) {
        _onSnapshot({ type: 'error', protocolVersion: 2, runId: _currentRunId, message });
    }
}

function deepFreezeArtifact<T>(value: T): Readonly<T> {
    if (typeof value !== 'object' || value === null || Object.isFrozen(value)) return value;
    for (const key of Reflect.ownKeys(value)) {
        deepFreezeArtifact((value as Record<PropertyKey, unknown>)[key]);
    }
    return Object.freeze(value);
}

function snapshotWithFrozenArtifactProvenance(
    message: WorkerSnapshotMessage,
): WorkerSnapshotMessage {
    const model = Object.freeze({ ...message.model });
    if (message.artifacts === undefined) return { ...message, model };
    const parsed: Partial<Record<keyof WorkerArtifactProvenanceV2, ArtifactProvenance>> = {};
    for (const key of Object.keys(message.artifacts) as Array<keyof WorkerArtifactProvenanceV2>) {
        const provenance = message.artifacts[key];
        if (provenance !== undefined) {
            parsed[key] = deepFreezeArtifact(parseArtifactProvenance(provenance));
        }
    }
    return {
        ...message,
        model,
        artifacts: Object.freeze(parsed) as WorkerArtifactProvenanceV2,
    };
}

// ── Initialization ──

function ensureWorker(): Worker {
    if (!_worker) {
        _worker = new Worker(
            new URL('./training.worker.ts', import.meta.url),
            { type: 'module' },
        );
        _worker.onerror = (event: ErrorEvent) => {
            emitWorkerError(`Worker error: ${event.message ?? 'unknown'}`);
        };
        _worker.onmessageerror = () => {
            emitWorkerError('Worker message deserialization error');
        };
    }
    return _worker;
}

/**
 * Get the Comlink proxy for RPC-style commands.
 */
export function getWorkerApi(): Comlink.Remote<TrainingWorkerApi> {
    if (!_comlinkApi) {
        const worker = ensureWorker();
        _comlinkApi = Comlink.wrap<TrainingWorkerApi>(worker);
    }
    return _comlinkApi;
}

/**
 * Set up the MessageChannel for streaming snapshot delivery.
 * Call this once after the worker is initialized.
 */
export async function setupStreamChannel(): Promise<void> {
    if (_streamPort) return; // Already set up

    const api = getWorkerApi();
    const channel = new MessageChannel();
    _streamPort = channel.port1;

    // Pass port2 to the worker via Comlink
    await api.setStreamPort(Comlink.transfer(channel.port2, [channel.port2]));

    // Listen for streamed messages on port1
    _streamPort.addEventListener('message', (event: MessageEvent<unknown>) => {
        handleWorkerMessage(event.data);
    });
    _streamPort.onmessageerror = () => {
        emitWorkerError('Stream port message deserialization error');
    };
    _streamPort.start();
}

// ── Message Handling ──

function handleWorkerMessage(value: unknown): void {
    // Validate message shape before processing.
    let msg: WorkerToMainMessage | null = null;
    try {
        if (isWorkerToMainMessage(value)) msg = value;
    } catch {
        msg = null;
    }
    if (msg === null) {
        emitWorkerError(
            'Received malformed message from worker: ' + describeMalformedMessage(value),
        );
        return;
    }

    // Scientific evidence is not visual-frame state: deliver it immediately
    // so cadence pairs cannot be overwritten by the rAF latest-wins slot.
    if (msg.type === 'evidence') {
        const evidence = parseWorkerToMainMessageV2(msg);
        if (evidence.type !== 'evidence') {
            emitWorkerError('Received malformed V2 evidence discriminator');
            return;
        }
        const generationId = evidenceGenerationId(evidence);
        if (generationId !== _currentRunId) return;
        if (_onSnapshot) _onSnapshot(evidence);
        return;
    }

    // Structured errors always surface, even if their originating generation
    // has already been replaced.
    if (msg.type === 'worker-error') {
        const error = parseWorkerToMainMessageV2(msg);
        if (error.type !== 'worker-error') {
            emitWorkerError('Received malformed V2 error discriminator');
            return;
        }
        if (_onSnapshot) _onSnapshot(error);
        return;
    }

    // Error messages always surface — even from stale runs — so async failures
    // after a reset are never silently dropped.
    if (msg.type !== 'error' && msg.runId < _currentRunId) return;

    if (msg.type === 'snapshot') {
        const snapshot = snapshotWithFrozenArtifactProvenance(msg);
        if (snapshot.runId === _currentRunId && _minimumSnapshotRevision > 0) {
            if (snapshot.model === undefined
                || snapshot.model.generationId !== snapshot.runId
                || !Number.isSafeInteger(snapshot.model.revision)
                || snapshot.model.revision < 0) {
                if (_streamPort) _streamPort.postMessage({
                    type: 'frameAck',
                    protocolVersion: WORKER_PROTOCOL_VERSION,
                });
                emitWorkerError('Received strict snapshot without a valid current model revision');
                return;
            }
            if (snapshot.model.revision < _minimumSnapshotRevision) {
                // Comlink replies and stream-port messages use independent
                // channels. A pre-restore frame can therefore arrive after
                // the restore RPC; acknowledge it without making it visible.
                if (_streamPort) _streamPort.postMessage({
                    type: 'frameAck',
                    protocolVersion: WORKER_PROTOCOL_VERSION,
                });
                return;
            }
        }
        // Drop out-of-order snapshots
        if (snapshot.snapshotId <= _latestSnapshotId && snapshot.runId === _currentRunId) return;
        _latestSnapshotId = snapshot.snapshotId;

        // Store as pending — will be applied on next rAF tick (latest-wins)
        _pendingSnapshot = snapshot;
    } else if (msg.type === 'sharedBuffers') {
        // Worker (re)allocated its SAB transport. Install views immediately
        // so the very next snapshot can read from them. Never queued to rAF
        // — we need this in place before any snapshot referring to it
        // arrives, and it carries no per-frame data.
        try {
            installSharedBuffers(msg);
        } catch {
            emitWorkerError('Received malformed shared-buffer handshake from worker');
        }
    } else {
        // Status/error messages are applied immediately
        if (_onSnapshot) _onSnapshot(msg);
    }
}

function describeMalformedMessage(value: unknown): string {
    try {
        return JSON.stringify(value) ?? String(value);
    } catch {
        return '[unserializable message]';
    }
}

function evidenceGenerationId(message: WorkerEvidenceMessageV2): number {
    if (message.liveSignal) return message.liveSignal.model.generationId;
    if (message.latestEvaluation) return message.latestEvaluation.model.generationId;
    const artifact = message.artifacts && Object.values(message.artifacts)[0];
    if (!artifact) throw new Error('validated evidence must contain an identity');
    return artifact.model.generationId;
}

// ── rAF Render Loop ──

// Build a minimal frame-buffer patch from a snapshot message. Only fields
// that are actually present in the message are written — this is essential
// for the cadence-gated snapshots, where the worker omits the grid on
// reuse frames and the main thread must retain the previously cached one.
function buildSnapshotFramePatch(
    msg: WorkerSnapshotMessage,
    currentRunId: number,
    sharedViews: SharedSnapshotViews | null,
    sharedViewsRunId: number | null,
    sharedOutputReadBuf: Float32Array | null,
    sharedNeuronReadBuf: Float32Array | null,
    sharedNeuronGridLayout: { count: number; gridSize: number } | null,
): FrameBufferPatch {
    const patch: FrameBufferPatch = {};
    const hasMulticlassBoundaryPayload = msg.multiclassClassGrid !== undefined;

    // AS-3 fast path: grid payloads were published through SharedArrayBuffers;
    // read them via the seqlock into our stable, non-shared read buffers and
    // point the frame buffer at those copies. We copy (rather than handing
    // the UI raw SAB views) because renderers paint across multiple rAF
    // ticks and can't tolerate the worker overwriting a view mid-paint.
    if (
        msg.sharedSeq !== undefined &&
        sharedViews &&
        sharedViewsRunId === msg.runId &&
        sharedViewsRunId === currentRunId &&
        sharedOutputReadBuf &&
        sharedNeuronReadBuf
    ) {
        const result = readSharedSnapshot(
            sharedViews,
            sharedOutputReadBuf,
            sharedNeuronReadBuf,
        );
        const strictFlags = FLAG_OUTPUT_GRID
            | (msg.artifacts?.neuronGrids === undefined ? 0 : FLAG_NEURON_GRIDS);
        const matchesStrictEnvelope = result !== null
            && result.seq === msg.sharedSeq
            && result.flags === strictFlags;
        if (result && matchesStrictEnvelope) {
            if ((result.flags & FLAG_OUTPUT_GRID) !== 0) {
                patch.outputGrid = sharedOutputReadBuf;
                patch.gridSize = msg.scalars.gridSize;
                patch.multiclassClassGrid = null;
                patch.multiclassConfidenceGrid = null;
                patch.multiclassBoundaryLayout = null;
            }
            if ((result.flags & FLAG_NEURON_GRIDS) !== 0) {
                patch.neuronGrids = sharedNeuronReadBuf;
                patch.neuronGridLayout =
                    msg.neuronGridLayout ?? sharedNeuronGridLayout;
            }
        }
        // If the seqlock read torn through all retries, skip the grid
        // update this frame — the UI will pick up the next consistent
        // publish. No inline fallback available (data isn't on the msg).
    } else {
        // Transferable postMessage path — grids arrived inline.
        if (msg.outputGrid !== undefined) {
            patch.outputGrid = msg.outputGrid.length > 0 ? msg.outputGrid : null;
            patch.gridSize = msg.outputGrid.length > 0 ? msg.scalars.gridSize : 0;
            if (!hasMulticlassBoundaryPayload && msg.outputGrid.length > 0) {
                patch.multiclassClassGrid = null;
                patch.multiclassConfidenceGrid = null;
                patch.multiclassBoundaryLayout = null;
            } else if (!hasMulticlassBoundaryPayload && msg.outputGrid.length === 0) {
                patch.multiclassClassGrid = null;
                patch.multiclassConfidenceGrid = null;
                patch.multiclassBoundaryLayout = null;
            }
        }
        if (msg.neuronGrids !== undefined) {
            patch.neuronGrids = msg.neuronGrids.length > 0 ? msg.neuronGrids : null;
            patch.neuronGridLayout = msg.neuronGrids.length > 0
                ? msg.neuronGridLayout ?? null
                : null;
        }
    }

    if (hasMulticlassBoundaryPayload) {
        patch.outputGrid = null;
        patch.neuronGrids = null;
        patch.neuronGridLayout = null;
    }
    if (msg.multiclassClassGrid !== undefined) {
        patch.multiclassClassGrid = msg.multiclassClassGrid;
    }
    if (msg.multiclassConfidenceGrid !== undefined) {
        patch.multiclassConfidenceGrid = msg.multiclassConfidenceGrid;
    }
    if (msg.multiclassBoundaryLayout !== undefined) {
        patch.multiclassBoundaryLayout = msg.multiclassBoundaryLayout;
        patch.gridSize = msg.multiclassBoundaryLayout.gridSize;
    }

    if (msg.weights !== undefined) patch.weights = msg.weights;
    if (msg.biases !== undefined) patch.biases = msg.biases;
    if (msg.weightLayout !== undefined) patch.weightLayout = msg.weightLayout;
    if (msg.weights !== undefined || msg.biases !== undefined || msg.weightLayout !== undefined) {
        patch.parameterProvenance = {
            model: msg.model,
            recipeFingerprint: msg.recipeFingerprint,
        };
        Object.freeze(patch.parameterProvenance);
    }
    if (msg.layerStats !== undefined) patch.layerStats = msg.layerStats;
    if (msg.activationHistogramBins !== undefined) {
        patch.activationHistogramBins = msg.activationHistogramBins;
    }
    if (msg.activationHistogramLayout !== undefined) {
        patch.activationHistogramLayout = msg.activationHistogramLayout;
    }
    if (msg.confusionMatrix !== undefined) {
        patch.confusionMatrix = msg.confusionMatrix;
        patch.multiclassConfusionMatrix = null;
    }
    if (msg.multiclassConfusionMatrix !== undefined) {
        patch.multiclassConfusionMatrix = msg.multiclassConfusionMatrix;
        patch.confusionMatrix = null;
    }
    attachStrictArtifactProvenance(msg, patch);
    return patch;
}

function rotateCommittedSharedReadBuffers(patch: FrameBufferPatch): void {
    if (_sharedOutputReadBuf !== null
        && _sharedOutputPublishedBuf !== null
        && patch.outputGrid === _sharedOutputReadBuf) {
        const previousPublished = _sharedOutputPublishedBuf;
        _sharedOutputPublishedBuf = _sharedOutputReadBuf;
        _sharedOutputReadBuf = previousPublished;
    }
    if (_sharedNeuronReadBuf !== null
        && _sharedNeuronPublishedBuf !== null
        && patch.neuronGrids === _sharedNeuronReadBuf) {
        const previousPublished = _sharedNeuronPublishedBuf;
        _sharedNeuronPublishedBuf = _sharedNeuronReadBuf;
        _sharedNeuronReadBuf = previousPublished;
    }
}

function patchHasOwn(patch: FrameBufferPatch, key: keyof FrameBufferPatch): boolean {
    return Object.prototype.hasOwnProperty.call(patch, key);
}

function attachStrictArtifactProvenance(
    msg: WorkerSnapshotMessage,
    patch: FrameBufferPatch,
): void {
    const artifacts = msg.artifacts;

    const boundaryMutated = patchHasOwn(patch, 'outputGrid')
        || patchHasOwn(patch, 'multiclassClassGrid')
        || patchHasOwn(patch, 'multiclassConfidenceGrid')
        || patchHasOwn(patch, 'multiclassBoundaryLayout');
    if (boundaryMutated) {
        const boundaryPresent = patch.outputGrid != null
            || patch.multiclassClassGrid != null
            || patch.multiclassConfidenceGrid != null;
        patch.decisionBoundaryProvenance = boundaryPresent
            ? artifacts!.decisionBoundary!
            : null;
    }

    const neuronMutated = patchHasOwn(patch, 'neuronGrids')
        || patchHasOwn(patch, 'neuronGridLayout');
    if (neuronMutated) {
        patch.neuronGridsProvenance = patch.neuronGrids != null
            ? artifacts!.neuronGrids!
            : null;
    }

    if (patchHasOwn(patch, 'layerStats')) {
        patch.layerStatsProvenance = patch.layerStats != null
            ? artifacts!.activationStatistics!
            : null;
        patch.layerStatsGradientRevision = patch.layerStats != null
            ? msg.layerStatsGradientRevision!
            : null;
    }

    const histogramMutated = patchHasOwn(patch, 'activationHistogramBins')
        || patchHasOwn(patch, 'activationHistogramLayout');
    if (histogramMutated) {
        patch.activationHistogramProvenance = patch.activationHistogramBins != null
            ? artifacts!.activationHistogram!
            : null;
    }

    const confusionMutated = patchHasOwn(patch, 'confusionMatrix')
        || patchHasOwn(patch, 'multiclassConfusionMatrix');
    if (confusionMutated) {
        const matrixPresent = patch.confusionMatrix != null
            || patch.multiclassConfusionMatrix != null;
        patch.confusionMatrixProvenance = matrixPresent
            ? artifacts!.confusionMatrix!
            : null;
        patch.confusionMatrixEvaluationId = matrixPresent
            ? msg.confusionMatrixEvaluationId!
            : null;
    }
}

function rafLoop(): void {
    if (_rafId === null) return; // Stopped

    if (_pendingSnapshot) {
        const msg = _pendingSnapshot;
        _pendingSnapshot = null;

        // Write heavy arrays to frame buffer
        if (msg.type === 'snapshot') {
            const patch = buildSnapshotFramePatch(
                msg,
                _currentRunId,
                _sharedViews,
                _sharedViewsRunId,
                _sharedOutputReadBuf,
                _sharedNeuronReadBuf,
                _sharedNeuronGridLayout,
            );
            updateFrameBuffer(patch, { requireArtifactProvenance: true });
            rotateCommittedSharedReadBuffers(patch);
        }

        // Notify the subscriber (typically updates useTrainingStore scalars)
        if (_onSnapshot) _onSnapshot(msg);

        // Ack snapshots to release the worker's back-pressure gate. Status/
        // error messages bypass the gate, so they don't need an ack.
        if (msg.type === 'snapshot' && _streamPort) {
            _streamPort.postMessage({
                type: 'frameAck',
                protocolVersion: WORKER_PROTOCOL_VERSION,
            });
        }
    }

    _rafId = requestAnimationFrame(rafLoop);
}

/**
 * Start the rAF render loop that applies pending snapshots.
 */
export function startRenderLoop(): void {
    if (_rafId !== null) return; // Already running
    _rafId = requestAnimationFrame(rafLoop);
}

/**
 * Stop the rAF render loop.
 */
export function stopRenderLoop(): void {
    if (_rafId !== null) {
        cancelAnimationFrame(_rafId);
        _rafId = null;
    }
    // Apply any final pending snapshot
    if (_pendingSnapshot && _onSnapshot) {
        const msg = _pendingSnapshot;
        _pendingSnapshot = null;
        if (msg.type === 'snapshot') {
            const patch = buildSnapshotFramePatch(
                msg,
                _currentRunId,
                _sharedViews,
                _sharedViewsRunId,
                _sharedOutputReadBuf,
                _sharedNeuronReadBuf,
                _sharedNeuronGridLayout,
            );
            updateFrameBuffer(patch, { requireArtifactProvenance: true });
            rotateCommittedSharedReadBuffers(patch);
        }
        _onSnapshot(msg);
        if (msg.type === 'snapshot' && _streamPort) {
            _streamPort.postMessage({
                type: 'frameAck',
                protocolVersion: WORKER_PROTOCOL_VERSION,
            });
        }
    }
}

// ── Streaming Commands ──

/**
 * Send a streaming command to the worker via the MessageChannel.
 */
export function postStreamCommand(cmd: MainToWorkerCommand): void {
    if (!_streamPort) {
        return;
    }
    _streamPort.postMessage(cmd);
}

// ── Run Lifecycle ──

/**
 * Increment the run ID. Call this when the network is reinitialized or reset.
 * Returns the new run ID.
 */
export function newRun(): number {
    _currentRunId++;
    _latestSnapshotId = -1;
    _minimumSnapshotRevision = 0;
    _pendingSnapshot = null;
    clearSharedBuffersIfRunMismatch();
    return _currentRunId;
}

/**
 * Set the run ID to a specific value (used to sync with worker's runId).
 */
export function newRunTo(targetRunId: number): void {
    _currentRunId = targetRunId;
    _latestSnapshotId = -1;
    _minimumSnapshotRevision = 0;
    _pendingSnapshot = null;
    clearSharedBuffersIfRunMismatch();
}

/**
 * Drop one visual frame queued before a same-generation restore commit and,
 * when provided, retain a minimum revision fence for delayed stream frames.
 * The snapshot ID fence is intentionally retained, and every discarded frame
 * is acknowledged so worker back-pressure cannot remain stuck.
 */
export function discardPendingSnapshot(
    runId: number,
    minimumRevision?: number,
): boolean {
    if (runId !== _currentRunId) return false;
    if (minimumRevision !== undefined) {
        if (!Number.isSafeInteger(minimumRevision) || minimumRevision < 0) {
            throw new RangeError('minimum snapshot revision must be a non-negative safe integer');
        }
        _minimumSnapshotRevision = Math.max(_minimumSnapshotRevision, minimumRevision);
    }
    if (_pendingSnapshot?.type !== 'snapshot' || _pendingSnapshot.runId !== runId) {
        return false;
    }
    _pendingSnapshot = null;
    if (_streamPort) _streamPort.postMessage({
        type: 'frameAck',
        protocolVersion: WORKER_PROTOCOL_VERSION,
    });
    return true;
}

/**
 * Get the current run ID.
 */
export function getCurrentRunId(): number {
    return _currentRunId;
}

// ── Subscription ──

/**
 * Register a callback for snapshot/status updates.
 * Only one callback is supported at a time.
 */
export function onSnapshot(callback: SnapshotCallback): () => void {
    _onSnapshot = callback;
    return () => {
        if (_onSnapshot === callback) _onSnapshot = null;
    };
}

// ── Cleanup ──

export function terminateWorker(): void {
    stopRenderLoop();
    if (_streamPort) {
        _streamPort.close();
        _streamPort = null;
    }
    if (_worker) {
        _worker.terminate();
        _worker = null;
        _comlinkApi = null;
    }
    _currentRunId = 0;
    _latestSnapshotId = -1;
    _minimumSnapshotRevision = 0;
    _pendingSnapshot = null;
    _onSnapshot = null;
    // SAB views outlive a single run (they're shared with the worker) but
    // a terminate invalidates everything, including the backing SABs once
    // the worker is gone. Drop our references so the GC can collect them
    // on the next run's handshake.
    tearDownSharedBuffers();
    resetFrameBuffer();
}
