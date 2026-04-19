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
} from '@nn-playground/shared';
import { isWorkerToMainMessage } from '@nn-playground/shared';
import { updateFrameBuffer, resetFrameBuffer } from './frameBuffer.ts';
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
let _sharedOutputReadBuf: Float32Array | null = null;
let _sharedNeuronReadBuf: Float32Array | null = null;
let _sharedNeuronGridLayout: { count: number; gridSize: number } | null = null;

function installSharedBuffers(msg: WorkerSharedBuffersMessage): void {
    _sharedViews = attachSharedSnapshotViews({
        control: msg.control,
        outputGrid: msg.outputGrid,
        neuronGrids: msg.neuronGrids,
        gridSize: msg.gridSize,
        neuronCount: msg.neuronGridLayout.count,
    });
    // Allocate reader-side destination arrays sized to the new shape.
    // These are regular (non-shared) Float32Arrays so downstream renderers
    // work on a stable copy; the cost is a single memcpy per snapshot.
    _sharedOutputReadBuf = new Float32Array(msg.gridSize * msg.gridSize);
    _sharedNeuronReadBuf = new Float32Array(
        Math.max(1, msg.neuronGridLayout.count * msg.gridSize * msg.gridSize),
    );
    _sharedNeuronGridLayout = msg.neuronGridLayout;
}

function tearDownSharedBuffers(): void {
    _sharedViews = null;
    _sharedOutputReadBuf = null;
    _sharedNeuronReadBuf = null;
    _sharedNeuronGridLayout = null;
}

// Callback for when a new snapshot is ready to be applied (called from rAF loop)
type SnapshotCallback = (msg: WorkerToMainMessage) => void;
let _onSnapshot: SnapshotCallback | null = null;

// Synthesize a WorkerErrorMessage and dispatch it through _onSnapshot so that
// bridge-level failures (onerror, onmessageerror) surface through the same
// path as worker-emitted errors.
function emitWorkerError(message: string): void {
    if (_onSnapshot) {
        _onSnapshot({ type: 'error', runId: _currentRunId, message });
    }
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

function handleWorkerMessage(msg: unknown): void {
    // Validate message shape before processing.
    if (!isWorkerToMainMessage(msg)) {
        emitWorkerError('Received malformed message from worker: ' + JSON.stringify(msg));
        return;
    }

    // Error messages always surface — even from stale runs — so async failures
    // after a reset are never silently dropped.
    if (msg.type !== 'error' && msg.runId < _currentRunId) return;

    if (msg.type === 'snapshot') {
        // Drop out-of-order snapshots
        if (msg.snapshotId <= _latestSnapshotId && msg.runId === _currentRunId) return;
        _latestSnapshotId = msg.snapshotId;

        // Store as pending — will be applied on next rAF tick (latest-wins)
        _pendingSnapshot = msg;
    } else if (msg.type === 'sharedBuffers') {
        // Worker (re)allocated its SAB transport. Install views immediately
        // so the very next snapshot can read from them. Never queued to rAF
        // — we need this in place before any snapshot referring to it
        // arrives, and it carries no per-frame data.
        installSharedBuffers(msg);
    } else {
        // Status/error messages are applied immediately
        if (_onSnapshot) _onSnapshot(msg);
    }
}

// ── rAF Render Loop ──

// Build a minimal frame-buffer patch from a snapshot message. Only fields
// that are actually present in the message are written — this is essential
// for the cadence-gated snapshots, where the worker omits the grid on
// reuse frames and the main thread must retain the previously cached one.
function framePatchFrom(msg: import('@nn-playground/shared').WorkerSnapshotMessage) {
    const patch: Parameters<typeof updateFrameBuffer>[0] = {};

    // AS-3 fast path: grid payloads were published through SharedArrayBuffers;
    // read them via the seqlock into our stable, non-shared read buffers and
    // point the frame buffer at those copies. We copy (rather than handing
    // the UI raw SAB views) because renderers paint across multiple rAF
    // ticks and can't tolerate the worker overwriting a view mid-paint.
    if (
        msg.sharedSeq !== undefined &&
        _sharedViews &&
        _sharedOutputReadBuf &&
        _sharedNeuronReadBuf
    ) {
        const result = readSharedSnapshot(
            _sharedViews,
            _sharedOutputReadBuf,
            _sharedNeuronReadBuf,
        );
        if (result) {
            if ((result.flags & FLAG_OUTPUT_GRID) !== 0) {
                patch.outputGrid = _sharedOutputReadBuf;
                patch.gridSize = msg.scalars.gridSize;
            }
            if ((result.flags & FLAG_NEURON_GRIDS) !== 0) {
                patch.neuronGrids = _sharedNeuronReadBuf;
                patch.neuronGridLayout =
                    msg.neuronGridLayout ?? _sharedNeuronGridLayout;
            }
        }
        // If the seqlock read torn through all retries, skip the grid
        // update this frame — the UI will pick up the next consistent
        // publish. No inline fallback available (data isn't on the msg).
    } else {
        // Legacy postMessage path — grids arrived inline.
        if (msg.outputGrid !== undefined) {
            patch.outputGrid = msg.outputGrid;
            patch.gridSize = msg.scalars.gridSize;
        }
        if (msg.neuronGrids !== undefined) {
            patch.neuronGrids = msg.neuronGrids;
            patch.neuronGridLayout = msg.neuronGridLayout ?? null;
        }
    }

    if (msg.weights !== undefined) patch.weights = msg.weights;
    if (msg.biases !== undefined) patch.biases = msg.biases;
    if (msg.weightLayout !== undefined) patch.weightLayout = msg.weightLayout;
    if (msg.layerStats !== undefined) patch.layerStats = msg.layerStats;
    if (msg.confusionMatrix !== undefined) patch.confusionMatrix = msg.confusionMatrix;
    return patch;
}

function rafLoop(): void {
    if (_rafId === null) return; // Stopped

    if (_pendingSnapshot) {
        const msg = _pendingSnapshot;
        _pendingSnapshot = null;

        // Write heavy arrays to frame buffer
        if (msg.type === 'snapshot') {
            updateFrameBuffer(framePatchFrom(msg));
        }

        // Notify the subscriber (typically updates useTrainingStore scalars)
        if (_onSnapshot) _onSnapshot(msg);

        // Ack snapshots to release the worker's back-pressure gate. Status/
        // error messages bypass the gate, so they don't need an ack.
        if (msg.type === 'snapshot' && _streamPort) {
            _streamPort.postMessage({ type: 'frameAck' });
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
            updateFrameBuffer(framePatchFrom(msg));
        }
        _onSnapshot(msg);
        if (msg.type === 'snapshot' && _streamPort) {
            _streamPort.postMessage({ type: 'frameAck' });
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
    _pendingSnapshot = null;
    return _currentRunId;
}

/**
 * Set the run ID to a specific value (used to sync with worker's runId).
 */
export function newRunTo(targetRunId: number): void {
    _currentRunId = targetRunId;
    _latestSnapshotId = -1;
    _pendingSnapshot = null;
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
    _pendingSnapshot = null;
    _onSnapshot = null;
    // SAB views outlive a single run (they're shared with the worker) but
    // a terminate invalidates everything, including the backing SABs once
    // the worker is gone. Drop our references so the GC can collect them
    // on the next run's handshake.
    tearDownSharedBuffers();
    resetFrameBuffer();
}
