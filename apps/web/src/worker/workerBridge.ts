// ── Worker Bridge ──
// Manages Communication with the training web worker.
// - Comlink proxy for request/response RPC (initialize, updateConfig, reset, step, etc.)
// - MessageChannel for high-frequency streamed snapshots during training
// - rAF-gated rendering loop that applies at most one snapshot per animation frame

import * as Comlink from 'comlink';
import type { TrainingWorkerApi } from './training.worker.ts';
import type { WorkerToMainMessage, MainToWorkerCommand } from '@nn-playground/shared';
import { updateFrameBuffer, resetFrameBuffer } from './frameBuffer.ts';

// ── Singleton state ──
let _worker: Worker | null = null;
let _comlinkApi: Comlink.Remote<TrainingWorkerApi> | null = null;
let _streamPort: MessagePort | null = null;
let _currentRunId = 0;
let _latestSnapshotId = -1;
let _rafId: number | null = null;
let _pendingSnapshot: WorkerToMainMessage | null = null;

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
    _streamPort.addEventListener('message', (event: MessageEvent<WorkerToMainMessage>) => {
        handleWorkerMessage(event.data);
    });
    _streamPort.onmessageerror = () => {
        emitWorkerError('Stream port message deserialization error');
    };
    _streamPort.start();
}

// ── Message Handling ──

function handleWorkerMessage(msg: WorkerToMainMessage): void {
    // Error messages always surface — even from stale runs — so async failures
    // after a reset are never silently dropped.
    if (msg.type !== 'error' && msg.runId < _currentRunId) return;

    if (msg.type === 'snapshot') {
        // Drop out-of-order snapshots
        if (msg.snapshotId <= _latestSnapshotId && msg.runId === _currentRunId) return;
        _latestSnapshotId = msg.snapshotId;

        // Store as pending — will be applied on next rAF tick (latest-wins)
        _pendingSnapshot = msg;
    } else {
        // Status/error messages are applied immediately
        if (_onSnapshot) _onSnapshot(msg);
    }
}

// ── rAF Render Loop ──

function rafLoop(): void {
    if (_rafId === null) return; // Stopped

    if (_pendingSnapshot) {
        const msg = _pendingSnapshot;
        _pendingSnapshot = null;

        // Write heavy arrays to frame buffer
        if (msg.type === 'snapshot') {
            updateFrameBuffer({
                outputGrid: msg.outputGrid ?? null,
                gridSize: msg.scalars.gridSize,
                neuronGrids: msg.neuronGrids ?? null,
                neuronGridLayout: msg.neuronGridLayout ?? null,
                weights: msg.weights ?? null,
                biases: msg.biases ?? null,
                weightLayout: msg.weightLayout ?? null,
                layerStats: msg.layerStats ?? null,
                confusionMatrix: msg.confusionMatrix ?? null,
            });
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
            updateFrameBuffer({
                outputGrid: msg.outputGrid ?? null,
                gridSize: msg.scalars.gridSize,
                neuronGrids: msg.neuronGrids ?? null,
                neuronGridLayout: msg.neuronGridLayout ?? null,
                weights: msg.weights ?? null,
                biases: msg.biases ?? null,
                weightLayout: msg.weightLayout ?? null,
                layerStats: msg.layerStats ?? null,
                confusionMatrix: msg.confusionMatrix ?? null,
            });
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
    resetFrameBuffer();
}
