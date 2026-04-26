# Worker Protocol

This document describes the message protocol between the main thread and the
training Web Worker in Neural Network Playground.

## Overview

Communication uses two channels:

- **Comlink RPC** (main → worker): request/response commands for setup,
  one-shot updates, point reads, demand sync, WebGPU toggling, and stream-port
  installation.
- **MessageChannel** (bidirectional): high-frequency streaming messages during
  training.  The main thread holds `port1`; the worker holds `port2`.

## Sequence Diagram

```
Main thread                          Worker (training.worker.ts)
    |                                          |
    |-- Comlink: initialize(configs) --------> |
    |<- { snapshot, runId } ------------------|
    |                                          |
    |-- Comlink: setStreamPort(port2) -------> |
    |   (port1 kept by workerBridge.ts)        |
    |                                          |
    |-- port1: startTraining { stepsPerFrame } |
    |<- port1: status { status: 'running' } ---|
    |                                          |
    |<- port1: snapshot { runId, snapshotId, scalars, ... } (repeat)
    |                                          |
    |-- port1: frameAck ---------------------->|   (releases back-pressure gate)
    |<- port1: snapshot ... -------------------|
    |                                          |
    |-- port1: stopTraining ----------------> |
    |<- port1: status { status: 'paused', pauseReason? } |
    |                                          |
    |-- Comlink: reset() --------------------> |
    |<- { snapshot, runId } ------------------|
```

## Comlink API

`training.worker.ts` exposes this Comlink surface as `TrainingWorkerApi`.
Legacy loop-control RPCs (`setRunning`, `isRunning`, `trainAndSnapshot`, and
`getSnapshot`) are not part of the current contract; continuous training runs
through the MessageChannel commands below.

| Method | Returns | Notes |
|---|---|---|
| `initialize(network, training, data, features)` | `{ snapshot, runId }` | Normalizes and validates config, rebuilds data/network, allocates fresh transport state. |
| `updateConfig(network, training, data, features, rebuild)` | `{ snapshot, runId }` | Normalizes and validates config; rebuilds only when required or requested. |
| `step(iterations = 1)` | `NetworkSnapshot` | Synchronous manual stepping for UI "step" actions/tests. |
| `reset()` | `{ snapshot, runId }` | Stops streaming, rebuilds data/network, returns a fresh snapshot. |
| `getTrainPoints()` | `DataPoint[]` | Returns current transformed training points for immediate UI render after init/config/reset. |
| `getTestPoints()` | `DataPoint[]` | Returns current transformed test points for immediate UI render after init/config/reset. |
| `updateDemand(demand)` | `void` | Validates with `normalizeVisualizationDemand`; invalid demand throws. Valid demand also marks expensive work due immediately. |
| `setWebGpuEnabled(enabled)` | `void` | Toggles the optional WebGPU grid path; disabling disposes the predictor and marks grids stale. |
| `setStreamPort(port)` | `void` | Installs the worker side of the MessageChannel and replays the SAB handshake if buffers already exist. |

## Messages: Worker → Main

All worker-to-main messages are validated by `isWorkerToMainMessage` before
being processed by `workerBridge.ts`.

### `snapshot`

Posted during active training at the rate controlled by `stepsPerFrame`.
Back-pressure: only one snapshot is in-flight at a time; the worker skips
posting while awaiting a `frameAck`.

| Field | Type | Description |
|---|---|---|
| `type` | `'snapshot'` | Discriminator |
| `runId` | `number` | Monotonically increasing; identifies the network build |
| `snapshotId` | `number` | Per-run counter; allows out-of-order drop |
| `scalars` | `SnapshotScalars` | Lightweight numeric summary (loss, accuracy, epoch, …) |
| `outputGrid` | `Float32Array?` | Flat decision-boundary heatmap (transferred) |
| `neuronGrids` | `Float32Array?` | Concatenated per-neuron grids (transferred) |
| `neuronGridLayout` | `{ count, gridSize }?` | Describes `neuronGrids` shape |
| `weights` | `Float32Array?` | Flat weight buffer (transferred) |
| `biases` | `Float32Array?` | Flat bias buffer (transferred) |
| `weightLayout` | `{ layerSizes }?` | Describes `weights` shape |
| `layerStats` | `LayerStats[]?` | Per-layer statistics (only when `needLayerStats`) |
| `historyPoint` | `HistoryPoint` | One point appended to the loss-history chart |
| `confusionMatrix` | `ConfusionMatrixData?` | Only when `needConfusionMatrix` |
| `sharedSeq` | `number?` | Present when grid payloads were published to SharedArrayBuffers instead of inline fields |

When SharedArrayBuffer transport is available, the worker first posts a
`sharedBuffers` handshake containing permanent SABs and layout metadata.
Subsequent snapshots may omit `outputGrid` and `neuronGrids`, set `sharedSeq`,
and let `workerBridge.ts` copy the latest consistent seqlock-protected data
into the frame buffer.

### `status`

Posted immediately on `startTraining` (-> `'running'`) and `stopTraining`
(-> `'paused'`). Automatic runtime stops may include `pauseReason`; manual
main-thread pauses currently set their reason locally in `useTraining.ts`.

| Field | Type | Description |
|---|---|---|
| `type` | `'status'` | Discriminator |
| `runId` | `number` | Current run ID |
| `status` | `'idle' \| 'running' \| 'paused'` | New training status |
| `pauseReason` | `PauseReason \| null?` | Optional reason for a paused/idle status. P0B worker runtime uses `'diverged'` for non-finite loss. |

### `error`

Posted by `postError()` inside the worker on uncaught exceptions, malformed
commands, and transport/runtime failures. Non-finite training/test loss is an
automatic paused status with `pauseReason: 'diverged'`, not an error message.

| Field | Type | Description |
|---|---|---|
| `type` | `'error'` | Discriminator |
| `runId` | `number` | Run ID at time of error (may be stale) |
| `message` | `string` | Human-readable description |

**Errors bypass the `frameAck` back-pressure gate.** They are dispatched
immediately via `_onSnapshot` without waiting for the render-loop tick,
ensuring failures surface to the UI regardless of training state.
Snapshots, by contrast, are queued as `_pendingSnapshot` and applied on the
next `requestAnimationFrame` tick after a `frameAck` is received.

## Messages: Main → Worker

All main-to-worker commands are validated by `isMainToWorkerCommand` before
being processed by the worker's `handleStreamCommand`.

### `startTraining`

| Field | Type | Description |
|---|---|---|
| `type` | `'startTraining'` | Discriminator |
| `stepsPerFrame` | `number` | Training steps executed per worker tick |

### `stopTraining`

| Field | Type | Description |
|---|---|---|
| `type` | `'stopTraining'` | Discriminator |

### `updateDemand`

| Field | Type | Description |
|---|---|---|
| `type` | `'updateDemand'` | Discriminator |
| `demand` | `VisualizationDemand` | New demand flags (which visuals to compute) |

`updateDemand` is runtime-validated by `isMainToWorkerCommand`, which delegates
to `normalizeVisualizationDemand`. Every boolean demand flag must be present,
and `testEvalInterval`, `trainEvalInterval`, and `gridInterval` must be finite
positive integers. The worker applies the same normalization on the Comlink
`updateDemand` RPC and throws `Invalid visualization demand.` if validation
fails.

Applying valid demand resets the cadence counters to their interval values and
marks grids stale. This makes the next snapshot recompute newly requested data
instead of waiting for the previous cadence schedule to expire.

### `updateSpeed`

| Field | Type | Description |
|---|---|---|
| `type` | `'updateSpeed'` | Discriminator |
| `stepsPerFrame` | `number` | New steps-per-tick value (hot-update without restart) |

### `frameAck`

| Field | Type | Description |
|---|---|---|
| `type` | `'frameAck'` | Discriminator |

Sent by the main thread after a snapshot has been applied to the frame buffer.
Releases the worker's back-pressure gate so the next snapshot may be posted.

## Back-pressure Contract

1. Worker sets `state.awaitingAck = true` immediately before each `postMessage`
   for a `snapshot`.
2. While `awaitingAck` is `true`, training continues but snapshot posting is
   skipped (UI coalesces to its render rate).
3. On `frameAck`, the worker clears `awaitingAck`, allowing the next snapshot.
4. On `stopTraining`, `reset`, or any path that calls `stopInternalLoop()`, the
   gate is reset unconditionally so a subsequent `startTraining` never stalls
   waiting for a `frameAck` that will never arrive.
5. **Errors are never gated.** `postError()` posts directly to `streamPort`
   regardless of `awaitingAck` state.

## Frame Version Behavior

Heavy typed-array payloads live in `worker/frameBuffer.ts`, outside React state.
`workerBridge.ts` writes only fields present on each streamed snapshot, so
cadence-gated frames retain previously cached arrays. Each domain has a version
counter:

| Counter | Incremented when |
|---|---|
| `frameVersion` | Any heavy frame-buffer domain changes |
| `outputGridVersion` | `outputGrid` is written |
| `neuronGridsVersion` | `neuronGrids` or `neuronGridLayout` is written |
| `paramsVersion` | `weights`, `biases`, or `weightLayout` is written |
| `layerStatsVersion` | `layerStats` is written |
| `confusionMatrixVersion` | `confusionMatrix` is written |

React components subscribe to the narrow version counter for the domain they
read, then imperatively read the current frame buffer during render/memo/paint.
Where a hook body reads mutable frame-buffer state, the version value is read
explicitly in that body so the dependency both documents and drives the
recomputation.
