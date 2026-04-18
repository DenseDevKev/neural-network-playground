# Worker Protocol

This document describes the message protocol between the main thread and the
training Web Worker in Neural Network Playground.

## Overview

Communication uses two channels:

- **Comlink RPC** (main → worker): request/response commands such as
  `initialize`, `updateConfig`, `reset`, `step`, and `setStreamPort`.
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
    |<- port1: status { status: 'paused' } ----|
    |                                          |
    |-- Comlink: reset() --------------------> |
    |<- { snapshot, runId } ------------------|
```

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

### `status`

Posted immediately on `startTraining` (→ `'running'`) and `stopTraining`
(→ `'paused'`).

| Field | Type | Description |
|---|---|---|
| `type` | `'status'` | Discriminator |
| `runId` | `number` | Current run ID |
| `status` | `'idle' \| 'running' \| 'paused'` | New training status |

### `error`

Posted by `postError()` inside the worker on any uncaught exception, NaN
divergence, or unknown command.

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
