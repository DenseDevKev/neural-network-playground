# AS-4 and AS-5 — Implementation Hand-off

**Audience:** an AI agent / engineer picking up the remaining two architectural-shift
items from the performance-optimization plan. QW-1..4 and AS-1..3 are already
landed on `main`; this file is the spec for AS-4 (WebGPU decision-boundary grid)
and AS-5 (canvas-based `NetworkGraph`).

**Ground rules** (consistent with the rest of this repo's work):

- Both items ship behind feature flags with a CPU / SVG fallback. A user on an
  older browser or with WebGPU disabled must still see a working visualization.
- No new UX regressions. Accessibility (ARIA labels, keyboard nav, hover
  tooltips) must be preserved.
- Keep changes scoped; don't refactor adjacent components unless required.
- All 325 tests must stay green. Add new tests alongside new code.
- Run `pnpm -r test` before and after. Run `pnpm --filter @nn-playground/web exec vite build`
  to confirm bundling survives. (`tsc` currently has pre-existing errors in
  `workerBridge.test.ts` unrelated to this work — ignore those; don't fix them
  as part of AS-4 / AS-5.)

---

## Shared context you need

### The snapshot pipeline (post AS-1..3)

1. **Engine** (`@/Users/kevincontreras/CascadeProjects/neural-network-playground/packages/engine/src/network.ts`) owns packed `Float64Array[]`
   weights/biases and a pre-allocated scratch for forward/backward. Grid
   prediction lives in `Network.predictGridInto(inputs, dst: Float32Array)`
   and `Network.predictGridWithNeuronsInto(inputs, outputDst, neuronDst)`.
2. **Worker** (`@/Users/kevincontreras/CascadeProjects/neural-network-playground/apps/web/src/worker/training.worker.ts`) owns the
   `Network`, runs training, and calls the grid predictors at a cadence
   controlled by `demand.gridInterval`.
3. **Transport** (new in AS-3, `@/Users/kevincontreras/CascadeProjects/neural-network-playground/apps/web/src/worker/sharedSnapshot.ts`):
   when cross-origin isolated, grid payloads go through a `SharedArrayBuffer`
   protected by a seqlock; otherwise they go through postMessage + Transferable.
4. **Bridge** (`@/Users/kevincontreras/CascadeProjects/neural-network-playground/apps/web/src/worker/workerBridge.ts`) receives snapshots,
   updates `_buffer` in `@/Users/kevincontreras/CascadeProjects/neural-network-playground/apps/web/src/worker/frameBuffer.ts`, and bumps
   `frameVersion` in `useTrainingStore`.
5. **UI** subscribes to `frameVersion` and reads typed arrays imperatively
   from `getFrameBuffer()` (see `@/Users/kevincontreras/CascadeProjects/neural-network-playground/apps/web/src/components/visualization/DecisionBoundary.tsx:184-208` and
   `@/Users/kevincontreras/CascadeProjects/neural-network-playground/apps/web/src/components/visualization/NetworkGraph.tsx:521-584`).

### Feature flags

Feature flags live in `@/Users/kevincontreras/CascadeProjects/neural-network-playground/packages/engine/src/types.ts` under `FeatureFlags`. For
user-visible toggles that affect the UI only, a better home is a small new
flags object in `@nn-playground/shared` or a `usePlaygroundStore` sub-slice.
Prefer a slim `featuresUI` slice with only `{ webgpuGrid: boolean; canvasNetworkGraph: boolean }`
and a settings panel checkbox for each. Default both to `true` and gate on
capability detection; if the detection fails, flip to the fallback at runtime.

---

## AS-4 — WebGPU decision-boundary grid

### Goal

Move the `Network.predictGridInto` / `predictGridWithNeuronsInto` computation
out of JS and onto the GPU via a WebGPU compute pipeline. On machines with
WebGPU, the grid forward pass runs in the low-single-digit-milliseconds range
regardless of grid size, vs. the ~11 ms / ~5.8 ms (no-neurons / neurons)
per-call numbers currently measured on CPU.

The engine CPU path stays authoritative for training (backprop, Adam, etc.) —
that's unchanged. Only the **visualization grid** forward pass moves to GPU.

### Why this is the right scope

The decision-boundary grid is `GRID_SIZE² × depth` purely data-parallel forward
passes with no gradient propagation. It's an ideal first WebGPU target:
zero kernel complexity, massive parallelism, trivial validation (compare to
CPU output).

Training forward/backward is **not** a good target — the per-step cost is
dominated by launch overhead, and getting Adam bit-exact on GPU is a
research project, not a quick win.

### Deliverables

Create `@/Users/kevincontreras/CascadeProjects/neural-network-playground/packages/engine/src/webgpu/` with:

- `detect.ts` — `detectWebGPU(): Promise<GPUDevice | null>`. Adapter request,
  device request, feature check. Cache the device singleton per process.
- `shaders/forward.wgsl` — a compute shader that reads flat weights + biases
  + activation type ID, runs a configurable-depth MLP, writes logits +
  per-neuron activations into output storage buffers.
- `gridPipeline.ts` — builds the `GPUComputePipeline`, manages bind group
  layouts, owns the permanent GPU buffers (they stay GPU-resident across
  frames and are updated in place via `queue.writeBuffer`).
- `predictGridGPU.ts` — the public API surface:
  ```ts
  class WebGPUGridPredictor {
      constructor(device: GPUDevice, layerSizes: number[], activation: string);
      updateWeights(flatWeights: Float32Array, flatBiases: Float32Array): void;
      predictGridInto(gridInputs: Float32Array, outputDst: Float32Array): Promise<void>;
      predictGridWithNeuronsInto(
          gridInputs: Float32Array,
          outputDst: Float32Array,
          neuronDst: Float32Array,
      ): Promise<void>;
      dispose(): void;
  }
  ```
- `__tests__/webgpu_parity.test.ts` — parity-tests the GPU predictor against
  `Network.predictGridInto` at element-wise tolerance 1e-4 (Float32 on GPU
  vs. Float64 on CPU). **Skip the test** (not fail) when `detectWebGPU()`
  returns null — CI runs on Linux without GPU. Use `it.skipIf(!device)`.

### Shader design (forward.wgsl)

- One workgroup covers a tile of grid pixels (e.g. 16×16 = 256 threads).
- Each thread handles one grid pixel through the full network.
- Inputs:
  - `@group(0) @binding(0) var<storage, read> weights: array<f32>;`
  - `@group(0) @binding(1) var<storage, read> biases: array<f32>;`
  - `@group(0) @binding(2) var<storage, read> gridInputs: array<f32>;`
  - `@group(0) @binding(3) var<storage, read_write> output: array<f32>;`
  - `@group(0) @binding(4) var<storage, read_write> neuronActs: array<f32>;`
  - `@group(0) @binding(5) var<uniform> meta: Meta;` with `layerSizes`,
    `layerOffsets`, `activationId`.
- Activations: inline a single `fn apply_activation(x: f32, id: u32) -> f32`
  switch over IDs {0=relu, 1=tanh, 2=sigmoid, 3=linear}. Match the exact
  formulas used in `Network` (`actRelu`, `actTanh`, …) from
  `@/Users/kevincontreras/CascadeProjects/neural-network-playground/packages/engine/src/network.ts`.
- Per-thread scratch: WGSL compute shaders don't have heap, so you must
  either (a) cap the hidden-layer width in the shader (e.g. 64) with a
  `var<private> scratch: array<f32, 64>;` and refuse to run for wider
  networks (falling back to CPU), or (b) use workgroup shared memory for
  intermediate activations. Option (a) is simpler and covers every
  playground network (max width is already small). Pick (a), assert the
  cap in `updateWeights`, and fall back to CPU for any network that exceeds
  it.

### Integration points

1. **Engine export**: add `WebGPUGridPredictor` to `@/Users/kevincontreras/CascadeProjects/neural-network-playground/packages/engine/src/index.ts`.

2. **Worker** (`@/Users/kevincontreras/CascadeProjects/neural-network-playground/apps/web/src/worker/training.worker.ts`):
   - Add `state.gpuGrid: WebGPUGridPredictor | null`.
   - After `buildDataAndNetwork()`, attempt to initialize the GPU predictor
     if `features.webgpuGrid` is on and `detectWebGPU()` succeeds. On any
     failure, set `state.gpuGrid = null` and log to console (the CPU path
     is the fallback).
   - Before every grid recompute in `computeSnapshot()`, call
     `state.gpuGrid.updateWeights(...)` using `network.getWeightsFlat().buffer`
     and `network.getBiasesFlat()`.
   - Replace the `network.predictGridInto(...)` / `predictGridWithNeuronsInto(...)`
     calls with the GPU variants when `state.gpuGrid` is non-null. GPU calls
     are `Promise<void>` — `computeSnapshot` must become `async` in that
     branch. Carefully coordinate with the train tick loop: the write into
     the SAB via `publishSharedSnapshot` must wait for the GPU work to
     finish before copying the destination buffer into the SAB. The simplest
     correctness-preserving pattern is:
     ```ts
     await state.gpuGrid.predictGridInto(inputs, state.outputGridBuffer);
     // …now state.outputGridBuffer is populated; rest of computeSnapshot
     //    continues synchronously.
     ```
     `trainTick` awaits `computeSnapshot` and posts the snapshot afterward.
     This serializes grid production but that's fine because the grid is
     already cadence-gated (`demand.gridInterval`), and the train steps
     already continue in parallel across tick budgets.
   - On shape change (any call to `buildDataAndNetwork`), `state.gpuGrid?.dispose()`
     and reinitialize.

3. **Capability plumbing**:
   - WebGPU in a worker requires `navigator.gpu` in the worker scope (Chrome 113+ has this).
   - Adapter request: `await navigator.gpu.requestAdapter({ powerPreference: 'low-power' })`.
     Low-power is a better choice than high-performance for a visualization
     so the user's laptop doesn't spin up the discrete GPU.
   - Device request: `await adapter.requestDevice()`.
   - Expose a one-time `capabilities` status message from the worker → bridge
     so the UI can show a "GPU-accelerated" badge (optional, but nice).

4. **UI toggle**: add `webgpuGrid: boolean` to the settings slice of
   `@/Users/kevincontreras/CascadeProjects/neural-network-playground/apps/web/src/store/usePlaygroundStore.ts`. Surface it in the same panel as
   the existing visualization demand toggles. Default `true`.

### Tests

- `packages/engine/src/webgpu/__tests__/webgpu_parity.test.ts` — round-trip
  test against a small network (e.g. `[2, 8, 8, 1]` tanh) with random
  weights. Compare every element, tolerance 1e-4.
- `packages/engine/src/webgpu/__tests__/webgpu_detect.test.ts` — assert
  `detectWebGPU()` returns `null` in the JSDOM test environment (i.e. the
  fallback path is always exercised in CI).
- Add an integration test in `apps/web/src/worker/*.test.ts` that pushes a
  snapshot with the GPU path disabled and verifies the frame buffer still
  populates. (Existing tests already cover the CPU path; make sure your
  changes haven't regressed them.)

### Bundle-size concern

WGSL shader source should be `?raw` imported (Vite supports this) so it
bundles as a string literal, not parsed. The GPU module should be
dynamically imported (`await import('./webgpu/predictGridGPU.ts')`) so
users without `webgpuGrid` on never pay for the pipeline-builder code.

### Rough LoC estimate

- `detect.ts`: 20
- `forward.wgsl`: 100
- `gridPipeline.ts`: 150
- `predictGridGPU.ts`: 120
- Worker wiring: +40 in `training.worker.ts`
- Tests: 100

**Total: ~530 lines**, contained entirely within `packages/engine/src/webgpu/`
plus one call site in the worker.

### Verification commands

```bash
# Run the engine test suite (parity test included when GPU present)
pnpm --filter @nn-playground/engine test

# Full test suite — must stay 325 green on CI (no-GPU environment)
pnpm -r test

# Dev server, click through a run, watch the perf panel — grid frames should
# measure near 0ms on the CPU side (work is off-thread on the GPU)
pnpm --filter @nn-playground/web dev
```

---

## AS-5 — Canvas-based NetworkGraph

### Goal

Today `@/Users/kevincontreras/CascadeProjects/neural-network-playground/apps/web/src/components/visualization/NetworkGraph.tsx` renders
the network topology as ~600 SVG `<path>` elements (one per edge) plus one
`<canvas>` per hidden-layer neuron (for its mini-heatmap). At 20 hidden neurons,
React commits ~650 DOM nodes per frame. Paint is fine; commit/diff is the
bottleneck.

Replace the main SVG layer with a single `<canvas>` that hand-draws edges and
neuron circles. Keep the mini-heatmap canvases as-is (they're already fast,
and Canvas-in-Canvas is messier than the gain justifies). Keep tooltip +
hover as DOM overlays — hit-testing lives in a small `findNodeAt(x, y)` /
`findEdgeAt(x, y)` helper.

### Why this is a meaningful win

- React commit cost drops from 600+ path nodes to a single canvas element.
- Style recalculation on hover (currently triggered by className toggles on
  paths) is eliminated.
- Animations (edge-pulse on weight updates, if you ever want to add them)
  become trivial on canvas and near-impossible on SVG.
- Zero layout thrash on window resize — canvas re-paints on `ResizeObserver`
  without triggering DOM reflow of 600 children.

### Deliverables

Refactor `@/Users/kevincontreras/CascadeProjects/neural-network-playground/apps/web/src/components/visualization/NetworkGraph.tsx`:

- Keep the `NetworkGraph` component's external API identical (it reads
  store state and takes no props).
- Replace the `<svg>` root and its `NetworkEdges` / `NetworkNodes` /
  `NetworkLabels` subtrees with:
  - A `<canvas>` for the main graph (edges + node discs + labels).
  - A layer of absolutely-positioned `<div>`s for the mini-heatmaps, one
    per neuron, positioned from the same `nodePositions` array you already
    compute. This keeps the existing `HeatmapCanvas` in play unchanged —
    just hosted in `<div>`s instead of `<foreignObject>`.
  - An absolutely-positioned `<div>` for the tooltip (already done).
- Hit-testing: a pair of pure functions
  ```ts
  function hitTestNode(x: number, y: number, nodePositions, radius): { layer, idx } | null;
  function hitTestEdge(x: number, y: number, edges, threshold): EdgeRef | null;
  ```
  Both called from a single `pointermove` handler on the canvas. For edges,
  use perpendicular-distance-to-line-segment. Threshold = 4 px.
- Accessibility: the canvas can't expose the network structure to screen
  readers. Add a visually-hidden `<ul>` that mirrors the topology with
  `aria-label` on each item. The canvas element should have `role="img"`
  and `aria-describedby="network-graph-desc"` pointing to a hidden `<p>`
  with a human summary (e.g. "Neural network: 2 inputs, 2 hidden layers of
  8 neurons each, 1 output. Weights update in real time."). Match the
  a11y pattern used by `@/Users/kevincontreras/CascadeProjects/neural-network-playground/apps/web/src/components/visualization/DecisionBoundary.tsx` (which
  already has an `aria-label` on its canvas).

### Paint plan

Exactly one paint per `frameVersion` bump (same trigger as today). Paint
function shape:

```ts
function paint(ctx, view) {
    ctx.save();
    // 1. Clear / background
    ctx.clearRect(0, 0, view.w, view.h);

    // 2. Edges (batched by color bucket to minimize fillStyle thrash)
    //    Bucket weights by sign × |w| quantile so we issue at most ~8
    //    stroke() calls total. For each bucket, strokeStyle = bucketColor,
    //    beginPath, moveTo/lineTo over every edge in the bucket, stroke().
    paintEdges(ctx, view.edges, view.flat, view.nodePositions);

    // 3. Hovered edge on top (single stroke with emphasis)
    if (view.hoveredEdge) paintHoveredEdge(ctx, view.hoveredEdge, view.nodePositions);

    // 4. Node rings (again batched: one bucket per bias sign × magnitude bin)
    paintNodes(ctx, view.nodePositions, view.flat);

    // 5. Labels (small, top-of-layer text)
    paintLabels(ctx, view.layerLabels, view.nodePositions);

    ctx.restore();
}
```

Typical frame: clearRect + ~8 stroke() calls for edges + 1 for hovered edge
+ ~4 fill() batches for node rings + N fillText calls for labels.

### State wiring

Move the `flat` / `neuronGrids` derivation from the current `useMemo`
closures into a plain function that takes the frame buffer as argument.
The component body becomes:

```ts
export function NetworkGraph() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [tooltip, setTooltip] = useState<TooltipData | null>(null);
    const [hoveredEdge, setHoveredEdge] = useState<EdgeRef | null>(null);
    const frameVersion = useTrainingStore(s => s.frameVersion);
    const layout = useMemo(() => computeLayout(...), [...]);
    // heatmaps still need React ownership (one canvas per neuron)
    const neuronGrids = useMemo(() => collectNeuronGrids(), [frameVersion, layout]);

    useEffect(() => { paintMainCanvas(canvasRef.current, layout, hoveredEdge); },
             [frameVersion, layout, hoveredEdge]);

    return (
        <div className="network-graph-container" ref={containerRef}>
            <canvas ref={canvasRef} onPointerMove={handlePointerMove} onPointerLeave={handlePointerLeave} />
            {neuronGrids.map(({ x, y, grid, gridSize, key }) => (
                <div key={key} className="network-graph-heatmap-slot" style={{ left: x - r, top: y - r, width: 2*r, height: 2*r }}>
                    <HeatmapCanvas grid={grid} gridSize={gridSize} />
                </div>
            ))}
            {tooltip && <Tooltip {...tooltip} />}
        </div>
    );
}
```

### DPR handling

```ts
const dpr = window.devicePixelRatio || 1;
if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
}
ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
```

Match the pattern used by `DecisionBoundary`.

### Feature flag

Add `canvasNetworkGraph: boolean` to the UI slice (default `true`). Keep
the SVG implementation around as `NetworkGraphSVG` for one release. The
public `NetworkGraph` component picks at runtime:

```ts
export function NetworkGraph() {
    const useCanvas = usePlaygroundStore(s => s.featuresUI.canvasNetworkGraph);
    return useCanvas ? <NetworkGraphCanvas /> : <NetworkGraphSVG />;
}
```

This lets you A/B the two quickly if a layout or interaction bug slips in.

### Tests

Update `@/Users/kevincontreras/CascadeProjects/neural-network-playground/apps/web/src/components/visualization/NetworkGraph.test.tsx`
(create if absent) to assert:

- Given a populated frame buffer, the component mounts without error.
- The canvas receives a valid 2D context and `clearRect` is called.
- Hovering the canvas at a node-position coordinate sets a tooltip
  (use `jsdom`'s pointer-event dispatch — or `fireEvent.pointerMove` with
  `clientX`/`clientY`).
- The accessibility `<ul>` is present and enumerates layer sizes.

Existing tests that bypass this component (store + worker tests) stay
unchanged.

### Rough LoC estimate

- `NetworkGraphCanvas.tsx`: 300
- Paint helpers (`paintEdges`, `paintNodes`, `paintLabels`, hit-tests): 150
- `NetworkGraphSVG.tsx`: renamed copy of current file, ~680 lines (no new
  code — just file rename + default-export swap)
- Component switcher in `NetworkGraph.tsx`: 20
- Tests: 80

**Total: ~550 lines of new code**, plus the existing file gets renamed. No
breaking changes to any other component.

### Verification commands

```bash
pnpm --filter @nn-playground/web test
pnpm -r test

# Visually compare the two implementations side-by-side by toggling the
# feature flag in the UI — network shape, edge colors, hover behaviour,
# tooltip content, and heatmap positioning must match exactly.
pnpm --filter @nn-playground/web dev
```

---

## Sequencing advice

**Do AS-5 first.** It's simpler (all DOM / Canvas2D, no capability detection),
its risk is contained to one component, and its win is UX-visible immediately
(lower input-latency during heavy training). AS-4 has a longer validation
tail because GPU parity bugs are subtle; doing it on top of a known-good
renderer is safer.

**Land each behind its flag**. Don't remove the flag for at least one
user-facing release after it's merged. If either turns out to have a
long-tail bug you can flip it off without a revert.

**Single commit per deliverable**. Commit messages suggested:

- `perf(webgpu): add WebGPU grid-prediction pipeline with CPU fallback (AS-4)`
- `perf(graph): canvas-based NetworkGraph renderer with SVG fallback (AS-5)`

---

## State of the repo at the point of hand-off

Last committed baseline: the five landed plan items (QW-1..4 + AS-1..3).

Green-field state:

```
packages/engine   184/184 tests
packages/shared    30/30 tests
apps/web          111/111 tests
```

Performance baselines to beat with AS-4:

| Metric                                 | Current (CPU) | Target (GPU)   |
|----------------------------------------|---------------|----------------|
| `predictGridInto` (100 iter, [2,32,32,32,1]) | 1088 ms       | < 50 ms        |
| `predictGridWithNeuronsInto` (50 iter) | 584 ms        | < 30 ms        |

Performance baselines to beat with AS-5:

| Metric                                 | Current (SVG) | Target (Canvas) |
|----------------------------------------|---------------|-----------------|
| React commit per frame, 20-hidden net  | ~4 ms         | < 0.5 ms        |
| Style recalc on hover                  | ~1 ms         | 0 ms            |
| DOM node count                         | ~650          | ~25 (heatmaps + overlays) |

Measure with Chrome DevTools Performance panel, Recording a 5-second window
of live training on a `[2, 16, 16, 1]` `tanh` net with every visualization on.
