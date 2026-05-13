# Performance Baseline

## Date

2026-05-11

## Scope

Wave 0 lightweight baseline using existing repository commands only. This file is intended to catch obvious future regressions; it is not a full benchmark suite.

## Commands

- `pnpm test:perf`
- `pnpm build`

## Startup / Build Size

`pnpm build` passed on 2026-05-11.

Relevant production output:

- `dist/index.html`: 1.29 kB, gzip 0.55 kB
- `dist/assets/training.worker-DVftvNQG.js`: 68.66 kB
- `dist/assets/index-BEeM6uz0.css`: 62.39 kB, gzip 10.91 kB
- `dist/assets/InspectionPanel-DSsWDBeL.js`: 5.57 kB, gzip 1.67 kB
- `dist/assets/engine-DZ1GTedS.js`: 5.68 kB, gzip 2.04 kB
- `dist/assets/CodeExportPanel-DU1TmhIE.js`: 7.25 kB, gzip 2.95 kB
- `dist/assets/RunHistoryPanel-0LPBYE86.js`: 8.58 kB, gzip 3.04 kB
- `dist/assets/react-j2mp3VYR.js`: 11.79 kB, gzip 4.21 kB
- `dist/assets/index-CP6enBVP.js`: 351.56 kB, gzip 107.17 kB

Build retained the existing Vite warning that some chunks are larger than
200 kB after minification.

## Fixed Training Scenario

`pnpm test:perf` passed on 2026-05-11 with 2 files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1125.7350 ms total for 100 iterations
- `predictGridInto`: 1086.5321 ms total for 100 iterations
- `predictGridWithNeurons`: 685.5998 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 585.2417 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.6819 ms
- Average `applyGradients` time (SGD): 1.4065 ms

## Decision Boundary Responsiveness

Pending Browser QA observation in `docs/qa/browser-qa/wave-0-baseline.md`.

## Worker Message Cadence / Demand Behavior

No dedicated Wave 0 measurement. Future Wave 5 work should add worker demand/cadence regression tests before behavior changes.

## Memory / Frame Buffer Behavior

No dedicated Wave 0 measurement. Current Wave 0 changes do not touch frame buffer, SharedArrayBuffer, WebGPU, worker protocol, or large-array transport.

## Notes

- Wave 0 is docs and repository packaging only.
- Wave 1 planned action cards are web-local navigation/focus UI only.
- Performance-sensitive runtime, worker, visualization data, or engine changes remain approval-gated by the roadmap.

## Wave 4 Activation Histogram Comparison

Date: 2026-05-11

Scope: approved activation histogram explorer using demand-gated, bounded layer-level bins.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the histogram slice. Relevant production output:

- `dist/assets/training.worker--UxrksoV.js`: 71.53 kB
- `dist/assets/index-DJn0joE3.css`: 63.58 kB, gzip 11.14 kB
- `dist/assets/InspectionPanel-D2P7wqJz.js`: 8.31 kB, gzip 2.42 kB
- `dist/assets/index-Do23QsAR.js`: 360.52 kB, gzip 109.51 kB

Compared with the Wave 0 baseline:

- Main app bundle increased from 351.56 kB to 360.52 kB, about 2.5%.
- Training worker bundle increased from 68.66 kB to 71.53 kB, about 4.2%.
- Inspection panel lazy chunk increased from 5.57 kB to 8.31 kB because it now renders the histogram explorer.
- The existing Vite large chunk warning remains.

`pnpm test:perf` passed after the histogram slice with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1083.7991 ms total for 100 iterations
- `predictGridInto`: 1060.3090 ms total for 100 iterations
- `predictGridWithNeurons`: 673.5523 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 576.1459 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 3.7961 ms
- Average `applyGradients` time (SGD): 1.4921 ms

These benchmark values remain within the roadmap warning thresholds compared with the Wave 0 baseline. The existing benchmark suite does not directly time activation histogram computation; runtime protection for this slice is covered by demand/cadence tests and by keeping compact bins in the frame buffer instead of React state.

## Wave 5 Runtime Hardening Comparison

Date: 2026-05-11

Scope: runtime and worker hardening through regression tests and documentation. Wave 5 did not change production runtime code, worker protocol fields, frame-buffer semantics, SharedArrayBuffer behavior, WebGPU transport, engine math, persistence, URL/config serialization, or public config shape.

Commands:

- `pnpm test`
- `pnpm lint`
- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the Wave 5 hardening slices. Relevant production output:

- `dist/assets/training.worker--UxrksoV.js`: 71.53 kB
- `dist/assets/index-DJn0joE3.css`: 63.58 kB, gzip 11.14 kB
- `dist/assets/InspectionPanel-D2P7wqJz.js`: 8.31 kB, gzip 2.42 kB
- `dist/assets/index-Do23QsAR.js`: 360.52 kB, gzip 109.51 kB

Compared with the Wave 4 activation histogram build, the relevant production bundle sizes were unchanged. The existing Vite large chunk warning remains.

`pnpm test:perf` passed after the Wave 5 hardening slices with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1148.0024 ms total for 100 iterations
- `predictGridInto`: 1148.9127 ms total for 100 iterations
- `predictGridWithNeurons`: 719.4125 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 604.6402 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.8508 ms
- Average `applyGradients` time (SGD): 1.6078 ms

These benchmark values remain within the roadmap warning thresholds compared with the Wave 0 and Wave 4 baselines. One earlier Wave 5 worker-lifecycle perf run was noisy, so it was repeated before commit; the repeated and final values did not indicate a sustained regression. Because Wave 5 changed tests and docs only, no runtime performance impact is expected from the completed slices.

## Wave 6A Run Comparison

Date: 2026-05-11

Scope: saved-run comparison summaries in the web History panel using existing `ExperimentRunRecordV1.summary` data only.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the Wave 6A comparison slice. Relevant production output:

- `dist/assets/training.worker--UxrksoV.js`: 71.53 kB
- `dist/assets/index-DJn0joE3.css`: 63.58 kB, gzip 11.14 kB
- `dist/assets/InspectionPanel-eH7iPmEC.js`: 8.31 kB, gzip 2.42 kB
- `dist/assets/RunHistoryPanel-DYvsV1ry.js`: 9.57 kB, gzip 3.24 kB
- `dist/assets/index-BFhnXlXc.js`: 360.52 kB, gzip 109.51 kB

Compared with the Wave 5 final build, the main app and worker chunks were unchanged. The lazy run-history chunk increased from 8.58 kB to 9.57 kB because it now renders comparison summaries. The existing Vite large chunk warning remains.

`pnpm test:perf` passed after the Wave 6A comparison slice with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1137.5487 ms total for 100 iterations
- `predictGridInto`: 1108.1885 ms total for 100 iterations
- `predictGridWithNeurons`: 708.8439 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 611.1152 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.3936 ms
- Average `applyGradients` time (SGD): 1.4319 ms

These benchmark values remain within the roadmap warning thresholds. Wave 6A did not touch engine, worker, frame-buffer, persistence, schema, URL/config serialization, or public config code.

## Wave 6B Saved Run Thumbnails

Date: 2026-05-11

Scope: generated, non-persisted SVG saved-run thumbnails using existing bounded `ExperimentRunRecordV1.history` data.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the Wave 6B thumbnail slice. Relevant production output:

- `dist/assets/training.worker--UxrksoV.js`: 71.53 kB
- `dist/assets/index-DJn0joE3.css`: 63.58 kB, gzip 11.14 kB
- `dist/assets/InspectionPanel-CUoBhCYz.js`: 8.31 kB, gzip 2.42 kB
- `dist/assets/RunHistoryPanel-ChJRRFsq.js`: 11.17 kB, gzip 3.79 kB
- `dist/assets/index-ZPK1SkKu.js`: 360.52 kB, gzip 109.51 kB

Compared with Wave 6A, the main app and worker chunks were unchanged. The lazy run-history chunk increased from 9.57 kB to 11.17 kB because it now renders SVG thumbnail helpers. The existing Vite large chunk warning remains.

`pnpm test:perf` passed after the Wave 6B thumbnail slice with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1137.8122 ms total for 100 iterations
- `predictGridInto`: 1093.4358 ms total for 100 iterations
- `predictGridWithNeurons`: 697.4345 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 592.1591 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.7540 ms
- Average `applyGradients` time (SGD): 1.4204 ms

These benchmark values remain within the roadmap warning thresholds. Wave 6B did not touch engine, worker, frame-buffer, persistence, schema, URL/config serialization, or public config code.

## Wave 6C Report Export

Date: 2026-05-11

Scope: richer markdown report export content using existing saved run config, summary, network-presence, and history fields.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the Wave 6C report export slice. Relevant production output:

- `dist/assets/training.worker--UxrksoV.js`: 71.53 kB
- `dist/assets/index-DJn0joE3.css`: 63.58 kB, gzip 11.14 kB
- `dist/assets/InspectionPanel-BjZkoXK9.js`: 8.31 kB, gzip 2.42 kB
- `dist/assets/RunHistoryPanel-DnKpviYY.js`: 12.31 kB, gzip 4.14 kB
- `dist/assets/index-Dk30pzTM.js`: 360.52 kB, gzip 109.51 kB

Compared with Wave 6B, the main app and worker chunks were unchanged. The lazy run-history chunk increased from 11.17 kB to 12.31 kB because the markdown export includes richer setup and metrics sections. The existing Vite large chunk warning remains.

`pnpm test:perf` passed after the Wave 6C report export slice with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1138.3189 ms total for 100 iterations
- `predictGridInto`: 1108.7949 ms total for 100 iterations
- `predictGridWithNeurons`: 690.4257 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 589.8016 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.6273 ms
- Average `applyGradients` time (SGD): 1.4548 ms

These benchmark values remain within the roadmap warning thresholds. Wave 6C did not touch engine, worker, frame-buffer, persistence, schema, URL/config serialization, or public config code.

## Wave 6D Dataset Parameter Lab

Date: 2026-05-11

Scope: bounded sample-count presets and accessible dataset settings summary in the existing Data panel.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the Wave 6D dataset parameter slice. Relevant production output:

- `dist/assets/training.worker--UxrksoV.js`: 71.53 kB
- `dist/assets/index-DJn0joE3.css`: 63.58 kB, gzip 11.14 kB
- `dist/assets/RunHistoryPanel-B2DeJ9nu.js`: 12.31 kB, gzip 4.14 kB
- `dist/assets/index-CYHfg_fP.js`: 361.36 kB, gzip 109.72 kB

Compared with Wave 6C, the worker and run-history chunks were unchanged. The main app chunk increased from 360.52 kB to 361.36 kB because the Data panel is in the main UI bundle. The existing Vite large chunk warning remains.

`pnpm test:perf` passed after the Wave 6D dataset parameter slice with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1160.4786 ms total for 100 iterations
- `predictGridInto`: 1112.4462 ms total for 100 iterations
- `predictGridWithNeurons`: 695.9391 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 597.1600 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.8577 ms
- Average `applyGradients` time (SGD): 1.5220 ms

These benchmark values remain within the roadmap warning thresholds. Wave 6D did not touch engine, worker, frame-buffer, persistence, schema, URL/config serialization, or public config code.

## Wave 6E Engine Checkpoint State

Date: 2026-05-13

Scope: engine-local runtime checkpoint and restore support for weights, biases, optimizer state, and step counters.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the engine checkpoint slice. Relevant production output:

- `dist/assets/training.worker-DIGdGEzY.js`: 74.75 kB
- `dist/assets/index-DJn0joE3.css`: 63.58 kB, gzip 11.14 kB
- `dist/assets/InspectionPanel-BeAX8XuO.js`: 8.31 kB, gzip 2.42 kB
- `dist/assets/RunHistoryPanel-xfRz3sls.js`: 12.31 kB, gzip 4.14 kB
- `dist/assets/index-B-6ikeZX.js`: 361.36 kB, gzip 109.72 kB

Compared with Wave 6D, the main app bundle was unchanged and the worker bundle increased from 71.53 kB to 74.75 kB because the worker imports the engine class that now includes checkpoint helpers. The existing Vite large chunk warning remains. The worker-bundle increase is below the roadmap 10% warning threshold.

`pnpm test:perf` passed after the engine checkpoint slice with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1195.5230 ms total for 100 iterations
- `predictGridInto`: 1120.8068 ms total for 100 iterations
- `predictGridWithNeurons`: 712.0676 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 588.5592 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.8142 ms
- Average `applyGradients` time (SGD): 1.5587 ms

These benchmark values remain within the roadmap warning thresholds. This slice does not add checkpoint capture to the training hot path yet; it only adds explicit engine copy/restore helpers used by later runtime slices.

## Wave 6E Checkpoint Timeline Protocol Metadata

Date: 2026-05-12

Scope: shared worker-protocol types and runtime guards for lightweight checkpoint timeline metadata on snapshot messages.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the protocol metadata slice. Relevant production output:

- `dist/assets/training.worker-DIGdGEzY.js`: 74.75 kB
- `dist/assets/index-DJn0joE3.css`: 63.58 kB, gzip 11.14 kB
- `dist/assets/InspectionPanel-Dh1yWAXN.js`: 8.31 kB, gzip 2.42 kB
- `dist/assets/RunHistoryPanel-vnZ6qPMo.js`: 12.31 kB, gzip 4.14 kB
- `dist/assets/index-CGYNOHsy.js`: 361.88 kB, gzip 109.90 kB

Compared with the prior Wave 6E engine checkpoint slice, the worker bundle remained unchanged and the main app bundle increased from 361.36 kB to 361.88 kB because the shared runtime guard now validates optional checkpoint metadata. The existing Vite large chunk warning remains, and the increase is below the roadmap 10% warning threshold.

`pnpm test:perf` passed after the protocol metadata slice with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1185.8776 ms total for 100 iterations
- `predictGridInto`: 1139.6066 ms total for 100 iterations
- `predictGridWithNeurons`: 699.6503 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 578.9710 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.8995 ms
- Average `applyGradients` time (SGD): 1.4268 ms

These benchmark values remain within the roadmap warning thresholds. This slice adds validation for optional scalar checkpoint metadata only; heavy checkpoint payloads are still worker-local and are not stored in React state.

## Wave 6E Worker Checkpoint Ring Buffer

Date: 2026-05-12

Scope: worker-local bounded checkpoint ring buffer, checkpoint metadata RPC, and restore RPC.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the worker checkpoint slice. Relevant production output:

- `dist/assets/training.worker-DPoonmTq.js`: 77.21 kB
- `dist/assets/index-DJn0joE3.css`: 63.58 kB, gzip 11.14 kB
- `dist/assets/InspectionPanel-DtpXDUcq.js`: 8.31 kB, gzip 2.42 kB
- `dist/assets/RunHistoryPanel-CZbKpP55.js`: 12.31 kB, gzip 4.14 kB
- `dist/assets/index-J2TlDb6i.js`: 361.88 kB, gzip 109.90 kB

Compared with the prior Wave 6E protocol metadata slice, the main app bundle was unchanged and the worker bundle increased from 74.75 kB to 77.21 kB because it now owns the bounded runtime checkpoint ring buffer. The increase is below the roadmap 10% warning threshold, and the existing Vite large chunk warning remains.

`pnpm test:perf` passed after the worker checkpoint slice with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1138.2182 ms total for 100 iterations
- `predictGridInto`: 1081.3733 ms total for 100 iterations
- `predictGridWithNeurons`: 693.0297 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 589.9068 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.5410 ms
- Average `applyGradients` time (SGD): 1.4516 ms

These benchmark values remain within the roadmap warning thresholds. The current perf benchmark does not directly measure checkpoint capture cadence; runtime protection is covered by worker tests for bounded metadata, restore behavior, and eviction.

## Wave 6E Checkpoint Timeline UI and Guard Fix

Date: 2026-05-12

Scope: web hook/store integration for checkpoint timeline metadata, accessible timeline controls, and a shared runtime-guard fix for omitted activation-histogram payloads represented as `undefined` optional fields.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the checkpoint timeline UI and guard fix. Relevant production output:

- `dist/assets/training.worker-DPoonmTq.js`: 77.21 kB
- `dist/assets/index-CkY1LDYX.css`: 64.48 kB, gzip 11.29 kB
- `dist/assets/InspectionPanel-Byj47Raa.js`: 8.31 kB, gzip 2.42 kB
- `dist/assets/RunHistoryPanel-BdKcAo3E.js`: 12.31 kB, gzip 4.14 kB
- `dist/assets/index-0S6cDKV4.js`: 364.37 kB, gzip 110.50 kB

Compared with the worker checkpoint ring-buffer slice, the worker bundle was unchanged. The main app bundle increased from 361.88 kB to 364.37 kB because the main training bar now renders timeline controls and hook/store wiring. The existing Vite large chunk warning remains, and the increase is below the roadmap 10% warning threshold.

`pnpm test:perf` passed after the checkpoint timeline UI and guard fix with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1141.9338 ms total for 100 iterations
- `predictGridInto`: 1122.8077 ms total for 100 iterations
- `predictGridWithNeurons`: 684.5251 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 587.9544 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.5803 ms
- Average `applyGradients` time (SGD): 1.4364 ms

These benchmark values remain within the roadmap warning thresholds. Browser QA found and verified the protocol guard fix; after the fix, training started and paused without console errors.

## Wave 7 Side-by-Side Model Arena Phase 1

Date: 2026-05-12

Scope: saved-run-only comparison UI in the lazy run-history panel. This slice does not change worker runtime, engine math, frame-buffer semantics, URL/config serialization, persistence schema, public config shape, or dependencies.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the saved-run arena slice. Relevant production output:

- `dist/assets/training.worker-DPoonmTq.js`: 77.21 kB
- `dist/assets/index-B3wCX0gO.css`: 66.18 kB, gzip 11.49 kB
- `dist/assets/InspectionPanel-DlOrFiUi.js`: 8.31 kB, gzip 2.42 kB
- `dist/assets/RunHistoryPanel-C2bti-N-.js`: 15.13 kB, gzip 4.84 kB
- `dist/assets/index-CZ39uNC2.js`: 364.37 kB, gzip 110.50 kB

Compared with the Wave 6E checkpoint timeline UI baseline, the worker and main app bundles were unchanged. The lazy `RunHistoryPanel` chunk increased from 12.31 kB to 15.13 kB because it now renders the saved-run arena selectors, model panes, comparison copy, and reused thumbnails. The increase is isolated to a lazy chunk and below the roadmap 10% production-build warning threshold for the whole app.

`pnpm test:perf` passed after the saved-run arena slice with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1286.9436 ms total for 100 iterations
- `predictGridInto`: 1278.7490 ms total for 100 iterations
- `predictGridWithNeurons`: 773.1364 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 653.8458 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 6.0251 ms
- Average `applyGradients` time (SGD): 1.5177 ms

These benchmark values are noisier than the immediately prior Wave 6E run but remain within the roadmap warning thresholds for this UI-only, lazy-panel slice. No runtime hot path was changed.

## Wave 7 Live Arena Scalar Runtime

Date: 2026-05-12

Scope: scalar-only live arena runtime contract and worker prototype. This slice adds side-tagged bounded arena summaries, frame-buffer version support for scalar summaries, and worker Comlink methods for initializing and stepping two model slots sequentially. It does not add paired heavy visualizations, URL/config serialization, persistence schema changes, public config shape changes, dependencies, multiple workers, or engine math changes.

Commands:

- `pnpm build`
- `pnpm test:perf`
- repeated `pnpm test:perf` because the first benchmark run was noisy

`pnpm build` passed after the scalar live arena runtime slice. Relevant production output:

- `dist/assets/training.worker-CUaUykZz.js`: 79.35 kB
- `dist/assets/index-B3wCX0gO.css`: 66.18 kB, gzip 11.49 kB
- `dist/assets/InspectionPanel-DVGMOlJx.js`: 8.31 kB, gzip 2.42 kB
- `dist/assets/RunHistoryPanel-DKXx4SLe.js`: 15.13 kB, gzip 4.84 kB
- `dist/assets/index-DE01l2hl.js`: 365.16 kB, gzip 110.67 kB

Compared with Wave 7 Phase 1, the lazy run-history chunk was unchanged. The worker bundle increased from 77.21 kB to 79.35 kB because it now includes scalar live-arena slot setup and sequential stepping helpers. The increase is below the roadmap 10% build-size warning threshold. The existing Vite large chunk warning remains.

The first `pnpm test:perf` run passed but exceeded the roadmap warning threshold on grid timings despite this slice not touching engine prediction code:

- `predictGrid`: 1749.2360 ms total for 100 iterations
- `predictGridInto`: 1867.7205 ms total for 100 iterations
- `predictGridWithNeurons`: 1457.9459 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 1410.4423 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.5880 ms
- Average `applyGradients` time (SGD): 3.5469 ms

The repeated `pnpm test:perf` run passed and returned to the established range:

- `predictGrid`: 1138.0670 ms total for 100 iterations
- `predictGridInto`: 1093.3958 ms total for 100 iterations
- `predictGridWithNeurons`: 696.0030 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 592.8361 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.4625 ms
- Average `applyGradients` time (SGD): 1.4232 ms

The repeat suggests the first perf run was machine noise. The implemented runtime slice does not alter engine prediction or gradient hot paths.

## Wave 7 Live Arena UI

Date: 2026-05-12

Scope: expose the scalar live arena prototype in the lazy Run History UI by wiring saved runs to existing `initializeArena` and `stepArena` worker APIs. This slice keeps arena data bounded to scalar summaries, leaves large model arrays in the worker/frame buffer, and does not change URL/config serialization, persistence schema, public config shape, dependencies, engine math, or paired heavy visualizations.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the live arena UI slice. Relevant production output:

- `dist/assets/training.worker-CUaUykZz.js`: 79.35 kB
- `dist/assets/index-B3wCX0gO.css`: 66.18 kB, gzip 11.49 kB
- `dist/assets/InspectionPanel-p_Q4BzX6.js`: 8.31 kB, gzip 2.42 kB
- `dist/assets/RunHistoryPanel-Dw3ve2v5.js`: 16.70 kB, gzip 5.14 kB
- `dist/assets/index-BPXCw-bD.js`: 366.75 kB, gzip 111.03 kB

Compared with the scalar runtime slice, the worker bundle was unchanged. The lazy `RunHistoryPanel` chunk increased from 15.13 kB to 16.70 kB because it now renders live arena controls and scalar summaries. The main app bundle increased from 365.16 kB to 366.75 kB because App/MainArea now pass the arena callbacks. These changes are below the roadmap 10% build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed after the live arena UI slice with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1093.0898 ms total for 100 iterations
- `predictGridInto`: 1076.1186 ms total for 100 iterations
- `predictGridWithNeurons`: 681.7128 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 579.4028 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 3.9796 ms
- Average `applyGradients` time (SGD): 1.4334 ms

These values are within the established range and below the roadmap warning thresholds. This UI slice does not alter engine prediction or gradient hot paths.

## Wave 7 Slow-Motion Backprop Engine Foundation

Date: 2026-05-12

Scope: engine-only dry-run backprop summary support for the approved slow-motion explanation mode foundation. This slice adds bounded scalar layer summaries and deterministic engine tests. It does not add worker RPCs, protocol fields, frame-buffer fields, UI state, URL/config serialization, persistence schema changes, public config shape changes, dependencies, or browser-visible behavior.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the engine foundation slice. Relevant production output:

- `dist/assets/training.worker-BYsEN3qS.js`: 82.68 kB
- `dist/assets/index-B3wCX0gO.css`: 66.18 kB, gzip 11.49 kB
- `dist/assets/InspectionPanel-rLCImout.js`: 8.31 kB, gzip 2.42 kB
- `dist/assets/RunHistoryPanel-kTi7VzC5.js`: 16.70 kB, gzip 5.14 kB
- `dist/assets/index-DmQTg3mQ.js`: 366.75 kB, gzip 111.02 kB

Compared with the Wave 7 live arena UI build, the main app and lazy run-history chunks were effectively unchanged. The worker bundle increased from 79.35 kB to 82.68 kB because the worker imports the engine class that now includes the dry-run backprop summary method. The increase is about 4.2%, below the roadmap 10% build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed after the engine foundation slice with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1137.3643 ms total for 100 iterations
- `predictGridInto`: 1147.3317 ms total for 100 iterations
- `predictGridWithNeurons`: 725.1739 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 606.6910 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.1593 ms
- Average `applyGradients` time (SGD): 1.5099 ms

These values remain within the established range and below the roadmap warning thresholds. The slice adds an explicit dry-run computation path but does not alter the live training hot path; tests prove the preview leaves the live network checkpoint, snapshot, and layer stats unchanged.

## Wave 7 Slow-Motion Backprop Worker RPC

Date: 2026-05-12

Scope: worker-only one-shot Comlink RPC for the slow-motion backprop explanation path. This slice exposes the already-bounded engine `BackpropExplanation` through `getBackpropExplanation()` and documents the RPC. It does not add UI state, streamed messages, frame-buffer fields, shared protocol guards, URL/config serialization, persistence schema changes, public config shape changes, dependencies, or raw activation/gradient arrays.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the worker RPC slice. Relevant production output:

- `dist/assets/training.worker-S7ocudoV.js`: 83.50 kB
- `dist/assets/index-B3wCX0gO.css`: 66.18 kB, gzip 11.49 kB
- `dist/assets/InspectionPanel-B9mxGWHm.js`: 8.31 kB, gzip 2.42 kB
- `dist/assets/RunHistoryPanel-CufaPasd.js`: 16.70 kB, gzip 5.14 kB
- `dist/assets/index-BkhH1qdr.js`: 366.75 kB, gzip 111.02 kB

Compared with the engine foundation build, the main app and lazy chunks were effectively unchanged. The worker bundle increased from 82.68 kB to 83.50 kB because it now exposes a Comlink one-shot RPC and batch preview helper. The increase is about 1.0%, below the roadmap 10% build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed after the worker RPC slice with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1080.3162 ms total for 100 iterations
- `predictGridInto`: 1072.9316 ms total for 100 iterations
- `predictGridWithNeurons`: 672.7654 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 579.6610 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 3.8635 ms
- Average `applyGradients` time (SGD): 1.5098 ms

These values remain within the established range and below the roadmap warning thresholds. The RPC is one-shot and not called from the live training loop, so no runtime cadence impact is expected from this slice.

## Wave 7 Slow-Motion Backprop Preview UI

Date: 2026-05-12

Scope: Inspection-panel UI for manually requesting the one-shot slow-motion backprop preview. This slice stores only the bounded scalar RPC response in local component state, renders layer-level summaries, and adds keyboard/error/loading coverage. It does not add streamed worker messages, frame-buffer fields, raw activation/gradient arrays, URL/config serialization, persistence schema changes, public config shape changes, dependencies, or training behavior changes.

Commands:

- `pnpm build`
- `pnpm test:perf` (run three times on 2026-05-12 because the first two runs were noisy, then rerun before commit on 2026-05-13)

`pnpm build` passed after the UI slice. Relevant production output:

- `dist/assets/training.worker-S7ocudoV.js`: 83.50 kB
- `dist/assets/index-1WlUmzVU.css`: 66.62 kB, gzip 11.56 kB
- `dist/assets/InspectionPanel-C0sjYXVR.js`: 10.98 kB, gzip 2.91 kB
- `dist/assets/RunHistoryPanel-CTUmNQkd.js`: 16.70 kB, gzip 5.14 kB
- `dist/assets/index-BtaMrfLu.js`: 366.75 kB, gzip 111.02 kB

Compared with the worker RPC build, the worker and main app bundles were effectively unchanged. The lazy Inspection panel chunk increased from 8.31 kB to 10.98 kB because it now renders the bounded backprop preview UI. The CSS bundle increased from 66.18 kB to 66.62 kB. These changes are below the roadmap build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed on all three runs. The first two runs were noisy:

- First run: `predictGrid` 1571.3100 ms, `predictGridInto` 1655.4161 ms, `predictGridWithNeurons` 973.7060 ms, `predictGridWithNeuronsInto` 709.0399 ms, Adam/L2/Clip applyGradients 5.3637 ms, SGD applyGradients 1.6433 ms.
- Second run: `predictGrid` 1429.3662 ms, `predictGridInto` 1321.5164 ms, `predictGridWithNeurons` 867.0705 ms, `predictGridWithNeuronsInto` 775.4257 ms, Adam/L2/Clip applyGradients 4.9737 ms, SGD applyGradients 2.4415 ms.

The third run returned to the established range:

- `predictGrid`: 1237.1647 ms total for 100 iterations
- `predictGridInto`: 1223.3140 ms total for 100 iterations
- `predictGridWithNeurons`: 771.2188 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 659.5370 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.5729 ms
- Average `applyGradients` time (SGD): 1.5759 ms

The 2026-05-13 pre-commit rerun also passed with 2 benchmark files and 4 benchmark tests:

- `predictGrid`: 1370.2497 ms total for 100 iterations
- `predictGridInto`: 1239.0260 ms total for 100 iterations
- `predictGridWithNeurons`: 750.3219 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 646.0797 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 6.1264 ms
- Average `applyGradients` time (SGD): 1.6279 ms

The UI slice does not touch engine prediction, gradient hot paths, worker cadence, frame-buffer semantics, or live training behavior. Browser QA passed on 2026-05-13 after restarting the local dev server and manually restoring the in-app browser to the local URL; see `docs/qa/browser-qa/wave-7-backprop-preview.md`.

## Wave 7 Loss Landscape Probe Engine Foundation

Date: 2026-05-13

Scope: engine-only deterministic dry-run loss-landscape probe. This slice adds bounded probe types and `Network.probeLossLandscape()` with a maximum `7x7` grid, maximum 64 evaluated samples, capped radius of 1.0, deterministic first-two-weight axes, and checkpoint-copy evaluation. It does not add worker RPCs, UI, frame-buffer fields, shared protocol changes, URL/config serialization, persistence schema changes, public config shape changes, dependencies, or training behavior changes.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the engine foundation slice. Relevant production output:

- `dist/assets/training.worker-CN1MGCA3.js`: 85.85 kB
- `dist/assets/index-1WlUmzVU.css`: 66.62 kB, gzip 11.56 kB
- `dist/assets/InspectionPanel-CmeGnjPP.js`: 10.98 kB, gzip 2.91 kB
- `dist/assets/RunHistoryPanel-u40Ox7Yv.js`: 16.70 kB, gzip 5.14 kB
- `dist/assets/index-DVvHT95l.js`: 366.75 kB, gzip 111.02 kB

Compared with the Wave 7 backprop preview UI build, the main app, CSS, and lazy UI chunks were effectively unchanged. The worker bundle increased from 83.50 kB to 85.85 kB because the worker imports the engine class that now includes the loss-landscape dry-run method. The increase is about 2.8%, below the roadmap 10% build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed after the engine foundation slice with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1182.8003 ms total for 100 iterations
- `predictGridInto`: 1141.2705 ms total for 100 iterations
- `predictGridWithNeurons`: 716.4242 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 656.4544 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 6.1383 ms
- Average `applyGradients` time (SGD): 1.5036 ms

These values remain within the established range and below the roadmap warning thresholds. The new probe is an explicit dry-run helper and is not called from the live training loop.

## Wave 7 Loss Landscape Probe Worker RPC

Date: 2026-05-13

Scope: worker-only one-shot Comlink RPC for the approved Loss Landscape Probe path. This slice exposes the already-bounded engine `probeLossLandscape()` result through `getLossLandscapeProbe()`, converts the engine `Float32Array` loss grid into a plain `number[]`, and documents the RPC. It does not add UI state, streamed messages, frame-buffer fields, shared protocol guards, URL/config serialization, persistence schema changes, public config shape changes, dependencies, or training behavior changes.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the worker RPC slice. Relevant production output:

- `dist/assets/training.worker-C2DakSc7.js`: 86.45 kB
- `dist/assets/index-1WlUmzVU.css`: 66.62 kB, gzip 11.56 kB
- `dist/assets/InspectionPanel-D13XD3j9.js`: 10.98 kB, gzip 2.91 kB
- `dist/assets/RunHistoryPanel-DFkFivXN.js`: 16.70 kB, gzip 5.14 kB
- `dist/assets/index-BL2FJMYe.js`: 366.75 kB, gzip 111.02 kB

Compared with the engine foundation build, the main app, CSS, and lazy UI chunks were effectively unchanged. The worker bundle increased from 85.85 kB to 86.45 kB because it now exposes a Comlink one-shot RPC and serializable response wrapper. The increase is about 0.7%, below the roadmap 10% build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed after the worker RPC slice with 2 benchmark files and 4 benchmark tests. The first run after implementation was slightly faster (`predictGrid` 1205.8439 ms, `predictGridInto` 1152.6893 ms, `predictGridWithNeurons` 709.6152 ms, `predictGridWithNeuronsInto` 597.6258 ms, Adam/L2/Clip 5.0184 ms, SGD 1.4061 ms). The pre-commit rerun after tightening raw-buffer test coverage also passed.

Observed pre-commit benchmark output:

- `predictGrid`: 1267.9825 ms total for 100 iterations
- `predictGridInto`: 1170.3269 ms total for 100 iterations
- `predictGridWithNeurons`: 724.9360 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 614.3189 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 5.8100 ms
- Average `applyGradients` time (SGD): 1.5234 ms

These values remain within the established range and below the roadmap warning thresholds. The RPC is one-shot, manual, and not called from the live training loop, so no runtime cadence impact is expected from this slice.
