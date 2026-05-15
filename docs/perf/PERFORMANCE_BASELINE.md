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

## Wave 7 Loss Landscape Probe UI

Date: 2026-05-13

Scope: Inspection-panel UI for manually requesting the one-shot Loss Landscape Probe. This slice stores only the bounded RPC response in local component state, renders a compact heatmap and text summary, and adds keyboard/error/loading coverage. It does not add streamed worker messages, frame-buffer fields, raw activation/loss arrays in React state, URL/config serialization, persistence schema changes, public config shape changes, dependencies, or training behavior changes.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the UI slice. Relevant production output:

- `dist/assets/training.worker-C2DakSc7.js`: 86.45 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/InspectionPanel-DcCt75L5.js`: 13.42 kB, gzip 3.49 kB
- `dist/assets/RunHistoryPanel-8zhhtp04.js`: 16.70 kB, gzip 5.14 kB
- `dist/assets/index-BUF1FoVn.js`: 366.75 kB, gzip 111.02 kB

Compared with the worker RPC build, the worker, run-history, and main app bundles were effectively unchanged. The lazy Inspection panel chunk increased from 10.98 kB to 13.42 kB because it now renders the bounded loss-surface controls, grid, and text summary. The CSS bundle increased from 66.62 kB to 67.02 kB. The existing Vite large chunk warning remains; the overall production build-size change is below the roadmap warning threshold.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1159.3244 ms total for 100 iterations
- `predictGridInto`: 1101.5146 ms total for 100 iterations
- `predictGridWithNeurons`: 695.7840 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 590.7820 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.3727 ms
- Average `applyGradients` time (SGD): 1.4332 ms

These values remain within the established range and below the roadmap warning thresholds. The UI slice does not touch engine prediction, gradient hot paths, worker cadence, frame-buffer semantics, or live training behavior. Browser QA passed on 2026-05-13 after starting the local dev server; see `docs/qa/browser-qa/wave-7-loss-landscape-probe.md`.

## Wave 7 Multiclass Softmax/Categorical-Loss Helpers

Date: 2026-05-14

Scope: engine-only math foundation for Multiclass Classification Mode. This slice adds a stable vector `softmax()` helper and categorical cross-entropy helpers with normalized-distribution validation and typed-array-friendly inputs. It does not wire the helpers into `Network`, app config, worker/runtime, UI, frame buffers, URL/config serialization, persistence schema, public config shape, code export, dependencies, deployment, or live training behavior.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the engine helper slice. Relevant production output:

- `dist/assets/training.worker-C2DakSc7.js`: 86.45 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-DZ1GTedS.js`: 5.68 kB, gzip 2.04 kB
- `dist/assets/InspectionPanel-DcCt75L5.js`: 13.42 kB, gzip 3.49 kB
- `dist/assets/RunHistoryPanel-8zhhtp04.js`: 16.70 kB, gzip 5.14 kB
- `dist/assets/index-BUF1FoVn.js`: 366.75 kB, gzip 111.02 kB

Compared with the Wave 7 Loss Landscape Probe UI build, bundle sizes were effectively unchanged. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1438.0162 ms total for 100 iterations
- `predictGridInto`: 1426.5795 ms total for 100 iterations
- `predictGridWithNeurons`: 817.1946 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 770.4850 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 7.3476 ms
- Average `applyGradients` time (SGD): 2.1358 ms

The run was slower than the prior loss-landscape UI benchmark, but this slice only exports standalone math helpers and they are not called by the live training loop, worker cadence path, frame-buffer path, WebGPU path, or prediction grid path. Treat this as benchmark noise unless future integrated multiclass runs reproduce the slowdown while exercising a new path.

## Wave 7 Private Multiclass Network Foundation

Date: 2026-05-14

Scope: private engine-only `Network` foundation for Multiclass Classification Mode. This slice lets engine tests exercise softmax output plus categorical cross-entropy through local casts, adds log-sum-exp categorical loss from output logits, validates categorical target distributions, and covers hidden-layer finite-difference gradients. It does not expose `ActivationType` or `LossType` publicly, change shared serialization, worker target encoding, protocol/frame-buffer paths, URL/config, persistence/run-history schema, code export, dependencies, deployment, or UI.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the private `Network` foundation slice. Relevant production output:

- `dist/assets/training.worker-DjkiQX_t.js`: 89.00 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-DZ1GTedS.js`: 5.68 kB, gzip 2.04 kB
- `dist/assets/InspectionPanel-DuOyQ9j7.js`: 13.42 kB, gzip 3.50 kB
- `dist/assets/RunHistoryPanel-DWVn3xNz.js`: 16.70 kB, gzip 5.14 kB
- `dist/assets/index-BgwmzzLI.js`: 366.75 kB, gzip 111.02 kB

Compared with the prior Multiclass helper build, the worker bundle increased from 86.45 kB to 89.00 kB because the worker imports the engine class that now contains the private multiclass `Network` path. The increase is about 3.0%, below the roadmap 10% build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1090.3664 ms total for 100 iterations
- `predictGridInto`: 1101.1419 ms total for 100 iterations
- `predictGridWithNeurons`: 688.5447 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 591.9035 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.0836 ms
- Average `applyGradients` time (SGD): 1.3856 ms

These values returned to the established benchmark range and remain below the roadmap warning thresholds. The private multiclass `Network` path is not exposed through current worker training configs or UI controls, so no live runtime cadence impact is expected until a separately approved public/shared integration slice wires it into app state and worker target encoding.

## Wave 7 Public/Shared Multiclass Contract Reservation

Date: 2026-05-14

Scope: public/shared contract reservation for Multiclass Classification Mode. This slice expands the engine `ActivationType`/`LossType` contracts to reserve `softmax` and `categoricalCrossEntropy`, keeps scalar activation/loss lookup separate from vector multiclass helpers, makes shared strict imports return a clear gated error for multiclass configs, keeps lenient URL decode on current runtime defaults, and adds web tests proving the future-only controls are not exposed. It does not add worker target encoding, UI controls, protocol/frame-buffer paths, persistence/run-history schema changes, URL/config output-size serialization, code export behavior, dependencies, deployment changes, or live training behavior.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the contract reservation slice. Relevant production output:

- `dist/assets/training.worker-goNyCWLR.js`: 89.72 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-CoiZi_rp.js`: 7.25 kB, gzip 2.95 kB
- `dist/assets/InspectionPanel-CankpUpg.js`: 13.42 kB, gzip 3.50 kB
- `dist/assets/RunHistoryPanel-BkavPI7R.js`: 16.70 kB, gzip 5.14 kB
- `dist/assets/index-B_bc6EYv.js`: 367.12 kB, gzip 111.13 kB

Compared with the private multiclass `Network` foundation build, the worker bundle increased from 89.00 kB to 89.72 kB, about 0.8%, below the roadmap 10% warning threshold. The main app bundle increased from 366.75 kB to 367.12 kB, about 0.1%. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1131.1425 ms total for 100 iterations
- `predictGridInto`: 1117.4338 ms total for 100 iterations
- `predictGridWithNeurons`: 699.5975 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 591.5526 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.1311 ms
- Average `applyGradients` time (SGD): 1.4898 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The new contract values are not exposed through current worker target encoding or UI controls, so no live runtime cadence impact is expected until a separately approved worker/UI slice wires multiclass configs into the running app.

## Wave 7 Worker Multiclass Target Guards

Date: 2026-05-15

Scope: worker-only Multiclass Classification Mode target encoding and compatibility guards. This slice lets direct worker calls use the exact bounded runtime shape `classification + outputSize 3 + softmax + categoricalCrossEntropy`, encodes labels as one-hot targets, keeps live arena scalar-only, and clears scalar grid caches when multiclass streamed snapshots explicitly omit scalar visualization data. It does not expose UI controls, change shared URL/config serialization, change persistence/run-history schema, change code export, add datasets or presets, add dependencies, change deployment, or add multiclass frame-buffer/protocol payloads.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the worker target-guard slice. Relevant production output:

- `dist/assets/training.worker-CB34B7zL.js`: 91.19 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-opI1oGqP.js`: 7.25 kB, gzip 2.95 kB
- `dist/assets/InspectionPanel-DkdZZarG.js`: 13.42 kB, gzip 3.49 kB
- `dist/assets/RunHistoryPanel-ZSHEaVkq.js`: 16.70 kB, gzip 5.14 kB
- `dist/assets/index-CiRxQBda.js`: 367.34 kB, gzip 111.18 kB

Compared with the public/shared multiclass contract reservation build, the worker bundle increased from 89.72 kB to 91.19 kB, about 1.6%, and the main app bundle increased from 367.12 kB to 367.34 kB, about 0.1%. Both are below the 10% roadmap build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1198.2515 ms total for 100 iterations
- `predictGridInto`: 1150.1891 ms total for 100 iterations
- `predictGridWithNeurons`: 743.6962 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 593.9905 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.5155 ms
- Average `applyGradients` time (SGD): 1.4182 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The slice does not add multiclass UI, URL/config output-size serialization, persistence, code export, datasets/presets, dependencies, or multiclass grid/protocol payloads. No browser QA was run because the feature remains worker-only and is not reachable from visible UI controls.

## Wave 7 Hidden Multiclass Dataset Helper

Date: 2026-05-15

Scope: engine-only hidden three-class cluster dataset helper for future Multiclass Classification Mode wiring. This slice adds deterministic bounded 3-class sample generation in `packages/engine/src/datasets.ts` and tests it directly from the engine module. It does not add the dataset to `DatasetType`, `generateDataset`, the package-level engine barrel, shared URL/config serialization, presets, worker/runtime, frame buffers, persistence/run-history schema, code export, dependencies, deployment, or UI.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the hidden dataset-helper slice. Relevant production output:

- `dist/assets/training.worker-CB34B7zL.js`: 91.19 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-opI1oGqP.js`: 7.25 kB, gzip 2.95 kB
- `dist/assets/InspectionPanel-DkdZZarG.js`: 13.42 kB, gzip 3.49 kB
- `dist/assets/RunHistoryPanel-ZSHEaVkq.js`: 16.70 kB, gzip 5.14 kB
- `dist/assets/index-CiRxQBda.js`: 367.34 kB, gzip 111.18 kB

Compared with the worker target-guard build, production bundle sizes were unchanged. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1090.1719 ms total for 100 iterations
- `predictGridInto`: 1261.3116 ms total for 100 iterations
- `predictGridWithNeurons`: 757.7095 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 600.3780 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.0243 ms
- Average `applyGradients` time (SGD): 1.6422 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The helper is not reachable from the live app or worker config path, so no runtime cadence or browser performance impact is expected from this slice.

## Wave 7 Multiclass Code Export Guard

Date: 2026-05-15

Scope: shared/web code-export guard for future Multiclass Classification Mode. This slice makes Pseudocode, NumPy, and TF.js exports truthful for an already-constructed `outputSize: 3`, `softmax`, `categoricalCrossEntropy` config, and removes the Code Export panel's hard-coded output-size override. It does not expose multiclass UI controls, datasets, presets, URL/config serialization, persistence/run-history schema, worker protocol/frame-buffer payloads, dependencies, deployment, or training behavior.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after the code-export guard slice. Relevant production output:

- `dist/assets/training.worker-CB34B7zL.js`: 91.19 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-D0b8h3A8.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-DC1_Y0oi.js`: 13.42 kB, gzip 3.50 kB
- `dist/assets/RunHistoryPanel-QnEaDprS.js`: 16.70 kB, gzip 5.14 kB
- `dist/assets/index-CSTehKAJ.js`: 367.34 kB, gzip 111.18 kB

Compared with the hidden dataset-helper build, the worker, main app, engine, CSS, inspection, and run-history chunks were unchanged. The lazy Code Export chunk increased from 7.25 kB in the earlier worker-guard build to 7.69 kB, about 6.1%, and remains below the 10% roadmap build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1094.3072 ms total for 100 iterations
- `predictGridInto`: 1101.6861 ms total for 100 iterations
- `predictGridWithNeurons`: 689.8360 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 585.9218 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.0932 ms
- Average `applyGradients` time (SGD): 1.4034 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The slice is code-export-only and does not touch engine training, worker cadence, frame-buffer transport, or live visualization paths.

## Wave 7 Stale Confusion Cache Guard

Date: 2026-05-15

Scope: test-stability cleanup plus a runtime/cache guard for future Multiclass Classification Mode. The test-stability slice widens waits around existing lazy/Suspense panel content under heavy load. The runtime slice clears stale binary confusion data from both the React snapshot and frame buffer when a fresh streamed snapshot omits confusion data, while preserving prior confusion during explicitly stale test-metrics snapshots. It does not change worker protocol fields, frame-buffer version semantics, URL/config serialization, persistence/run-history schema, public config shape, engine math, dependencies, deployment, or visible UI.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `1529904` and `6ef96c2`. Relevant production output:

- `dist/assets/training.worker-CB34B7zL.js`: 91.19 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-DG3lU0-G.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-D_SGlGP9.js`: 13.42 kB, gzip 3.49 kB
- `dist/assets/RunHistoryPanel-C9e_zIIf.js`: 16.70 kB, gzip 5.14 kB
- `dist/assets/index-DNwr_meN.js`: 367.43 kB, gzip 111.20 kB

Compared with the code-export guard build, the worker, lazy Code Export, engine, CSS, inspection, and run-history chunk sizes remain effectively unchanged. The main app bundle changed from 367.34 kB to 367.43 kB, about 0.02%, well below the 10% roadmap build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1115.9496 ms total for 100 iterations
- `predictGridInto`: 1093.1002 ms total for 100 iterations
- `predictGridWithNeurons`: 686.7536 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 582.5746 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.1261 ms
- Average `applyGradients` time (SGD): 1.4488 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. One intermediate perf run before the final verification was noisy (`predictGridWithNeurons` 1048.0556 ms and `predictGridWithNeuronsInto` 717.0967 ms), but the repeat final run returned to the recent range. The runtime slice only changes cache invalidation for omitted binary confusion data and should not affect engine training throughput.

## Wave 7 Run-History Persistence Guard

Date: 2026-05-15

Scope: test-stability cleanup plus a capture-time persistence guard for future Multiclass Classification Mode. The test-stability slice narrows the legacy `MainArea` Code Export assertion to panel placement while `CodeExportPanel.test.tsx` continues to cover detailed export tabs and generated code. The persistence guard prevents hidden unsupported multiclass configs from producing run-history records that the existing strict persistence validator would reject later. It does not change URL/config serialization, persistence/run-history schema, public config shape, worker protocol, frame-buffer semantics, engine math, dependencies, deployment, or visible UI.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `ec2ba94` and `4035f9d`. Relevant production output:

- `dist/assets/training.worker-CB34B7zL.js`: 91.19 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-KaN8hkYP.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-B-yDKSL1.js`: 13.42 kB, gzip 3.50 kB
- `dist/assets/RunHistoryPanel-DXKu_yh-.js`: 16.71 kB, gzip 5.15 kB
- `dist/assets/index-wHcTtHlA.js`: 367.43 kB, gzip 111.20 kB

Compared with the stale-confusion guard build, the worker, lazy Code Export, engine, CSS, inspection, and main app chunk sizes remain effectively unchanged. The lazy Run History chunk changed from 16.70 kB to 16.71 kB, about 0.06%, below the 10% roadmap build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1069.2586 ms total for 100 iterations
- `predictGridInto`: 1058.1727 ms total for 100 iterations
- `predictGridWithNeurons`: 672.0584 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 575.4159 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 3.8577 ms
- Average `applyGradients` time (SGD): 1.5844 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The slice does not touch engine training throughput, worker cadence, visualization payloads, or browser-visible rendering.

## Wave 7 Visualization Non-Exposure Guard

Date: 2026-05-15

Scope: UI-only visualization fallback guard for future Multiclass Classification Mode. `DecisionBoundary` now avoids binary decision-boundary copy and Negative/Positive legends when current classification config or point labels indicate non-binary output. `ConfusionMatrix` now separates "no test data" from "test data exists but no matrix is available yet" with neutral demand-gated copy, after review found that missing matrices can be normal while the panel waits for fresh binary metrics. This slice does not change URL/config serialization, persistence/run-history schema, public config shape, worker protocol, frame-buffer semantics, engine math, dependencies, deployment, or training behavior.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `d90ec15`. Relevant production output:

- `dist/assets/training.worker-CB34B7zL.js`: 91.19 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-025rTLae.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-Dvl_owys.js`: 13.42 kB, gzip 3.50 kB
- `dist/assets/RunHistoryPanel-NHBwCCM0.js`: 16.71 kB, gzip 5.15 kB
- `dist/assets/index-ClV2IhNJ.js`: 368.30 kB, gzip 111.37 kB

Compared with the run-history persistence guard build, the worker, lazy Code Export, engine, CSS, inspection, and run-history chunks remain effectively unchanged. The main app chunk changed from 367.43 kB to 368.30 kB, about 0.24%, below the 10% roadmap build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1072.2905 ms total for 100 iterations
- `predictGridInto`: 1062.1561 ms total for 100 iterations
- `predictGridWithNeurons`: 674.3473 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 577.0119 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 3.9098 ms
- Average `applyGradients` time (SGD): 1.4554 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The slice is UI/fallback-copy only and does not touch engine training throughput, worker cadence, frame-buffer transport, or heavy visualization payload generation.

## Wave 7 Public Scalar-Control Guard

Date: 2026-05-15

Scope: public-control guard for future Multiclass Classification Mode. `setDataset` now resets `network.outputSize` to `1` when public dataset/problem controls select the existing scalar classification/regression paths, and component/store tests assert that Network, Hyperparams, Data, and built-in preset flows do not expose or retain `softmax`, `categoricalCrossEntropy`, or multi-output runtime config. This slice does not change URL/config serialization, persistence/run-history schema, public config shape, worker protocol, frame-buffer semantics, engine math, dependencies, deployment, or training behavior beyond scalar public dataset normalization.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `4099c24`. Relevant production output:

- `dist/assets/training.worker-CB34B7zL.js`: 91.19 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-C1Cmj0f9.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-BqR7RALx.js`: 13.42 kB, gzip 3.49 kB
- `dist/assets/RunHistoryPanel-Djg__emT.js`: 16.71 kB, gzip 5.15 kB
- `dist/assets/index-B0NOmPKH.js`: 368.31 kB, gzip 111.38 kB

Compared with the visualization non-exposure guard build, the worker, lazy Code Export, engine, CSS, inspection, and run-history chunks remain effectively unchanged. The main app chunk changed from 368.30 kB to 368.31 kB, about 0.003%, below the 10% roadmap build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1073.4571 ms total for 100 iterations
- `predictGridInto`: 1064.9546 ms total for 100 iterations
- `predictGridWithNeurons`: 678.7174 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 576.5990 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 3.9368 ms
- Average `applyGradients` time (SGD): 1.4758 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The slice affects store normalization and tests only, and does not touch engine training throughput, worker cadence, frame-buffer transport, or heavy visualization payload generation.

## Wave 7 URL/Config Outbound Guard

Date: 2026-05-15

Scope: app-boundary outbound URL/config guard for future Multiclass Classification Mode. `syncToUrl()` now leniently normalizes current store config and fails closed to defaults before URL encoding, and Config Panel JSON export now validates current config strictly before creating a blob. Hidden unsupported multiclass state cannot publish `softmax`, `categoricalCrossEntropy`, or multi-output config through current app URL sync or JSON export. This slice does not change URL/config keys or schema, persistence/run-history schema, public config shape, worker protocol, frame-buffer semantics, engine math, dependencies, deployment, or training behavior.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `9a4235d`. Relevant production output:

- `dist/assets/training.worker-CB34B7zL.js`: 91.19 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-yT0kjw6V.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-BJ2CGnxX.js`: 13.42 kB, gzip 3.49 kB
- `dist/assets/RunHistoryPanel-BB7ZDyoe.js`: 16.71 kB, gzip 5.15 kB
- `dist/assets/index-CmVkBTE8.js`: 368.46 kB, gzip 111.44 kB

Compared with the public scalar-control guard build, the worker, lazy Code Export, engine, CSS, inspection, and run-history chunks remain effectively unchanged. The main app chunk changed from 368.31 kB to 368.46 kB, about 0.04%, below the 10% roadmap build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1076.7760 ms total for 100 iterations
- `predictGridInto`: 1060.2178 ms total for 100 iterations
- `predictGridWithNeurons`: 669.2138 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 578.0207 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.0562 ms
- Average `applyGradients` time (SGD): 1.4136 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The slice affects app-local config boundary checks only, and does not touch engine training throughput, worker cadence, frame-buffer transport, or heavy visualization payload generation.

## Wave 7 Public Dataset/Preset Non-Exposure Guard

Date: 2026-05-15

Scope: test-only registry guard for future Multiclass Classification Mode. Shared preset tests now assert built-in presets use only the current public dataset allow-list, `outputSize: 1`, non-softmax output activation, and non-categorical loss. Lesson registry tests assert lesson presets stay scalar-facing. This slice does not change production code, URL/config format, public config shape, persistence/run-history schema, worker protocol, frame-buffer semantics, engine math, dependencies, deployment, or training behavior.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `2897a26`. Relevant production output:

- `dist/assets/training.worker-CB34B7zL.js`: 91.19 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-yT0kjw6V.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-BJ2CGnxX.js`: 13.42 kB, gzip 3.49 kB
- `dist/assets/RunHistoryPanel-BB7ZDyoe.js`: 16.71 kB, gzip 5.15 kB
- `dist/assets/index-CmVkBTE8.js`: 368.46 kB, gzip 111.44 kB

Compared with the URL/config outbound guard build, production output remained unchanged. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1074.8065 ms total for 100 iterations
- `predictGridInto`: 1058.6659 ms total for 100 iterations
- `predictGridWithNeurons`: 672.2782 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 577.4767 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 3.9552 ms
- Average `applyGradients` time (SGD): 1.4398 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The slice is test-only and does not touch engine training throughput, worker cadence, frame-buffer transport, or heavy visualization payload generation.

## Wave 7 Experiment Memory Serialized Network Guard

Date: 2026-05-15

Scope: shared persistence-validation guard for future Multiclass Classification Mode. Experiment-memory validation now checks the embedded `SerializedNetwork.config` against the current public compatibility boundary before preserving a run record. The accepted serialized network payload is still cloned unchanged, and `network: null` records remain valid. This slice does not change URL/config format, public config shape, run-history schema version, worker protocol, frame-buffer semantics, engine math, dependencies, deployment, training behavior, or visualization payload transport.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `f83ac4b`. Relevant production output:

- `dist/assets/training.worker-CB34B7zL.js`: 91.19 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-CJV4H9e4.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-DQDh5g45.js`: 13.42 kB, gzip 3.49 kB
- `dist/assets/RunHistoryPanel-CJPPRxN5.js`: 17.01 kB, gzip 5.21 kB
- `dist/assets/index-ZZGBORoL.js`: 369.84 kB, gzip 111.75 kB

Compared with the Config Panel import rejection guard build, the worker, CSS, lazy Code Export, engine, inspection, and main app chunks remain effectively unchanged. The lazy Run History chunk changed from 16.71 kB to 17.01 kB, about 1.8%, below the 10% roadmap build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1074.3888 ms total for 100 iterations
- `predictGridInto`: 1055.8439 ms total for 100 iterations
- `predictGridWithNeurons`: 675.2009 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 570.6535 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.0443 ms
- Average `applyGradients` time (SGD): 1.5450 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The slice affects shared validation only and does not touch engine training throughput, worker cadence, frame-buffer transport, or heavy visualization payload generation.

## Wave 7 Serialized Network Payload Validation

Date: 2026-05-15

Scope: shared persistence-validation guard for future Multiclass Classification Mode. Experiment-memory validation now checks serialized-network weights and biases for expected layer counts, matrix/vector dimensions, sparse entries, and finite numeric values before preserving a run record. Accepted payloads are still cloned unchanged, and `network: null` records remain valid. This slice does not change URL/config format, public config shape, run-history schema version, worker protocol, frame-buffer semantics, engine math, dependencies, deployment, training behavior, or visualization payload transport.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `82f0a61`. Relevant production output:

- `dist/assets/training.worker-CB34B7zL.js`: 91.19 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-eumM5NXQ.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-BSMb-XzD.js`: 13.42 kB, gzip 3.50 kB
- `dist/assets/RunHistoryPanel-DmgXe0gZ.js`: 17.96 kB, gzip 5.43 kB
- `dist/assets/index-22sgq67Z.js`: 369.84 kB, gzip 111.75 kB

Compared with the prior experiment-memory serialized-network guard build, the worker, CSS, lazy Code Export, engine, inspection, and main app chunks remain effectively unchanged. The lazy Run History chunk changed from 17.01 kB to 17.96 kB, about 5.6%, below the 10% roadmap build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1071.4690 ms total for 100 iterations
- `predictGridInto`: 1065.7098 ms total for 100 iterations
- `predictGridWithNeurons`: 699.5400 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 575.6738 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 3.9841 ms
- Average `applyGradients` time (SGD): 1.6491 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The slice affects shared validation only and does not touch engine training throughput, worker cadence, frame-buffer transport, or heavy visualization payload generation.

## Wave 7 Shared Multiclass URL/Config Contract

Date: 2026-05-15

Scope: opt-in shared URL/import validation for the future public Multiclass Classification Mode. Shared serialization now accepts and round-trips exactly one bounded 3-class config when callers explicitly pass the multiclass option: classification data, `outputSize: 3`, `softmax`, and `categoricalCrossEntropy`. Default public app paths remain scalar-only. This slice does not add visible UI controls, presets, persistence/run-history schema changes, worker protocol changes, frame-buffer fields, dependencies, deployment changes, training-behavior changes, or visualization payload transport.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `16e696e`. Relevant production output:

- `dist/assets/training.worker-BFdCjK-A.js`: 91.69 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-DoyOs-i4.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-CKR7X0DG.js`: 13.42 kB, gzip 3.50 kB
- `dist/assets/RunHistoryPanel-DiKj5nuT.js`: 17.96 kB, gzip 5.43 kB
- `dist/assets/index-BLRFr57A.js`: 370.76 kB, gzip 112.04 kB

Compared with the serialized-network payload-validation build, the lazy Code Export, engine, inspection, run-history, and CSS chunks remain effectively unchanged. The main app chunk changed from 369.84 kB to 370.76 kB, about 0.25%, and the worker chunk changed from 91.19 kB to 91.69 kB, about 0.55%; both are below the 10% roadmap build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1070.7663 ms total for 100 iterations
- `predictGridInto`: 1061.2701 ms total for 100 iterations
- `predictGridWithNeurons`: 671.1200 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 577.2953 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 3.9806 ms
- Average `applyGradients` time (SGD): 1.4759 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The slice affects shared serialization validation only and does not touch engine training throughput, worker cadence, frame-buffer transport, or heavy visualization payload generation.

## Wave 7 Opt-In Multiclass Experiment-Memory Eligibility

Date: 2026-05-15

Scope: explicit opt-in shared experiment-memory validation for the future public Multiclass Classification Mode. Shared experiment memory can now accept the exact approved 3-class config and matching serialized-network payloads when callers pass the multiclass option. Default public app save/load/capture/history paths remain scalar-only. This slice does not add visible UI controls, presets, persistence/run-history schema version changes, worker protocol changes, frame-buffer fields, dependencies, deployment changes, training-behavior changes, or visualization payload transport.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `37b8824`. Relevant production output:

- `dist/assets/training.worker-BFdCjK-A.js`: 91.69 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-CUKpNtMy.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-K85eKO-k.js`: 13.42 kB, gzip 3.50 kB
- `dist/assets/RunHistoryPanel-QLHf-fqu.js`: 18.06 kB, gzip 5.46 kB
- `dist/assets/index-Ca4Kgis4.js`: 370.76 kB, gzip 112.04 kB

Compared with the shared multiclass URL/config contract build, the worker, CSS, lazy Code Export, engine, inspection, and main app chunks remain effectively unchanged. The lazy Run History chunk changed from 17.96 kB to 18.06 kB, about 0.6%, below the 10% roadmap build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1163.3106 ms total for 100 iterations
- `predictGridInto`: 1111.3547 ms total for 100 iterations
- `predictGridWithNeurons`: 687.8551 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 580.6636 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.2266 ms
- Average `applyGradients` time (SGD): 1.5468 ms

These values remain below the roadmap performance warning thresholds compared with the prior shared-config run. The largest observed benchmark delta was `predictGrid`, about 8.6%, below the 20% warning threshold; this slice does not touch engine training, worker cadence, frame-buffer transport, or heavy visualization payload generation.

## Wave 7 Hidden Multiclass Dataset Contract

Date: 2026-05-15

Scope: engine-local metadata contract for the hidden deterministic three-class cluster generator. The contract records the bounded class count, labels, default generation parameters, output size, softmax activation, and categorical cross-entropy loss for future public multiclass wiring. It remains out of the package barrel and does not change `DatasetType`, shared serialization, public dataset generation, presets, lessons, UI controls, persistence defaults, worker/protocol behavior, frame-buffer fields, dependencies, deployment behavior, training behavior, or visualization payload transport.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `ea48342`. Relevant production output:

- `dist/assets/training.worker-BFdCjK-A.js`: 91.69 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-CUKpNtMy.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-K85eKO-k.js`: 13.42 kB, gzip 3.50 kB
- `dist/assets/RunHistoryPanel-QLHf-fqu.js`: 18.06 kB, gzip 5.46 kB
- `dist/assets/index-Ca4Kgis4.js`: 370.76 kB, gzip 112.04 kB

Compared with the opt-in experiment-memory eligibility build, production output was unchanged. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1162.2222 ms total for 100 iterations
- `predictGridInto`: 1103.9761 ms total for 100 iterations
- `predictGridWithNeurons`: 679.0895 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 622.4592 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.3995 ms
- Average `applyGradients` time (SGD): 1.4579 ms

These values remain below the roadmap performance warning thresholds. The slice adds metadata and tests only and does not touch engine training throughput, worker cadence, frame-buffer transport, or heavy visualization payload generation.

## Wave 7 Binary Confusion Matrix Guard

Date: 2026-05-15

Scope: visualization-local guard for future Multiclass Classification Mode. The Confusion Matrix panel now refuses to render binary confusion matrices when the current network config or explicit train/test labels indicate unsupported multiclass state. The guard keeps normal scalar binary matrices and demand-gated unavailable states intact. This slice does not change URL/config format, public config shape, persistence/run-history schema, worker protocol, frame-buffer semantics, engine math, dependencies, deployment, training behavior, or multiclass visualization payload transport.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `b7c3499`. Relevant production output:

- `dist/assets/training.worker-CB34B7zL.js`: 91.19 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-CJxUHbWK.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-BWFvgAAd.js`: 13.42 kB, gzip 3.49 kB
- `dist/assets/RunHistoryPanel-XQ4O8pIB.js`: 16.71 kB, gzip 5.15 kB
- `dist/assets/index-Cr2dBbDY.js`: 369.84 kB, gzip 111.75 kB

Compared with the public runtime config guard build, the worker, CSS, lazy Code Export, engine, inspection, and run-history chunks remain effectively unchanged. The main app chunk changed from 369.27 kB to 369.84 kB, about 0.15%, below the 10% roadmap build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1080.2500 ms total for 100 iterations
- `predictGridInto`: 1071.1034 ms total for 100 iterations
- `predictGridWithNeurons`: 672.1155 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 577.5240 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 3.9272 ms
- Average `applyGradients` time (SGD): 1.3876 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The slice affects visualization rendering only and does not touch engine training throughput, worker cadence, frame-buffer transport, or heavy visualization payload generation.

## Wave 7 Config Panel Import Rejection Guard

Date: 2026-05-15

Scope: test-only guard for future Multiclass Classification Mode. The Config Panel import test now asserts hidden unsupported multiclass JSON is rejected through the existing strict validation path before applying a preset, resetting training state, or mutating public scalar network/training settings. This slice does not change production code, URL/config format, public config shape, persistence/run-history schema, worker protocol, frame-buffer semantics, engine math, dependencies, deployment, training behavior, or visualization payload transport.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `2545487`. Relevant production output:

- `dist/assets/training.worker-CB34B7zL.js`: 91.19 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-CJxUHbWK.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-BWFvgAAd.js`: 13.42 kB, gzip 3.49 kB
- `dist/assets/RunHistoryPanel-XQ4O8pIB.js`: 16.71 kB, gzip 5.15 kB
- `dist/assets/index-Cr2dBbDY.js`: 369.84 kB, gzip 111.75 kB

Compared with the binary confusion matrix guard build, production output remained unchanged. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1076.4446 ms total for 100 iterations
- `predictGridInto`: 1060.0425 ms total for 100 iterations
- `predictGridWithNeurons`: 670.3855 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 577.6365 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 3.9816 ms
- Average `applyGradients` time (SGD): 1.5374 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The slice is test-only and does not touch engine training throughput, worker cadence, frame-buffer transport, or heavy visualization payload generation.

## Wave 7 Public Store Setter Non-Exposure Guard

Date: 2026-05-15

Scope: app-local store guard for future Multiclass Classification Mode. Public playground store setters now reject hidden multiclass-only loss/activation values while preserving already-valid scalar state. If a pre-existing hidden value is already present, the setter path clamps back to the current scalar public contract. This slice does not change URL/config format, public config shape, persistence/run-history schema, worker protocol, frame-buffer semantics, engine math, dependencies, deployment, or training behavior.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `25c74a8`. Relevant production output:

- `dist/assets/training.worker-CB34B7zL.js`: 91.19 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-D99IjLhJ.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-Cug68S1n.js`: 13.42 kB, gzip 3.50 kB
- `dist/assets/RunHistoryPanel-CkxyfifQ.js`: 16.71 kB, gzip 5.15 kB
- `dist/assets/index-DMbCVJbM.js`: 369.02 kB, gzip 111.55 kB

Compared with the worker/API data-path guard build, the worker, CSS, lazy Code Export, engine, inspection, and run-history chunks remain effectively unchanged. The main app chunk changed from 368.46 kB to 369.02 kB, about 0.15%, below the 10% roadmap build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1104.1724 ms total for 100 iterations
- `predictGridInto`: 1098.0573 ms total for 100 iterations
- `predictGridWithNeurons`: 679.5977 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 596.8127 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.1150 ms
- Average `applyGradients` time (SGD): 1.4340 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The slice affects app-local scalar store setter guards only, and does not touch engine training throughput, worker cadence, frame-buffer transport, or heavy visualization payload generation.

## Wave 7 Public Runtime Config Guard

Date: 2026-05-15

Scope: app-local public training hook guard for future Multiclass Classification Mode. The public `useTraining` hook now validates the current playground config with strict shared import validation before creating/using the worker API, initializing public training, updating config, or sending stop commands for a running config sync. Direct worker multiclass tests remain available for bounded private worker coverage, but the public hook refuses hidden multiclass app state. This slice does not change URL/config format, public config shape, persistence/run-history schema, worker protocol, frame-buffer semantics, engine math, dependencies, deployment, or training behavior for valid scalar configs.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `a1dfd50`. Relevant production output:

- `dist/assets/training.worker-CB34B7zL.js`: 91.19 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-BM-7YnZG.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-zPTCC9od.js`: 13.42 kB, gzip 3.49 kB
- `dist/assets/RunHistoryPanel-B6R8j-di.js`: 16.71 kB, gzip 5.15 kB
- `dist/assets/index-BBaTuv22.js`: 369.27 kB, gzip 111.63 kB

Compared with the public store setter guard build, the worker, CSS, lazy Code Export, engine, inspection, and run-history chunks remain effectively unchanged. The main app chunk changed from 369.02 kB to 369.27 kB, about 0.07%, below the 10% roadmap build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1088.8360 ms total for 100 iterations
- `predictGridInto`: 1084.8675 ms total for 100 iterations
- `predictGridWithNeurons`: 669.1058 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 574.0163 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 3.9068 ms
- Average `applyGradients` time (SGD): 1.5140 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The slice affects app-local public runtime config validation only, and does not touch engine training throughput, worker cadence, frame-buffer transport, or heavy visualization payload generation.

## Wave 7 Worker/API Multiclass Data-Path Guard

Date: 2026-05-15

Scope: test-only worker/API guard for future Multiclass Classification Mode. Worker tests now assert that the approved direct multiclass worker config still sources train/test points from the current public binary dataset generator, not the hidden three-class helper. Existing engine tests continue to assert `generateThreeClassClusters` stays out of the package-level engine barrel export. This slice does not change production code, URL/config format, public config shape, persistence/run-history schema, worker protocol, frame-buffer semantics, engine math, dependencies, deployment, or training behavior.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `2534b4b`. Relevant production output:

- `dist/assets/training.worker-CB34B7zL.js`: 91.19 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-yT0kjw6V.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-BJ2CGnxX.js`: 13.42 kB, gzip 3.49 kB
- `dist/assets/RunHistoryPanel-BB7ZDyoe.js`: 16.71 kB, gzip 5.15 kB
- `dist/assets/index-CmVkBTE8.js`: 368.46 kB, gzip 111.44 kB

Compared with the previous test-only guard build, production output remained unchanged. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1077.5484 ms total for 100 iterations
- `predictGridInto`: 1059.6036 ms total for 100 iterations
- `predictGridWithNeurons`: 672.3662 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 573.6489 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 3.9601 ms
- Average `applyGradients` time (SGD): 1.3872 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The slice is test-only and does not touch engine training throughput, worker cadence, frame-buffer transport, or heavy visualization payload generation.

## Wave 7 Multiclass Boundary Transport

Date: 2026-05-15

Scope: bounded worker/protocol/frame-buffer transport for future Multiclass Classification Mode. The worker now computes demand-gated class-index and winning-confidence grids for the approved hidden 3-class softmax runtime, validates all-or-nothing snapshot payloads, stores arrays in the frame buffer behind a dedicated version counter, and keeps large arrays out of React state. This slice does not change public controls, default URL/config serialization, public config shape, persistence/run-history schema, dependencies, deployment, SAB/WebGPU transport, visible UI, or public dataset/preset exposure.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `2088bd3`. Relevant production output:

- `dist/assets/training.worker-CFzxpQkv.js`: 93.78 kB
- `dist/assets/index-DZR84vix.css`: 67.02 kB, gzip 11.64 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-Cd0KZ6ln.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-g6UxHFZZ.js`: 13.42 kB, gzip 3.50 kB
- `dist/assets/RunHistoryPanel-8Rht9xaF.js`: 18.06 kB, gzip 5.46 kB
- `dist/assets/index-WI4i1D_P.js`: 373.29 kB, gzip 112.50 kB

Compared with the hidden dataset contract build, the worker bundle increased from 91.69 kB to 93.78 kB, about 2.3%, and the main app bundle increased from 370.76 kB to 373.29 kB, about 0.7%. Both remain below the 10% roadmap build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1110.8832 ms total for 100 iterations
- `predictGridInto`: 1097.1431 ms total for 100 iterations
- `predictGridWithNeurons`: 684.6906 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 574.9719 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.0609 ms
- Average `applyGradients` time (SGD): 1.5421 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The existing perf suite does not directly time `predictMulticlassBoundaryInto`; runtime protection for this slice is covered by demand/cadence worker tests, protocol guards, frame-buffer version tests, and bounded inline transfer of class/confidence grids only when requested.

## Wave 7 Multiclass Decision Boundary Renderer

Date: 2026-05-15

Scope: read-only visualization renderer for future Multiclass Classification Mode. The `DecisionBoundary` component can now render bounded class/confidence grids already held in the frame buffer, adds class legend and confidence summary text, and preserves the scalar snapshot fallback. This slice does not change worker protocol, frame-buffer semantics, SharedArrayBuffer/WebGPU transport, URL/config serialization, persistence/run-history schema, public config shape, public controls, dependencies, deployment, engine math, or training behavior.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `9414244`. Relevant production output:

- `dist/assets/training.worker-CFzxpQkv.js`: 93.78 kB
- `dist/assets/index-BotKS8Aj.css`: 67.38 kB, gzip 11.68 kB
- `dist/assets/engine-BT-TiDoL.js`: 5.74 kB, gzip 2.07 kB
- `dist/assets/CodeExportPanel-CcEDnnW3.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/InspectionPanel-BPhjphFy.js`: 13.42 kB, gzip 3.50 kB
- `dist/assets/RunHistoryPanel-BgBp1Hoe.js`: 18.06 kB, gzip 5.46 kB
- `dist/assets/index-CeJs9-s2.js`: 376.41 kB, gzip 113.46 kB

Compared with the multiclass boundary transport build, the worker bundle stayed at 93.78 kB, CSS increased from 67.02 kB to 67.38 kB (about 0.5%), and the main app bundle increased from 373.29 kB to 376.41 kB (about 0.8%). These changes remain below the 10% roadmap build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1126.5585 ms total for 100 iterations
- `predictGridInto`: 1105.0490 ms total for 100 iterations
- `predictGridWithNeurons`: 693.9335 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 595.6118 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.3604 ms
- Average `applyGradients` time (SGD): 1.3771 ms

These values remain within the established benchmark range and below the roadmap warning thresholds. The renderer iterates only bounded frame-buffer grids already produced by the worker and does not introduce new runtime data collection or hot-path training work.

## Wave 7 Multiclass Confusion Readout

Date: 2026-05-15

Scope: UI-only derived 3-class confusion readout using existing frame-buffer weights/biases and test points. The initial readout implementation imported `Network` into `ConfusionMatrix`, which made the manual `engine` chunk grow from the prior 5.74 kB range to 38.72 kB. The follow-up lightweight readout path computes a transient local forward pass from frame-buffer parameters instead, keeping the engine chunk small while preserving the no-worker-metric scope. This slice does not change worker protocol, frame-buffer semantics, URL/config format, public config shape, persistence/run-history schema, dependencies, deployment, engine math, or training behavior.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed after `624eaa6`. Relevant production output:

- `dist/assets/training.worker-CFzxpQkv.js`: 93.78 kB
- `dist/assets/index-Q85g2pVY.css`: 67.62 kB, gzip 11.76 kB
- `dist/assets/engine-C35zPxVS.js`: 5.79 kB, gzip 2.09 kB
- `dist/assets/CodeExportPanel-gFVJSU72.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/react-j2mp3VYR.js`: 11.79 kB, gzip 4.21 kB
- `dist/assets/InspectionPanel-DEzbHzkU.js`: 13.42 kB, gzip 3.50 kB
- `dist/assets/RunHistoryPanel-Bg6IE3HT.js`: 18.06 kB, gzip 5.46 kB
- `dist/assets/index-CQmF4GHR.js`: 382.83 kB, gzip 115.18 kB

Compared with the multiclass decision-boundary renderer build, the worker bundle stayed at 93.78 kB, CSS increased from 67.38 kB to 67.62 kB (about 0.36%), the engine chunk changed from 5.74 kB to 5.79 kB (about 0.9%), and the main app chunk changed from 376.41 kB to 382.83 kB (about 1.7%). These changes remain below the 10% roadmap build-size warning threshold. The existing Vite large chunk warning remains.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1171.1160 ms total for 100 iterations
- `predictGridInto`: 1154.5979 ms total for 100 iterations
- `predictGridWithNeurons`: 737.1576 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 632.1893 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.4114 ms
- Average `applyGradients` time (SGD): 1.5683 ms

These values are slower than the immediately prior boundary-renderer perf sample but remain within the established noisy local benchmark range and below the roadmap warning thresholds. The readout runs only when the paused hidden 3-class confusion panel is rendered, uses bounded test points plus existing frame-buffer params, and does not touch engine training throughput, worker cadence, or large visualization transport.

## Wave 7 Public Multiclass Dataset Contract

Date: 2026-05-15

Scope: engine/shared contract slice for the first public multiclass dataset
path. The engine `DatasetType` and `generateDataset` now recognize the
approved bounded `three-class-clusters` generator, while shared URL/import and
experiment-memory validation only preserve that dataset when the full explicit
multiclass pairing is present. Dataset-only or partial multiclass URLs fall
back to scalar defaults. This slice does not add public UI controls, presets,
worker protocol fields, frame-buffer fields, persistence schema changes,
dependencies, deployment changes, engine training-behavior changes, or visible
runtime behavior.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed on 2026-05-15 with the existing Vite chunk-size warning.
Relevant production output:

- `dist/assets/training.worker-DjoNRhXG.js`: 94.62 kB
- `dist/assets/index-Q85g2pVY.css`: 67.62 kB, gzip 11.76 kB
- `dist/assets/engine-DIxxLc7J.js`: 6.15 kB, gzip 2.21 kB
- `dist/assets/CodeExportPanel-DPMr_X-H.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/react-j2mp3VYR.js`: 11.79 kB, gzip 4.21 kB
- `dist/assets/InspectionPanel-pQIxLNuU.js`: 13.42 kB, gzip 3.49 kB
- `dist/assets/RunHistoryPanel-BLkg-qRJ.js`: 18.06 kB, gzip 5.46 kB
- `dist/assets/index-et0WGhH2.js`: 383.56 kB, gzip 115.35 kB

Compared with the multiclass confusion readout build, the worker bundle changed
from 93.78 kB to 94.62 kB, about 0.9%, the engine chunk changed from 5.79 kB
to 6.15 kB, about 6.2%, and the main app chunk changed from 382.83 kB to
383.56 kB, about 0.2%. These changes remain below the 10% roadmap build-size
warning threshold.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1210.4854 ms total for 100 iterations
- `predictGridInto`: 1110.5422 ms total for 100 iterations
- `predictGridWithNeurons`: 700.5530 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 598.3708 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.1164 ms
- Average `applyGradients` time (SGD): 1.4686 ms

These values remain within the established noisy local benchmark range and
below roadmap warning thresholds. The slice only adds a deterministic dataset
route and shared validation guards; it does not add new hot-path training work,
worker cadence changes, or heavy visualization payloads.

## Wave 7 Public Multiclass Store URL Transition

Date: 2026-05-15

Scope: web store transition slice for the approved multiclass tuple. The store
can now preserve `three-class-clusters`, `classification`, `outputSize: 3`,
`softmax`, and `categoricalCrossEntropy` together through dataset selection,
synthetic approved preset application, URL sync, and URL load. This slice does
not add visible Data Panel or Preset Panel controls, public built-in presets,
worker protocol fields, frame-buffer fields, persistence schema changes,
dependencies, deployment changes, official 3x3 worker metrics, or new training
behavior.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed on 2026-05-15 with the existing Vite chunk-size warning.
Relevant production output:

- `dist/assets/training.worker-DjoNRhXG.js`: 94.62 kB
- `dist/assets/index-Q85g2pVY.css`: 67.62 kB, gzip 11.76 kB
- `dist/assets/engine-DIxxLc7J.js`: 6.15 kB, gzip 2.21 kB
- `dist/assets/CodeExportPanel-DXv-JB0n.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/react-j2mp3VYR.js`: 11.79 kB, gzip 4.21 kB
- `dist/assets/InspectionPanel-C2sFMeY3.js`: 13.42 kB, gzip 3.50 kB
- `dist/assets/RunHistoryPanel-mFOc1p8b.js`: 18.06 kB, gzip 5.46 kB
- `dist/assets/index-D78bAaVM.js`: 383.78 kB, gzip 115.41 kB

Compared with the public multiclass dataset contract build, the worker and
engine chunks stayed the same size, while the main app chunk changed from
383.56 kB to 383.78 kB, about 0.06%. These changes remain below the 10%
roadmap build-size warning threshold.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1063.7482 ms total for 100 iterations
- `predictGridInto`: 1063.5782 ms total for 100 iterations
- `predictGridWithNeurons`: 671.3207 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 574.3940 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 3.9252 ms
- Average `applyGradients` time (SGD): 1.4245 ms

These values remain within the established noisy local benchmark range and
below roadmap warning thresholds. The slice only changes web store normalization
and URL opt-in calls; it does not add runtime hot-path work, worker cadence
changes, or heavy visualization payloads.

## Wave 7 Public Multiclass Runtime Acceptance

Date: 2026-05-15

Scope: runtime acceptance slice for the exact approved multiclass tuple. The
training hook now validates public runtime configs with the existing shared
multiclass opt-in guard, and the worker validates only
`three-class-clusters`, `classification`, `outputSize: 3`, `softmax`, and
`categoricalCrossEntropy` before mutating worker state. The worker now sources
real three-class samples for that tuple and keeps malformed tuples rejected.
Live arena remains scalar-only. This slice does not add visible controls,
public built-in presets, Config Panel JSON import/export, run-history
save/restore, worker protocol fields, frame-buffer fields, persistence schema
changes, dependencies, deployment changes, official 3x3 worker metrics, or raw
probability-grid transport.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed on 2026-05-15 with the existing Vite chunk-size warning.
Relevant production output:

- `dist/assets/training.worker-rMnDb0A3.js`: 94.59 kB
- `dist/assets/index-Q85g2pVY.css`: 67.62 kB, gzip 11.76 kB
- `dist/assets/engine-DIxxLc7J.js`: 6.15 kB, gzip 2.21 kB
- `dist/assets/CodeExportPanel-D0HffnMV.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/react-j2mp3VYR.js`: 11.79 kB, gzip 4.21 kB
- `dist/assets/InspectionPanel-SiKCqEa6.js`: 13.42 kB, gzip 3.49 kB
- `dist/assets/RunHistoryPanel-00aKrwOR.js`: 18.06 kB, gzip 5.46 kB
- `dist/assets/index-BWRyg_P5.js`: 383.80 kB, gzip 115.41 kB

Compared with the public multiclass store URL transition build, the worker
chunk changed from 94.62 kB to 94.59 kB, the engine chunk stayed at 6.15 kB,
and the main app chunk changed from 383.78 kB to 383.80 kB, about 0.01%.
These changes remain below the 10% roadmap build-size warning threshold.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1096.4730 ms total for 100 iterations
- `predictGridInto`: 1087.3352 ms total for 100 iterations
- `predictGridWithNeurons`: 686.9347 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 587.7726 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.0566 ms
- Average `applyGradients` time (SGD): 1.4411 ms

These values remain within the established noisy local benchmark range and
below roadmap warning thresholds. The slice changes validation and dataset
acceptance gates only; multiclass boundary arrays remain bounded and
demand/cadence-gated in the existing worker path.

## Wave 7 Public Multiclass Config JSON

Date: 2026-05-15

Scope: Config Panel JSON import/export slice for the exact approved multiclass
tuple. The panel now uses the existing shared multiclass opt-in validator for
JSON export and import, preserving only `three-class-clusters`,
`classification`, `outputSize: 3`, `softmax`, and
`categoricalCrossEntropy`. Malformed partial multiclass configs remain
rejected. This slice does not add visible dataset controls, public built-in
presets, worker protocol fields, frame-buffer fields, run-history schema
changes, persistence migrations, dependencies, deployment changes, official 3x3
worker metrics, or raw probability-grid transport.

Commands:

- `pnpm build`
- `pnpm test:perf`

`pnpm build` passed on 2026-05-15 with the existing Vite chunk-size warning.
Relevant production output:

- `dist/assets/training.worker-rMnDb0A3.js`: 94.59 kB
- `dist/assets/index-Q85g2pVY.css`: 67.62 kB, gzip 11.76 kB
- `dist/assets/engine-DIxxLc7J.js`: 6.15 kB, gzip 2.21 kB
- `dist/assets/CodeExportPanel-Cl8FFv6C.js`: 7.69 kB, gzip 3.07 kB
- `dist/assets/react-j2mp3VYR.js`: 11.79 kB, gzip 4.21 kB
- `dist/assets/InspectionPanel-i-N0Eknv.js`: 13.42 kB, gzip 3.50 kB
- `dist/assets/RunHistoryPanel-BALK3S7M.js`: 18.06 kB, gzip 5.46 kB
- `dist/assets/index-CASDv05U.js`: 383.84 kB, gzip 115.43 kB

Compared with the public multiclass runtime acceptance build, the worker and
engine chunks stayed the same size, while the main app chunk changed from
383.80 kB to 383.84 kB, about 0.01%. These changes remain below the 10%
roadmap build-size warning threshold.

`pnpm test:perf` passed with 2 benchmark files and 4 benchmark tests.

Observed benchmark output:

- `predictGrid`: 1094.6214 ms total for 100 iterations
- `predictGridInto`: 1084.8148 ms total for 100 iterations
- `predictGridWithNeurons`: 693.1046 ms total for 50 iterations
- `predictGridWithNeuronsInto`: 593.2782 ms total for 50 iterations
- Average `applyGradients` time (Adam, L2, Clip): 4.0385 ms
- Average `applyGradients` time (SGD): 1.4134 ms

These values remain within the established noisy local benchmark range and
below roadmap warning thresholds. The slice only changes Config Panel validation
and tests; it does not add runtime hot-path work, worker cadence changes, or
heavy visualization payloads.
