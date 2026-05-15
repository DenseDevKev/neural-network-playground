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
