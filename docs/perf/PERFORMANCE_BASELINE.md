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
