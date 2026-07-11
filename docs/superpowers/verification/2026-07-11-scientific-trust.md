# Scientific Trust Verification Record

## Baseline

- Captured: `2026-07-11T02:14:55-04:00`
- Branch: `codex/scientific-trust-v2`
- Base snapshot: `6b83026925b68a5be42ed9e2d600cb5f210aed59`
- Environment: same isolated worktree and development machine required for the post-change comparison.

### Repository gates

| Gate | Result | Evidence |
|---|---|---|
| `pnpm test` | PASS | 74 files, 958 tests: engine 320, shared 125, web 513 |
| `pnpm lint` | PASS | ESLint exit 0, no findings |
| `pnpm build` | PASS | TypeScript and Vite exit 0; existing `>200 kB` chunk warning recorded |
| `pnpm test:perf` | PASS | 2 benchmark files, 4 tests on each of five runs |
| `git diff --check` | PASS | exit 0, no output |

### Five-run raw performance measurements

Times are milliseconds per operation. Grid totals printed by the benchmark were divided by their declared iteration counts: 100 for standard grids and 50 for neuron grids.

| Run | predictGrid | predictGridInto | Adam + L2 + clip | SGD | predictGridWithNeurons | predictGridWithNeuronsInto |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.375342 | 9.288689 | 3.9737 | 1.3456 | 11.924420 | 10.273716 |
| 2 | 9.322705 | 9.454270 | 4.0115 | 1.3694 | 11.977600 | 10.199000 |
| 3 | 9.504869 | 9.277257 | 4.0267 | 1.3895 | 11.986086 | 10.190458 |
| 4 | 9.327635 | 9.280918 | 3.9383 | 1.3745 | 11.963984 | 10.205682 |
| 5 | 9.535484 | 9.405236 | 3.9302 | 1.4301 | 12.004992 | 10.180192 |
| **Median** | **9.375342** | **9.288689** | **3.9737** | **1.3745** | **11.977600** | **10.199000** |

The post-change gate is at most 120% of each median above, plus the new absolute gates of at most 250 ms for a maximum-recipe paired evaluation and at most 500 ms for a maximum-recipe save capture.

## Implementation slices

Pending.

## Post-change repository gates

Pending.

## Browser scenarios

Pending.

## Limitations

Pending final verification.
