# NN.FORGE release-roadmap verification record

**Recorded:** September 6, 2026  
**Repository:** `DenseDevKev/neural-network-playground` (private source repository)  
**Authoritative intake `main`:** `ae09b9863ae90f8fb2f62545834fcc138755ba9a`  
**Execution branch:** `codex/nn-forge-release-roadmap`  
**Latest fully qualified code SHA before this documentation-only consolidation:** `fbae4b98a71b86bdf704a3d8896ccc8f0b92b4a0`  
**Qualification run:** `34001183202`

This is a branch qualification record, not a claim that the code has been merged or deployed. At the time of this record, `main` has not been advanced and no successful live Pages receipt has been recorded.

## Result

`fbae4b98a71b86bdf704a3d8896ccc8f0b92b4a0` passed every release-verification job:

| Job | Result |
|---|---|
| source-evidence | Pass |
| correctness | Pass |
| browsers (preview) | Pass |
| browsers (subpath) | Pass |
| browsers (recovery) | Pass |
| performance | Pass |

The release performance pass uses the original same-development-machine regression semantics reconstructed from the July 11 Scientific Trust baseline: five baseline and five candidate runs on one runner, candidate engine medians <= 120% of same-runner baseline medians, plus independent fixed worker budgets of 250 ms forced paired evaluation and 500 ms save capture. The historical absolute engine constants were not modified.

## Correctness

Correctness job `101400345182` ran on Ubuntu 24.04 with Node 20.20.2 and pnpm 9.15.9.

Observed results:

- infrastructure helper tests: **88 passed**, zero failed/skipped;
- lint: pass;
- typecheck: pass across engine/shared/web and web test TypeScript projects;
- engine tests: **497 passed**;
- shared tests: **334 passed**;
- web tests: **1,043 passed**;
- total package tests: **1,874 passed**;
- production build: pass;
- JavaScript bundle guard: pass;
- tracked-source drift check after build: pass.

### JavaScript gzip contract

| Dimension | Actual gzip bytes | Fixed maximum | Result |
|---|---:|---:|---|
| Application entry | 146,940 | 152,245 | Pass |
| InspectionPanel | 5,414 | 7,373 | Pass |
| All JavaScript including worker | 226,417 | 234,161 | Pass |

The generic Vite 200 kB chunk warning remains visible and non-fatal; the reviewed release contract is the executable gzip check above. CSS and fonts are outside those JavaScript dimensions.

Build artifact:

- ID: `9979590065`
- name: `release-build-fbae4b98a71b86bdf704a3d8896ccc8f0b92b4a0`
- ZIP digest: `sha256:d518c4ef2954fe65c9a87a03071ff8e63bf55aa195f99b150620f79acdebb5a2`
- size: 1,371,728 bytes

## Browser qualification

The Playwright reports were parsed from the retained report artifacts rather than inferred from job status.

### Isolated preview

- total scenarios: 48
- expected/passed: **42**
- skipped: **6**
- unexpected: **0**
- flaky: **0**
- report `ok`: true

The six skips are intentional: four external-hosting-only contracts plus two fault-enabled cases excluded from a normal build.

Artifact:

- ID: `9979602592`
- digest: `sha256:35aa9f165094dcb0bccfdf1f3c40fc1a49ca800743994b955950e1eac8cb5832`

### Non-isolated project-subpath fixture

- total scenarios: 48
- expected/passed: **46**
- skipped: **2**
- unexpected: **0**
- flaky: **0**
- report `ok`: true

Only the two fault-enabled cases are intentionally skipped. The hosting contracts execute against `http://127.0.0.1:4174/neural-network-playground/` and verify actual project-path worker/lazy resources, non-isolated operation, evidence stepping, and V2 URL reload semantics.

Artifact:

- ID: `9979613493`
- digest: `sha256:ad7784901199346a96351d1fc686cd2e3cd5acfeeaa216fe4e723a79ff3f8119`

### Fault-enabled recovery

Chromium and WebKit both passed the original recovery scenario. The workflow then preserved the fault report, rebuilt without fault injection, and passed the fault-disabled normal-build check.

Artifact:

- ID: `9979568486`
- digest: `sha256:0b7fd2807e96853c109c6c4ff945d85f292a4ef63e70189f232d595839766c12`

The prior WebKit failure was resolved by same-origin pinned font delivery. The strict console/page/resource checks were not suppressed.

## Performance policy provenance

The July 11 Scientific Trust baseline described the engine release criterion as a comparison on the **same isolated development machine**: establish five baseline runs and require the post-change median to remain within **120% of the baseline median**. The later frozen absolute engine constants are that historical development-machine calibration; they are useful locally but are not hardware-independent measurements.

The release workflow now implements the original comparison semantics explicitly:

1. candidate and exact intake baseline are checked out in the same job;
2. five complete performance runs are collected for each;
3. order alternates baseline-first and candidate-first;
4. exact host metadata, SHAs, exits, and raw logs are retained;
5. `scripts/compare-performance-reference.mjs` parses all ten logs and fails closed for missing/ambiguous evidence;
6. engine medians must be <= `baseline median * 1.20`;
7. both baseline and candidate worker medians must remain <= 250/500 ms fixed scientific-trust budgets.

No engine or shared implementation source was changed as part of this policy correction, and the old absolute constants were not raised.

### TDD evidence for the comparator

- `f6aa6937e1ada6e8d50880b86a346f11f519c7bf`: test-only RED. The existing 77 helper tests passed and 11 new comparator tests failed because the implementation module did not exist.
- `27e0b2448aec60bcb22afcf0213f4debe9032478`: implementation; 10/11 comparator tests passed. The one failure was a test fixture that omitted the mandatory web-performance PASS summary.
- `5d637eb176acc6598dfca727cdac7c3e3dd8e62a`: corrected fixture isolation; helper suite passed **88/88**.
- `11facd9790a39630afa0d81379b92fa1e11f21bb`: release workflow uses the paired comparator.
- `784651123bc71aaffe580e988732dfc701d1dda9`: dedicated reference workflow uses the same comparator.
- `fbae4b98a71b86bdf704a3d8896ccc8f0b92b4a0`: lint-only CLI stream correction; comparator policy unchanged.

A dedicated reference run `34001077012` passed before the final code qualification, proving the comparator against a real ten-run collection rather than only synthetic parser tests.

## Exact `fbae4b9` performance result

Performance job `101400345218` ran on macOS 15.7.9, `macos-15-arm64`, Node 20.20.2, pnpm 9.15.9. Exact baseline: `ae09b9863ae90f8fb2f62545834fcc138755ba9a`.

Five-run medians reported by the comparator:

| Measurement | Baseline median | Candidate median | Change | Allowed candidate | Result |
|---|---:|---:|---:|---:|---|
| predictGrid | 16.6419 ms | 18.6696 ms | +12.1843% | 19.97028 ms | Pass |
| predictGridInto | 18.9647 ms | 19.5467 ms | +3.0689% | 22.75764 ms | Pass |
| predictGridWithNeurons | 22.0760 ms | 21.2197 ms | -3.8789% | 26.4912 ms | Pass |
| predictGridWithNeuronsInto | 21.9224 ms | 19.5882 ms | -10.6476% | 26.30688 ms | Pass |
| Adam + L2 + clipping | 5.4373 ms | 5.2640 ms | -3.1872% | 6.52476 ms | Pass |
| SGD zero-gradient adapter | 1.2648 ms | 1.3682 ms | +8.1752% | 1.51776 ms | Pass |
| Forced paired evaluation | 9.0782 ms | 8.5604 ms | -5.7038% | 250 ms | Pass |
| Save capture | 12.0391 ms | 10.8588 ms | -9.8039% | 500 ms | Pass |

The raw individual engine runs still report failures against the historical frozen development-machine constants on this VM. That is expected evidence and remains in the logs. The release decision is the paired same-runner comparison above, which is the original documented regression rule. Individual failures were not hidden or rewritten.

Performance artifact:

- ID: `9979614460`
- name: `release-performance-fbae4b98a71b86bdf704a3d8896ccc8f0b92b4a0`
- digest: `sha256:9c9d39ee5044c3c8e6ba115c7d1fe54fe9af5ced7a1cb01d907342ff8461dd51`

## Source evidence

Source artifact:

- ID: `9979539505`
- digest: `sha256:48a1a94c0b7f2a73c6ec03da76280ce734e81b917c47dd706e7e65565ffe99e2`

The workflow records the exact source SHA and a tracked-source archive. Qualification jobs use read-only repository permission and do not deploy.

## Branch audit

Before this verification record was consolidated, the release branch diff was reviewed against intake `main`.

Observed boundaries:

- no engine implementation changes;
- no shared-package schema/protocol/persistence implementation changes;
- no prototype-state import;
- no historical branch merge;
- product changes limited to the reproduced skip-link state fix and same-origin font delivery;
- remaining changes are regression tests, release/browser infrastructure, comparator/bundle tooling, pinned font dependency/license data, and documentation.

## Remaining release steps

This documentation consolidation itself creates a new branch SHA, so that SHA must run the same release-verification workflow before merge. After it is green:

1. re-read `main` and branch refs;
2. review the exact final diff;
3. open a PR to `main`;
4. merge the qualified branch;
5. require `main` CI to pass;
6. let the existing deployment workflow rebuild/deploy the exact tested `main` SHA;
7. record the actual `page_url` and run external Chromium/WebKit verification.

Precision Lab implementation begins only from the accepted release baseline. Its historical prototype is unavailable in GitHub, so no unseen-reference pixel-parity claim is permitted.
