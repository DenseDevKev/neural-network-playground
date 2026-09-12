# Precision Lab completion — September 12, 2026

This release closes the outstanding Precision Lab integration, P7 presentation migration and saved-run concurrency defect. Final source identity, remote checks, merge and deployment are recorded on [PR #36](https://github.com/DenseDevKev/neural-network-playground/pull/36), because those receipts occur after the source commit exists.

## Branch reconciliation

| Branch/source | Disposition |
|---|---|
| `main` at `98f29b86` | Accepted release baseline used as merge base |
| `codex/nn-forge-release-roadmap` at `2cd5b896` | Already merged through PR #35 |
| `codex/nn-forge-precision-lab` at `6cf88bde` | Complete product candidate; latest intake qualification `34639261446` passed all six jobs |
| `codex/nn-forge-precision-apply` at `93466490` | Transport only; continuation, workflow, browser and plan patches match `2083187`, `3b56795`, `9501428`, `3c8e48a` exactly; no unique product change to import |
| Local browser-smoke, responsive-polish and refactor branch heads | Ancestors of accepted main |
| Local cockpit snapshot `f9da752` | Product fixes survive or are superseded by V2 scientific-state handling; unique audit report/screenshots remain archival |
| Stale local tracking ref `integration/neural-playground` at `918f235` | Removed upstream; old import coverage is superseded by V2 round-trip/strict-schema tests; no production delta beyond an obsolete whitespace change |

No archived snapshot, transport blob or uncommitted original-checkout file was imported. Historical refs and the original working files remain intact.

## Completion changes

- Web Locks serialize saved-run hydration and mutations across tabs. Each mutation reads the latest persisted envelope while holding the lock. No schema, scientific identity, worker-capture, capacity or retry-artifact contract changed.
- Rejected cleanup follows selected bytes after index changes. Whole-file deletion verifies the originally selected incompatible/legacy bytes. Mutations cannot silently promote rejected duplicates into accepted runs.
- PrecisionLabContent holds the live content exports after dead shell adapters/components and unused CSS are retired. Meaningful preset, configuration reset, boundary and ownership tests target the current components.
- Vite/Vitest and compatible vulnerable transitive development packages are patched. Audit reports zero critical/high/low findings, with two moderate entries for one unused Vitest development-server route explicitly documented in `BUGS-TO-REVIEW.md`.
- The complete six-job qualification now also runs on pushed main commits.

## Local evidence before publication

- Original independent-store regressions failed, then passed after the fix. Follow-up guards also had failing regression evidence before implementation.
- Original controlled two-tab stale-save scenario failed in Chromium and WebKit, then passed. A separate browser scenario proves one held origin lock and one waiting writer during asynchronous envelope validation, followed by both successful records surviving.
- Store and related-hook tests: 40 passing. Existing desktop/mobile exact-artifact retry journeys pass in both browsers.
- Complete package tests: 497 engine + 334 shared + 1,139 web = 1,970 passing. Obsolete component-only tests were retired; applicable behavior moved to current consumers.
- Infrastructure tests: 156 passing. Lint, typecheck and production build pass.
- Full Chromium/WebKit production preview: 86 passed, 6 intentional hosting/fault-mode skips, zero failures or flaky results.
- Gzip: entry 151,570 / 152,245 bytes; InspectionPanel 5,431 / 7,373; total JavaScript 234,007 / 234,161. Caps are unchanged.

These local receipts do not substitute for the final upstream source SHA. The final preview, project-subpath, recovery, five-pair performance, main CI and live deployment results are linked from PR #36 after they execute. No finite suite establishes absence of every possible future defect.
