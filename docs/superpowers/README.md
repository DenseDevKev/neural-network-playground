# NN.FORGE plan index and reconciliation

Reconciled September 6, 2026. `main` is the product authority; code on an execution branch is not a released product. Start with the [current roadmap](plans/2026-09-05-nn-forge-release-roadmap.md) and [recorded qualification](verification/2026-09-05-release-roadmap.md).

## Historical requirements, not an automatic backlog

| Source | Classification | Current disposition |
|---|---|---|
| [May 22 Build/Run design](specs/2026-05-22-nn-forge-build-run-instrument-design.md) | Implemented and verified for the current automated contracts; some earlier presentation rules superseded | The application composes BuildRunShell. Profile-driven visibility supersedes universal specialist-panel visibility. Saved recipe application is not parameter restoration. |
| [July 11 scientific-trust design](specs/2026-07-11-nn-forge-scientific-trust-design.md) and [implementation plan](plans/2026-07-11-nn-forge-scientific-trust.md) | Implemented; dated hardware observations remain hardware-specific | The [scientific-trust verification record](verification/2026-07-11-scientific-trust.md) maps all twelve tasks to committed implementation slices. The original engine criterion was same-development-machine median <=120% of baseline; release verification now enforces that relationship explicitly on paired same-runner samples while retaining the frozen historical constants. |
| [July 16 Precision Lab design](specs/2026-07-16-precision-lab-production-integration-design.md) and [July 17 implementation plan](plans/2026-07-17-precision-lab-production-integration.md) | Still applicable as the next gated presentation design; not the current production shell | Existing scientific adapters are prerequisites. Task 1's gzip guard is implemented. The replacement shell, pinned boundary rail, selection deck, production-backed compact evidence, and final responsive integration remain future work. Current user-facing terminology is `Workspace profile`; do not blindly restore the historical `Audience mode` label. |
| Historical screenshot, visual-comparison, and real-device criteria | Acceptance evidence only when the relevant current artifact/device/reference exists | A checked-in screenshot or passing unit test is not proof of current physical-device or visual-reference acceptance. The Precision Lab prototype referenced by the old plan is not present in GitHub, so no pixel-parity claim may be made against it. |
| Old remote branches, dirty-worktree instructions, and superseded audits | Archival only unless a specific unique reference is needed | Never merge historical product code wholesale or delete branches as documentation cleanup. |
| Previous Mac's local-only plans and prototype tree | Not available as repository evidence | Do not claim these artifacts exist on GitHub or block unrelated release work on them. |

The original design and plan documents retain their original text/checklists for historical traceability. This index records current disposition instead of rewriting every old checkbox.

## Rules for continuation

Read current code and exact run logs before declaring work implemented, verified, failed, or released. Separate a product change from its regression-test commit, workflow/tooling changes, and later documentation.

Preserve the V2 recipe, model, training step, checkpoints, saved evidence, export selection, and shared hash across navigation/profile/disclosure changes. Keep batch/EMA and same-revision full-split evaluations separate, with age and drift. Saved runs contain evidence and a recipe, not trained parameters. Checkpoints do not guarantee an identical future shuffle trajectory.

`pnpm test:perf` retains historical absolute development-machine calibration. Release performance is the documented same-runner relative contract: five baseline/candidate samples on one host, engine candidate medians <=120% of baseline medians, and fixed 250/500 ms worker budgets. Do not raise historical constants merely because a different machine is slower, and do not call a candidate green without the paired comparator.

Advancement of `main` requires exact candidate qualification. Public Pages publication remains an explicit release action and must be followed by live browser verification against GitHub's actual deployed `page_url`.
