# NN.FORGE plan index and reconciliation

Reconciled September 5, 2026. `main` is the product authority; code on an
execution branch is not a released product. Start with the
[current roadmap](plans/2026-09-05-nn-forge-release-roadmap.md) and
[recorded qualification](verification/2026-09-05-release-roadmap.md).

## Historical requirements, not an automatic backlog

| Source | Classification | Current disposition |
|---|---|---|
| [May 22 Build/Run design](specs/2026-05-22-nn-forge-build-run-instrument-design.md) | Implemented and verified for the current automated contracts; some earlier presentation rules superseded | The application composes BuildRunShell. Profile-driven visibility supersedes universal specialist-panel visibility. Saved recipe application is not parameter restoration. |
| [July 11 scientific-trust design](specs/2026-07-11-nn-forge-scientific-trust-design.md) and [implementation plan](plans/2026-07-11-nn-forge-scientific-trust.md) | Implemented; dated performance/manual observations need their own fresh verification | The [scientific-trust verification record](verification/2026-07-11-scientific-trust.md) maps all twelve tasks to committed implementation slices. Unchecked historical boxes do not reopen the entire program. |
| [July 16 Precision Lab design](specs/2026-07-16-precision-lab-production-integration-design.md) and [July 17 implementation plan](plans/2026-07-17-precision-lab-production-integration.md) | Still applicable as a later, gated presentation design; not the shipped shell | Existing scientific adapters are implemented prerequisites. The replacement shell, pinned boundary rail, selection deck, and responsive integration are not thereby implemented. Task 1's gzip guard has been implemented independently on the release branch; tasks 2–14 have not. |
| Historical screenshot, visual-comparison, and real-device criteria | Implemented features needing fresh verification where an applicable check has not been run | A checked-in screenshot or passing unit test is not proof of current physical-device or visual-reference acceptance. |
| Old remote branches, dirty-worktree instructions, and superseded audits | Archival only unless a specific unique reference is needed | Never merge historical product code wholesale or delete branches as documentation cleanup. |
| Previous Mac's local-only plans and prototype tree | Not inspected; outside release evidence | Do not claim these artifacts exist on GitHub or block unrelated release work on them. |

This dated index supplies reconciliation in one place. The original five design
and plan documents retain their original text and checkboxes; no blanket
completion edits have been made to them.

## Rules for the next engineer

Read current code and the exact run logs before declaring work implemented,
verified, failed, or released. Separate the production commit from its test-harness
commit and from later documentation commits. A successful main CI run does not
prove that optional performance or fault-injection commands ran.

Preserve the V2 recipe, model, training step, checkpoints, saved evidence, export
selection, and shared hash across navigation/profile/disclosure changes. Keep
batch/EMA and same-revision full-split evaluations separate, with age and drift.
Saved runs contain evidence and a recipe, not trained parameters. Checkpoints do
not guarantee an identical future shuffle trajectory.

Public Pages publication and advancement of main remain owner-gated. A failed
performance baseline is evidence to investigate, not permission to call the
candidate's performance gate green.
