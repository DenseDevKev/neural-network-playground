# Neural Network Playground 2.0 Superpowers Execution Tracker

This artifact is the persistent source of truth for the Superpowers-aware
roadmap execution. Update it whenever a task, review, verification command, or
blocker changes.

## Current Status

- [x] Roadmap corrected into a Superpowers execution plan - [REQ-SUPERPOWERS](#req-superpowers), [REQ-AGENTS](#req-agents)
- [x] P0A pure stop-condition evaluator implemented - [REQ-P0A](#req-p0a)
- [x] P0A focused tests passed - [REQ-P0A](#req-p0a), [REQ-TDD](#req-tdd)
- [x] P0A reviewed by subagent - [REQ-AGENTS](#req-agents)
- [x] P0A staged/committed or otherwise integrated - [REQ-VERIFY](#req-verify)
- [x] P0B runtime pause plumbing implemented - [REQ-P0B](#req-p0b)
- [x] P0B spec compliance review complete - [REQ-AGENTS](#req-agents), [REQ-P0B](#req-p0b)
- [x] P0B code quality review complete - [REQ-AGENTS](#req-agents), [REQ-P0B](#req-p0b)
- [x] P0B verification complete - [REQ-VERIFY](#req-verify)
- [x] Explicit approval received to start P0C - [REQ-NO-P0C-WITHOUT-APPROVAL](#req-no-p0c-without-approval)
- [x] P0C deterministic explanations implemented - [REQ-P0C](#req-p0c)
- [x] P0C minimal UI implemented - [REQ-P0C](#req-p0c)
- [x] P0C focused and full verification complete - [REQ-P0C](#req-p0c), [REQ-VERIFY](#req-verify)
- [x] Explicit approval received to start P1 - [REQ-AGENTS](#req-agents), [REQ-P1](#req-p1)
- [x] P1 lesson schema/registry implemented - [REQ-P1](#req-p1)
- [x] P1 XOR migration and conservative 3-lesson batch implemented - [REQ-P1](#req-p1)
- [x] P1 final verification complete - [REQ-VERIFY](#req-verify)
- [x] P1 final reviews complete - [REQ-AGENTS](#req-agents)
- [x] P2-P4 read-only recon complete - [REQ-AGENTS](#req-agents), [REQ-P2](#req-p2), [REQ-P3](#req-p3), [REQ-P4](#req-p4)
- [x] P2 experiment memory implemented - [REQ-P2](#req-p2)
- [x] P3 data understanding implemented - [REQ-P3](#req-p3)
- [x] P4 on-demand prediction trace implemented - [REQ-P4](#req-p4)
- [x] P2-P4 final reviews complete - [REQ-AGENTS](#req-agents)
- [x] P2-P4 final verification complete - [REQ-VERIFY](#req-verify)

## Current Gate

P2-P4 are complete. Do not expand P2 into exact training continuation, P3 into
dataset schema expansion/custom brush, or P4 into streamed interpretability
payloads without a separate approved plan.

## Prompt Requirement Index

<a id="req-superpowers"></a>
### REQ-SUPERPOWERS

Use the Superpowers methodology explicitly:

- `superpowers:brainstorming` for product scope control.
- `superpowers:writing-plans` for executable implementation plans.
- `superpowers:dispatching-parallel-agents` for read-only recon and disjoint implementation slices only.
- `superpowers:subagent-driven-development` as the preferred execution model.
- `superpowers:test-driven-development` for behavior changes.
- `superpowers:using-git-worktrees` when local worktrees are appropriate; use isolated branches/tasks otherwise.
- `superpowers:verification-before-completion` before claiming any task, phase, test, build, or review complete.

<a id="req-agents"></a>
### REQ-AGENTS

Use coordinator, read-only recon agents, implementation workers, reviewer
agents, and a merge steward. Parallel agents are allowed only when ownership,
state, contracts, and tests are disjoint. Implementation for P0B is serial-only.

<a id="req-file-ownership"></a>
### REQ-FILE-OWNERSHIP

Maintain explicit file ownership and collision rules. Shared protocol,
serialization/config, worker runtime, Zustand stores, visualization demand, and
major layout files are locked unless assigned. Workers must stop rather than
edit outside their allowlist.

<a id="req-p0a"></a>
### REQ-P0A

Pure stop-condition foundation:

- Define `PauseReason`.
- Define `StopCondition`.
- Add a pure stop-condition evaluator.
- Unit-test target loss, target accuracy, unavailable accuracy, plateau,
  divergence, max steps, NaN, Infinity, deterministic priority, and reset state.
- No UI, protocol, store, or serialization changes unless explicitly assigned.

<a id="req-p0b"></a>
### REQ-P0B

Runtime plumbing:

- Connect the evaluator to the training runtime.
- Add pause reason propagation.
- Update store/runtime state as needed.
- Manual pause uses `pauseReason: 'manual'`.
- Automatic stop-condition pauses use matching non-manual reasons.
- Reset clears pause reason and stop-condition tracking.
- Resume after automatic pause requires explicit user resume.
- Resume does not clear historical metrics unless reset is triggered.
- Divergence catches `NaN` and `Infinity` immediately.
- `targetAccuracy` is ignored when accuracy is unavailable.
- If multiple stop conditions trigger, use deterministic priority.
- In P0B, default runtime behavior is non-finite divergence only. Do not add new
  user controls, share URL fields, persistence, or serialization.

<a id="req-p0c"></a>
### REQ-P0C

Explanation rules and minimal UI:

- Add deterministic explanation rules.
- Add a minimal "Why did this happen?" surface.
- Use existing metrics/diagnostics only.
- Do not add lessons, custom datasets, or expensive worker data.

<a id="req-p1"></a>
### REQ-P1

Lesson system:

- Keep lesson schema/registry web-local under `apps/web/src/lessons/`.
- Define `LessonDefinition`, `LessonStep`, and `LessonTarget`.
- Migrate the existing XOR guided lesson into the registry.
- Add only the conservative 3-lesson seed set: XOR, Single Neuron, Regression Plane.
- Do not add Feature Engineering in P1.
- Do not change shared exports, serialization, worker protocol, store shape, presets, or app-wide layout model.

<a id="req-p2"></a>
### REQ-P2

Experiment memory:

- Add local-first run history with typed, versioned records.
- Use actual exported project types instead of persistent `unknown` payloads.
- Persist bounded local history and recover corrupt/future-version storage as empty history.
- Capture current config, summary metrics, bounded loss history, and serialized network parameters.
- Restore means restore saved config and show saved metrics/history/report; it does not claim exact training continuation.
- Export a report from saved run data.

<a id="req-p3"></a>
### REQ-P3

Data understanding:

- Make train/test split visible using existing runtime train/test arrays.
- Reshuffle means changing existing `data.seed` through the existing config flow.
- Do not add split-only seeds, split membership persistence, custom dataset brush, or dataset-specific schema expansion in this slice.
- Keep the visualization textual/accessibility-safe and not color-only.

<a id="req-p4"></a>
### REQ-P4

Interpretability:

- Start with one safe on-demand prediction trace path.
- Add engine pure-copy trace support.
- Add a Comlink one-shot worker RPC only if it crosses worker/UI.
- Add minimal inspection UI for a single sample trace.
- Do not add streamed histograms, gradient payloads, frame-buffer domains, or SAB changes in this slice.

<a id="req-tdd"></a>
### REQ-TDD

Every behavior change must follow TDD:

1. Write the failing test first.
2. Run the targeted command and confirm the expected failure.
3. Implement the minimal code.
4. Re-run the targeted command and confirm pass.
5. Add regression coverage.
6. Run package-level tests.
7. Run coordinator final verification.

<a id="req-verify"></a>
### REQ-VERIFY

Before phase completion, run the actual discovered verification commands:

- `pnpm --filter @nn-playground/shared test`
- `pnpm --filter @nn-playground/web test`
- `pnpm lint`
- `pnpm build`

Use targeted package/file commands during TDD.

<a id="req-no-p0c-without-approval"></a>
### REQ-NO-P0C-WITHOUT-APPROVAL

Do not start P0C until P0B has tests, review, verification, and explicit
approval to proceed.

## Linked Checklist

### P0A: Pure Stop-Condition Foundation

- [x] Define `PauseReason` - [REQ-P0A](#req-p0a)
- [x] Define `StopCondition` - [REQ-P0A](#req-p0a)
- [x] Add pure evaluator - [REQ-P0A](#req-p0a)
- [x] Add deterministic pause priority - [REQ-P0A](#req-p0a)
- [x] Test target loss - [REQ-P0A](#req-p0a), [REQ-TDD](#req-tdd)
- [x] Test target accuracy - [REQ-P0A](#req-p0a), [REQ-TDD](#req-tdd)
- [x] Test unavailable accuracy - [REQ-P0A](#req-p0a), [REQ-TDD](#req-tdd)
- [x] Test plateau patience - [REQ-P0A](#req-p0a), [REQ-TDD](#req-tdd)
- [x] Test divergence with `NaN` - [REQ-P0A](#req-p0a), [REQ-TDD](#req-tdd)
- [x] Test divergence with `Infinity` - [REQ-P0A](#req-p0a), [REQ-TDD](#req-tdd)
- [x] Test max steps - [REQ-P0A](#req-p0a), [REQ-TDD](#req-tdd)
- [x] Test deterministic priority - [REQ-P0A](#req-p0a), [REQ-TDD](#req-tdd)
- [x] Test reset helper/fresh state - [REQ-P0A](#req-p0a), [REQ-TDD](#req-tdd)
- [x] Run focused P0A verification - [REQ-VERIFY](#req-verify)
- [x] Complete P0A read-only review - [REQ-AGENTS](#req-agents)
- [x] Stage/commit or otherwise integrate P0A files - [REQ-VERIFY](#req-verify)

### P0B: Runtime Plumbing

- [x] Move/export protocol-facing `PauseReason` - [REQ-P0B](#req-p0b)
- [x] Add optional `pauseReason` to worker status protocol - [REQ-P0B](#req-p0b)
- [x] Validate pause reason in protocol guard - [REQ-P0B](#req-p0b), [REQ-TDD](#req-tdd)
- [x] Update worker protocol documentation - [REQ-P0B](#req-p0b)
- [x] Add training-store `pauseReason` state - [REQ-P0B](#req-p0b)
- [x] Ensure streamed snapshots do not clear pause reason - [REQ-P0B](#req-p0b), [REQ-TDD](#req-tdd)
- [x] Wire manual pause to `manual` - [REQ-P0B](#req-p0b)
- [x] Keep config-sync stop reasonless - [REQ-P0B](#req-p0b)
- [x] Clear pause reason on play/resume - [REQ-P0B](#req-p0b)
- [x] Clear pause reason on reset - [REQ-P0B](#req-p0b)
- [x] Reset worker stop-condition state on reset/rebuild - [REQ-P0B](#req-p0b)
- [x] Evaluate runtime stop conditions only while running - [REQ-P0B](#req-p0b)
- [x] Convert non-finite train loss to `diverged` pause - [REQ-P0B](#req-p0b)
- [x] Convert non-finite test loss to `diverged` pause - [REQ-P0B](#req-p0b)
- [x] Post final snapshot before paused status - [REQ-P0B](#req-p0b)
- [x] Stop render loop on worker automatic pause - [REQ-P0B](#req-p0b)
- [x] Set worker errors to `pauseReason: 'error'` - [REQ-P0B](#req-p0b)
- [x] Add protocol tests first - [REQ-P0B](#req-p0b), [REQ-TDD](#req-tdd)
- [x] Add store tests first - [REQ-P0B](#req-p0b), [REQ-TDD](#req-tdd)
- [x] Add hook lifecycle tests first - [REQ-P0B](#req-p0b), [REQ-TDD](#req-tdd)
- [x] Add runtime divergence tests first - [REQ-P0B](#req-p0b), [REQ-TDD](#req-tdd)
- [x] Add bridge final-snapshot flush test - [REQ-P0B](#req-p0b), [REQ-TDD](#req-tdd)
- [x] Complete P0B spec compliance review - [REQ-AGENTS](#req-agents)
- [x] Complete P0B code quality review - [REQ-AGENTS](#req-agents)
- [x] Run P0B final verification - [REQ-VERIFY](#req-verify)

### P0C: Explanation Rules and Minimal UI

- [x] Confirm explicit approval to start P0C - [REQ-NO-P0C-WITHOUT-APPROVAL](#req-no-p0c-without-approval)
- [x] Add deterministic explanation rules - [REQ-P0C](#req-p0c)
- [x] Add minimal "Why did this happen?" UI - [REQ-P0C](#req-p0c)
- [x] Test stable ordered explanation output - [REQ-P0C](#req-p0c), [REQ-TDD](#req-tdd)
- [x] Test top explanation rendering - [REQ-P0C](#req-p0c), [REQ-TDD](#req-tdd)
- [x] Test related panel ids are safe when present - [REQ-P0C](#req-p0c), [REQ-TDD](#req-tdd)
- [x] Run P0C verification - [REQ-VERIFY](#req-verify)

### P1: Lesson System

#### P1A: Schema and Registry

- [x] Define web-local `LessonDefinition` and `LessonStep` - [REQ-P1](#req-p1)
- [x] Define `LessonTarget` as `data | network | hyperparams | transport` - [REQ-P1](#req-p1)
- [x] Add `LESSON_DEFINITIONS` registry - [REQ-P1](#req-p1)
- [x] Add `DEFAULT_LESSON_ID` - [REQ-P1](#req-p1)
- [x] Add `getLessonDefinition` and `getLessonPreset` - [REQ-P1](#req-p1)
- [x] Test registry ids, preset references, required text, valid targets, and no runtime predicates/config snapshots - [REQ-P1](#req-p1), [REQ-TDD](#req-tdd)

#### P1B: XOR Migration

- [x] Remove hardcoded XOR lesson constants from `GuidedLessonPanel` - [REQ-P1](#req-p1)
- [x] Render guided lesson state from registry data - [REQ-P1](#req-p1)
- [x] Preserve XOR preset application, reset, target order, tab/phase focus, finish, and unmount cleanup - [REQ-P1](#req-p1), [REQ-TDD](#req-tdd)

#### P1C: Conservative Lesson Content

- [x] Add `lesson-xor-hidden-layers` - [REQ-P1](#req-p1)
- [x] Add `lesson-single-neuron-linear-separator` - [REQ-P1](#req-p1)
- [x] Add `lesson-regression-plane-baseline` - [REQ-P1](#req-p1)
- [x] Add minimal in-panel selector for the seed lessons - [REQ-P1](#req-p1)
- [x] Test each lesson appears, starts, applies its preset, shows correct progress, and reaches run-phase transport - [REQ-P1](#req-p1), [REQ-TDD](#req-tdd)

#### P1D: Verification and Gate

- [x] Run full P1 verification - [REQ-VERIFY](#req-verify)
- [x] Complete P1 read-only spec compliance review - [REQ-AGENTS](#req-agents)
- [x] Complete P1 read-only code quality review - [REQ-AGENTS](#req-agents)
- [x] Complete P1 read-only schema invariant review - [REQ-AGENTS](#req-agents)
- [x] Complete P1 read-only merge-steward review - [REQ-AGENTS](#req-agents)
- [x] Do not expand beyond conservative 3 lessons until schema gate remains stable - [REQ-P1](#req-p1)

### P2: Experiment Memory

- [x] Define typed `ExperimentRunRecordV1` using actual exported project types - [REQ-P2](#req-p2), [REQ-TDD](#req-tdd)
- [x] Add versioned storage envelope and validators - [REQ-P2](#req-p2), [REQ-TDD](#req-tdd)
- [x] Enforce bounded record and history lengths - [REQ-P2](#req-p2), [REQ-TDD](#req-tdd)
- [x] Add corrupt/future-version localStorage recovery - [REQ-P2](#req-p2), [REQ-TDD](#req-tdd)
- [x] Add run capture helper for config, summary, bounded history, and serialized network - [REQ-P2](#req-p2)
- [x] Add run history UI panel - [REQ-P2](#req-p2)
- [x] Add restore-config behavior without exact training continuation claims - [REQ-P2](#req-p2)
- [x] Add Markdown report export - [REQ-P2](#req-p2)

### P3: Data Understanding

- [x] Add `data.seed` reshuffle action - [REQ-P3](#req-p3), [REQ-TDD](#req-tdd)
- [x] Add DataPanel reshuffle control through existing config-change flow - [REQ-P3](#req-p3)
- [x] Add accessible train/test count readout - [REQ-P3](#req-p3), [REQ-TDD](#req-tdd)
- [x] Add minimal split overlay mode without new persistence - [REQ-P3](#req-p3)
- [x] Defer dataset parameter lab/schema expansion - [REQ-P3](#req-p3)
- [x] Defer custom dataset brush - [REQ-P3](#req-p3)

### P4: Interpretability

- [x] Add engine pure-copy prediction trace - [REQ-P4](#req-p4), [REQ-TDD](#req-tdd)
- [x] Add on-demand worker `getPredictionTrace` RPC - [REQ-P4](#req-p4), [REQ-TDD](#req-tdd)
- [x] Add minimal InspectionPanel trace UI - [REQ-P4](#req-p4), [REQ-TDD](#req-tdd)
- [x] Avoid streamed diagnostics/frame-buffer/SAB expansion - [REQ-P4](#req-p4)
- [x] Defer activation histograms - [REQ-P4](#req-p4)
- [x] Defer layer-level gradient-flow overlay - [REQ-P4](#req-p4)
- [x] Defer neuron/edge histories - [REQ-P4](#req-p4)

### P5: Advanced Labs

- [ ] Defer side-by-side model arena to separate spec
- [ ] Defer slow-motion backprop to separate spec
- [ ] Defer multi-class mode to separate spec
- [ ] Defer loss landscape probes to separate spec

### Gate

- [x] Do not start P0C without explicit approval - [REQ-NO-P0C-WITHOUT-APPROVAL](#req-no-p0c-without-approval)
- [x] Do not start P1 without explicit approval - [REQ-AGENTS](#req-agents), [REQ-P1](#req-p1)
- [x] Do not expand beyond conservative 3 lessons until schema gate remains stable - [REQ-P1](#req-p1)
- [x] Do not expand P4 beyond on-demand trace in this slice - [REQ-P4](#req-p4)

## Agent Review Log

| Agent | Reasoning | Scope | Result | Blockers / Notes |
|---|---|---|---|---|
| Bernoulli | Not recorded | P0A read-only review | APPROVED | P0A coverage present. Non-blocking note: P0B config shape must avoid duplicate plateau/divergence conditions because P0A rejects duplicates. |
| Curie | Not recorded | P0B read-only reconnaissance | COMPLETED | Identified required P0B edit map, protocol need, manual/config-sync distinction, and high-risk files. |
| Euler | High | P0B spec/semantics review | NEEDS_CHANGES | Require shared protocol `pauseReason`, hook-local cleanup on automatic pause, manual/config-sync distinction, non-finite loss as `diverged`, reset/resume clearing tests. |
| Hilbert | High | P0B quality/integration review | NEEDS_CHANGES | Require shared `PauseReason` guard/export, final snapshot before paused status, protocol validation tests, and store/hook lifecycle tests. |
| Franklin | High | P0B spec compliance review | APPROVED | Confirmed P0B scope: shared protocol, runtime non-finite divergence pause, manual/config-sync distinction, reset/resume semantics, final snapshot ordering, and no serialization/persistence/UI controls. |
| Fermat | High | P0B code quality review | NEEDS_CHANGES | Code passed structurally; blocker was unrelated dirty files in the working tree. Resolved by staging only P0A/P0B/tracker files and leaving unrelated files unstaged. |
| Plato | High | P0B staged merge-steward review | NEEDS_CHANGES | Staged files matched scope and unrelated files were excluded. Requested tracker checklist update and direct bridge coverage for queued final snapshot flush. Both were addressed before final verification rerun. |
| Einstein | High | P0B final staged merge-steward review | APPROVED | Confirmed previous blockers resolved, staged diff clean, unrelated files excluded, and no serialization/persistence/UI controls/dependencies/snapshots staged. |
| Hegel | High | P0C read-only rules/metrics reconnaissance | COMPLETED | Recommended a pure explanation module, existing metrics only, no worker protocol/store changes, and no LossChart parity claims. |
| Schrodinger | High | P0C read-only UI reconnaissance | COMPLETED | Recommended placing a minimal explanation surface under the existing loss chart and avoiding stylesheet edits because unrelated CSS files are dirty. |
| Harvey | High | P0C spec compliance review | NEEDS_CHANGES | Found invalid `relatedPanelIds` value `training` and requested regression coverage for safe panel ids. Fixed by constraining ids to `LeftTabId | RightTabId` and adding a regression test. |
| Banach | High | P0C code quality review | APPROVED | Approved staged diff with non-blocking notes. Locale-sensitive tie-break and duplicated accessible label were addressed after review. Snapshot re-render note accepted as safe for the small P0C rule set. |
| Boyle | High | P0C final spec compliance review | APPROVED | Confirmed deterministic explanations, minimal UI, existing metrics only, required tests, and no protocol/store/serialization/dependency/P1 scope creep. |
| Huygens | High | P0C final merge-steward review | APPROVED | Confirmed staged files match the P0C allowlist, unrelated dirty files are unstaged, no public exports/protocol/config/store/persistence/dependency changes, no snapshots, and no P1 work. |
| Descartes | High | P1 schema/registry planning recon | COMPLETED | Recommended web-local lesson schema/registry because lessons depend on web layout, phase, and highlight concepts. |
| Poincare | High | P1 schema/registry planning recon | COMPLETED | Confirmed `apps/web/src/lessons` ownership and warned against shared package exports for P1. |
| Singer | High | P1 XOR migration planning recon | COMPLETED | Documented behavior to preserve: XOR preset, reset, target order, tab/phase focus, finish, and unmount cleanup. |
| Dalton | High | P1 XOR migration planning recon | COMPLETED | Confirmed `transport` is a lesson target but not a layout tab, so lesson targets must remain UI-specific. |
| Kant | High | P1 content planning recon | COMPLETED | Proposed Feature Engineering, but noted it requires a new `features` lesson target. Deferred by user choice. |
| Averroes | High | P1 content planning recon | COMPLETED | Recommended conservative beginner lessons and a minimal in-panel selector. |
| Epicurus | High | P1 verification gate planning recon | COMPLETED | Recommended web-local invariant tests, full verification commands, and review/merge-steward gates. |
| Jason | High | P1 verification gate planning recon | COMPLETED | Confirmed verified command syntax and collision rules for P1. |
| Noether | High | P1 schema/registry review | APPROVED | Confirmed web-local schema, four-value `LessonTarget`, registry exports, conservative 3 lessons, and invariant tests. |
| Boole | High | P1 schema/registry review | APPROVED | Confirmed exact conservative lesson ids, no `features` target, and no shared/package contract churn. |
| Godel | High | P1 XOR migration review | NEEDS_CHANGES | Found selector overflow risk from three long nowrap buttons in the fixed-width guided panel. Fixed with a vertical labelled button group and wrapped button text. |
| Planck | High | P1 XOR migration review | APPROVED | Confirmed XOR behavior, target order, phase progression, finish/unmount cleanup, and no CSS/layout/store/shared changes. |
| Euclid | High | P1 content review | APPROVED | Confirmed exact conservative 3-lesson set, no `features` target, local selector, per-lesson progress, and transport/run coverage. |
| Sagan | High | P1 content review | APPROVED | Confirmed preset wiring, no Feature Engineering, no `features` target, and guided panel coverage for selection/start/progress. |
| Dewey | High | P1 final checklist review | APPROVED | Confirmed staged files match P1 allowlist, no shared/store/worker/serialization/preset changes, and no P2 work. |
| Hypatia | High | P1 merge-steward review | NEEDS_CHANGES | Requested tracker final-review checkboxes and schema-stability gate be marked complete. Code scope and verification evidence were otherwise clean. |
| Kuhn | High | P1 final merge-steward re-review | APPROVED | Confirmed tracker gate fixes, staged allowlist, no shared/protocol/store/serialization/preset/dependency/snapshot changes, and clean cached diff. |
| Bacon | High | P2 data/persistence recon | COMPLETED | Mapped typed shared model, validators, storage envelope, web persistence, capture helper, and restore/report risks. |
| Ramanujan | High | P2 UI/report recon | COMPLETED | Recommended a right-region history/report panel after persistence contract lands and warned layout edits must be coordinator-owned. |
| Anscombe | High | P3 data/schema lifecycle recon | COMPLETED | Confirmed P3 should reuse `data.seed`, avoid schema migration, and derive counts from existing train/test arrays. |
| Carver | High | P3 visualization/UI recon | COMPLETED | Recommended accessible DataPanel split readout, disabled reshuffle during loading, and compact split visualization. |
| Cicero | High | P4 engine/protocol recon | COMPLETED | Recommended engine pure-copy trace and one-shot Comlink RPC, avoiding `WorkerToMainMessage`, frame buffer, and SAB changes. |
| Peirce | High | P4 UI recon | COMPLETED | Recommended extending `InspectionPanel`, local state, native controls, aria-live output, and in-flight gating. |
| Aquinas | High | P2-P4 spec compliance review | APPROVED | Confirmed P2 local-first memory, P3 seed-based split understanding, and P4 on-demand trace match plan with no streamed/frame-buffer/SAB expansion. |
| Zeno | High | P2-P4 code quality review | NEEDS_CHANGES -> APPROVED | Initially flagged restore wording, stale params, and localStorage write failures. Re-review approved after config-only restore copy, frame-buffer parameter capture, empty-param nulling, and guarded persistence writes. |
| Leibniz | High | P2-P4 merge-steward review | NEEDS_CHANGES -> APPROVED | Initially blocked on restore semantics mismatch. Re-review approved after `Restore config` UI/copy aligned with locked P2 decision. |

## Verification Log

### P0A

| Command | Outcome | Notes |
|---|---|---|
| `pnpm --filter @nn-playground/web test src/worker/stopConditions.test.ts src/worker/trainingLoop.test.ts` | PASS | 2 files, 17 tests. Focused verification run after P0A review. |
| `pnpm --filter @nn-playground/web test` | PASS | Reported from prior P0A implementation turn: 42 files, 237 tests. Re-run before final integration if stale. |
| `pnpm lint` | PASS | Reported from prior P0A implementation turn. Re-run before final integration if stale. |
| `pnpm build` | PASS | Reported from prior P0A implementation turn with existing Vite chunk-size warning. Re-run before final integration if stale. |

### P0B

| Command | Outcome | Notes |
|---|---|---|
| `pnpm --filter @nn-playground/shared test src/__tests__/workerProtocol.test.ts` | PASS | Targeted shared protocol red/green and final rerun: 9 tests. |
| `pnpm --filter @nn-playground/web test src/worker/stopConditions.test.ts src/store/useTrainingStore.test.ts src/hooks/useTraining.test.tsx` | PASS | Initial targeted P0B red/green: 30 tests. |
| `pnpm --filter @nn-playground/web test src/worker/workerBridge.test.ts` | PASS | Added bridge-level final-snapshot flush coverage: 10 tests. |
| `pnpm --filter @nn-playground/web test src/worker/stopConditions.test.ts src/store/useTrainingStore.test.ts src/hooks/useTraining.test.tsx src/worker/workerBridge.test.ts` | PASS | Expanded targeted final rerun: 40 tests. |
| `pnpm --filter @nn-playground/shared test` | PASS | Full shared suite: 4 files, 59 tests. |
| `pnpm --filter @nn-playground/web test` | PASS | Full web suite: 42 files, 244 tests. |
| `pnpm lint` | PASS | ESLint passed. |
| `pnpm build` | PASS | Production build passed with existing Vite chunk-size warning. |

### P0C

| Command | Outcome | Notes |
|---|---|---|
| `pnpm --filter @nn-playground/web test src/explanations/trainingExplanations.test.ts src/components/visualization/TrainingExplanationPanel.test.tsx` | EXPECTED FAIL | Initial TDD red run failed because P0C implementation files did not exist yet. |
| `pnpm --filter @nn-playground/web test src/explanations/trainingExplanations.test.ts src/components/visualization/TrainingExplanationPanel.test.tsx` | PASS | Initial P0C red/green after implementation: 2 files, 5 tests. |
| `pnpm --filter @nn-playground/web test src/components/layout/MainArea.test.tsx src/explanations/trainingExplanations.test.ts src/components/visualization/TrainingExplanationPanel.test.tsx` | PASS | Adjacent layout wiring verification before review fix: 3 files, 8 tests. |
| `pnpm --filter @nn-playground/web test src/explanations/trainingExplanations.test.ts src/components/visualization/TrainingExplanationPanel.test.tsx src/components/layout/MainArea.test.tsx` | PASS | Final focused P0C verification after review fixes and polish: 3 files, 9 tests. |
| `pnpm --filter @nn-playground/web test` | PASS | Full web suite after P0C changes: 44 files, 251 tests. |
| `pnpm --filter @nn-playground/shared test` | PASS | Full shared suite unchanged by P0C: 4 files, 59 tests. |
| `pnpm lint` | PASS | ESLint passed. |
| `pnpm build` | PASS | Production build passed with existing Vite chunk-size warning. |

### P1

| Command | Outcome | Notes |
|---|---|---|
| `pnpm --filter @nn-playground/web test src/lessons/lessonRegistry.test.ts` | EXPECTED FAIL | Initial TDD red run failed because `lessonRegistry.ts` did not exist. |
| `pnpm --filter @nn-playground/web test src/components/controls/GuidedLessonPanel.test.tsx` | EXPECTED FAIL | Initial TDD red run failed because the registry import did not exist. |
| `pnpm --filter @nn-playground/web test src/lessons/lessonRegistry.test.ts` | PASS | Registry invariants passed: 1 file, 6 tests. |
| `pnpm --filter @nn-playground/web test src/components/controls/GuidedLessonPanel.test.tsx` | PASS | Guided lesson panel migration/content tests passed: 1 file, 3 tests. |
| `pnpm --filter @nn-playground/web test src/lessons/lessonRegistry.test.ts src/components/controls/GuidedLessonPanel.test.tsx` | PASS | Combined P1 targeted suite passed: 2 files, 9 tests. |
| `pnpm --filter @nn-playground/web test` | EXPECTED FAIL | First full-web run caught an accessibility regression in the lesson selector markup. |
| `pnpm --filter @nn-playground/web test src/App.test.tsx` | PASS | Accessibility regression fixed by changing the selector container to a labelled group: 1 file, 10 tests. |
| `pnpm --filter @nn-playground/web test` | PASS | Full web suite passed after the accessibility fix: 45 files, 259 tests. |
| `pnpm --filter @nn-playground/shared test` | PASS | Full shared suite unchanged by P1: 4 files, 59 tests. |
| `pnpm lint` | PASS | ESLint passed. |
| `pnpm test` | PASS | Workspace test suite passed: engine 277 tests, shared 59 tests, web 259 tests. |
| `pnpm build` | PASS | Production build passed with existing Vite chunk-size warning. |
| `pnpm --filter @nn-playground/web test src/components/controls/GuidedLessonPanel.test.tsx src/lessons/lessonRegistry.test.ts src/App.test.tsx` | PASS | Focused rerun after reviewer-requested selector overflow fix: 3 files, 19 tests. |
| `pnpm --filter @nn-playground/web test` | PASS | Full web rerun after selector overflow fix: 45 files, 259 tests. |
| `pnpm lint` | PASS | ESLint rerun passed after selector overflow fix. |
| `pnpm build` | PASS | Production build rerun passed after selector overflow fix with existing Vite chunk-size warning. |

### P2-P4

| Command | Outcome | Notes |
|---|---|---|
| `pnpm --filter @nn-playground/shared test src/__tests__/experimentMemory.test.ts` | EXPECTED FAIL | Initial P2 TDD red run failed because the shared experiment memory module did not exist. |
| `pnpm --filter @nn-playground/shared test src/__tests__/experimentMemory.test.ts` | PASS | Shared P2 model/validator tests passed: 1 file, 5 tests. |
| `pnpm --filter @nn-playground/web test src/store/experimentMemoryStore.test.ts src/store/experimentRunCapture.test.ts` | EXPECTED FAIL | Initial P2 TDD red run failed because web persistence/capture modules did not exist. |
| `pnpm --filter @nn-playground/web test src/store/experimentMemoryStore.test.ts src/store/experimentRunCapture.test.ts` | PASS | Web persistence/capture tests passed: 2 files, 6 tests. |
| `pnpm --filter @nn-playground/web test src/store/useLayoutStore.test.ts src/components/controls/RunHistoryPanel.test.tsx` | EXPECTED FAIL | Initial P2 UI red run failed because `history` tab/panel did not exist. |
| `pnpm --filter @nn-playground/web test src/store/useLayoutStore.test.ts src/components/controls/RunHistoryPanel.test.tsx` | PASS | P2 history tab/panel tests passed: 2 files, 14 tests. |
| `pnpm --filter @nn-playground/web test src/store/usePlaygroundStore.test.ts src/components/controls/DataPanel.test.tsx src/components/visualization/DecisionBoundary.test.tsx` | EXPECTED FAIL | Initial P3 red run failed because seed reshuffle and split readout controls did not exist. |
| `pnpm --filter @nn-playground/web test src/store/usePlaygroundStore.test.ts src/components/controls/DataPanel.test.tsx src/components/visualization/DecisionBoundary.test.tsx` | PASS | P3 store/DataPanel/boundary tests passed: 3 files, 12 tests. |
| `pnpm --filter @nn-playground/engine test src/__tests__/network.test.ts` | EXPECTED FAIL | Initial P4 engine red run failed because `tracePrediction` did not exist. |
| `pnpm --filter @nn-playground/engine test src/__tests__/network.test.ts` | PASS | P4 engine trace tests passed: 55 network tests. |
| `pnpm --filter @nn-playground/web test src/components/controls/InspectionPanel.test.tsx` | EXPECTED FAIL | Initial P4 UI red run failed because trace controls did not exist. |
| `pnpm --filter @nn-playground/web test src/components/controls/InspectionPanel.test.tsx` | PASS | P4 inspection UI tests passed: 1 file, 3 tests. |
| `pnpm --filter @nn-playground/web test src/worker/training.worker.test.ts src/components/controls/InspectionPanel.test.tsx` | PASS | P4 worker RPC and UI regression tests passed: 2 files, 5 tests. |
| `pnpm --filter @nn-playground/web test src/store/experimentMemoryStore.test.ts src/store/experimentRunCapture.test.ts src/components/controls/RunHistoryPanel.test.tsx` | PASS | P2 targeted rerun passed: 3 files, 10 tests. |
| `pnpm --filter @nn-playground/web test src/store/experimentRunCapture.test.ts src/store/experimentMemoryStore.test.ts src/components/controls/RunHistoryPanel.test.tsx` | EXPECTED FAIL | Review-fix TDD red run caught missing frame-buffer capture, uncaught storage writes, and restore-copy mismatch. |
| `pnpm --filter @nn-playground/web test src/store/experimentRunCapture.test.ts src/store/experimentMemoryStore.test.ts src/components/controls/RunHistoryPanel.test.tsx` | PASS | Review-fix targeted rerun passed: 3 files, 13 tests. |
| `pnpm --filter @nn-playground/shared test` | PASS | Full shared suite: 5 files, 64 tests. |
| `pnpm --filter @nn-playground/engine test` | PASS | Full engine suite: 12 files, 279 tests. |
| `pnpm --filter @nn-playground/web test` | EXPECTED FAIL | First full web run caught missing `HistoryContent` in an existing integration mock. |
| `pnpm --filter @nn-playground/web test src/__tests__/training.integration.test.tsx` | PASS | Integration mock fixed: 1 file, 6 tests. |
| `pnpm --filter @nn-playground/web test` | PASS | Full web suite passed after fixes: 49 files, 280 tests. |
| `pnpm lint` | PASS | ESLint passed. |
| `pnpm test` | PASS | Workspace test suite passed: engine 279 tests, shared 64 tests, web 280 tests. |
| `pnpm build` | EXPECTED FAIL | First final build caught two TypeScript-only guard/narrowing issues. |
| `pnpm build` | PASS | Production build passed after type guard fixes with existing Vite chunk-size warning. |
| `git diff --check` | PASS | No whitespace errors. |
| `pnpm lint` | PASS | Final lint rerun after TypeScript guard fixes passed. |
| `pnpm test` | PASS | Final workspace test rerun after TypeScript guard fixes passed. |

## File Ownership And Collision Notes

Current unrelated dirty/untracked files to avoid unless explicitly assigned:

- None at P1 start.

Current P0A files:

- `apps/web/src/worker/stopConditions.ts`
- `apps/web/src/worker/stopConditions.test.ts`

P0B high-risk files:

- `packages/shared/src/workerProtocol.ts`
- `packages/shared/src/types.ts`
- `packages/shared/src/index.ts`
- `apps/web/src/worker/training.worker.ts`
- `apps/web/src/hooks/useTraining.ts`
- `apps/web/src/store/useTrainingStore.ts`

P0C files:

- `apps/web/src/explanations/trainingExplanations.ts`
- `apps/web/src/explanations/trainingExplanations.test.ts`
- `apps/web/src/components/visualization/TrainingExplanationPanel.tsx`
- `apps/web/src/components/visualization/TrainingExplanationPanel.test.tsx`
- `apps/web/src/components/layout/MainArea.tsx`
- `apps/web/src/components/layout/MainArea.test.tsx`

P1 files:

- `apps/web/src/lessons/types.ts`
- `apps/web/src/lessons/lessonRegistry.ts`
- `apps/web/src/lessons/lessonRegistry.test.ts`
- `apps/web/src/components/controls/GuidedLessonPanel.tsx`
- `apps/web/src/components/controls/GuidedLessonPanel.test.tsx`

P2-P4 files:

- `packages/shared/src/experimentMemory.ts`
- `packages/shared/src/__tests__/experimentMemory.test.ts`
- `apps/web/src/store/experimentMemoryStore.ts`
- `apps/web/src/store/experimentMemoryStore.test.ts`
- `apps/web/src/store/experimentRunCapture.ts`
- `apps/web/src/store/experimentRunCapture.test.ts`
- `apps/web/src/components/controls/RunHistoryPanel.tsx`
- `apps/web/src/components/controls/RunHistoryPanel.test.tsx`
- `apps/web/src/components/controls/DataPanel.tsx`
- `apps/web/src/components/controls/DataPanel.test.tsx`
- `apps/web/src/components/visualization/DecisionBoundary.tsx`
- `apps/web/src/components/visualization/DecisionBoundary.test.tsx`
- `packages/engine/src/network.ts`
- `packages/engine/src/types.ts`
- `packages/engine/src/__tests__/network.test.ts`
- `apps/web/src/worker/training.worker.ts`
- `apps/web/src/worker/training.worker.test.ts`
- `apps/web/src/components/controls/InspectionPanel.tsx`
- `apps/web/src/components/controls/InspectionPanel.test.tsx`

No worker may edit outside its allowlist, change serialization unexpectedly, add
dependencies, or perform unrelated cleanup.

## Canonical Prompt

The canonical execution prompt follows. This is preserved as the source material
for the checklist above.

```md
PLEASE IMPLEMENT THIS PLAN:
# Neural Network Playground 2.0 Superpowers Execution Plan

## 1. Superpowers Usage Strategy
Applied skills:
- `superpowers:brainstorming`: product scope control only.
- `superpowers:writing-plans`: convert each roadmap slice into executable task plans.
- `superpowers:dispatching-parallel-agents`: read-only recon and only disjoint implementation slices.
- `superpowers:subagent-driven-development`: preferred implementation model.
- `superpowers:test-driven-development`: required for all behavior changes.
- `superpowers:using-git-worktrees`: use in local filesystem/Codex Desktop. In Codex Cloud or unsuitable environments, use isolated branches/tasks with the same ownership, review, and merge-steward rules.
- `superpowers:verification-before-completion`: required before claiming any task, phase, test, build, or review is complete.

Parallel agents are allowed only when file ownership, runtime state, protocol/API contracts, and tests are disjoint, and a coordinator owns integration.

Parallel agents are forbidden when they would edit the same files, shared serialization/config schema, worker protocol, Zustand store shape, dependent unmerged work, or anything requiring whole-system design judgment.

## 2. Agent Operating Model
### Coordinator Agent
Owns plan, sequencing, file ownership, task graph, merge order, and verification.

Responsibilities:
- Inspect repo state first.
- Refresh the file map before assigning workers.
- Create/update Superpowers plan files.
- Assign narrow worker scopes.
- Maintain file ownership table.
- Decide parallelism.
- Review every worker summary and diff.
- Run final verification.

The coordinator does not casually edit implementation code while workers are active.

### Read-Only Recon Agents
May inspect only. Use for:
- Lesson/preset/config flow.
- Worker/protocol/runtime state.
- Visualization demand/frame-buffer flow.
- Serialization/share-link/local persistence.
- Test/build command discovery.

Output must include files, existing patterns, risks, task boundaries, and files unsafe for parallel edits.

### Implementation Worker Agents
Each worker receives:
- Goal.
- Exact files allowed to modify.
- Exact files allowed to inspect.
- Forbidden files.
- Required failing tests first.
- Required commands.
- Expected output format.
- Stop conditions.

Each returns:
`DONE`, `DONE_WITH_CONCERNS`, `NEEDS_CONTEXT`, or `BLOCKED`, plus changed files, tests, commands/results, contract/API changes, and risks.

### Reviewer Agents
Use two read-only reviewers per task:
1. Spec Compliance Reviewer: checks exact requested behavior, missing requirements, extra behavior, scope creep.
2. Code Quality Reviewer: checks naming, cohesion, performance, test quality, maintainability, shared-state risk.

### Merge Steward
Runs after workers finish. Checks diffs, conflicts, contracts, serialization, store shape, persistence, tests, build, bundle-impacting imports, and unexpected dependencies.

## 3. Current Last-Discovered Map, To Be Refreshed Before Execution
- Stores: `apps/web/src/store/usePlaygroundStore.ts`, `apps/web/src/store/useTrainingStore.ts`, `apps/web/src/store/useLayoutStore.ts`.
- Worker/runtime: `apps/web/src/worker/training.worker.ts`, `apps/web/src/worker/workerBridge.ts`, `apps/web/src/worker/trainingLoop.ts`, `apps/web/src/hooks/useTraining.ts`.
- Protocol/contracts: `packages/shared/src/workerProtocol.ts`, `packages/shared/src/types.ts`, `packages/shared/src/index.ts`.
- Serialization/config: `packages/shared/src/serialization.ts`, `packages/shared/src/constants.ts`.
- Lessons/presets: `apps/web/src/components/controls/GuidedLessonPanel.tsx`, `packages/shared/src/presets.ts`.
- Visualization demand/frame-buffer: `apps/web/src/components/layout/deriveVisualizationDemand.ts`, `apps/web/src/worker/frameBuffer.ts`, `apps/web/src/worker/sharedSnapshot.ts`, `apps/web/src/worker/frameBufferLayout.ts`.
- Existing tests: `apps/web/src/worker/*.test.ts`, `apps/web/src/store/*.test.ts`, `apps/web/src/components/controls/GuidedLessonPanel.test.tsx`, `packages/shared/src/__tests__/serialization.test.ts`, `packages/shared/src/__tests__/workerProtocol.test.ts`, `packages/engine/src/__tests__/*.test.ts`.
- Package scripts discovered: root `dev`, `build`, `test`, `test:engine`, `test:perf`, `bench`, `lint`, `clean`; web `dev`, `build`, `preview`, `test`; engine `test`, `test:watch`, `test:perf`, `bench`; shared `test`.

No worker gets an allowlist based on stale or guessed paths.

## 4. File Ownership and Collision Rules
| Area | Classification | Rule |
|---|---|---|
| `packages/shared/src/workerProtocol.ts` | Locked | Coordinator-only or dedicated serial protocol task. |
| `packages/shared/src/serialization.ts`, `types.ts`, `constants.ts` | Locked | Schema/default changes are coordinator-owned. |
| `apps/web/src/worker/training.worker.ts`, `workerBridge.ts`, `useTraining.ts` | Locked | Serial runtime task only. |
| `apps/web/src/store/usePlaygroundStore.ts`, `useTrainingStore.ts` | Locked | One store-shape task at a time. |
| `frameBuffer.ts`, `sharedSnapshot.ts`, `deriveVisualizationDemand.ts` | Locked | Demand/buffer changes require tests and perf guardrails. |
| App shell/layout composition | Locked | Coordinator-approved edits only. |
| New isolated UI components | Owner | Parallel-safe after contracts land. |
| Lesson content-only files | Owner | Parallel-safe if registry/schema files are not edited. |

Rules:
- No unrelated cleanup.
- No broad export renames.
- No broad snapshot/test churn without explanation.
- No dependency additions unless assigned.
- Only one active implementation worker may edit shared contract files.

## 5. Agent Stop Conditions
Workers must stop with `NEEDS_CONTEXT` or `BLOCKED` if:
- They need to edit outside allowlist.
- A path is wrong.
- A command does not exist.
- Shared serialization is needed unexpectedly.
- Worker protocol changes are needed unexpectedly.
- Store changes are needed unexpectedly.
- Tests fail outside scope.
- Another active worker owns the same file.
- Product/design decisions are missing.

Workers must not improvise around these constraints.

## 6. Corrected Roadmap Sequencing
### P0A: Pure Stop-Condition Foundation
Scope:
- Define `PauseReason`.
- Define `StopCondition`.
- Add pure stop-condition evaluator.
- Unit-test evaluator behavior.

Rules:
- No UI changes.
- No worker protocol changes unless absolutely required.
- No serialization changes unless stop conditions are saved in config.
- P0 stop conditions start as internal/default runtime behavior unless explicitly assigned as user-configurable controls.
- Do not add new controls, share URL fields, or persistence for stop conditions in P0A/P0B unless assigned.

Acceptance:
- Evaluator handles target loss, target accuracy, plateau, divergence, max steps, NaN, and Infinity.
- Tests prove each stop condition.
- Deterministic priority is tested.
- No unrelated files touched.

### P0B: Runtime Plumbing
Scope:
- Connect evaluator to training runtime.
- Add pause reason propagation.
- Update store/runtime state as needed.

Runtime semantics:
- Stop conditions evaluate only while training is running.
- Manual pause uses `pauseReason: 'manual'`.
- Stop-condition pause uses the matching non-manual `PauseReason`.
- Reset clears `pauseReason`, stop-condition counters, plateau tracking, and divergence tracking.
- Resume after stop-condition pause is allowed only by explicit user resume.
- Resume does not clear historical metrics unless reset is triggered.
- Plateau requires enough steps to satisfy patience.
- Divergence catches `NaN` and `Infinity` immediately.
- `targetAccuracy` is ignored/disabled when accuracy is unavailable.
- If multiple stop conditions trigger, use deterministic priority.

Acceptance:
- Training pauses with correct `pauseReason`.
- Manual pause still works.
- Resume/reset behavior is defined and tested.
- Worker/store tests pass.

### P0C: Explanation Rules and Minimal UI
Scope:
- Deterministic explanation rules.
- Minimal "Why did this happen?" UI.
- Existing metrics/diagnostics only.

Rules:
- No lesson system.
- No custom datasets.
- No expensive worker data.
- Explanations are deterministic and testable.

Acceptance:
- Stable ordered output from fixed context.
- UI displays top explanation.
- Related panel ids are optional and safe if absent.
- Component tests cover one stop reason and one diagnostic explanation.

### P1: Lesson System
- Define `LessonDefinition` and `LessonStep`.
- Migrate current XOR guided lesson.
- Add only 2-3 lessons first.
- Add remaining lessons after schema proves stable.

### P2: Experiment Memory
- Run history records.
- Restore run.
- Bounded versioned local persistence.
- Exportable experiment report.

Typing rule:
- Replace `unknown` in `RunHistoryRecord` with actual exported project types after repo inspection.
- Do not ship persistent records with untyped `unknown` payloads unless wrapped in a versioned schema validator.

### P3: Data Understanding
- Train/test split visualizer.
- Reshuffle controls.
- Dataset parameter lab.
- Custom dataset brush later.

### P4: Interpretability
- On-demand prediction trace.
- Activation histograms.
- Layer-level gradient-flow overlay.
- Neuron/edge histories only if payload cost remains controlled.

### P5: Advanced Labs
Defer into separate specs:
- Side-by-side model arena.
- Slow-motion backprop.
- Multi-class mode.
- Loss landscape probes.

## 7. Parallelization Map
| Slice | Classification | Parallelism |
|---|---|---|
| P0A | `SERIAL_ONLY` implementation | Read-only recon first only. |
| P0B | `SERIAL_ONLY` | Runtime/store/protocol risk. |
| P0C | `SERIAL_ONLY` or `PARALLEL_READ_ONLY` | Serialize implementation. |
| P1 | `PARALLEL_IMPLEMENTATION_RISKY` | Schema serial first; content-only later. |
| P2 | `PARALLEL_IMPLEMENTATION_RISKY` | Data model/persistence serial first; UI/report later. |
| P3 | `PARALLEL_READ_ONLY` first | Dataset/split contracts stabilize serially. |
| P4 | `SERIAL_ONLY` for protocol/RPC | UI components parallel only after protocol lands. |
| P5 | `SERIAL_ONLY` | Separate specs required. |

## 8. Deterministic Pause Priority
Automatic stop-condition priority:

```ts
const PAUSE_REASON_PRIORITY: PauseReason[] = [
  'diverged',
  'max-steps',
  'target-loss-reached',
  'target-accuracy-reached',
  'plateau',
];
```

`manual` and `error` are outside automatic priority:
- `manual`: explicit user pause.
- `error`: runtime/worker exception.

P0A must test this priority.

## 9. TypeScript Data Models
P0A may keep these app-internal until P0B proves shared contract need.

```ts
export type PauseReason =
  | 'target-loss-reached'
  | 'target-accuracy-reached'
  | 'plateau'
  | 'diverged'
  | 'max-steps'
  | 'manual'
  | 'error';

export type StopCondition =
  | { kind: 'targetLoss'; threshold: number }
  | { kind: 'targetAccuracy'; threshold: number }
  | { kind: 'plateau'; metric: 'loss' | 'accuracy'; minDelta: number; patienceSteps: number }
  | { kind: 'divergence'; lossMultiplier?: number; nanOrInfinity?: boolean; patienceSteps?: number }
  | { kind: 'maxSteps'; steps: number };

export interface StopConditionContext {
  step: number;
  trainLoss: number;
  testLoss: number;
  trainAccuracy?: number;
  testAccuracy?: number;
}

export interface StopConditionState {
  bestLoss: number | null;
  bestAccuracy: number | null;
  plateauStartStep: number | null;
  divergenceStartStep: number | null;
}

export interface StopConditionEvaluation {
  pauseReason: Exclude<PauseReason, 'manual' | 'error'> | null;
  nextState: StopConditionState;
}

export interface ExplanationContext {
  trainLoss: number;
  testLoss: number;
  trainAccuracy?: number;
  testAccuracy?: number;
  pauseReason?: PauseReason;
  testMetricsStale?: boolean;
  step: number;
}

export interface ExplanationRuleDescriptor {
  id: string;
  priority: number;
  title: string;
  explanation: string;
  suggestedAction?: string;
  relatedPanelIds?: string[];
}

export interface RuntimeExplanationRule extends ExplanationRuleDescriptor {
  when: (context: ExplanationContext) => boolean;
}

export interface LessonDefinition {
  id: string;
  title: string;
  summary: string;
  presetId?: string;
  estimatedMinutes?: number;
  steps: LessonStep[];
}

export interface LessonStep {
  id: string;
  title: string;
  body: string;
  targetPanelId?: string;
  expectedOutcome?: string;
  successCheck?: string;
  explanationRuleIds?: string[];
}

export interface RunHistoryRecord {
  id: string;
  createdAt: number;
  label?: string;
  config: unknown;
  finalMetrics: unknown;
  bestMetrics?: unknown;
  stepCount: number;
  pauseReason?: PauseReason;
  thumbnailDataUrl?: string;
}

export interface PredictionTraceResult {
  sampleId?: string;
  input: number[];
  target: number[];
  output: number[];
  prediction: number | number[];
  lossContribution: number;
  layers: Array<{
    layerIndex: number;
    activations: number[];
    preActivations?: number[];
  }>;
}
```

Lesson definitions may reference `explanationRuleIds`; they must not contain inline runtime predicate functions.

## 10. TDD Requirements
Every implementation task must:
1. Write failing test first.
2. Run targeted command and confirm expected failure.
3. Implement minimal code.
4. Run targeted command and confirm pass.
5. Add regression coverage.
6. Run package-level tests.
7. Coordinator runs final verification.

No implementation-first workers.

## 11. Verified Command Matrix
Commands discovered from root/package `package.json`, `pnpm-workspace.yaml`, CI, and test files.

| Actual Command | Target Area | Owner | When | Proves |
|---|---|---|---|---|
| `pnpm lint` | Whole repo | Coordinator | Before merge | Matches CI lint step. |
| `pnpm test` | All workspace tests | Coordinator | Before phase completion | Matches CI test step. |
| `pnpm build` | Web production build | Coordinator | Before phase completion/deploy | Matches CI/deploy build step. |
| `pnpm test:engine` | Engine package | Engine worker/coordinator | Engine changes | Root engine test script. |
| `pnpm test:perf` | Engine perf tests | Coordinator | Perf-sensitive engine changes | Root perf script. |
| `pnpm --filter @nn-playground/engine test` | Engine package | Worker | Engine task completion | Package script. |
| `pnpm --filter @nn-playground/shared test` | Shared package | Worker | Shared contract changes | Package script. |
| `pnpm --filter @nn-playground/web test` | Web package | Worker | Web/store/UI/worker changes | Package script. |
| `pnpm --filter @nn-playground/engine test src/__tests__/network.test.ts` | Targeted engine test | Worker | TDD red/green | Verified targeted syntax. |
| `pnpm --filter @nn-playground/shared test src/__tests__/serialization.test.ts` | Serialization | Worker/coordinator | Config/schema changes | Verified targeted syntax. |
| `pnpm --filter @nn-playground/shared test src/__tests__/workerProtocol.test.ts` | Protocol guards | Worker/coordinator | Protocol changes | Actual targeted script form. |
| `pnpm --filter @nn-playground/web test src/worker/workerBridge.test.ts` | Worker bridge | Worker | Runtime/bridge changes | Verified targeted syntax. |
| `pnpm --filter @nn-playground/web test src/store/useTrainingStore.test.ts` | Training store | Worker | Store-shape changes | Actual targeted script form. |
| `pnpm --filter @nn-playground/web test src/components/controls/GuidedLessonPanel.test.tsx` | Guided lesson UI | Worker | Lesson UI changes | Actual targeted script form. |
| `pnpm --filter @nn-playground/web test src/worker/stopConditions.test.ts` | P0A stop evaluator | P0A worker | P0A red/green | New targeted test command. |

Substitution note:
- Do not use `pnpm --filter <pkg> test -- <file>` here. It did not narrow suites consistently.
- Use `pnpm --filter <pkg> test <file>`.

## 12. P0A Execution Plan
Editable files:
- Create `apps/web/src/worker/stopConditions.ts`.
- Create `apps/web/src/worker/stopConditions.test.ts`.

Inspect-only files:
- `packages/engine/src/types.ts`
- `apps/web/src/worker/training.worker.ts`
- `apps/web/src/store/useTrainingStore.ts`
- `apps/web/src/worker/*.test.ts`

Forbidden for P0A:
- `packages/shared/src/workerProtocol.ts`
- `packages/shared/src/serialization.ts`
- `packages/shared/src/types.ts`
- `apps/web/src/store/*`
- `apps/web/src/worker/training.worker.ts`
- Any UI/component files

TDD steps:
1. Write failing tests for target loss, target accuracy, unavailable accuracy, plateau patience, NaN divergence, Infinity divergence, max steps, deterministic priority, and reset helper if included.
2. Run:
   `pnpm --filter @nn-playground/web test src/worker/stopConditions.test.ts`
3. Implement minimal evaluator.
4. Re-run:
   `pnpm --filter @nn-playground/web test src/worker/stopConditions.test.ts`
5. Run adjacent regression:
   `pnpm --filter @nn-playground/web test src/worker/stopConditions.test.ts src/worker/trainingLoop.test.ts`
6. Review: Spec Compliance, then Code Quality.
7. Coordinator verification:
   `pnpm --filter @nn-playground/web test`
   `pnpm lint`
   `pnpm build`

Do not proceed to P0B until P0A passes tests, review, and verification.

## 13. Performance Guardrails
- No expensive interpretability data streamed by default.
- New visualization demand flags default false.
- Activation histograms compute only when visible/requested.
- Prediction trace is on-demand RPC, not continuous stream.
- Gradient overlay starts layer-level only.
- New worker payloads must define size/frequency expectations.
- New frame-buffer domains require targeted tests and perf sanity checks.
- Track current main bundle chunk warning; advanced panels are code-splitting candidates.

## 14. Compatibility and Migration
- Existing share URLs must continue to load.
- Existing serialized configs must normalize through defaults.
- New config fields must be optional.
- Stop conditions are not added to serialized app config unless user-configurable or required for share/restore.
- If stop conditions are added to config later, they must be optional, defaulted through normalization, version-compatible, and covered by old/minimal config tests.
- Run history/local persistence must be versioned and bounded.
- No breaking public package APIs unless explicitly planned.

## 15. Merge Steward Diff Checklist
Before merge, inspect:
- Changed files vs assigned allowlist.
- Public exports changed.
- Worker protocol changed.
- Config/serialization changed.
- Store shape changed.
- Local persistence changed.
- Bundle-impacting imports added.
- Tests added for each behavior change.
- Deleted or broadly updated snapshots.
- Unexpected dependency additions.

Any mismatch blocks merge until explained or fixed.

## 16. Linear / Ticket-Ready Breakdown
Do not create Linear tickets yet.

| Epic | Task | Dependencies | Files | Tests | Risk |
|---|---|---|---|---|---|
| P0 Foundation | P0A pure evaluator | None | `apps/web/src/worker/stopConditions.ts` | `stopConditions.test.ts` | Medium |
| P0 Foundation | P0B runtime plumbing | P0A | Worker/store/protocol only if assigned | Worker/store tests | High |
| P0 Foundation | P0C explanations UI | P0B | Explanation module + minimal component | Unit/component tests | Medium |
| P1 Lessons | Lesson schema + XOR migration | P0 stable | Lesson files + guided panel | Guided lesson tests | Medium |
| P2 Memory | Run history persistence | P1 stable | Store/persistence module | Store/persistence tests | High |
| P2 Memory | Export report | Run history | Report module/UI | Unit/component tests | Medium |
| P3 Data | Split visualizer | P2 stable | Data/boundary UI | Component tests | Medium |
| P4 Interpretability | Prediction trace RPC | Demand policy stable | Engine/worker/protocol/UI | Engine/worker/UI tests | High |

## 17. First Execution Target
Execute only P0A first after leaving Plan Mode.

Sequence:
1. Refresh file map.
2. Set up local worktree or isolated branch/task depending on environment.
3. Run P0A TDD.
4. Spec Compliance Review.
5. Code Quality Review.
6. Coordinator verification.
7. Stop. Do not start P0B without explicit approval.
```

## P0B Corrected Continuation Prompt

This P0B correction supersedes the earlier P0B details where they conflict.

```md
# P0B Runtime Pause Reasons Plan

## Summary
High-reasoning subagents reviewed the prior P0B plan and both returned `NEEDS_CHANGES`. The corrected plan below resolves their blockers before implementation.

P0B remains serial-only. It wires P0A's evaluator into runtime pause behavior without adding user controls, share URL fields, persistence, lesson logic, or UI panels.

## Corrections From Review
- `PauseReason` must be a shared protocol contract, not web-local only.
- Worker-driven automatic pause must stop hook-local playback state, not just update Zustand.
- Manual pause and config-sync stop must stay distinguishable.
- Non-finite train/test loss should pause with `pauseReason: 'diverged'`, not use the fatal worker error overlay.
- Worker should post the final snapshot first, then post paused status with the reason.
- Reset/resume clearing semantics must be explicit and tested.

## Implementation Plan
1. Shared protocol
   - Add `PauseReason` to `packages/shared/src/types.ts`.
   - Export it from `packages/shared/src/index.ts`.
   - Add `pauseReason?: PauseReason | null` to `WorkerStatusMessage`.
   - Add protocol validation for valid/invalid pause reasons in `workerProtocol.ts`.
   - Update `docs/worker-protocol.md`.

2. Runtime behavior
   - Keep P0B default runtime condition to non-finite divergence only.
   - Use P0A evaluator so `NaN` or `Infinity` in train/test loss returns `diverged`.
   - Do not activate target loss, target accuracy, plateau, or max steps at runtime yet.
   - Evaluate only while worker training is running.
   - When automatic divergence triggers, post the latest snapshot, then post status `{ status: 'paused', pauseReason: 'diverged' }`.

3. Store and hook lifecycle
   - Add `pauseReason: PauseReason | null` to the training store.
   - Manual pause sets `manual` locally.
   - Config-sync internal stops do not set `manual`.
   - Worker automatic pause sets the received reason, clears `isPlayingRef`, and stops the render loop.
   - Worker error sets `error`.
   - Play/resume clears the visible pause reason.
   - Reset clears pause reason and worker stop-condition state.
   - Streamed snapshots must not clear pause reason.
```

## Update Rule

After each task:

- check off completed items only after verification;
- add exact verification command results;
- add reviewer results;
- record blockers;
- preserve the canonical prompt as the source of truth;
- do not mark a phase complete before `superpowers:verification-before-completion`.
