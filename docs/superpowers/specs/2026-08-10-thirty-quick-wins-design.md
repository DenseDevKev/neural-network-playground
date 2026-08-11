# NN Playground Thirty Quick Wins Design

**Date:** 2026-08-10

**Status:** Approved for implementation

**Scope:** Implement and verify the exact thirty quick wins selected by the
multi-agent product, code, accessibility, and validation audit. The work is a
targeted hardening and usability program for the current production app, not a
replacement shell or a continuation of the unimplemented Precision Lab plan.

## Intent

Make the existing NN Playground safer, easier to learn, more accessible, and
more trustworthy to ship. Each win must remain independently understandable and
testable. The program preserves the current scientific model, experiment
document, worker protocol, persistence safety, audience profiles, Build/Run
composition, and visual language.

Completion means all thirty numbered outcomes below exist in production code,
their acceptance checks pass, and the full repository gates prove the combined
result. A plan, partial implementation, or passing subset is not completion.

## Product and engineering constraints

- Preserve the unrelated modified Scientific Trust plan, `V0.1 BUILD DOCU.md`,
  the responsive-polish draft, and the entire `prototypes/` directory unless a
  numbered task explicitly targets a prototype lint file.
- Preserve the worker as the authority for trained parameters and scientific
  evidence. Do not change engine math, objective semantics, dataset generation,
  experiment schema versions, checkpoint guarantees, or saved-run retention.
- Preserve audience profiles as visibility and guidance choices only. They must
  never mutate the experiment, training state, or saved evidence.
- Preserve the existing compatibility behavior for invalid URLs, imports,
  legacy saved runs, rejected records, and persistence failures.
- Add no production dependency unless an acceptance criterion cannot be met with
  the existing stack. `terser` is already declared and locked; repairing the
  local installation is an execution prerequisite, not a package-feature task.
- All behavior changes use test-driven development: a focused test must fail for
  the expected missing behavior before production code is changed.
- Configuration-only changes require an executable validation command even when
  a unit test is not appropriate.
- All new copy must be concise, plain-language, and consistent with the current
  scientific-trust vocabulary.
- Use the existing CSS tokens, component patterns, accessible names, and
  responsive breakpoints. Do not introduce a new design system.
- Keep Playwright retries at zero and do not weaken assertions, performance
  thresholds, accessibility checks, or timeouts to obtain a pass.

## The thirty required outcomes

### Correctness and green gates

1. **Safe global shortcuts.** Modified shortcuts and key repeats are ignored by
   the app handler. Command/Control+R remains browser refresh; plain R resets
   exactly once; Space and ArrowRight retain their current unmodified behavior;
   editable targets remain excluded.
2. **Retry-safe stream setup.** A rejected worker `setStreamPort` call closes
   both new ports, leaves no installed stream port, and permits a later setup
   attempt to succeed.
3. **Truthful async error recovery.** Error-boundary retry callbacks may return a
   promise. The fallback clears only after success; rejection keeps an
   actionable error state and creates no unhandled rejection.
4. **Live URL navigation.** Valid hash changes and browser back/forward navigation
   load through the existing validated URL path after startup. Invalid hashes
   enter the existing compatibility state. The listener is lifecycle-owned and
   removed on cleanup.
5. **No stale configuration error.** A successful copy, import, or export clears
   any earlier configuration error before showing success.
6. **Lesson reset disclosure.** Before a lesson begins, visible copy explains
   that its setup replaces the current recipe and resets training. The action
   label communicates the reset.
7. **Human-readable run names.** Saving a run accepts an optional user-facing
   title, supplies a useful deterministic default, persists the title through
   reload, and continues to validate the existing bounded title field.
8. **One repository typecheck.** A root command checks web production code,
   engine code and benchmarks, shared code and tests, and web tests. Current
   benchmark globals, Node types, stale fixtures, and readonly test mutations are
   corrected without weakening types. CI runs the command as a named gate.
9. **Green repository lint.** Prototype Node tests receive the correct Node
   global environment or explicit imports while browser files remain protected
   from accidental globals. Root lint reports zero errors and warnings.
10. **Trustworthy grid performance gate.** All four prediction paths are measured
    and asserted even when one fails. Measurement uses repeatable warmup and
    robust sampling suitable for the defined runner. The implementation is
    optimized if the controlled baseline demonstrates a real regression; caps
    are not simply raised to hide one.

### Onboarding, usability, and accessibility

11. **First-visit lesson cue.** A fresh user with no lesson progress or saved-run
    history sees one dismissible call to start a short lesson. Dismissal persists
    locally and the cue does not repeatedly interrupt established users.
12. **Observable lesson steps.** Every guided lesson step names an action or
    observation. Steps may show a lightweight Done state derived from existing
    state, but Next always remains available and learners are never trapped.
13. **Explicit run-pair comparison.** Users can select any two accepted saved
    runs for comparison. The heading names both runs, deletion reconciles the
    selection, and the existing dataset/objective comparability guard remains.
14. **Build/Run explanation.** An accessible explanation states that Build edits
    the recipe, Run trains and inspects it, and switching views alone neither
    starts nor resets training. It is available to pointer, keyboard, and touch.
15. **Unambiguous workspace label.** The visible `Mode` label becomes
    `Workspace` while the stored audience enum and accessible description remain
    compatible. Copy states that the choice changes visible tools and guidance
    only.
16. **Discoverable shortcut reference.** Space, ArrowRight, and R are listed in a
    keyboard-accessible help surface that remains reachable when inline shortcut
    badges are hidden responsively. The list is sourced from the same explicit
    shortcut definitions used by the handler or otherwise contract-tested
    against it.
17. **Meaningful range controls.** Train ratio and noise have explicit labels,
    stable IDs, associated visible outputs, and unit-bearing accessible value
    text. Train ratio communicates both train and test percentages.
18. **Quiet metric announcements.** Continuously changing header metrics are not
    one polite live region. Values remain readable, while lifecycle,
    configuration, pause, completion, and error announcements continue through
    the existing announcer exactly once per meaningful transition.
19. **Complete worker-error modal.** The worker crash alert dialog contains
    keyboard focus, removes the background from interaction and assistive
    technology, and restores the prior focus when the error clears.
20. **Mobile graph and evidence targets.** At 390 by 844 CSS pixels, visible graph
    toolbar, graph mode, edge-filter, and decision-overlay actions provide at
    least a 44 by 44 CSS-pixel hit area without global horizontal overflow.

### Resilience and validation depth

21. **Compact mobile outcome.** At compact widths, the current full-evaluation
    step and primary task outcome remain reachable from a compact disclosure
    without horizontal overflow or reinstating the full desktop metric bar.
22. **Three new concepts.** Learning rate, train/test split, and epoch join the
    typed concept catalog with plain definitions, extended explanations,
    examples, related concepts, profile visibility, and accessible help at their
    primary UI locations.
23. **Consistent state-effect copy.** Reset, reshuffle, lesson start, preset
    application, and saved-recipe application use a shared `Changes:` and
    `Preserves:` copy pattern that accurately describes recipe, data, weights,
    checkpoints, saved evidence, and view state.
24. **Stable timed state.** `useTimedState` has a stable setter identity and
    documented current-prop semantics when default value or duration changes
    during an active timer. Cleanup remains complete.
25. **Stable panel persistence IDs.** Collapsible panels persist by explicit,
    collision-resistant IDs rather than visible titles. Existing saved keys are
    migrated or read compatibly so a copy-only rename does not lose state.
26. **Cross-tab run-memory synchronization.** Storage events for accepted and
    legacy run-memory keys queue a rehydration in other tabs. Same-tab writes are
    not duplicated and queued store operations remain serialized.
27. **Independent performance results.** Root performance execution always runs
    both engine and web suites, reports both outcomes, and exits nonzero if either
    fails.
28. **Production-browser accessibility scans.** Chromium and WebKit run Axe in a
    representative desktop and compact production state. Serious and critical
    violations fail the suite; any exception is narrow, documented, and tied to
    an issue.
29. **390-pixel responsive coverage.** The existing compact Playwright checks run
    at both 320 by 844 and 390 by 844, covering overflow, critical target size,
    focus restoration, reachable drawers, and the compact outcome disclosure.
30. **Production failure-and-recovery journey.** A deterministic, test-only fault
    injection exercises worker startup/runtime failure in the built app, proves
    the accessible recovery UI and focus behavior, and returns to an operational
    workspace without console or page errors.

## Component and data-flow design

The program changes existing boundaries rather than creating a parallel app.
`App.tsx` continues to own global shortcuts, URL lifecycle integration, drawer
composition, training commands, and the worker-error overlay. Worker streaming
remains in `workerBridge.ts`. Run persistence remains in
`experimentMemoryStore.ts`, with focused UI state in `RunHistoryPanel.tsx` or a
small extracted hook when that keeps save and comparison responsibilities clear.

Header help, metric disclosure, and audience copy remain in the current header
and layout system. Lesson progress and completion checks extend the existing
lesson definitions with optional declarative completion predicates; they do not
create a second tutorial state store. Accessibility additions reuse the current
announcer, tooltip/help, modal, and CSS token patterns.

Repository gates stay explicit and independently diagnosable. Typecheck, lint,
unit/integration tests, build, engine performance, web performance, and browser
projects report separate conclusions. Browser fault injection must be compiled
only for test mode and cannot expose a production user control.

## Error handling

- Worker stream setup owns every port it creates until installation succeeds and
  closes both ends on any setup failure.
- Async retries preserve their latest error and fallback UI until success.
- Hash navigation reuses compatibility state rather than throwing or silently
  retaining the prior experiment.
- Cross-tab storage failures follow the existing persistence-error path and do
  not delete or overwrite recoverable bytes.
- Naming and comparison errors remain local to the History surface and never
  discard a captured run.
- Browser fault injection is deterministic, isolated to automated test builds,
  and removed from the ordinary production path.

## Verification strategy

Each numbered outcome has a focused red-green test or executable configuration
check. After every coherent batch, run the affected package tests plus web
typecheck and lint. Completion requires a fresh clean run of:

```bash
pnpm typecheck
pnpm lint
pnpm test
pnpm build
pnpm test:perf
pnpm test:e2e
git diff --check
```

The final completion audit maps every number above to its implementation,
focused test, and acceptance evidence. No narrow focused test may be used to
claim the whole program is complete.

## Non-goals

- Implementing the previous Precision Lab production-integration plan.
- Replacing the current Build/Run shell, design system, stores, or worker.
- Changing model math, datasets, scientific metric definitions, schemas, or
  persistence limits.
- Deleting, migrating, or silently modifying existing user data.
- Adding broad analytics, accounts, cloud persistence, collaboration, or a new
  onboarding framework.
