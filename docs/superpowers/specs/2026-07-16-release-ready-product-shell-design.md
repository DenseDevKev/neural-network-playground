# Release-Ready Product Shell Design

**Date:** 2026-07-16

**Status:** Approved implementation design

**Scope:** Advanced Tools, terminology, audience profiles, and release validation

> **Implementation handoff:** This is the approved design, not a release-result
> record. Current behavior and extension rules live in
> [`../../architecture/product-shell.md`](../../architecture/product-shell.md),
> implementation gates remain in
> [`../plans/2026-07-16-release-ready-product-shell.md`](../plans/2026-07-16-release-ready-product-shell.md),
> and final measurements are added only after release verification.

## Intent

Make NN·FORGE easier to enter without splitting it into separate products. Beginner, Explore, and Lab are visibility and guidance profiles over one shell, one experiment document, and one training runtime. Advanced Tools is a progressive-disclosure boundary within that shared shell. The terminology catalog supplies the same accurate definitions wherever the shell explains a concept.

This design completes the product-shell roadmap after the InspectionPanel and DecisionBoundary adapter pilots. It does not change engine mathematics, the worker protocol, V2 experiment documents, URL encoding, checkpoints, or saved-run formats.

## Baseline and browser diagnosis

The pre-change validation baseline is:

- `pnpm lint`: pass.
- `pnpm test`: pass; 1,587 tests (engine 480, shared 325, web 782).
- `pnpm build`: pass; main chunk 141,153 bytes gzip, Inspection chunk 5,496 bytes gzip, total JavaScript 219,883 bytes gzip.
- `pnpm test:e2e`: pass; all six Chromium/WebKit training, preset, checkpoint, and saved-run scenarios pass unchanged in 12.1 seconds on an uncontended machine.
- The first performance run had one 1.36% threshold miss while `syspolicyd`, `trustd`, Codex, and WindowServer heavily contended for CPU. Performance must therefore be rerun repeatedly on an idle host and judged from the recorded medians, not from that contaminated sample.

The earlier browser failures were environment-specific time-budget exhaustion, not a reproduced browser incompatibility. WebKit timed out during `page.goto('/')` even though localhost responded in milliseconds; Chromium progressed through the workflows but exhausted the same global budget during later actions. The unchanged suite now passes both engines. We will preserve its semantic assertions and avoid retries, skips, and longer sleeps. New shell coverage will wait on observable application state.

## State and compatibility boundaries

- `usePlaygroundStore` continues to own the shareable V2 experiment document, preparation/import/URL compatibility, and visualization-demand delivery cache.
- `useTrainingStore` continues to own runtime status, evidence references, configuration synchronization, checkpoints, errors, and session speed.
- The worker remains the authoritative producer of scientific artifacts.
- `frameBuffer` remains the accepted main-thread typed-array/provenance cache.
- `useLayoutStore` owns only local workspace navigation: Build/Run, selected modules/evidence, code tab, audience profile, and Advanced Tools disclosure.
- `experimentMemoryStore` continues to own saved, rejected, and pending run artifacts.
- Profile and disclosure preferences are persisted only in `nn-playground-layout`. They are not added to the URL, V2 document, checkpoints, exports, or saved runs.
- Existing deprecated layout aliases remain synchronized for compatibility.

Persisted layout hydration is additive and sanitized. Missing or invalid profiles become Explore. Missing or invalid disclosure state uses the profile default. A valid explicit Lab collapse survives reload. If a stored target requires Advanced Tools, hydration opens the disclosure rather than discarding that target.

## Shared capability model

The shell uses a pure profile table rather than scattered profile checks:

```ts
type AudienceMode = 'beginner' | 'explore' | 'lab';
type BuildModuleId = 'data' | 'network' | 'features' | 'hyperparams' | 'config';

interface AudienceProfile {
    label: string;
    description: string;
    coreBuildModules: readonly BuildModuleId[];
    coreEvidenceViews: readonly EvidenceViewId[];
    advancedDefaultOpen: boolean;
    guidanceLevel: 'high' | 'standard' | 'compact';
}
```

Profile defaults:

| Profile | Core Build modules | Core evidence | Advanced default | Guidance |
|---|---|---|---|---|
| Beginner | Data, Network | Boundary, Loss | Closed | High |
| Explore | Data, Network, Features, Hyperparameters | Boundary, Loss, Confusion | Closed | Standard |
| Lab | Data, Network, Features, Hyperparameters | Boundary, Loss, Confusion | Open | Compact |

Current Recipe/Run, Topology, transport, Presets, Lessons, and History stay available in every profile. Advanced Tools exposes the complete applicable union: Features, Hyperparameters, Configuration, Confusion, Inspect, and Code. Hidden components are unmounted, but their recipe values are never reset. Current Recipe remains visible and therefore keeps hidden configured values discoverable.

Changing profile updates only `audienceMode` and that profile's disclosure default. It must not mutate the experiment recipe, URL, runtime, checkpoints, saved runs, export state, or drawer state. Closing Advanced Tools while an advanced target is active atomically falls back to Data in Build or Boundary in Run and synchronizes legacy aliases. Direct lesson or explanation navigation to a hidden target opens Advanced Tools without changing profile.

## Advanced Tools interaction

Advanced Tools is an inline shell disclosure, not the old miscellaneous More drawer.

- The Header exposes one button with `aria-expanded` and `aria-controls`.
- The controlled region contains a short description before advanced modules or evidence tabs.
- In Build, opening reveals the profile-hidden build modules plus Configuration.
- In Run, opening reveals profile-hidden evidence plus Inspect and Code.
- Opening alone does not request diagnostic worker artifacts; demand follows the resolved, visible evidence view only.
- The evidence tablist contains only visible tabs. It uses roving `tabIndex`, Arrow Left/Right, Home, and End.
- Inspect and Code navigation opens Advanced Tools atomically.
- Explicit collapse or Escape returns focus to the disclosure trigger before advanced content unmounts. Hydration never steals focus.
- Motion is optional and disabled under `prefers-reduced-motion: reduce`.
- Responsive layouts keep the disclosure and profile switch reachable at 200% zoom and narrow viewports.

`resolveVisibleEvidenceView` is the single resolver for rendering, accessibility context, and visualization demand. Legacy `history` resolves to Boundary. The unused history-drawer demand argument is removed. Production-consumer searches found no legacy direct InspectionPanel path outside the App shell, so App is the sole visibility-demand owner.

## Terminology catalog

The catalog is a React-free typed module with stable identifiers and no provider, CMS, or i18n framework:

```ts
type ConceptId =
    | 'data-loss'
    | 'training-objective'
    | 'decision-boundary'
    | 'activation'
    | 'gradient'
    | 'checkpoint';

interface ConceptEntry {
    id: ConceptId;
    canonicalTerm: string;
    plainDefinition: string;
    extendedExplanation?: string;
    aliases: readonly string[];
    related: readonly ConceptId[];
    profiles: readonly AudienceMode[];
    difficulty: 'beginner' | 'intermediate' | 'advanced';
    examples?: readonly string[];
    uiTarget?: RecipeSectionId | EvidenceViewId;
    documentationUrl?: string;
}
```

Pure APIs provide exact ID lookup, case-insensitive canonical/alias lookup, profile filtering, and a stable ordered list. Type construction plus tests validate unique IDs and aliases, valid related IDs, and complete entries.

An accessible `ConceptHelp` disclosure integrates definitions into repeated high-value surfaces such as Current Run, Loss, decision-boundary context, Inspection, experiment state, and Header metrics. The trigger is a real button with an accessible name and `aria-expanded`; content is available on click/focus and never hover-only. The plain definition is always present in the DOM while open. Extended explanations and examples are optional. Existing scientific labels stay precise; profile guidance changes density, not the underlying term or calculation.

The initial six concepts intentionally bound payload and editorial risk while providing a complete extension pattern. Search/onboarding can consume the pure catalog APIs later without creating another source of truth.

## Accessibility and safety

- Profile selection is a labeled native select and has adjacent text explaining that modes change visible tools only.
- Profile changes are announced through a polite live region.
- Disclosure state, controlled region, focus restoration, roving tabs, and Escape behavior are covered by component tests.
- Every profile receives automated axe coverage in representative Build and Run states.
- Keyboard-only, narrow viewport, zoom/text-resize, touch target, contrast, focus visibility, and reduced-motion checks are documented in the release QA record.
- No profile weakens validation or training safety. Experimental/destructive operations, if introduced later, must remain explicitly labeled; this slice does not invent such operations.

## Performance and loading

Profile tables and the catalog are small static modules. Advanced visualization components retain their existing lazy boundaries; hidden advanced content is not mounted. The implementation avoids duplicating UI trees per profile and avoids copying typed arrays.

Release gates compare exact zlib gzip sizes to the recorded baseline and retain the pilot budgets: Inspection lazy gzip no more than +2 KiB, main entry no more than +1 KiB, total JavaScript no more than +2%. Performance commands are run repeatedly on an idle host, with medians reported and any contaminated samples identified rather than hidden.

## Verification contract

Focused TDD precedes each production slice. Final acceptance requires lint, all package tests, production build, repeated performance gates, Chromium/WebKit smoke, profile/disclosure cross-browser invariants, axe checks, responsive/keyboard checks, `git diff --check`, consumer searches, and documentation updates. No test may be skipped or weakened to obtain a pass.
