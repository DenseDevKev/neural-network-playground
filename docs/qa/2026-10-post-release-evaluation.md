# Post-release evaluation — October 2026

Carries out section 20 of the [living execution plan](../../NN-FORGE-LIVING-EXECUTION-PLAN.md) against the Signal Atelier build on `main`.

## Scope and honesty boundary

- Date: 2026-10-02. Source: `main` at `ec3a6194b1714ceb9cad1ad996300f52a8b91c2d` (PR #38 Signal Atelier plus PR #39; the later commit only removes a brainstorm file).
- Method: `pnpm build`, `vite preview`, and scratch Playwright scripts kept outside the repository, driving Chromium 1194 at 1440x900, 390x844 and 360x844 (touch emulation). Chromium only; WebKit was not run here.
- **This is an agent walkthrough, not user research.** It is not evidence of real usage, learner comprehension, or frequency of problems. The plan's "Gather actual usage/feedback evidence" and "Rank problems by frequency" items remain open (`[ ]`). Severities below are the walkthrough author's estimates of impact, not measured frequencies.
- Build health: `pnpm build` succeeded; `pnpm test:bundle` reported entry 150,841 / 152,245, inspection 6,591 / 7,373, total JavaScript gzip 234,088 / 234,161 bytes, matching the PR #39 figures. The deployed site was not visited (the page URL is not recorded in the repository).

## Bundle-budget context

Total JavaScript headroom is **73 bytes** (entry 1,404, inspection 782 bytes). Any new JavaScript feature, copy, or component will fail `pnpm test:bundle` unless something else shrinks, and the caps may not be raised without separate justification (plan section 21). Consequences for the findings below: items that are CSS, static assets or deletions fit today; items that add JavaScript need the separate bundle-size work to land first. Recommendations mark which kind each is. (Numbers for the parallel bundle-size and test-runner branches are not recorded here; see their PRs.)

## Walkthrough of the 12 evaluation items

| # | Item | Result |
|---|---|---|
| 1 | Experiment creation | Works. Setup has Data / Inputs & layers / Training sections, 11 datasets, sample counts, noise, split and seed with a deterministic preview, plus six presets. One shared draft with Apply/Cancel; leaving Setup with a pending draft raises an "Apply your setup changes?" dialog (Stay / Discard / Apply). See F2. Screenshots: [setup](browser-qa/2026-10-01-setup-dataset.png), [guard](browser-qa/2026-10-06-draft-guard.png). |
| 2 | Architecture construction | Works. Added a hidden layer and switched activation to Tanh; Apply restarted training at step 0 and the summary became `X₁, X₂ -> [4] -> [4] -> [4] -> 1 output`. Per-layer neuron steppers, 9 input features, 8 activations, 4 initializers. Screenshots: [draft](browser-qa/2026-10-02-architecture-draft.png), [applied](browser-qa/2026-10-02-architecture-applied.png). |
| 3 | Training controls | Works. Play/Pause, Step, Reset and Speed (1×–50×) behave consistently; status text moves Ready → Training → "Paused manually"; Step advances exactly one step (9,600 → 9,601). Screenshot: [controls](browser-qa/2026-10-03-training-controls.png). See F1 for the default experiment's slow start. |
| 4 | Evidence comprehension | Good. Results → Learning progress shows batch EMA, full-split train/test loss and objective on shared axes; the current-run card names the evidence ("Full evaluation 195 at step 9,601"). Screenshot: [learning progress](browser-qa/2026-10-04-learning-progress.png). |
| 5 | Train/test distinction | Clear in labels ("Train data loss (full split)", "Test data loss (full split)", "150 training samples / 150 held-out samples", confusion matrix "full test split"); a gap value (+0.0300) is shown. The prediction grid is labelled "A sampled field, not a full-split evaluation." Screenshot: [errors & confusion](browser-qa/2026-10-05-errors-confusion.png). |
| 6 | Evaluation age / recipe drift | Age is explicit while running ("Batch trend through step 291" next to "Full evaluation 6 at step 250") and collapses to "Evaluation matches the current step" when paused. Recipe drift ("Current recipe differs from trained snapshot.") is covered by `apps/web/src/components/controls/CurrentRunCard.test.tsx`; the walkthrough could not provoke it from the UI because Apply restarts training and clears evidence. Drift is therefore unobserved in the browser here, not verified absent. |
| 7 | Network inspection | Works. Selecting Hidden 1 · neuron 2 shows bias, activation statistics, and strongest signed inputs/outputs with the step basis ("Weights at step 910; activation grid at step 910"). The Inspect tab offers sample traces; it needs a chosen sample before showing values. Screenshots: [selection](browser-qa/2026-10-07-network-selection.png), [inspect](browser-qa/2026-10-07-inspect.png). |
| 8 | Save / compare | Works with friction (F4, F5). Two saved runs (SGD vs Adam) compared: headline "has lower test data loss by 0.7098", shared-axes stored histories, differences table (optimizer differs, test accuracy 100% vs 44.7%). Screenshots: [saved runs](browser-qa/2026-10-08-saved-runs-two.png), [compare](browser-qa/2026-10-08-compare-two-runs.png). |
| 9 | Code export | Works. Pseudocode, NumPy and TF.js tabs show the current architecture and training settings with a parameter-snapshot note ("Parameter snapshot: step 0", "Evaluation evidence downloads are in Saved runs"). Screenshot: [code](browser-qa/2026-10-09-code-export.png). |
| 10 | Sharing | Works. The URL carries a V2 recipe fragment (936 characters for a small network); a fresh browser context opened it and showed the same architecture at step 0 with a fresh model. Clipboard permission was not available to the scratch script, so "Copy setup link" output was not read. See F3. Screenshots: [share dialog](browser-qa/2026-10-10-share-dialog.png), [fresh context](browser-qa/2026-10-10-shared-fresh-context.png). |
| 11 | Mobile 390px | No page-level horizontal overflow on Setup, Network, Results (including trained) or Inspect; transport usable by touch; the only sub-44px interactive elements are checkbox inputs whose labels measure 328x44. See F6. Screenshots: [results](browser-qa/2026-10-11-mobile-390-results-trained.png), [setup](browser-qa/2026-10-11-mobile-390-setup-training.png), [network](browser-qa/2026-10-11-mobile-390-network.png), [inspect](browser-qa/2026-10-11-mobile-390-inspect.png). |
| 12 | Mobile 360px | Same overflow and target results as 390px. Visible layout defects: F6. Screenshots: [results](browser-qa/2026-10-11-mobile-360-results-trained.png), [setup](browser-qa/2026-10-11-mobile-360-setup-training.png), [network](browser-qa/2026-10-11-mobile-360-network.png), [inspect](browser-qa/2026-10-11-mobile-360-inspect.png). |

## Findings

No blocking defects were found. Console output on every load contained one error (F7).

### F1 — Default experiment shows almost no learning for the first minute (medium, product/pedagogy)

- Repro: load `/`, press Play at the default 5× speed, open Results.
- Observed: the default Circle experiment (SGD, learning rate 0.03) reads training loss 0.6838, test loss 0.7182 and test accuracy 44.7% (all predictions in one class) at step 890; at 1× speed, 294 steps took about 4 seconds with accuracy still 44.7%. After about 9,600 steps accuracy reached 100%. Switching the optimizer to Adam reached loss 0.0030 by step 910. The app's own labels are honest, but a first-time learner sees a long plateau.
- Walkthrough estimate only; whether real learners abandon here is unknown.
- Budget: copy or lesson changes are JavaScript bytes; changing defaults would alter recipe identity and URLs and needs separate scientific/schema review. Not recommended without owner approval.

### F2 — Leaving Setup with a draft shows a modal, but sibling-tab changes were easy to lose track of (low)

- Repro: Setup → Training, change Optimizer, click Results.
- Observed: the guard dialog appears with Stay / Discard changes / Apply changes (works as designed; the unified draft is a deliberate decision). Noted only because Cancel and the guard are the sole ways to drop changes across three sections.
- No action proposed.

### F3 — "Share setup" opens whichever Export/import tab was used last (low-medium)

- Repro: Utilities → Export / import → Code → TF.js; close; click "Share setup".
- Observed: the dialog opened on the Code tab showing TF.js source, not "Setup & sharing" with the link controls. A user looking for the link must find the "Setup & sharing" tab.
- Likely fix is small logic (a few JavaScript bytes), so it depends on bundle headroom.

### F4 — Saved-run default names omit the optimizer (low)

- Repro: save an SGD run and an Adam run of the same architecture.
- Observed: both are named `Circle · 2-4-4-1 · step N` (only the step differs). Distinguishing runs in the list requires opening Compare or renaming. The compare table does list optimizer differences.
- Naming text is JavaScript bytes.

### F5 — "Save run" navigates away before saving (low)

- Repro: click Save run in the header.
- Observed: opens the Saved runs page and requires a second "Save current run" click; the page header still reads "Circle experiment". Matches the saved-run design (name entry) but may surprise a user expecting a one-click save. Needs usability evidence before changing.

### F6 — Mobile transport/status labels crowd at 360px (low-medium, visual)

- Repro: 360x844, train briefly, open Results ([screenshot](browser-qa/2026-10-11-mobile-360-results-trained.png)).
- Observed: "Step 755" and "Epoch 50" touch with no gap and sit beside "Paused manually"; the sentence "Full evaluation at step 755. Evaluation matches the current step." renders with an uneven gap. Tabs "Errors & confusion" clip (horizontally scrollable, acceptable). No page overflow.
- CSS-only fix is plausible and does not count against JavaScript caps; initial CSS grew 1,290 gzip bytes in PR #39 and has no cap in `pnpm test:bundle` output, but should be watched.

### F7 — `/favicon.ico` returns 404 on every load (low)

- Repro: load the preview with devtools open; the console shows `Failed to load resource: ... 404`. `apps/web/index.html` declares no icon and `apps/web/public/` contains only `font-licenses.txt`.
- A static asset or an inline `<link rel="icon">` data URI resolves it; no JavaScript change.
- No spec or script in the repository mentions a favicon; whether the browser specs already filter this console message was not checked.

### Observations that need no action

- Recipe drift could not be provoked from the UI (item 6); unit coverage exists.
- 14x14 checkbox inputs on Results have 328x44 labels, so touch targets are met.
- Header, tab and transport targets measured at or above 44px on both mobile widths.

## Bundle-budget implications

| Finding | Kind of change | Fits in 73 bytes? |
|---|---|---|
| F7 favicon | static asset or HTML link | Yes (no JavaScript) |
| F6 mobile spacing | CSS | Yes for JavaScript caps; CSS has no cap but should be reported |
| F3 share tab | small logic | Probably not; needs headroom |
| F4 run names | string logic | Probably not; needs headroom |
| F1 / F5 | copy, flow or defaults | No; needs headroom plus owner/scientific review |

## Proposed next milestone (needs owner approval)

**Milestone: "First-run and share polish" — a small, bounded pass on the first minutes of use, after the bundle-size work gives JavaScript headroom.**

- Contents: F7 favicon, F6 mobile 360px spacing, F3 Share opens the link tab, F4 optimizer in default run names; F1 and F5 get a documented decision (leave as is, or propose copy) instead of silent change.
- Preconditions: owner approval; the parallel bundle-size work merged with its new headroom cited from its PR; actual usage observation from at least a few real users to confirm or reorder these findings (the plan's usage-evidence item stays open).
- Exit criteria: each item has a regression test or screenshot on `main`, unchanged JavaScript caps pass, and no schema, protocol or scientific default changes.
- Not proposed: new features, new datasets, default-recipe changes, or any cap increase.

This is a proposal only; the owner decides.
