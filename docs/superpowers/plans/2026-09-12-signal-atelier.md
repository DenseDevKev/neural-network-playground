# Signal Atelier implementation plan

User approved execution of the complete 20-screen vision on September 12, 2026. This document is the execution index; visual references and feature descriptions are in the task's outputs/nn-forge-vision gallery and vision-notes.md.

## Global constraints

- React 19, Vite, Zustand, existing engine/shared packages and V2 scientific protocols remain the foundation.
- System theme initially; explicit System/Light/Dark preference persists locally.
- One session-only draft across Dataset, Network and Training; Apply commits one complete recipe; Cancel discards all unsubmitted changes.
- Keep scientific provenance, model/recipe identity, exact pending save retry, cross-tab saved-run coordination and session checkpoint guarantees.
- One App-owned training/save/boundary/network-selection controller. App alone derives visualization demand. Local shell navigation never uses the experiment URL fragment.
- Light canvas #F7F6F2, ink #202225, secondary #666970, rule #D7D7D2, accent #D64C28, action #C74424/white. Dark canvas #17191B, controls #202326, ink #EEEDE8, secondary #BCC0C2, rule #43484D, accent/action #EF653F with dark text. Inter/Space Grotesk. Body16, labels14, headings32. Flat surfaces and square neuron maps.
- Preserve all eleven datasets, nine features, zero-six hidden layers, one-sixteen neurons/layer, eight hidden activations, four initializations, all optimizers/schedules/losses/regularization/clipping.
- Gzip caps entry152245, inspection7373, allJS234161 bytes; historical paired performance factor <=1.2. No weakened accounting.
- Original checkout is untouched. One integration PR then main CI/Pages and public verification. No partial public rollout.

### Task 1: Design system and navigation

Create shared tokens/primitives and AtelierShell. Header NN·FORGE/Playground/Saved runs/theme/Lessons/utilities. Workspace Setup/Network/Results/Inspect. Utility export/import, checkpoints, guidance, shortcuts/help. Default Network paused; optional lesson cue. Persist new local navigation key, read old layout key without deleting it, adapt old lesson/explanation navigation. System theme before first paint, live OS changes, manual override and storage failure. Document design and state ownership.

### Task 2: Unified setup draft

Implement screens02/03/04/20 with one App-owned draft controller. Dataset previews, all datasets/sample/noise/split/data seed. Network features/layers/widths/activation/init/model seed; derived output. Full training controls including optimizer params, schedules, objective, regularization, clipping. Permit incomplete numeric text and cross-field-invalid intermediate values. Canonical full-candidate validation. Apply disabled unchanged/invalid/busy; expected-base identity rejects stale drafts; one config transaction with source setup. Clear draft only after matching worker acknowledgement; preparation failures preserve input; sync failures retain existing recovery. Dirty navigation Apply/Discard/Stay and beforeunload. Existing inline shortcuts enter the editor instead of mutating active recipe. Focused tests cover cancellation, multi-tab edit, cross-field correction, stale base, duplicate submit and worker acknowledgement.

### Task 3: Network and results

Screens01/05/06/07/15/16/19: main data→hidden maps→prediction; real engine visuals, Canvas and SVG fallback, shared geometry/theme, zoom/Fit/pan/weights-activations/filter/selection. Inspector shows bias/grid statistics/signed influences and separate parameter/grid steps. Transport retains play/pause/step/reset/speed/epoch/reasons/checkpoints. Results separate EMA/full evaluations with current basis; boundary overlays and show-test/discretize; actual confusion metrics. Multiclass full tuple/three outputs; regression continuous legends and losses without classification controls. No stale evidence masquerading as current.

### Task 4: Saved runs and recovery

Screens08/09/17: bounded20-record desktop table/mobile list; name/rename/apply recipe/download/delete and max120-codepoint titles. Exactly two selections then explicit comparison, aligned configuration/evidence, differences-only and stored histories. Compare ranking only matching dataset/objective. Exact pending artifact retry/download/discard; prevent new capture until resolved. Keep legacy/rejected/incompatible recovery, cross-tab locks, errors and no automatic deletion. Confirm deletion and explain recipe reset. Retain list/selection on comparison return.

### Task 5: Inspection

Screens12/13: Trace/Activations/Gradients tabs. Sample coordinates/target/layer activations/output/loss/step; histograms/statistics with sampling provenance; worker backprop summary and local parameter grid with actual axes/counts/objectives/best offset. Pause required for one-off diagnostic calls. Preserve latest request/model guards, duplicate protection and loading/error/retry. Probes do not update weights.

### Task 6: Lessons

Screens10/11: all ten existing lessons and canonical preset refs. Library outline/summary/duration; reset notice; controller alive through lesson targets; updated targets and Apply instructions. Current-session completion, Previous/Continue/Restart/Exit; canonical concept help and explanation actions. Desktop side panel/mobile compact guide; controls reachable. Optional first-visit cue.

### Task 7: Export and checkpoints

Screens14/18: existing code exports with guarded parameters; setup JSON and URL; distinct saved evidence. Stage import before explicit apply; invalid files retain active experiment; invalid initial URL retains compatibility view. Clipboard fallback selectable text/download. Checkpoint selected metadata distinct current; explicit paused restore with parameters/optimizer-only guarantee and revision/evidence validation.

### Task 8: Responsive and qualification

All screens: >=1200 horizontal,760-1199 stacked,below760 focused. Every feature mobile-accessible,44px targets, safe area/keyboard handling, no page overflow; local graph/table scroll. Complete loading/empty/invalid/error/unavailable/recovery states, readable legends, stable layout, reduced motion/forced colors and WCAG2.2AA. Capture all20 states both themes; inspect1440/1280/1024/768/390/360 and200%zoom. Actual engine values rather than mock numbers. Tests: infrastructure helpers, lint,typecheck,allunit,build,bundle,Chromium/WebKit normal/subpath/recovery,fault-disabled rebuild,pairedperformance. Update navigation assertions without dropping scientific checks. Retire unused shell paths with consumer proof. Full integration review, PR/main/Pages/public asset verification and release receipts.
