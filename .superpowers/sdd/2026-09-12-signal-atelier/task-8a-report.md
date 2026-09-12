# Task 8A — legacy retirement and complete Setup

## Result
Retired 41 unreachable source/test files. Before deletion, resolved relative imports in every non-test web source and checked the entire removal set had no consumer outside that set. ConfigPanel, export utilities, dataset preview model/canvas, visualization models/controllers/canvas, evidence selectors, checkpoints, current/saved runs, guidance and accessible announcers remain. BuildControls was already absent. No dependencies or scientific schema/default changes were introduced.

Setup now exposes all seven canonical recipe presets through “Start from a preset” in every tab. Selecting stages a complete cloned recipe, clears obsolete raw numeric text, and preserves the original session base. Apply remains the sole publication/preparation route and waits for the exact existing worker acknowledgement; Cancel discards every staged section. Canonical presets were a real access gap left by retiring PresetPanel; the lesson library alone was not a complete substitute.

Reformatted Setup into readable JSX sections, centralized friendly scientific labels (including engine activation labels), linked numeric validation errors to stable input IDs, including canonical indexed layer paths, and retained the global cross-field list. All eleven datasets, nine features, zero–six layers, exact unclamped 1–16 neuron validation, initialization/activation variants, optimizer/schedule parameters, regularization/clipping and regression losses remain available. Support copy is 14px, numeric/select values 16px with 44px controls. Existing parent sample-preset style is included in the owned setup.css commit.

## Coverage mapping
- Old DataPanel, FeaturesPanel, NetworkConfigPanel and HyperparamPanel presentation/transaction suites: new Setup tests cover complete dataset/feature access, task contracts, zero/six-layer bounds, unclamped invalid numbers with local accessible errors, cross-tab validation/Cancel, valid high noise previews, precision retention and optimizer/schedule/loss/penalty/clipping parameter access. Existing useRecipeDraft tests retain exact publication/acknowledgement, raw incomplete text, stale base, preparation failure and sync retry assertions. Store schema/recipe transaction and engine suites remain authoritative for validation/scientific behavior.
- Old PresetPanel/PresetCard: new draft tests cover all 49 catalog source/destination transitions, no pre-Apply preparation, raw text replacement, Cancel, combined preset plus edits publication, locked submitted selection and stale-base rejection. Old immediate panel reset/callback/loading tests are obsolete because Setup has one staged transaction and shared Apply instead.
- Old DecisionBoundary wrapper: kept its entire scientific drawing/pixel/provenance test file using a test-local composition of the production model hook and production canvas. Shared controller test now composes canvas and evidence directly and retains one-canvas identity and view-command behavior; only the retired rail's expansion button assertion was removed.
- Old TrainingControls/Header/UIFlows: live App/integration, trainingLifecycle, shared save controller, keyboard shortcut and CheckpointPanel suites remain; parent owns updated transport/browser validation. Removed tests tied only to old toolbar layout/tooltips/progress widgets.
- Old ExperimentStateContext and precision recipe-strip/model: retained evidenceSelectors, network/boundary model and Atelier integration suites own scientific generation/freshness and recipe/controller state. Old closed-drawer/audience-shell text/layout tests were tied to a removed interface. Parent owns visibility/demand gates for the new mobile regions.
- Old LoadingState/FirstVisitLessonCue/PrecisionLabShell presentation-only tests retired with their unused components. No scientific computation tests removed.
- Removed only obsolete mock declarations in App.test.tsx, appShell.integration.test.tsx and training.integration.test.tsx.

## Validation
- Web unit suite: 88 files, 998 tests passed after source retirement.
- Focused Setup/useRecipeDraft/boundary suites: 4 files, 37 tests passed.
- Final Setup suite after local-error/index and complete-dataset assertion additions: 7 tests passed.
- Web source and test TypeScript checks passed.
- Repository ESLint passed (temporary formatter script removed from lint scope by retaining it as a text scratch artifact).
- git diff --check passed.
- No build, browser, performance or bundle gate run here; parent owns those final gates and caps (entry 152245, inspection 7373, total 234161 gzip).

## Removed paths
- `apps/web/src/components/layout/PrecisionLabContent.tsx`
- `apps/web/src/components/layout/PrecisionLabContent.test.tsx`
- `apps/web/src/components/layout/ExperimentStateContext.tsx`
- `apps/web/src/components/layout/ExperimentStateContext.test.tsx`
- `apps/web/src/components/layout/Header.test.tsx`
- `apps/web/src/components/layout/Header.tsx`
- `apps/web/src/components/layout/TrainingProgressBar.tsx`
- `apps/web/src/components/layout/TrainingProgressBar.test.tsx`
- `apps/web/src/components/layout/deriveVisualizationDemand.ts`
- `apps/web/src/components/layout/deriveVisualizationDemand.test.ts`
- `apps/web/src/components/layout/UIFlows.integration.test.tsx`
- `apps/web/src/components/controls/AdvancedRecipeNotice.tsx`
- `apps/web/src/components/controls/FeaturesPanel.tsx`
- `apps/web/src/components/controls/FeaturesPanel.test.tsx`
- `apps/web/src/components/controls/TrainingControls.tsx`
- `apps/web/src/components/controls/TrainingControls.test.tsx`
- `apps/web/src/components/controls/FirstVisitLessonCue.tsx`
- `apps/web/src/components/controls/FirstVisitLessonCue.test.tsx`
- `apps/web/src/components/controls/NetworkConfigPanel.test.tsx`
- `apps/web/src/components/controls/NetworkConfigPanel.tsx`
- `apps/web/src/components/controls/PresetPanel.test.tsx`
- `apps/web/src/components/controls/PresetPanel.tsx`
- `apps/web/src/components/controls/HyperparamPanel.test.tsx`
- `apps/web/src/components/controls/HyperparamPanel.tsx`
- `apps/web/src/components/controls/PresetCard.test.tsx`
- `apps/web/src/components/controls/PresetCard.tsx`
- `apps/web/src/components/controls/DataPanel.previews.test.tsx`
- `apps/web/src/components/controls/DataPanel.test.tsx`
- `apps/web/src/components/controls/DataPanel.tsx`
- `apps/web/src/components/common/LoadingState.test.tsx`
- `apps/web/src/components/common/LoadingState.tsx`
- `apps/web/src/components/visualization/PinnedBoundaryRail.tsx`
- `apps/web/src/components/visualization/DecisionBoundary.tsx`
- `apps/web/src/components/layout/precisionLab/PrecisionLabShell.test.tsx`
- `apps/web/src/components/layout/precisionLab/usePrecisionLabRecipeModel.test.tsx`
- `apps/web/src/components/layout/precisionLab/precisionLabRecipeModel.ts`
- `apps/web/src/components/layout/precisionLab/PrecisionLabRecipeStrip.test.tsx`
- `apps/web/src/components/layout/precisionLab/precisionLabRecipeModel.test.ts`
- `apps/web/src/components/layout/precisionLab/PrecisionLabShell.tsx`
- `apps/web/src/components/layout/precisionLab/usePrecisionLabRecipeModel.ts`
- `apps/web/src/components/layout/precisionLab/PrecisionLabRecipeStrip.tsx`
