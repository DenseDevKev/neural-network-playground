# Neural Network Playground Feature Brainstorm

## 1. Quick Summary Of The Current Project

Neural Network Playground 2.0 appears to be a browser-based educational neural network lab inspired by TensorFlow Playground. It uses a pnpm monorepo with a React 19 + Vite web app, Zustand stores, and a custom framework-agnostic TypeScript engine. Training runs in a Web Worker through a Comlink command API plus a MessageChannel streaming protocol. The worker can stream frame-buffered predictions, parameters, neuron grids, layer stats, and confusion-matrix data while applying back-pressure so the UI remains responsive.

The app already supports 8 classification datasets and 2 regression datasets, feature toggles, hidden-layer editing, activation/loss/optimizer/regularization controls, learning-rate schedules, weight initialization, gradient clipping, presets, URL sharing, JSON config import/export, code export to pseudocode/NumPy/TF.js, a guided XOR lesson, multiple workspace layouts, decision-boundary heatmaps, loss/accuracy charts with basic diagnostics, confusion matrix, per-neuron mini heatmaps, and a layer inspection panel. Recent commits suggest the project is actively improving performance and visualization plumbing, especially around frame-buffer layout helpers, snapshot hydration, and graph rendering.

The strongest natural feature directions are educational explanations, deeper inspection/debugging, richer dataset editing, experiment comparison, and visual training controls. These fit the existing architecture because the app already separates durable configuration state, volatile runtime state, shared serialization/code-export utilities, and engine internals cleanly.

## 2. Feature Ideas Grouped By Theme

### Theme A: Visual Inspection And Debugging

1. Gradient Flow Overlay
2. Activation Distribution Explorer
3. Per-Sample Prediction Trace
4. Weight History Sparklines
5. Decision Boundary Difference View
6. Loss Landscape Probe

### Theme B: Training Controls And Time Travel

7. Training Checkpoints And Timeline Scrubber
8. Stop Conditions
9. Slow-Motion Backprop Step
10. Training Speed Profiler
11. Batch Explorer

### Theme C: Dataset Selection And Editing

12. Custom Dataset Brush
13. Dataset Parameter Lab
14. Train/Test Split Visualizer
15. Outlier And Noise Lab

### Theme D: Architecture Editing And Model Design

16. Direct-Manipulation Network Editor
17. Per-Layer Activation Controls
18. Multi-Class Classification Mode
19. Architecture Templates

### Theme E: Experiment Comparison And History

20. Experiment Run History
21. Side-By-Side Model Arena
22. Preset And Lesson Builder

### Theme F: Learning, Explanation, And Sharing

23. Guided Lesson Library
24. Contextual "Why Did This Happen?" Explanations
25. Export Experiment Report

## 3. Detailed Feature Backlog

### Theme A: Visual Inspection And Debugging

#### 1. Gradient Flow Overlay

- **User value:** Helps users see where learning is strong, weak, exploding, or vanishing across the network.
- **What the user would see or do:** Toggle a "Gradients" overlay on the network graph. Edges or neuron rings glow by recent gradient magnitude, with hover values and warnings for near-zero or very large gradients.
- **Complexity:** Medium.
- **Technical risks or dependencies:** Current `LayerStats` only exposes per-layer mean absolute gradient. Per-edge or per-neuron gradients would require extending the engine snapshot and worker frame buffer carefully to avoid large per-frame payloads.

#### 2. Activation Distribution Explorer

- **User value:** Makes saturation, dead ReLUs, and activation spread visible instead of abstract.
- **What the user would see or do:** Open an inspection tab with mini histograms per layer, plus a selected-neuron histogram over the training set. Hovering a neuron in the graph highlights its distribution.
- **Complexity:** Medium.
- **Technical risks or dependencies:** Requires collecting activation samples or histogram bins in the engine/worker. Needs demand-gated computation like `needLayerStats` so it does not slow normal training.

#### 3. Per-Sample Prediction Trace

- **User value:** Lets learners follow one point through the model and understand how the final prediction is produced.
- **What the user would see or do:** Click a data point in the decision-boundary view. A side panel shows the input features, each layer's activations, final logit/output, predicted class/value, target, and loss contribution.
- **Complexity:** Medium.
- **Technical risks or dependencies:** The engine already has forward-pass internals, but the public worker/API would need a trace method that returns intermediate activations for one selected point.

#### 4. Weight History Sparklines

- **User value:** Shows how individual weights and biases evolve, helping users connect training dynamics to network behavior.
- **What the user would see or do:** Hover or pin an edge/node in the network graph to see a tiny sparkline of that weight or bias over recent steps.
- **Complexity:** Medium.
- **Technical risks or dependencies:** Needs bounded history storage for selected parameters only. Recording all weights every frame would be memory-heavy for larger networks.

#### 5. Decision Boundary Difference View

- **User value:** Makes each training burst understandable by showing what changed, not just the latest output.
- **What the user would see or do:** Toggle "Delta" on the boundary panel to see regions that became more positive, more negative, or unchanged since the previous pause/checkpoint.
- **Complexity:** Small to Medium.
- **Technical risks or dependencies:** Requires retaining one previous output grid in the frame buffer or training store. Works best with the existing grid payload and does not require engine changes.

#### 6. Loss Landscape Probe

- **User value:** Gives power users intuition for local minima, sharp valleys, and why learning rate matters.
- **What the user would see or do:** Pause training and click "Probe landscape" to render a small 2D contour map around the current weights along two random or chosen directions. The current model appears as a marker.
- **Complexity:** Large.
- **Technical risks or dependencies:** Computationally expensive because it evaluates many perturbed networks. Should run on demand in the worker, possibly at low resolution with progress updates and cancellation.

### Theme B: Training Controls And Time Travel

#### 7. Training Checkpoints And Timeline Scrubber

- **User value:** Lets users replay learning and compare early, middle, and final model behavior.
- **What the user would see or do:** A timeline below the transport controls stores periodic checkpoints. Users scrub to a prior step, inspect the boundary/weights/loss, and optionally resume from that checkpoint.
- **Complexity:** Large.
- **Technical risks or dependencies:** Requires serializing/restoring network state in the worker and keeping memory bounded. The engine already supports serialization, but UI and worker lifecycle handling would need care.

#### 8. Stop Conditions

- **User value:** Makes experiments repeatable and prevents runaway training.
- **What the user would see or do:** Set "stop when test loss below X", "stop when accuracy above Y", "stop on plateau", or "stop on divergence". The training loop pauses automatically and reports why.
- **Complexity:** Small to Medium.
- **Technical risks or dependencies:** LossChart already computes plateau/divergence diagnostics in the UI; moving a simpler version into the worker or training store would make it reliable during background training.

#### 9. Slow-Motion Backprop Step

- **User value:** Turns one training step into an understandable sequence instead of a black box.
- **What the user would see or do:** Click "Explain next step" to animate forward pass, loss calculation, backward gradients, and weight update. The network graph highlights each phase.
- **Complexity:** Large.
- **Technical risks or dependencies:** The engine would need an instrumented training-step mode that exposes intermediate values without disrupting the optimized hot path.

#### 10. Training Speed Profiler

- **User value:** Helps users understand why some settings feel slower and gives developers a built-in performance sanity check.
- **What the user would see or do:** A small diagnostics panel shows steps/sec, frame latency, grid recompute interval, test-eval staleness, renderer mode, and WebGPU/SAB availability.
- **Complexity:** Medium.
- **Technical risks or dependencies:** Some signals already exist in comments and worker demand settings, but they need surfaced cleanly without confusing beginners. Could live behind an advanced toggle.

#### 11. Batch Explorer

- **User value:** Shows that training uses mini-batches, not the whole dataset every update.
- **What the user would see or do:** During step mode, the boundary panel highlights the current batch points. A small batch strip shows batch size, class balance, and batch loss.
- **Complexity:** Medium.
- **Technical risks or dependencies:** The worker currently owns shuffled indices and batch selection. It would need to stream the current batch indices or points at low cost.

### Theme C: Dataset Selection And Editing

#### 12. Custom Dataset Brush

- **User value:** Lets users create their own problems and immediately test whether a network can learn them.
- **What the user would see or do:** Switch to "Draw data", paint positive/negative points on the boundary canvas, erase points, adjust labels, then train on that custom dataset.
- **Complexity:** Large.
- **Technical risks or dependencies:** Requires a `custom` dataset type, serialization/URL support, import/export handling, validation, and potentially a compact encoding for shared custom datasets.

#### 13. Dataset Parameter Lab

- **User value:** Helps users learn how data geometry changes model difficulty.
- **What the user would see or do:** Advanced controls for dataset-specific parameters such as spiral turns, ring count, moon separation, class imbalance, Gaussian distance, checkerboard cells, or regression bump width.
- **Complexity:** Medium.
- **Technical risks or dependencies:** Current `DataConfig` has shared fields only. This needs a typed dataset-parameter extension and migration-safe serialization defaults.

#### 14. Train/Test Split Visualizer

- **User value:** Makes generalization concrete by showing what the model trains on versus what it is judged on.
- **What the user would see or do:** A split mode colors training points and test points differently, with a small summary of train/test counts and class balance. Users can reshuffle while keeping the same dataset settings.
- **Complexity:** Small to Medium.
- **Technical risks or dependencies:** The UI already has "Show test data" and train/test arrays. Reshuffling independently from regeneration may require separating data seed from split seed.

#### 15. Outlier And Noise Lab

- **User value:** Teaches robustness, overfitting, regularization, Huber loss, and why noisy data can mislead models.
- **What the user would see or do:** Add a few outliers, flip labels, or inject localized noise. The app can show which points dominate loss or remain misclassified.
- **Complexity:** Medium.
- **Technical risks or dependencies:** Depends on either custom dataset support or extra dataset mutation metadata. Per-sample loss display would pair well with the prediction trace feature.

### Theme D: Architecture Editing And Model Design

#### 16. Direct-Manipulation Network Editor

- **User value:** Makes architecture editing feel tangible and visual instead of form-based.
- **What the user would see or do:** Click plus buttons between layers to add layers, drag layer-size handles, remove a layer from the graph, or use keyboard controls on focused layers.
- **Complexity:** Medium.
- **Technical risks or dependencies:** Current store actions already support layer add/remove and neuron count changes. Main risk is keeping the canvas graph accessible and avoiding accidental edits while panning/zooming.

#### 17. Per-Layer Activation Controls

- **User value:** Lets advanced users compare architectures like ReLU early layers with tanh later layers.
- **What the user would see or do:** Each hidden layer has its own activation selector, visible in the graph and export output.
- **Complexity:** Large.
- **Technical risks or dependencies:** The current engine type has one hidden activation for all hidden layers. This requires changing `NetworkConfig`, engine activation dispatch, serialization, presets, code export, tests, and URL compatibility.

#### 18. Multi-Class Classification Mode

- **User value:** Expands the playground beyond binary boundaries and introduces softmax, one-hot targets, and richer confusion matrices.
- **What the user would see or do:** Choose 3-class datasets, see a multi-color decision map, train with softmax cross-entropy, and inspect a 3x3 confusion matrix.
- **Complexity:** Large.
- **Technical risks or dependencies:** The UI, datasets, prediction grid, metrics, code export, confusion matrix, and many assumptions currently center on a single output. The engine has `outputSize`, but product behavior assumes binary/regression.

#### 19. Architecture Templates

- **User value:** Gives users quick starts for common model shapes without overwhelming them.
- **What the user would see or do:** Choose templates such as "Linear", "Shallow wide", "Deep narrow", "Regularized", "Fast demo", or "Hard spiral". Each template updates layers, activation, and hyperparameters.
- **Complexity:** Small.
- **Technical risks or dependencies:** Fits existing `PRESETS` and store actions. Needs careful naming so templates complement, rather than duplicate, full presets.

### Theme E: Experiment Comparison And History

#### 20. Experiment Run History

- **User value:** Lets users compare attempts and avoid losing a promising configuration.
- **What the user would see or do:** After reset/run, save a run card with config, final metrics, best test loss, step count, and a thumbnail of the boundary. Users can restore or export a saved run.
- **Complexity:** Medium.
- **Technical risks or dependencies:** Needs localStorage/indexedDB persistence strategy, thumbnail capture, config migration, and a cap on stored runs.

#### 21. Side-By-Side Model Arena

- **User value:** Makes trade-offs obvious by comparing two models on the same dataset in real time.
- **What the user would see or do:** Duplicate current experiment into Model A and Model B, tweak one setting, then train both side by side with separate boundaries, losses, and metrics.
- **Complexity:** Large.
- **Technical risks or dependencies:** Requires either multiple workers or a worker that owns multiple network states. Store and frame-buffer architecture would need a model identity dimension.

#### 22. Preset And Lesson Builder

- **User value:** Lets educators create reusable classroom demos and share them.
- **What the user would see or do:** Save the current config as a named preset, add lesson steps with target panels and explanatory text, then export/share it as JSON or URL state.
- **Complexity:** Medium.
- **Technical risks or dependencies:** Builds on `PRESETS`, `GuidedLessonPanel`, URL sharing, and config import/export. Needs validation so user-authored lessons cannot reference missing tabs or invalid configs.

### Theme F: Learning, Explanation, And Sharing

#### 23. Guided Lesson Library

- **User value:** Helps beginners learn concepts in a deliberate sequence instead of experimenting randomly.
- **What the user would see or do:** Pick lessons like "Why XOR needs hidden layers", "Overfitting noisy data", "Learning rate too high", "Feature engineering helps", "Regularization smooths boundaries", or "Regression versus classification".
- **Complexity:** Medium.
- **Technical risks or dependencies:** Existing `GuidedLessonPanel` handles one XOR lesson. Generalizing it to multiple lesson definitions is straightforward, but content quality matters.

#### 24. Contextual "Why Did This Happen?" Explanations

- **User value:** Turns surprising outcomes into teachable moments.
- **What the user would see or do:** When loss plateaus, divergence appears, test loss rises while train loss falls, or gradients vanish, the UI offers a concise explanation and suggested knobs to try.
- **Complexity:** Small to Medium.
- **Technical risks or dependencies:** Should use deterministic rules from metrics/stats rather than remote AI. Needs careful copy so it is educational without being noisy.

#### 25. Export Experiment Report

- **User value:** Makes the playground useful for homework, demos, and sharing results.
- **What the user would see or do:** Click "Export report" to download a Markdown or HTML summary containing config, final metrics, loss chart image, boundary image, network diagram, and generated code.
- **Complexity:** Medium.
- **Technical risks or dependencies:** Requires canvas capture for visuals, stable report formatting, and maybe a shared report generator in `packages/shared`.

## 4. Prioritized Top 10 List

1. **Guided Lesson Library** - highest educational value and builds directly on the existing XOR lesson and presets.
2. **Contextual "Why Did This Happen?" Explanations** - high beginner value with modest implementation risk.
3. **Stop Conditions** - practical, focused, and fits existing training metrics.
4. **Experiment Run History** - makes the app more useful for real experimentation without changing the engine.
5. **Per-Sample Prediction Trace** - strong learning value and a natural next inspection feature.
6. **Gradient Flow Overlay** - visually powerful and aligned with existing layer stats and graph rendering.
7. **Train/Test Split Visualizer** - small/medium feature that clarifies generalization.
8. **Direct-Manipulation Network Editor** - improves UX and makes architecture feel tangible.
9. **Dataset Parameter Lab** - expands experimentation while reusing existing dataset generators.
10. **Export Experiment Report** - makes sharing and classroom use much stronger.

## 5. Three Wow Factor Features

1. **Side-By-Side Model Arena** - train two models on the same dataset and watch boundaries, losses, and gradients diverge in real time.
2. **Slow-Motion Backprop Step** - a visual movie of forward pass, loss, gradients, and weight updates for one selected step.
3. **Loss Landscape Probe** - on-demand contour visualization around the current weights to show why optimization behaves the way it does.

## 6. Three Beginner-Friendly Learning Features

1. **Guided Lesson Library** - preset-backed walkthroughs for foundational ideas like XOR, overfitting, learning rate, and regularization.
2. **Contextual "Why Did This Happen?" Explanations** - rule-based explanations when the app detects plateau, divergence, overfitting, or underfitting.
3. **Per-Sample Prediction Trace** - click one point and see how features become layer activations, prediction, and loss.

## 7. Three Advanced/Technical Features For Power Users

1. **Gradient Flow Overlay** - inspect per-layer, per-neuron, or per-edge gradient magnitudes while training.
2. **Loss Landscape Probe** - evaluate local optimization geometry around the current parameters.
3. **Per-Layer Activation Controls** - allow mixed activation architectures, with corresponding updates to serialization and export.

## 8. Three UX Improvements That Would Make The App Easier To Use

1. **Direct-Manipulation Network Editor** - edit layers and neurons from the graph instead of jumping to side controls.
2. **Experiment Run History** - save, restore, and compare recent experiments so exploration feels less disposable.
3. **Train/Test Split Visualizer** - make data partitions and class balance visible where users already inspect the boundary.

## 9. Suggested Next Implementation Order

1. Generalize `GuidedLessonPanel` into a lesson-definition system and add 4-6 lessons.
2. Add rule-based explanations for plateau, divergence, overfitting, underfitting, and stale test metrics.
3. Add stop conditions using existing metrics and a simple worker/store pause reason.
4. Add experiment run history with local persistence and restore-from-config.
5. Add train/test split visualization and optional reshuffle controls.
6. Add per-sample prediction trace through a focused worker RPC.
7. Add a gradient flow overlay, starting with layer-level stats and expanding only if performance allows.
8. Add direct graph editing for layer count and neurons per layer.
9. Add dataset-specific parameter controls for the existing synthetic generators.
10. Add exportable experiment reports once the comparison/history concepts are stable.

This order starts with high-value features that mostly reuse existing UI and data contracts, then moves toward features that require deeper worker, engine, or serialization changes.
