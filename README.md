# NN·FORGE — Signal Atelier

An interactive, browser-based neural network playground for learning — inspired by [TensorFlow Playground](https://playground.tensorflow.org/).

Build, train, and inspect neural networks in a spacious data → network → prediction workspace. Every plot and metric comes from the local training engine. No account or server computation is required.

![Neural Network Playground screenshot](https://raw.githubusercontent.com/DenseDevKev/neural-network-playground/main/screenshot.png)

## Features

- **Eleven datasets** — Nine classification datasets (including Three-Class), plus Plane and Multi-Gauss regression.
- **Unified Setup** — Edit Dataset, Network, and Training together; Apply commits one validated recipe and Cancel discards the draft.
- **Network workspace** — Square neuron activation maps, signed connection filters, local pan/zoom, and a focused neuron inspector.
- **Live training and Results** — Play/Pause, Step, Reset, speed, session checkpoints, prediction overlays, learning histories, and task-aware confusion metrics.
- **Trace, Activations, and Gradients** — Inspect real samples, activation distributions, backpropagation, and diagnostic parameter probes with provenance.
- **Local saved runs** — Save, rename, download, apply recipes, and compare two records with compatible-evidence checks and exact-artifact save recovery.
- **Ten guided lessons** — Learn from single neurons through XOR, softmax, regression, regularization, and noisy data.
- **Exports and sharing** — Pseudocode, NumPy, TensorFlow.js, configuration JSON, and shareable V2 setup links. Imported JSON is previewed before explicit Apply.
- **System, Light, and Dark** — Follow the device or remember a chosen theme; plots repaint without changing training.
- **Guidance and help** — More, Standard, or Compact explanations; every feature stays reachable.
- **Responsive layouts** — Full desktop composition, dedicated tablet graph, and focused mobile regions with 44px controls.

## 🏗️ Architecture

```
neural-network-playground/
├── apps/
│   └── web/              # React + Vite frontend
│       └── src/
│           ├── components/   # Controls, visualizations, and layout
│           ├── concepts/     # Typed terminology catalog
│           ├── productShell/ # Navigation types and legacy adapters
│           ├── hooks/        # React hooks, including useTraining
│           ├── store/        # Zustand state management
│           ├── worker/       # Web Worker integration
│           └── styles/       # CSS
├── packages/
│   ├── engine/           # Pure TypeScript neural network engine
│   │   ├── network.ts    # Network class (forward, backward, gradients)
│   │   ├── datasets.ts   # Dataset generators
│   │   ├── activations.ts
│   │   ├── losses.ts
│   │   ├── optimizers.ts
│   │   └── __tests__/    # Numerical correctness tests
│   └── shared/           # Shared utilities (presets, URL state, code export)
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Custom engine (no TF.js) | Full transparency — every operation explainable and debuggable |
| Web Worker training | UI stays responsive; training runs off the main thread |
| Zustand store | Lightweight, minimal boilerplate, great React integration |
| Monorepo (pnpm workspaces) | Clean separation: engine has zero DOM deps, testable in isolation |
| One product shell | Navigation preserves one experiment, training runtime, and scientific identity |

See [Product-shell architecture](docs/architecture/product-shell.md),
[state ownership](docs/architecture/state-ownership.md), and the
[Signal Atelier design contract](docs/architecture/signal-atelier-design.md).

## Workspace

New visitors start at Playground → Network with the default experiment paused.
Use Setup to stage a complete recipe, Results to inspect learning and prediction,
and Inspect for paused diagnostics. Saved runs and Lessons are separate header
destinations. The utility menu contains exports/imports, checkpoints, guidance,
and help. Switching views or themes never resets or pauses training.

Guidance More, Standard, and Compact preserve the former Beginner, Explore, and
Lab explanation densities. They do not hide tools. Navigation and theme are local
preferences; shared experiment URLs contain the experiment document only. Existing
local layout preferences migrate without deleting the old key or saved records.

## 🚀 Quick Start

### Prerequisites

- [Node.js](https://nodejs.org/) v20+
- [pnpm](https://pnpm.io/) v9+

### Development

```bash
# Clone the repo
git clone https://github.com/DenseDevKev/neural-network-playground.git
cd neural-network-playground

# Install dependencies
pnpm install --frozen-lockfile

# Start dev server
pnpm dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Other commands

```bash
pnpm test          # Run all tests
pnpm test:engine   # Run engine tests only
pnpm build         # Production build
pnpm test:bundle   # Enforce reviewed JavaScript gzip limits after build
pnpm lint          # Lint all files
pnpm typecheck     # Check all TypeScript packages
pnpm test:perf     # Run repository performance gates
pnpm test:e2e      # Run Chromium and WebKit against a built app
pnpm test:e2e:recovery # Exercise fault-enabled recovery; rebuild normally afterward
```

## Project status and roadmap

`main` is the authoritative released product. Execution-branch work is not a
release. The [complete Atelier plan](docs/superpowers/plans/2026-09-12-signal-atelier.md)
and [feature acceptance checklist](docs/qa/signal-atelier-acceptance.md) describe
this redesign. Historical plans and receipts remain under
[plan reconciliation](docs/superpowers/README.md); dated measurements are not
current release proof.

## 🚢 Deployment

The app is a fully static SPA — no backend and no runtime environment variables.
Once the owner has approved public publication and enabled Pages with GitHub
Actions, successful CI for a pushed `main` SHA permits deployment of that exact
tested commit. A private source repository does not make its Pages site private.
The separate release-verification workflow never publishes the application.
For self-hosting or custom base-path configuration, see
[docs/deployment.md](docs/deployment.md).

## 🧪 Testing

The repository combines engine/shared unit tests, React component and
integration tests, performance gates, and Playwright browser tests. Build before a
local end-to-end run because Playwright serves the production output through
Vite preview.

```bash
node --test scripts/*.test.mjs
pnpm typecheck
pnpm lint
pnpm test
pnpm build
pnpm test:bundle
pnpm test:perf
pnpm test:e2e
pnpm test:e2e:recovery
```

Engine coverage includes:

- **Networks** — Forward/backward pass, gradient computation, weight updates
- **Activations** — All 8 activation functions and their derivatives
- **Losses** — MSE, cross-entropy, Huber loss
- **Optimizers** — SGD, SGD+Momentum, Adam
- **Datasets** — All 11 dataset generators
- **Features** — Feature transform pipeline
- **PRNG** — Deterministic random number generation

See [QA Checklist](docs/qa/QA_CHECKLIST.md) for focused-test commands,
cross-browser coverage, accessibility checks, and release evidence expectations.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and PR guidelines.

## Typography

Inter and Space Grotesk are supplied by exactly pinned Fontsource packages and
served from the application origin. Native reloads must not depend on Google
Fonts or weaken cross-origin isolation. The original families and requested
weights remain; the production build includes `font-licenses.txt`.

## 📄 License

Application code: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
Fonts retain their [SIL Open Font License notices](apps/web/public/font-licenses.txt).
Inspired by [TensorFlow Playground](https://github.com/tensorflow/playground) by Daniel Smilkov & Shan Carter.
