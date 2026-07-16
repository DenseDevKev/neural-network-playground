# Neural Network Playground 2.0

An interactive, browser-based neural network playground for learning — inspired by [TensorFlow Playground](https://playground.tensorflow.org/).

Build, train, and visualize neural networks in real time. Experiment with different architectures, datasets, and hyperparameters to develop intuition for how neural networks learn.

![Neural Network Playground screenshot](https://raw.githubusercontent.com/DenseDevKev/neural-network-playground/main/screenshot.png)

## ✨ Features

- **9 classification datasets** — Eight binary datasets plus Three-Class Clusters
- **2 regression datasets** — Plane, Gaussian
- **Live training** — Watch the decision boundary and loss curve update in real time
- **Mini neuron heatmaps** — See what each neuron learned inside the network graph
- **Code export** — Export your network as Pseudocode, NumPy, or TensorFlow.js
- **Advanced inspection** — Per-layer gradient magnitudes, weight stats, activation distributions
- **Beginner, Explore, and Lab profiles** — Change visible tools and guidance without changing the experiment
- **Advanced Tools** — Reveal specialist configuration, diagnostics, and export surfaces on demand
- **Contextual concept help** — Consistent definitions for loss, objectives, boundaries, activations, gradients, and checkpoints
- **Presets** — One-click configurations for common learning scenarios
- **URL sharing** — Share your exact playground state via URL
- **Config import/export** — Save and load configurations as JSON
- **Responsive design** — Works on desktop, tablet, and mobile

## 🏗️ Architecture

```
neural-network-playground/
├── apps/
│   └── web/              # React + Vite frontend
│       └── src/
│           ├── components/   # Controls, visualizations, and layout
│           ├── concepts/     # Typed terminology catalog
│           ├── productShell/ # Profiles, shell types, and visibility rules
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
│   │   └── __tests__/    # 144+ unit tests
│   └── shared/           # Shared utilities (presets, URL state, code export)
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Custom engine (no TF.js) | Full transparency — every operation explainable and debuggable |
| Web Worker training | UI stays responsive; training runs off the main thread |
| Zustand store | Lightweight, minimal boilerplate, great React integration |
| Monorepo (pnpm workspaces) | Clean separation: engine has zero DOM deps, testable in isolation |
| One product shell | Profiles control visibility and guidance over one experiment and runtime |

See [Product-shell architecture](docs/architecture/product-shell.md) for profile,
Advanced Tools, persistence, and terminology-extension rules.

## 🧭 Experience profiles

Profiles are flexible workspace views, not separate products or permission tiers.
Switching profile never changes the recipe, model, training step, checkpoints,
saved runs, exports, or share URL.

| Profile | Core Build tools | Core Run evidence | Advanced Tools default |
|---|---|---|---|
| Beginner | Data, Network | Boundary, Loss | Closed |
| Explore | Data, Network, Features, Hyperparameters | Boundary, Loss, Confusion | Closed |
| Lab | Data, Network, Features, Hyperparameters | Boundary, Loss, Confusion | Open |

Explore is the default. Advanced Tools always provides the complete applicable
set of additional configuration, confusion, inspection, and code-export tools.
Profile and disclosure preferences are local workspace settings; they are not
included in shared experiment URLs.

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
pnpm install

# Start dev server
pnpm dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

### Other commands

```bash
pnpm test          # Run all tests
pnpm test:engine   # Run engine tests only
pnpm build         # Production build
pnpm lint          # Lint all files
pnpm test:perf     # Run repository performance gates
pnpm test:e2e      # Run Chromium and WebKit smoke tests against a built app
```

## 🚢 Deployment

The app is a fully static SPA — no backend and no runtime environment variables.
After CI passes for a pushed `main` SHA, GitHub Actions rebuilds and deploys
that exact tested commit to GitHub Pages.
For self-hosting or custom base-path configuration, see
[docs/deployment.md](docs/deployment.md).

## 🧪 Testing

The repository combines engine/shared unit tests, React component and
integration tests, performance gates, and Playwright smoke tests. Build before a
local end-to-end run because Playwright serves the production output through
Vite preview.

```bash
pnpm --filter @nn-playground/web exec tsc --noEmit
pnpm lint
pnpm test
pnpm build
pnpm test:perf
pnpm test:e2e
```

Engine coverage includes:

- **Networks** — Forward/backward pass, gradient computation, weight updates
- **Activations** — All 8 activation functions and their derivatives
- **Losses** — MSE, cross-entropy, Huber loss
- **Optimizers** — SGD, SGD+Momentum, Adam
- **Datasets** — All 10 dataset generators
- **Features** — Feature transform pipeline
- **PRNG** — Deterministic random number generation

See [QA Checklist](docs/qa/QA_CHECKLIST.md) for focused-test commands,
cross-browser coverage, accessibility checks, and release evidence expectations.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and PR guidelines.

## 📄 License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) — inspired by [TensorFlow Playground](https://github.com/tensorflow/playground) by Daniel Smilkov & Shan Carter.
