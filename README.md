# Neural Network Playground 2.0

An interactive, browser-based neural network playground for learning — inspired by [TensorFlow Playground](https://playground.tensorflow.org/).

Build, train, and visualize neural networks in real time. Experiment with different architectures, datasets, and hyperparameters to develop intuition for how neural networks learn.

![Neural Network Playground screenshot](https://raw.githubusercontent.com/DenseDevKev/neural-network-playground/main/screenshot.png)

## ✨ Features

- **8 classification datasets** — Circle, XOR, Gaussian, Spiral, Moons, Checkerboard, Rings, Heart
- **2 regression datasets** — Plane, Gaussian
- **Live training** — Watch the decision boundary and loss curve update in real time
- **Mini neuron heatmaps** — See what each neuron learned inside the network graph
- **Code export** — Export your network as Pseudocode, NumPy, or TensorFlow.js
- **Advanced inspection** — Per-layer gradient magnitudes, weight stats, activation distributions
- **Presets** — One-click configurations for common learning scenarios
- **URL sharing** — Share your exact playground state via URL
- **Config import/export** — Save and load configurations as JSON
- **Responsive design** — Works on desktop, tablet, and mobile

## 🏗️ Architecture

```
neural-network-playground/
├── apps/
│   └── web/              # React + Vite frontend
│       ├── components/   # UI components (controls, visualization, layout)
│       ├── hooks/        # Custom React hooks (useTraining)
│       ├── store/        # Zustand state management
│       ├── worker/       # Web Worker for off-thread training
│       └── styles/       # CSS
├── packages/
│   ├── engine/           # Pure TypeScript neural network engine
│   │   ├── network.ts    # Network class (forward, backward, gradients)
│   │   ├── datasets.ts   # Dataset generators
│   │   ├── activations.ts
│   │   ├── losses.ts
│   │   ├── optimizers.ts
│   │   └── __tests__/    # 144+ unit tests
│   └── shared/           # Shared utilities (presets, URL state, code export)
└── legacy/               # Original implementation (preserved)
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Custom engine (no TF.js) | Full transparency — every operation explainable and debuggable |
| Web Worker training | UI stays responsive; training runs off the main thread |
| Zustand store | Lightweight, minimal boilerplate, great React integration |
| Monorepo (pnpm workspaces) | Clean separation: engine has zero DOM deps, testable in isolation |

## 🚀 Quick Start

### Prerequisites

- [Node.js](https://nodejs.org/) v18+
- [pnpm](https://pnpm.io/) v8+

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
pnpm format        # Format with Prettier
```

## 🧪 Testing

The engine package has comprehensive unit tests covering:

- **Networks** — Forward/backward pass, gradient computation, weight updates
- **Activations** — All 8 activation functions and their derivatives
- **Losses** — MSE, cross-entropy, Huber loss
- **Optimizers** — SGD, SGD+Momentum, Adam
- **Datasets** — All 10 dataset generators
- **Features** — Feature transform pipeline
- **PRNG** — Deterministic random number generation

```bash
pnpm test:engine
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and PR guidelines.

## 📄 License

[Apache 2.0](LICENSE) — inspired by [TensorFlow Playground](https://github.com/tensorflow/playground) by Daniel Smilkov & Shan Carter.
