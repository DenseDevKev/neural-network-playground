# Contributing to Neural Network Playground

Thanks for your interest in contributing! This guide covers setup, workflow, and coding standards.

## 📋 Prerequisites

- [Node.js](https://nodejs.org/) v20+
- [pnpm](https://pnpm.io/) v9+

## 🛠️ Setup

```bash
git clone https://github.com/DenseDevKev/neural-network-playground.git
cd neural-network-playground
pnpm install
pnpm dev
```

## 📁 Project structure

| Path | Purpose |
|---|---|
| `apps/web/` | React + Vite frontend |
| `packages/engine/` | Pure TypeScript neural network engine (zero DOM deps) |
| `packages/shared/` | Shared utilities (presets, URL serialization, code export) |

## 🔄 Development workflow

1. **Create a branch** from `main`
2. **Make changes** — run `pnpm dev` and test in browser
3. **Run tests** — `pnpm test` (all) or `pnpm test:engine` (engine only)
4. **Lint** — `pnpm lint`
5. **Submit a PR** with a clear description of what and why

## 📝 Coding standards

- **TypeScript** — Strict mode, no `any` unless truly necessary
- **CSS** — BEM-style class names (`.block__element--modifier`)
- **Components** — Functional React components with hooks
- **Engine** — Zero browser dependencies; must be testable in Node
- **Tests** — Add tests for new engine features; Vitest with `describe`/`it`

## 🧪 Testing

```bash
# Run all tests
pnpm test

# Run engine tests with watch mode
cd packages/engine
pnpm test:watch

# Run a specific test file
npx vitest run src/__tests__/network.test.ts
```

### Adding tests

- Engine tests go in `packages/engine/src/__tests__/`
- Use descriptive `describe` blocks and `it` names
- Use the PRNG with a fixed seed for deterministic tests

## 🏛️ Architecture decisions

| Decision | Why |
|---|---|
| Custom engine vs TF.js | Educational transparency — every operation is visible |
| Web Worker training | Keeps UI responsive during heavy computation |
| Zustand over Redux | Simpler API, less boilerplate, better for this scale |
| pnpm workspaces | Engine is testable in isolation, shared code reused cleanly |
| No SSR | This is a pure client-side interactive tool |

## 🚀 Releases

Merging to `main` automatically triggers the deployment pipeline:

1. The `.github/workflows/ci.yml` CI workflow runs lint, engine tests, and
   a production build to confirm the branch is green.
2. On success, `.github/workflows/deploy.yml` builds `apps/web/dist/` and
   deploys it to GitHub Pages.

There is no manual release step. If you need to deploy from a fork, see
[docs/deployment.md](docs/deployment.md) for the fork-and-deploy flow.

## 🙏 Code of conduct

Be respectful, constructive, and welcoming. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).
