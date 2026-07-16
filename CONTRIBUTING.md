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
3. **Run focused tests** while iterating, then `pnpm test` for the complete suite
4. **Type-check, lint, and build** the production app
5. **Run browser and performance gates** when the change affects UI, runtime, or loading behavior
6. **Submit a PR** with a clear description of what and why

## 📝 Coding standards

- **TypeScript** — Strict mode, no `any` unless truly necessary
- **CSS** — BEM-style class names (`.block__element--modifier`)
- **Components** — Functional React components with hooks
- **Engine** — Zero browser dependencies; must be testable in Node
- **Tests** — Add tests for new engine features; Vitest with `describe`/`it`

## 🧪 Testing

```bash
# Type-check the web app and imported workspace packages
pnpm --filter @nn-playground/web exec tsc --noEmit

# Run all tests
pnpm test

# Run engine tests with watch mode
cd packages/engine
pnpm test:watch

# Run a focused web test directly (do not use `pnpm test -- <file>`;
# that package command still selects the complete web suite)
NODE_OPTIONS=--experimental-require-module pnpm --filter @nn-playground/web exec vitest run src/path/to/example.test.ts --pool=forks --reporter=dot

# Build before Playwright because it serves apps/web/dist with Vite preview
pnpm build
pnpm test:e2e

# Performance-sensitive changes
pnpm test:perf
```

### Adding tests

- Engine tests go in `packages/engine/src/__tests__/`
- Web tests live beside their components or in `apps/web/src/__tests__/`
- Use descriptive `describe` blocks and `it` names
- Use the PRNG with a fixed seed for deterministic tests
- Use semantic roles, accessible names, and observable state in Playwright; do
  not mask failures with sleeps, retries, skips, or weakened assertions

See [docs/qa/QA_CHECKLIST.md](docs/qa/QA_CHECKLIST.md) for the full local and
release verification matrix.

## 🏛️ Architecture decisions

| Decision | Why |
|---|---|
| Custom engine vs TF.js | Educational transparency — every operation is visible |
| Web Worker training | Keeps UI responsive during heavy computation |
| Zustand over Redux | Simpler API, less boilerplate, better for this scale |
| pnpm workspaces | Engine is testable in isolation, shared code reused cleanly |
| No SSR | This is a pure client-side interactive tool |
| One product shell | Beginner, Explore, and Lab share state and behavior; profiles only control visibility and guidance |

Product-shell state, visibility, and terminology-extension rules are documented
in [docs/architecture/product-shell.md](docs/architecture/product-shell.md).

## 🚀 Releases

Merging to `main` automatically triggers the release pipeline:

1. `.github/workflows/ci.yml` runs lint, the complete test suite, a production
   build, and Chromium/WebKit smoke tests for the pushed `main` SHA.
2. Only a successful CI push run triggers `.github/workflows/deploy.yml`.
3. Deployment checks out that exact tested SHA, rebuilds `apps/web/dist/`, and
   publishes it to GitHub Pages.

Maintainers can still invoke the deploy workflow manually. If you need to
deploy from a fork, see [docs/deployment.md](docs/deployment.md) for the
fork-and-deploy flow.

## 🙏 Code of conduct

Be respectful, constructive, and welcoming. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).
