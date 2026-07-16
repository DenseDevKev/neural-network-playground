# Deployment Guide

Neural Network Playground is a fully static single-page application (SPA).
There is no backend, no runtime environment variables, and no build-time
secrets — the entire app runs in the browser.

## GitHub Pages (recommended)

### Prerequisites

- A GitHub account
- The repository forked to your account (or push access to the original)
- GitHub Pages enabled in the repository Settings

### Fork-and-deploy flow

1. **Fork** the repository on GitHub.
2. Open your fork's **Actions** tab and enable workflows for the fork.
3. Go to your fork's **Settings → Pages**.
4. Under **Source**, select **GitHub Actions**.
5. Push a commit to `main`. The `CI` workflow must pass lint, tests, build, and
   Chromium/WebKit smoke for that commit.
6. A successful CI push run allows `.github/workflows/deploy.yml` to build and
   deploy. It checks out the exact tested SHA and will:
   - Install dependencies with pnpm 9
   - Build the app (`pnpm build` → `apps/web/dist/`)
   - Upload the `dist` folder as a Pages artifact
   - Deploy to `https://<your-username>.github.io/<repo-name>/`

Maintainers can also trigger **Actions → Deploy to GitHub Pages → Run
workflow**. A manual dispatch builds the explicitly selected ref and is the
intentional escape hatch for forks or recovery; it does not claim a preceding
CI result.

Automatic deployment does not race CI: failed, cancelled, pull-request, and
non-`main` CI runs cannot start the deploy job. A completed CI run for a stale
`main` SHA is also ignored when a newer commit has already reached `main`.

### Notes

- The workflow uses **Node 20** and **pnpm 9** — these match the declared
  `engines` in `package.json`.
- Automatic deploys rebuild the exact `workflow_run.head_sha` that passed CI;
  they never silently deploy a newer untested `main` commit.
- No secrets are needed.
- The `vite.config.ts` uses `VITE_BASE` when provided and defaults to
  `base: './'`, so all asset paths are relative and the app works correctly
  in any subdirectory URL.

## Self-hosting on any static file server

The production build output is a standard set of static files in
`apps/web/dist/`. Any web server that can serve static files works
(Nginx, Apache, Caddy, S3 + CloudFront, Netlify, Vercel, etc.).

### Build locally

```bash
# Install dependencies (Node >= 20, pnpm >= 9 recommended)
pnpm install

# Produce a production build
pnpm build
```

The output lands in `apps/web/dist/`. Copy that directory to your host.

### Nginx example

```nginx
server {
    listen 80;
    server_name example.com;
    root /var/www/neural-network-playground;
    index index.html;

    # All routes fall back to index.html (hash routing handles the rest)
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

### Serving from a sub-path

The default `vite.config.ts` sets `base: './'`, which makes all asset
URLs relative. This means the app works whether it is hosted at
`https://example.com/` or `https://example.com/tools/nn-playground/`
without any extra configuration.

If you need an **absolute** base path (e.g. for a reverse proxy that
rewrites paths), override it at build time:

```bash
VITE_BASE=/tools/nn-playground/ pnpm build
```

The build already reads `VITE_BASE`, so no config edit is needed.

## Environment

- **No backend required.** All training runs entirely in the browser using
  a Web Worker. There is no API server, database, or authentication.
- **No runtime environment variables required.** The build works with default
  settings out of the box. `VITE_BASE` is optional build-time configuration
  for hosts that require an absolute asset base path.
- **Privacy.** No data leaves the browser — training data, weights, and
  network configurations are never transmitted to any server.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Blank page after deploy | Wrong base path | Ensure `base: './'` in `vite.config.ts` |
| Assets 404 on sub-path | Absolute asset URLs | Keep `base: './'` (relative assets) |
| Old version still showing | Browser cache | Hard-refresh or clear cache |
| Worker fails silently | COOP/COEP headers | Set `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` (required for `SharedArrayBuffer`; the app works without them but some features may be limited) |
