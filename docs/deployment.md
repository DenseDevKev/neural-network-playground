# Deployment Guide

Neural Network Playground is a fully static single-page application (SPA).
There is no backend, no environment variables, and no build-time secrets —
the entire app runs in the browser.

## GitHub Pages (recommended)

### Prerequisites

- A GitHub account
- The repository forked to your account (or push access to the original)
- GitHub Pages enabled in the repository Settings

### Fork-and-deploy flow

1. **Fork** the repository on GitHub.
2. Go to your fork's **Settings → Pages**.
3. Under **Source**, select **GitHub Actions**.
4. Push any commit to `main` (or trigger the workflow manually via
   **Actions → Deploy to GitHub Pages → Run workflow**).
5. The `.github/workflows/deploy.yml` workflow will:
   - Install dependencies with pnpm 9
   - Build the app (`pnpm build` → `apps/web/dist/`)
   - Upload the `dist` folder as a Pages artifact
   - Deploy to `https://<your-username>.github.io/<repo-name>/`

The deploy workflow runs automatically on every push to `main`.

### Notes

- The workflow uses **Node 20** and **pnpm 9** — these match the declared
  `engines` in `package.json`.
- No environment variables or secrets are needed.
- The `vite.config.ts` sets `base: './'`, so all asset paths are relative
  and the app works correctly in any subdirectory URL.

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

Then update `vite.config.ts`:

```ts
base: process.env.VITE_BASE ?? './',
```

## Environment

- **No backend required.** All training runs entirely in the browser using
  a Web Worker. There is no API server, database, or authentication.
- **No environment variables required.** The build works with default
  settings out of the box.
- **Privacy.** No data leaves the browser — training data, weights, and
  network configurations are never transmitted to any server.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Blank page after deploy | Wrong base path | Ensure `base: './'` in `vite.config.ts` |
| Assets 404 on sub-path | Absolute asset URLs | Keep `base: './'` (relative assets) |
| Old version still showing | Browser cache | Hard-refresh or clear cache |
| Worker fails silently | COOP/COEP headers | Set `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` (required for `SharedArrayBuffer`; the app works without them but some features may be limited) |
