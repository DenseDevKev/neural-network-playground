# Deployment Guide

NN.FORGE is a static React/Vite SPA. Training runs in a browser worker; there is
no backend, runtime API key, or required runtime environment variable.

The [roadmap](superpowers/plans/2026-09-05-nn-forge-release-roadmap.md) and
[verification record](superpowers/verification/2026-09-05-release-roadmap.md)
distinguish the authoritative `main` product, branch qualification, and an actual
published release. A successful build is not proof of a deployed site. The JavaScript gzip check
measures JavaScript only; it does not claim a total font/CSS transfer or startup-time budget.

## GitHub Pages: owner decision first

For a private repository, verify account eligibility for Pages and explicitly
approve the website's audience. Keeping source private does not automatically
make the delivered website private. Do not change repository visibility to work
around a Pages setting or entitlement failure.

After approval, the owner selects **Settings → Pages → Source → GitHub Actions**.
The initial September 5 deployment failed with `HttpError: Not Found` and
`Ensure GitHub Pages has been enabled`; that result was a repository-settings
blocker, not an application build failure. Do not alter code to conceal it.

### Existing deployment chain

A successful `CI` push run for the current `main` SHA permits
`.github/workflows/deploy.yml` to rebuild and deploy that exact tested SHA.
Failed, cancelled, pull-request, non-main, and stale-main runs are excluded by
the workflow's existing conditions. The artifact is `apps/web/dist/`.

Manual dispatch builds the selected ref and does not itself prove preceding CI.
Before a manual release, record the selected full SHA and its completed
qualification results. Do not silently substitute a newer head for a pinned
release candidate.

The separate `Release verification` workflow is read-only and does not deploy.
Its independent jobs retain failures instead of allowing one failed job to hide
other results. Main's CI success does not imply that its performance or
fault-enabled recovery checks ran; verify the actual job list.

The expected default project address is:

```text
https://densedevkev.github.io/neural-network-playground/
```

This example is not a deployment receipt. The successful deploy job's `page_url`
is authoritative. Record the actual deployment SHA, run/attempt, artifact ID and
digest, URL, timestamp, and subsequent live browser results.

## Build and self-host

Use the repository toolchain and frozen lockfile:

```bash
pnpm install --frozen-lockfile
pnpm typecheck
pnpm lint
pnpm test
pnpm build
pnpm test:bundle
```

Serve the complete `apps/web/dist/` directory. Preserve the generated asset names
and `font-licenses.txt`; do not publish a partial directory or an injected-fault
build. Inter and Space Grotesk are packaged dependencies and load from the app's
origin rather than Google Fonts. No runtime font CDN is required.

`apps/web/vite.config.ts` defaults to `base: './'`. A project URL must retain its
trailing slash so relative assets resolve below the project directory. For a
host needing an absolute base, use the existing build-time override:

```bash
VITE_BASE=/tools/nn-playground/ pnpm build
pnpm test:bundle
```

Hash state describes the experiment; it is not server-side routing. Missing JS,
CSS, or font assets must produce real HTTP errors, not an HTML fallback with
status 200. Ensure the server uses JavaScript and font MIME types correctly.

## Verify the real destination

Install Playwright's Chromium and WebKit browsers in the verification environment.
With `PLAYWRIGHT_BASE_URL` unset, `pnpm test:e2e` starts an isolated local preview.
An explicit base URL disables preview startup; an unavailable target fails rather
than falling back to a different app. Do not also set `PLAYWRIGHT_PORT`.

To exercise a non-isolated host before publication, build normally, start this
fixture in one terminal, and leave it running for the test command:

```bash
node scripts/serve-release-fixture.mjs apps/web/dist 4174
```

In another terminal:

```bash
PLAYWRIGHT_BASE_URL=http://127.0.0.1:4174/neural-network-playground/ pnpm test:e2e
```

After owner-authorized publication, run against the exact confirmed `page_url`:

```bash
PLAYWRIGHT_BASE_URL=https://densedevkev.github.io/neural-network-playground/ pnpm test:e2e
```

Use fresh automated contexts, not the owner's stored browser profile. Preserve
reports, named skips, browser versions, target metadata, and resource evidence.
The external-only deployment checks must execute on the non-isolated fixture
and selected Pages contract. An HTTP 200 alone is not browser verification.

## Isolation and transport

Development and preview deliberately set COOP/COEP headers. The worker uses
shared buffers where supported and naturally falls back to transferable messages
when the document is not isolated. Non-isolation on Pages is expected and is not
by itself a broken-worker diagnosis. Do not weaken the isolated preview to make
resource errors disappear.

Training data and model computation remain in the browser. The hosting server
receives ordinary static asset requests. Sharing/exporting is a user action that
can disclose an experiment recipe; a shared URL does not resume trained weights.

## Fault-injection builds are never release artifacts

With external target overrides unset:

```bash
pnpm test:e2e:recovery
# Preserve its report before a subsequent run overwrites it.
env -u VITE_E2E_FAULTS pnpm build
pnpm test:bundle
pnpm exec playwright test tests/e2e/worker-recovery.spec.ts --grep '@fault-disabled'
```

The clean rebuild and fault-disabled assertions are mandatory after recovery
testing. The examples use POSIX environment syntax; other shells must remove the
same variables explicitly. Never upload the recovery bundle to Pages.

## Troubleshooting and maintenance boundaries

| Observation | Required response |
|---|---|
| Pages deployment says to enable Pages | Confirm owner consent, eligibility, and the repository setting; preserve the failed run |
| Project assets return 404 | Inspect the actual `page_url`, trailing slash, emitted relative URLs, and uploaded directory |
| Font reload fails | Run the real font-delivery and recovery cases; do not mock font responses or suppress console errors |
| SharedArrayBuffer is unavailable | Check actual isolation; the transferable fallback must still train and publish paired evidence |
| Engine timing limits fail | Retain raw output and compare the pinned baseline on the same measured host; do not raise limits |
| Vite emits its 200 kB chunk warning | Record it; it is not an executable budget failure or an automatic code-splitting mandate |
| GitHub annotates Node action-runtime deprecation | Separate the action's own runtime from the Node version selected for project commands |

The existing workflows select Node 20 and pnpm 9 for project commands. Project
support remains declared by `package.json`; this work does not change it.
