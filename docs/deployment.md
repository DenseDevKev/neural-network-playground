# Deployment and release operations

NN.FORGE is a static Vite application. The authoritative deploy path is the checked-in `.github/workflows/deploy.yml`; do not create a parallel ad-hoc deployment mechanism.

## Release chain

The intended sequence is:

1. qualify the exact release branch SHA with `Release verification`;
2. review/merge that qualified SHA into `main`;
3. require normal `main` CI to pass;
4. let `deploy.yml` consume the exact successful CI SHA;
5. rebuild that tested SHA;
6. upload the Pages artifact;
7. deploy through `actions/deploy-pages`;
8. record GitHub's actual `page_url` and deployment provenance;
9. run external Chromium/WebKit checks against that exact URL.

A branch qualification is not deployment, and an expected URL is not evidence of a successful deploy.

## GitHub Pages configuration

The repository's deployment workflow uses GitHub Actions as the Pages source. Repository Pages settings must permit that workflow to create/update the site.

Publishing a GitHub Pages site is an external release decision. A private source repository does not by itself make the Pages website private; confirm the intended audience before enabling publication.

Do not make the source repository public merely to work around Pages configuration.

## Project base path

`apps/web/vite.config.ts` defaults to relative assets (`base: './'`) unless `VITE_BASE` is explicitly supplied. The checked-in external fixture validates actual production bytes beneath:

`/neural-network-playground/`

Do not assume a root-hosted application during release validation.

## Pre-merge release qualification

The exact candidate SHA must have a green release-verification run with:

- source-evidence;
- correctness;
- preview browsers;
- project-subpath browsers;
- recovery;
- paired performance qualification.

Correctness includes helper tests, lint, typecheck, all package tests, build, JavaScript bundle limits, and tracked-source cleanliness.

### Performance semantics

Local `pnpm test:perf` retains the historical development-machine absolute engine constants. Those constants are not rewritten during deployment/release qualification.

The release workflow restores the documented same-machine regression contract by running the exact intake baseline and candidate five times each on one Apple Silicon runner. Candidate engine medians must remain within +20% of baseline medians, while forced paired evaluation and save capture remain fixed <=250/500 ms budgets.

A slow host that causes both individual revisions to miss the frozen development-machine constants is not by itself a release regression. Conversely, the release still fails if the paired comparator exceeds +20% or either worker budget is exceeded.

## Build provenance

The release-verification correctness job records:

- exact source SHA;
- Node and pnpm versions;
- SHA-256 manifest of every production dist file;
- a retained build artifact.

The deployment workflow should rebuild the exact `main` SHA whose CI succeeded. Record the deployment run, source SHA, artifact/digest, and returned `page_url`.

## Static hosting expectations

GitHub Pages normally does not provide the local preview's COOP/COEP headers. NN.FORGE must therefore work through its transferable/non-isolated worker path on Pages. The checked-in project-subpath fixture deliberately omits those isolation headers and is the pre-publication approximation of that hosting mode.

Do not add fake isolation headers to tests merely to make the environment easier than the deployment target.

## Font delivery

Inter and Space Grotesk are bundled and served from the application origin. Production must not depend on Google Fonts CSS or `fonts.gstatic.com`.

Live verification should confirm:

- expected local font resources return successfully beneath the project base;
- no Google font requests occur;
- no font-related console/page errors occur through reload.

## Deployment workflow behavior

The existing deployment workflow waits for successful CI on `main` (or an explicit dispatch), checks out the tested revision, builds, configures Pages, uploads a Pages artifact, and invokes `actions/deploy-pages`.

If `actions/deploy-pages` reports a repository/settings `Not Found` or asks to enable Pages, treat that as a Pages configuration blocker. Do not claim the application itself failed deployment routing and do not change application code solely to hide a repository-settings failure.

## Live verification

After a successful deployment, use the **actual** `page_url` returned by GitHub:

```bash
PLAYWRIGHT_BASE_URL=<actual-page-url> pnpm test:e2e
```

The external target resolver requires an HTTPS public target and preserves its project path.

Required live checks include:

- Chromium initial load;
- WebKit initial load;
- no page/console errors;
- local font resources;
- training worker resource;
- naturally non-isolated fallback where applicable;
- manual step and paired evaluation evidence;
- Inspection lazy surface;
- Code / NumPy export surface;
- History and saved-run semantics;
- Configuration surface;
- canonical V2 shared recipe URL;
- fresh-context reload preserving recipe/architecture but creating a fresh runtime;
- skip-link keyboard/pointer integrity;
- project path never collapsing to origin root.

If the live environment exposes a hosting-specific failure, reproduce it with the real URL and add a bounded regression before changing product code.

## Expected URL versus deployment receipt

For this repository, the conventional GitHub Pages project URL is expected to resemble:

`https://densedevkev.github.io/neural-network-playground/`

This is only an expectation. The authoritative URL is the `page_url` from a successful deployment job.

## Rollback

If a deployment is bad:

1. identify the last known-good deployed `main` SHA;
2. revert through normal Git history rather than editing generated Pages output;
3. require CI on the rollback commit;
4. let the same deployment workflow publish the tested rollback;
5. repeat live Chromium/WebKit verification.

Never repair a deployment by manually mutating generated `dist` files outside source control.
