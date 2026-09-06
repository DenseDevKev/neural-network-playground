# NN.FORGE consolidated release and product roadmap

**Updated:** September 6, 2026  
**Product authority before this release:** `main` at `ae09b9863ae90f8fb2f62545834fcc138755ba9a`  
**Execution branch:** `codex/nn-forge-release-roadmap`  
**Latest fully qualified code SHA before this documentation-only consolidation:** `fbae4b98a71b86bdf704a3d8896ccc8f0b92b4a0`

This file is the canonical execution order. A checked implementation on a branch is not a release until it is accepted into `main`, and a `main` commit is not a deployed release until the actual deployment succeeds and its live URL is verified.

## Product invariants

Preserve the product state model:

`Current recipe -> Trained snapshot -> Active run -> Evidence -> Saved runs`

The following are release constraints, not optional presentation preferences:

- Navigation, workspace profile, disclosure, and focus changes must not mutate recipe, model generation/revision, step, checkpoints, saved records, code-export selection, or the experiment URL.
- The V2 hash is experiment state. Shell navigation must not replace it.
- Batch/EMA signals remain distinct from full-split paired train/test evaluations.
- Evaluation age and recipe drift remain visible where applicable.
- Saved runs contain recipe/evidence, not trained parameters. Applying a saved recipe is not model restoration.
- Checkpoint restoration does not promise an identical future stochastic trajectory.
- Scientific evaluation cadence may not be reduced merely to improve presentation performance.
- `useTraining` remains the worker-command owner; UI layout state remains separate from experiment state.
- Historical branches/prototypes are evidence only unless a specific missing dependency is established.
- The end state has one production shell, not competing Build/Run and Precision Lab products.

## Release baseline sequence

| Stage | Status | Exit condition |
|---|---|---|
| R0 — Source authority | Complete | Exact intake SHA, isolated execution branch, no reliance on missing local workspace |
| R1 — Explicit browser targets | Complete | Local defaults retained; external project-path targets validated and fail closed |
| R2 — Hosting/state contracts | Complete | Real static subpath worker/lazy loads, shared recipe reload, skip-link integrity, WebKit reload safety |
| R3 — Repeatable qualification | Complete on `fbae4b9` | Correctness, browser, recovery, bundle, and comparative performance gates all green |
| R4 — Consolidated handoff | Complete with this documentation wave | Roadmap, QA, deployment, and verification evidence agree |
| R5 — Accept into `main` | Pending | Documentation SHA requalified, reviewed PR merged, resulting `main` CI green |
| R6 — Live publication | Pending / owner-authorized | Accepted `main` deployed through existing Pages workflow and live Chromium/WebKit verification passes |
| P1 — Executable JS bundle limits | Complete | Fixed reviewed JavaScript gzip caps enforced in CI |
| P2–P7 — Precision Lab integration | Pending | One production presentation shell with all scientific/state/accessibility contracts retained |

## Release qualification already implemented

### Explicit Playwright destinations

`scripts/playwright-target.mjs` preserves the existing local preview and supports an explicit external base URL without silently falling back to localhost. It preserves project subpaths and rejects credentials, malformed targets, public insecure HTTP targets, query/fragment contamination, whitespace/backslash ambiguity, and conflicting explicit port/base settings.

### Real static project-subpath fixture

`scripts/serve-release-fixture.mjs` serves actual `apps/web/dist` bytes at `/neural-network-playground/` without COOP/COEP. It serves correct MIME types, preserves canonical trailing-slash behavior, returns real 404s, rejects traversal/symlink escape, and permits GET/HEAD only.

### Hosting and shared-state contracts

The browser suite verifies:

- real training worker and lazy chunks beneath the project path;
- naturally non-isolated operation;
- paired evaluation after a manual step;
- Inspection, Code, History, and Configuration surfaces;
- canonical V2 share URL reload into the same recipe/architecture with a fresh runtime;
- project path/origin preservation.

### Skip-link state integrity

The old `#main-content` anchor navigation replaced the V2 experiment fragment. The regression now exercises keyboard and pointer activation after a real recipe and training step. The fix prevents default fragment navigation and focuses the existing main landmark without changing experiment state.

### Same-origin font delivery

Inter and Space Grotesk are exactly pinned through Fontsource and served from the application origin. Real Chromium/WebKit font tests require requested faces/weights, native reloads, no Google Fonts requests, and no font/page/console errors. The original strict recovery assertion remains intact. Font license notices ship with the build.

### Executable JavaScript size contract

`pnpm test:bundle` measures the actual application entry, exactly one InspectionPanel chunk, and all JavaScript including the training worker. It fails closed for missing/ambiguous/symlink/traversal inputs and one-byte overruns.

Reviewed limits and `fbae4b9` measurements:

| Dimension | Actual gzip bytes | Maximum |
|---|---:|---:|
| Application entry | 146,940 | 152,245 |
| InspectionPanel | 5,414 | 7,373 |
| All JavaScript including worker | 226,417 | 234,161 |

## Performance contract — resolved semantics

The July 11 Scientific Trust baseline did **not** define the frozen engine timing numbers as universal hardware-independent CI limits. It required five baseline runs on the same isolated development machine and accepted a post-change median when it remained at or below **120% of that same-machine baseline median**.

The old absolute engine constants remain unchanged as historical/local calibration. Release qualification now restores the original semantics reproducibly:

1. check out the exact intake baseline and candidate on one `macos-15` Apple Silicon runner;
2. collect five complete runs for each revision, alternating execution order;
3. retain raw logs, exits, host metadata, exact SHAs, and outliers;
4. compare each engine candidate median against `baseline median * 1.20`;
5. independently require forced paired evaluation <= 250 ms and save capture <= 500 ms.

This is implemented by `scripts/compare-performance-reference.mjs` and enforced by both release verification and the dedicated reference workflow. It does **not** relax or rewrite engine benchmark constants, and no engine/shared implementation was changed to obtain a release pass.

On qualified code SHA `fbae4b9`, release run `34001183202` passed the comparative gate. The largest engine movement was `predictGrid` at +12.1843%, below the +20% limit. All other engine metrics were closer or faster, and worker medians remained far inside the fixed scientific-trust budgets.

## Final release-baseline steps

### R5.1 — Requalify this documentation SHA

The documentation commit that contains this roadmap must run the same release-verification workflow. Required jobs:

- source evidence;
- helper tests, lint, typecheck, all unit tests, build, bundle guard, tracked-source cleanliness;
- Chromium/WebKit isolated preview;
- Chromium/WebKit non-isolated project-subpath fixture;
- fault-enabled recovery, clean rebuild, fault-disabled check;
- five-pair same-runner performance comparison.

### R5.2 — Review and merge

After all jobs are green:

1. re-read current `main` and branch refs;
2. compare the exact candidate against `main`;
3. confirm no unexpected engine/shared/schema/protocol/persistence changes;
4. open a PR from `codex/nn-forge-release-roadmap` to `main`;
5. review the PR/checks;
6. merge only the qualified candidate;
7. record the resulting `main` SHA;
8. require normal `main` CI to pass.

### R6 — Deploy the accepted `main`

Use the existing `.github/workflows/deploy.yml`. It waits for successful `main` CI, checks out the exact tested SHA, rebuilds it, uploads a Pages artifact, and deploys that artifact. Do not add a second deployment mechanism.

A successful deployment must record its tested/deployed SHA, workflow run, Pages artifact/digest, and actual `page_url`. Then run the checked-in browser suite against the actual deployment URL. Expected project URL is only a hypothesis until GitHub reports a successful `page_url`.

## Precision Lab — next product milestone

The July 16 design and July 17 implementation plan remain the only later product direction carried forward. Their internal `AudienceMode` state may remain an implementation detail, but current user-facing terminology is **Workspace profile**. Do not regress the accessible name back to the historical label merely to match the old plan.

The referenced prototype is not present in GitHub. Therefore functional/spec acceptance proceeds, but no pixel-parity claim may be made against an unseen visual reference.

### P1 — Bundle limits

Complete, as described above.

### P2 — Display-safe shell

- Keep exactly one `useTraining` owner.
- Create presentation-focused typed adapters rather than passing raw worker envelopes.
- Preserve experiment/runtime/checkpoint/saved-run/export/hash invariants.
- Implement shell/profile/disclosure/focus/compact-layout models with tests first.
- Compose a Precision Lab presentation from real production state only.

### P3 — Real network selection

- Define stable layer/neuron/edge selection identities.
- Invalidate selection on architecture/generation changes.
- Rank strongest paths deterministically.
- Use real typed-array/model data and preserve Canvas plus accessible fallback behavior.
- Avoid copying live numerical grids into persistent React state.
- Support pointer, keyboard, clear, profile/disclosure, and compact interaction.

### P4 — One canonical live boundary

- Maintain exactly one live decision-boundary renderer in Build/Run.
- Separate current live boundary from pinned evidence.
- Preserve evaluation step/age/drift provenance.
- Update visualization-demand derivation without reducing required scientific evaluation cadence.

### P5 — Production-backed previews/evidence

- Generate all dataset previews from deterministic production generators.
- Adapt Loss/Confusion/training controls to compact presentation without provenance loss or horizontal overflow.
- Preserve exact pending save artifacts on retry; do not silently recapture a different model state.
- Keep saved-run parameter limitations and Apply Saved Recipe semantics explicit.

### P6 — Production acceptance

Required acceptance includes:

- full unit/integration/E2E/build/bundle/recovery/performance suites;
- Chromium and WebKit at zero retries;
- required viewports `1437x742`, `735x860`, and `320x844`;
- five seconds of training at 50 steps/frame with stable major-region bounds within 1 CSS px where stability is required;
- zero Chromium post-start CLS for the acceptance scenario;
- 44 px required touch targets, keyboard operation, visible focus, reduced motion, 200% zoom, semantic regions, and no accessibility regressions;
- JavaScript bundle caps, 250/500 ms worker budgets, and the accepted same-runner engine regression policy.

### P7 — One final production shell

Migrate every consumer before removing the existing presentation. Remove obsolete shell/styles/adapters only after consumer proof. The accepted end state contains one production shell and unchanged scientific/V2 state contracts.

## Explicitly out of scope without a new reviewed milestone

- accounts or cloud backend;
- collaboration/multiplayer;
- generic AI features;
- new dataset program or new neural-network math;
- broad engine/framework rewrite;
- wholesale historical-branch/prototype import;
- arbitrary benchmark-limit increases;
- trained-parameter persistence in saved runs;
- deterministic-future claims for checkpoint restoration;
- making the repository public merely to simplify deployment.

After the Precision Lab release, use the product and collect actual friction/usage evidence before defining another large milestone.
