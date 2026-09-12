# Maintenance and qualification guide

This is maintenance infrastructure, not a second product roadmap. Source authority, milestone status and release acceptance live in `NN-FORGE-LIVING-EXECUTION-PLAN.md`. State/controller ownership lives in `docs/architecture/state-ownership.md`. The approved work and its review are `2026-09-10-plan.md` and `2026-09-10-review.md` in this directory.

## Environment contract

The root manifest declares Node `>=20` and pnpm `>=9`. The current workflow selects Node `20` and pnpm `9`, and its source/build/performance receipts record the actual resolved versions. Observed intake CI used Node `20.20.2` and pnpm `9.15.9`; the isolated maintenance container used Node `22.16.0` and pnpm `9.15.9`. Those observations are not a guarantee for future floating-major resolution.

Do not silently upgrade a runtime, regenerate a lockfile, or add dependencies during cleanup. Install with `pnpm install --frozen-lockfile`; the committed lockfile and package manifests must match recovered dependencies. An incomplete offline store is an environment error, not a successful install. Recovered dependency archives can omit binaries/assets: a successful local compile does not prove complete font delivery or browser fidelity. GitHub's fresh install and exact-source artifacts remain the acceptance environment.

## Development checks

Run from the repository root:

```sh
node --test scripts/*.test.mjs
pnpm lint
pnpm typecheck
pnpm test
pnpm build
pnpm test:bundle
git diff --check
```

For a focused web suite use `pnpm --filter @nn-playground/web exec vitest run <src/path.test.tsx> --pool=forks`. Test helpers belong under `apps/web/src/test`. Use `makeSavedRunRecord` for the shared saved-artifact fixture; it fixes IDs/timestamps and clones nested recipe state. Keep expected results independent of production serializers. Restore mocks, clocks and subscriptions; do not replace completion conditions with arbitrary sleeps.

The unchanged gzip caps are entry 152245 bytes, InspectionPanel 7373 bytes and total JavaScript including worker 234161 bytes. The comparator implements the accepted five-pair performance contract against `ae09b9863ae90f8fb2f62545834fcc138755ba9a`. Do not raise limits or substitute a historical absolute diagnostic for that accepted comparison.

## Command receipts

```sh
node scripts/run-with-evidence.mjs tests -- pnpm test
node scripts/run-with-evidence.mjs bundle -- pnpm test:bundle
node scripts/summarize-qualification.mjs focused
```

The wrapper accepts a safe receipt name and a command/argument array after `--`; it does not evaluate a shell string. It streams raw stdout/stderr to `<name>.log` and records source commit/tree, tracked-source status, command, Node version, timestamps, exit status and signal in `<name>.json`. `NN_FORGE_EVIDENCE_DIR` overrides the default `qualification-evidence` directory. An unfinished command remains `running`; a missing executable fails; cancellation remains interrupted even when a child handles termination and exits zero. An uncatchable process kill may leave incomplete evidence, never a fabricated pass.

The summarizer supports `focused`, `preview`, `subpath`, `recovery`, and `performance`. Run it only after that mode's required commands; missing steps/reports result in non-passing evidence. It independently checks current tracked-source cleanliness, validates enumerated Playwright counts/result statuses, and retains skipped/flaky/interrupted distinctions. It limits detailed failures to eight and excerpts to 600 characters, retaining total/omitted counts. Raw logs and reports remain available separately.

Playwright JSON is written alongside list/HTML output as `playwright-results.json`. Recovery preserves its fault-enabled JSON under `recovery-evidence/` before rebuilding and generating the fault-disabled report. Retries, browser assertions, traces and failure limits are unchanged.

## GitHub connector recovery sequence

1. Re-read main and the active branch ref. Fetch the workflow run for its exact SHA; do not rely on an earlier chat checkpoint.
2. Read job/step conclusions before downloading logs. A job can fail its bundle or browser gate while correctness passes.
3. Fetch the small `precision-summary-<mode>-<sha>` artifact first. Its `summary.json` includes source identity, step exits, bounded browser failures and bundle/performance results. It is a summary, not authority to override a failing GitHub job.
4. For more context fetch `precision-command-evidence-<mode>-<sha>`, the existing `precision-browser-<mode>-<sha>`, or `precision-performance-<sha>`. Full screenshots/traces stay in the browser artifact. Build provenance is retained when the build succeeds even if the later bundle guard fails.
5. Recover `precision-source-<sha>`, verify its archive hash and Git tree, and compare changed blobs before writing. Build a coherent Git tree/commit through the connector. Re-read the ref before a non-forced update; reconcile concurrent movement instead of overwriting it.
6. Inspect exact-head CI after publication. A reporting error, missing artifact or unreadable connector response is not proof of an application defect or a pass. Never merge/deploy based on missing evidence.

## Boundaries

Maintenance tests alone do not qualify a release. The September 12 completion resolves legacy-shell retirement and the saved-run overwrite; final exact-commit acceptance and deployment receipts live in completion PR #36 and the living execution plan. This maintenance workflow itself does not modify repository visibility or Pages settings.
