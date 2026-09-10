import { URL } from 'node:url';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

const workflow = readFileSync(new URL('../.github/workflows/precision-lab-verification.yml', import.meta.url), 'utf8');
test('Precision Lab qualification stays read-only and cannot deploy or change visibility', () => {
    assert.match(workflow, /permissions:\s*\n\s+contents: read/);
    assert.doesNotMatch(workflow, /contents: write|pages: write|deploy-pages|visibility:/);
});
test('Precision Lab includes helpers, correctness, exact build evidence and recovery', () => {
    for (const contract of ['node --test scripts/*.test.mjs', 'pnpm test', 'pnpm lint', 'pnpm typecheck', 'pnpm build', 'pnpm test:bundle', 'precision-build-evidence/dist.sha256', 'mode: [preview, subpath, recovery]', 'pnpm test:e2e:recovery', "--grep '@fault-disabled'"]) {
        assert.ok(workflow.includes(contract), `Missing qualification: ${contract}`);
    }
});
test('Precision Lab reuses the accepted five-pair performance policy and exact baseline', () => {
    assert.match(workflow, /performance:\s*\n\s+runs-on: macos-15/);
    assert.ok(workflow.includes('for pass in 1 2 3 4 5'));
    assert.ok(workflow.includes('ae09b9863ae90f8fb2f62545834fcc138755ba9a'));
    assert.ok(workflow.includes('scripts/compare-performance-reference.mjs'));
    assert.ok(workflow.includes("order='candidate baseline'"));
});
test('qualification commands keep their original exit status and record receipts', () => {
    for (const command of [
        'helpers -- node --test scripts/*.test.mjs', 'tests -- pnpm test',
        'lint -- pnpm lint', 'types -- pnpm typecheck', 'build -- pnpm build',
        'bundle -- pnpm test:bundle', 'preview -- pnpm exec playwright test --max-failures=5',
        'subpath -- pnpm exec playwright test --max-failures=5',
        'recovery -- pnpm test:e2e:recovery', "fault-disabled -- pnpm exec playwright test tests/e2e/worker-recovery.spec.ts --grep '@fault-disabled'",
    ]) assert.ok(workflow.includes(`scripts/run-with-evidence.mjs ${command}`), `Missing receipt: ${command}`);
    assert.doesNotMatch(workflow, /continue-on-error:\s*true/);
});
test('build provenance survives a failing bundle gate but not a failed build', () => {
    assert.match(workflow, /Record exact build and verify tracked source\n\s+if: always\(\) && steps\.build\.outcome == 'success'/);
});
test('both recovery reports and compact summaries survive failure', () => {
    assert.ok(workflow.includes('mv playwright-results.json recovery-evidence/playwright-results.json'));
    assert.ok(workflow.includes('precision-summary-${{ matrix.mode }}-${{ github.sha }}'));
    for (const mode of ['focused', 'performance']) assert.ok(workflow.includes(`scripts/summarize-qualification.mjs ${mode}`));
    assert.match(workflow, /Summarize browser qualification\n\s+if: always\(\)/);
});
test('Playwright adds machine evidence without changing retry and failure artifacts', () => {
    const config = readFileSync(new URL('../playwright.config.ts', import.meta.url), 'utf8');
    assert.ok(config.includes("['json', { outputFile: 'playwright-results.json' }]"));
    assert.ok(config.includes('retries: 0'));
    assert.ok(config.includes("trace: 'retain-on-failure'"));
});
