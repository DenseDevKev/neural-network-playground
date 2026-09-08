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
