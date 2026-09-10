import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ESLint } from 'eslint';

const eslint = new ESLint();
const rule = '@typescript-eslint/no-explicit-any';
const source = 'export const value: any = 1;';

for (const filePath of [
    'apps/web/src/production-policy-probe.ts',
    'apps/web/src/testing/e2eFaults.ts',
    'packages/engine/src/production-policy-probe.ts',
    'packages/shared/src/production-policy-probe.ts',
]) {
    test(`rejects explicit any in production: ${filePath}`, async () => {
        const [result] = await eslint.lintText(source, { filePath });
        assert.equal(result.messages.filter((message) => message.ruleId === rule && message.severity === 2).length, 1);
        assert.equal(result.fatalErrorCount, 0);
    });
}

for (const filePath of [
    'apps/web/src/hooks/policy.test.tsx',
    'packages/engine/src/__tests__/policy.ts',
    'packages/shared/src/policy.spec.ts',
    'apps/web/src/test/policy-fixture.ts',
    'tests/e2e/policy.spec.ts',
]) {
    test(`permits intentionally malformed test fixtures: ${filePath}`, async () => {
        const [result] = await eslint.lintText(source, { filePath });
        assert.equal(result.messages.filter((message) => message.ruleId === rule).length, 0);
        assert.equal(result.fatalErrorCount, 0);
    });
}
