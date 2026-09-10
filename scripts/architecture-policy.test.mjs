import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ESLint } from 'eslint';

const eslint = new ESLint();
const rule = 'nn-forge/dependency-boundaries';
const cases = [
    ['engine rejects UI aliases', 'packages/engine/src/probe.ts', "import '@nn-playground/web';", true],
    ['shared rejects normalized UI paths', 'packages/shared/src/probe.ts', "import '../../../apps/web/src/store/../store/useTrainingStore.ts';", true],
    ['domain rejects React', 'packages/engine/src/probe.ts', "export * from 'react';", true],
    ['engine cannot depend on shared', 'packages/engine/src/probe.ts', "import '@nn-playground/shared';", true],
    ['shared can depend on engine', 'packages/shared/src/probe.ts', "import '@nn-playground/engine';", false],
    ['production rejects fixtures', 'apps/web/src/components/probe.ts', "import '../test/scientificTrustFixtures.ts';", true],
    ['production rejects literal lazy fixtures', 'apps/web/src/probe.ts', "void import('./test/scientificTrustFixtures.ts');", true],
    ['production rejects prototype reexports', 'apps/web/src/probe.ts', "export * from '../../../prototypes/example.ts';", true],
    ['production rejects test types', 'apps/web/src/probe.ts', "export type T = import('./test/fixture.ts').T;", true],
    ['new UI owner rejected even with renamed import', 'apps/web/src/components/probe.ts', "import { useTraining as own } from '../hooks/useTraining.ts'; export { own };", true],
    ['new UI owner rejects lazy controller import', 'apps/web/src/components/probe.ts', "void import(`../hooks/useSaveCurrentRun.ts`);", true],
    ['new owner rejects require controller import', 'apps/web/src/components/probe.ts', "export const module = require('../hooks/useSaveCurrentRun.ts');", true],
    ['controller types remain display-safe', 'apps/web/src/components/probe.ts', "import type { SaveCurrentRunController } from '../hooks/useSaveCurrentRun.ts'; export type C = SaveCurrentRunController;", false],
    ['inline type specifier remains allowed', 'apps/web/src/components/probe.ts', "import { type TrainingHook } from '../hooks/useTraining.ts'; export type H = TrainingHook;", false],
    ['production rejects test framework', 'apps/web/src/probe.ts', "import 'vitest';", true],
    ['inline type reexport remains allowed', 'apps/web/src/components/probe.ts', "export { type TrainingHook } from '../hooks/useTraining.ts';", false],
    ['App retains training ownership', 'apps/web/src/App.tsx', "import { useTraining } from './hooks/useTraining.ts'; export { useTraining };", false],
    ['known standalone renderer retained', 'apps/web/src/components/visualization/NetworkGraphSVG.tsx', "import { useNetworkSelectionController } from './useNetworkSelectionController.ts'; export { useNetworkSelectionController };", false],
    ['renderer exception does not allow training ownership', 'apps/web/src/components/visualization/NetworkGraphSVG.tsx', "import { useTraining } from '../../hooks/useTraining.ts'; export { useTraining };", true],
    ['unqualified legacy history wrapper retained', 'apps/web/src/components/controls/RunHistoryPanel.tsx', "import { useSaveCurrentRun } from '../../hooks/useSaveCurrentRun.ts'; export { useSaveCurrentRun };", false],
    ['ordinary lazy UI remains allowed', 'apps/web/src/probe.ts', "void import('./components/EducationContent.ts');", false],
    ['benchmarks may import Vitest', 'packages/engine/src/__benchmarks__/policy.bench.ts', "import 'vitest';", false],
    ['production rejects benchmark entry', 'packages/engine/src/probe.ts', "import './__benchmarks__/performance.bench.ts';", true],
    ['tests may import controllers', 'apps/web/src/components/probe.test.ts', "import { useTraining } from '../hooks/useTraining.ts'; export { useTraining };", false],
];

for (const [name, filePath, source, blocked] of cases) {
    test(name, async () => {
        const [result] = await eslint.lintText(source, { filePath });
        assert.equal(result.fatalErrorCount, 0, JSON.stringify(result.messages));
        assert.equal(result.messages.filter((message) => message.ruleId === rule).length, blocked ? 1 : 0);
    });
}
