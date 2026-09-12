import { test, expect, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { LESSON_DEFINITIONS, type LessonDefinition } from '../../apps/web/src/lessons/lessonRegistry';
import { ready, workspace, utility, applyDataset } from './atelier-helpers';

const transport = (page: Page) => page.locator('section[aria-label="Training controls"]');
async function identity(page: Page) {
    return { url: page.url(), generation: await transport(page).getAttribute('data-model-generation'), revision: await transport(page).getAttribute('data-model-revision'), step: await transport(page).getAttribute('data-model-step') };
}
const setupSection = (page: Page, name: string) => page.getByRole('navigation', { name: 'Setup sections' }).getByRole('button', { name, exact: true });
async function startLesson(page: Page, label: 'Start lesson and reset' | 'Restart lesson and reset') {
    const generation = Number(await transport(page).getAttribute('data-model-generation'));
    await page.getByRole('button', { name: label, exact: true }).click();
    await expect(transport(page)).toHaveAttribute('data-model-generation', String(generation + 1));
    await ready(page);
}
async function apply(page: Page) {
    await page.getByRole('button', { name: 'Apply changes', exact: true }).click();
    await expect(page.getByRole('region', { name: 'Experiment setup' })).toHaveAttribute('aria-busy', 'false');
    await expect(page.getByRole('button', { name: 'Cancel', exact: true })).toBeDisabled();
    await ready(page);
}

test.beforeEach(async ({ page }) => { await page.goto('./'); await ready(page); });

for (const colorScheme of ['light', 'dark'] as const) {
    test(`system ${colorScheme}, live preference, override and persistence preserve model`, async ({ page }) => {
        await page.emulateMedia({ colorScheme });
        await page.reload(); await ready(page);
        await expect(page.locator('html')).toHaveAttribute('data-theme', colorScheme);
        const before = await identity(page);
        const other = colorScheme === 'light' ? 'dark' : 'light';
        await page.emulateMedia({ colorScheme: other });
        await expect(page.locator('html')).toHaveAttribute('data-theme', other);
        expect(await identity(page)).toEqual(before);
        await page.getByLabel('Color theme').selectOption(colorScheme);
        await expect(page.locator('html')).toHaveAttribute('data-theme', colorScheme);
        expect(await identity(page)).toEqual(before);
        await page.reload(); await ready(page);
        await expect(page.getByLabel('Color theme')).toHaveValue(colorScheme);
        await expect(page.locator('html')).toHaveAttribute('data-theme', colorScheme);
    });
}

test('theme remains usable when theme storage is denied', async ({ page }) => {
    await page.addInitScript(() => {
        const get = Storage.prototype.getItem; const set = Storage.prototype.setItem;
        Storage.prototype.getItem = function (key) { if (key === 'nn-playground-theme') throw new DOMException('Denied', 'SecurityError'); return get.call(this, key); };
        Storage.prototype.setItem = function (key, value) { if (key === 'nn-playground-theme') throw new DOMException('Denied', 'SecurityError'); return set.call(this, key, value); };
    });
    await page.reload(); await ready(page); const before = await identity(page);
    await page.getByLabel('Color theme').selectOption('dark');
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
    await page.getByLabel('Color theme').selectOption('light');
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
    expect(await identity(page)).toEqual(before);
});

test('three setup sections retain raw drafts and accept one combined paused transaction', async ({ page }) => {
    await workspace(page, 'Setup'); const before = await identity(page);
    await page.getByLabel('Samples', { exact: true }).fill('');
    await setupSection(page, 'Inputs & layers').click();
    await page.getByLabel('Model seed', { exact: true }).fill('123');
    await setupSection(page, 'Training').click();
    await page.getByLabel('Learning rate', { exact: true }).fill('0.025');
    await setupSection(page, 'Data').click();
    await expect(page.getByLabel('Samples', { exact: true })).toHaveValue('');
    await expect(page.getByRole('button', { name: 'Apply changes', exact: true })).toBeDisabled();
    expect(await identity(page)).toEqual(before);
    await page.getByLabel('Samples', { exact: true }).fill('300'); await apply(page);
    await expect(transport(page)).toHaveAttribute('data-model-generation', String(Number(before.generation) + 1));
    await expect(transport(page)).toHaveAttribute('data-model-step', '0');
    await setupSection(page, 'Inputs & layers').click(); await expect(page.getByLabel('Model seed', { exact: true })).toHaveValue('123');
    await setupSection(page, 'Training').click(); await expect(page.getByLabel('Learning rate', { exact: true })).toHaveValue('0.025');
    await page.getByLabel('Batch size', { exact: true }).fill('300');
    await setupSection(page, 'Data').click(); await page.getByLabel('Samples', { exact: true }).fill('100');
    await setupSection(page, 'Training').click(); await expect(page.getByLabel('Batch size', { exact: true })).toHaveValue('300');
    await expect(page.getByRole('button', { name: 'Apply changes', exact: true })).toBeDisabled();
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
    await expect(page.getByLabel('Batch size', { exact: true })).not.toHaveValue('300');
});

test('dirty navigation supports Stay, Discard, and Apply', async ({ page }) => {
    for (const decision of ['Stay', 'Discard changes', 'Apply changes']) {
        await workspace(page, 'Setup'); await setupSection(page, 'Data').click();
        const initial = await page.getByLabel('Data seed', { exact: true }).inputValue();
        await page.getByLabel('Data seed', { exact: true }).fill(String(Number(initial) + 1));
        const before = await identity(page);
        await page.getByRole('tab', { name: 'Network', exact: true }).click();
        const dialog = page.getByRole('alertdialog'); await expect(dialog).toBeVisible();
        await dialog.getByRole('button', { name: decision, exact: true }).click();
        await expect(dialog).toBeHidden();
        if (decision === 'Stay') {
            await expect(page.getByRole('region', { name: 'Experiment setup' })).toBeVisible();
            expect(await identity(page)).toEqual(before);
            await page.getByRole('button', { name: 'Cancel', exact: true }).click();
        } else {
            await expect(page.getByRole('tab', { name: 'Network', exact: true })).toHaveAttribute('aria-selected', 'true');
            if (decision === 'Apply changes') await expect(transport(page)).toHaveAttribute('data-model-generation', String(Number(before.generation) + 1));
            else expect(await identity(page)).toEqual(before);
        }
    }
});

for (const dataset of ['XOR', 'Three-Class', 'Plane']) {
    test(`${dataset} engine diagnostics retain paused identity and task-aware trace`, async ({ page }) => {
        await applyDataset(page, dataset);
        await workspace(page, 'Inspect');
        await page.getByRole('button', { name: 'Run one training step' }).click();
        await expect(transport(page)).toHaveAttribute('data-model-step', '1');
        await workspace(page, 'Inspect'); const before = await identity(page);
        await page.getByRole('button', { name: 'Trace prediction', exact: true }).click();
        await expect(page.locator('.inspection-panel')).toContainText('Trace from training sample 0 · model step 1');
        const output = page.locator('.inspection__trace-result .inspection__stat-row').filter({ has: page.getByText('Output', { exact: true }) }).locator('.inspection__stat-value').first();
        const values = (await output.innerText()).split(', ').map(Number);
        expect(values).toHaveLength(dataset === 'Three-Class' ? 3 : 1);
        expect(values.every(Number.isFinite)).toBe(true);
        if (dataset === 'Three-Class') { expect(values.every((n) => n >= 0 && n <= 1)).toBe(true); expect(values.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 3); }
        await page.getByLabel('Sample', { exact: true }).selectOption('test');
        await expect(page.locator('.inspection__trace-result')).toHaveCount(0);
        await page.getByRole('button', { name: 'Trace prediction', exact: true }).click();
        await expect(page.locator('.inspection-panel')).toContainText('Trace from test sample 0 · model step 1');
        await page.getByRole('tab', { name: 'Activations', exact: true }).click();
        await expect(page.locator('.inspection-panel')).toContainText(/Activation statistics across \d+ of \d+ training examples/);
        await page.getByRole('tab', { name: 'Gradients', exact: true }).click();
        await page.getByRole('button', { name: 'Preview backprop', exact: true }).click();
        await expect(page.getByRole('region', { name: 'Slow-motion backprop preview' })).toContainText('Preview from step 1');
        await page.getByRole('button', { name: 'Probe loss surface', exact: true }).click();
        await expect(page.getByRole('region', { name: 'Loss landscape probe', exact: true })).toContainText('Probe from step 1');
        await expect(page.getByRole('region', { name: 'Loss landscape probe', exact: true }).getByRole('img')).toBeVisible();
        expect(await identity(page)).toEqual(before);
        await transport(page).getByRole('button', { name: /(?:Start|Resume) training/, exact: true }).click();
        await expect(page.locator('.inspection__pause').getByRole('button', { name: 'Pause training', exact: true })).toBeVisible();
        await expect(page.getByRole('button', { name: 'Preview backprop', exact: true })).toBeDisabled();
        await page.locator('.inspection__pause').getByRole('button', { name: 'Pause training', exact: true }).click();
        await expect(page.getByRole('button', { name: 'Preview backprop', exact: true })).toBeEnabled();
        await workspace(page, 'Setup'); await setupSection(page, 'Training').click();
        if (dataset === 'Plane') await expect(page.getByRole('combobox', { name: 'Data loss', exact: true })).toHaveValue('mean-squared-error');
        else await expect(page.getByRole('combobox', { name: 'Data loss', exact: true })).toHaveCount(0);
    });
}

test('export files roundtrip through staged review and invalid files preserve the model', async ({ page }) => {
    await page.getByRole('button', { name: 'Run one training step' }).click();
    await expect(transport(page)).toHaveAttribute('data-model-step', '1');
    await utility(page, 'Export / import');
    const downloadPromise = page.waitForEvent('download'); await page.getByRole('button', { name: 'Export JSON setup', exact: true }).click();
    const download = await downloadPromise; const json = await readFile((await download.path())!, 'utf8');
    const document = JSON.parse(json); expect(document.schemaVersion).toBe(2);
    const changed = structuredClone(document);
    changed.recipe.model.seed += 1;
    changed.recipe.training.learningRate = 0.07;
    changed.view.showTestData = !document.view.showTestData;
    changed.view.discretizeOutput = !document.view.discretizeOutput;
    const before = await identity(page);
    await page.getByLabel('Import setup JSON file').setInputFiles({ name: 'invalid.json', mimeType: 'application/json', buffer: Buffer.from('{') });
    await expect(page.getByRole('alert')).toBeVisible(); expect(await identity(page)).toEqual(before);
    await page.getByLabel('Import setup JSON file').setInputFiles({ name: 'roundtrip.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(changed)) });
    await expect(page.getByRole('region', { name: 'Imported setup review' })).toBeVisible(); expect(await identity(page)).toEqual(before);
    await page.getByRole('button', { name: 'Apply imported setup' }).click();
    await expect(page.getByText('Imported setup applied. Training is paused.', { exact: true })).toBeVisible();
    await expect(transport(page)).toHaveAttribute('data-model-step', '0');
    await expect.poll(async () => Number((await identity(page)).generation)).toBeGreaterThan(Number(before.generation));
    const againPromise = page.waitForEvent('download'); await page.getByRole('button', { name: 'Export JSON setup', exact: true }).click();
    expect(JSON.parse(await readFile((await (await againPromise).path())!, 'utf8'))).toEqual(changed);
    await page.getByRole('tab', { name: 'Code', exact: true }).click();
    const codePromise = page.waitForEvent('download'); await page.getByRole('button', { name: 'Download code' }).click();
    expect((await readFile((await (await codePromise).path())!, 'utf8')).length).toBeGreaterThan(100);
    await page.evaluate(() => Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: () => Promise.reject(new DOMException('Denied', 'NotAllowedError')) } }));
    await page.getByRole('tab', { name: 'Setup & sharing', exact: true }).click();
    await page.getByRole('button', { name: /Copy setup link/i }).click();
    await expect(page.getByLabel('Selectable setup link')).toHaveValue(/^http/);
    await page.getByRole('tab', { name: 'Code', exact: true }).click();
    await page.getByRole('button', { name: /Copy Code/ }).click();
    await expect(page.getByText(/Could not copy code/)).toBeVisible();
    await expect(page.locator('.code-export__code')).toBeVisible();
});

test('code parameter snapshots cannot retain learned weights after a new recipe', async ({ page }) => {
    await page.getByRole('button', { name: 'Run one training step' }).click();
    await expect(transport(page)).toHaveAttribute('data-model-step', '1');
    await utility(page, 'Export / import'); await page.getByRole('tab', { name: 'Code', exact: true }).click();
    await expect(page.getByText('Generated from the current setup.', { exact: false })).toContainText('Parameter snapshot: step 1.');
    await page.getByRole('button', { name: 'Close Export / import', exact: true }).click();
    await applyDataset(page, 'Plane');
    await utility(page, 'Export / import'); await page.getByRole('tab', { name: 'Code', exact: true }).click();
    await expect(page.getByText('Generated from the current setup.', { exact: false })).not.toContainText('Parameter snapshot: step 1.');
    await expect(page.locator('.code-export__code')).toContainText(/linear|regression/i);
});

for (const lesson of LESSON_DEFINITIONS as readonly LessonDefinition[]) {
    test(`complete lesson journey: ${lesson.title}`, async ({ page }) => {
        await page.getByRole('button', { name: 'Lessons', exact: true }).click();
        await page.locator('.lesson-list').getByRole('button', { name: new RegExp(lesson.title) }).click();
        await startLesson(page, 'Start lesson and reset');
        const panel = page.getByRole('complementary', { name: 'Guided lesson mode' });
        await expect(panel.getByRole('heading', { name: lesson.steps[0].title, exact: true })).toBeVisible();
        await expect(transport(page)).toHaveAttribute('data-model-step', '0');
        for (const [index, step] of lesson.steps.entries()) {
            await expect(panel.getByRole('heading', { name: step.title, exact: true })).toBeVisible();
            if (index > 0) {
                await panel.getByRole('button', { name: 'Previous', exact: true }).click();
                await expect(panel.getByRole('heading', { name: lesson.steps[index - 1].title, exact: true })).toBeVisible();
                await panel.getByRole('button', { name: 'Next lesson step', exact: true }).click();
            }
            await panel.getByRole('button', { name: 'Show me →', exact: true }).click();
            if (step.tab) {
                const section = step.tab === 'data' ? 'Data' : step.tab === 'hyperparams' ? 'Training' : 'Inputs & layers';
                await expect(setupSection(page, section)).toHaveAttribute('aria-current', 'page');
            } else {
                await expect(page.getByRole('tab', { name: 'Results', exact: true })).toHaveAttribute('aria-selected', 'true');
                await expect(page.getByRole('tab', { name: step.evidenceView === 'loss' ? 'Learning progress' : 'Prediction', exact: true })).toHaveAttribute('aria-selected', 'true');
                const old = Number(await transport(page).getAttribute('data-model-step'));
                await page.getByRole('button', { name: 'Run one training step' }).click();
                await expect(transport(page)).toHaveAttribute('data-model-step', String(old + 1));
            }
            if (step.id === 'inspect-noise-controls') {
                await page.getByLabel('Noise (%)', { exact: true }).fill('12'); await apply(page);
            }
            if (step.id === 'retry-with-smaller-steps' || step.id === 'simplify-or-regularize') {
                if (step.id === 'retry-with-smaller-steps') await page.getByLabel('Learning rate', { exact: true }).fill('0.02');
                else { const penalty = page.getByRole('combobox', { name: 'Penalty', exact: true }); await penalty.selectOption((await penalty.inputValue()) === 'l2' ? 'l1' : 'l2'); }
                await apply(page); await workspace(page, 'Results');
                await page.getByRole('tab', { name: 'Learning progress', exact: true }).click();
                await page.getByRole('button', { name: 'Run one training step' }).click();
                await expect(transport(page)).toHaveAttribute('data-model-step', '1');
            }
            if ('completion' in step) await expect(panel.getByRole('status')).toHaveText('Done');
            if (index < lesson.steps.length - 1) await panel.getByRole('button', { name: 'Next lesson step', exact: true }).click();
        }
        await panel.getByRole('button', { name: 'Finish guided lesson' }).click();
        await expect(page.getByRole('region', { name: 'Selected lesson details' })).toBeVisible();
        await startLesson(page, 'Start lesson and reset');
        await panel.getByText('Restart lesson', { exact: true }).click();
        await startLesson(page, 'Restart lesson and reset');
        await expect(transport(page)).toHaveAttribute('data-model-step', '0');
        await panel.getByRole('button', { name: 'Exit lesson', exact: true }).click();
        await expect(page.getByRole('region', { name: 'Selected lesson details' })).toBeVisible();
    });
}
