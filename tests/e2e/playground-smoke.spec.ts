import { Buffer } from 'node:buffer';
import { expect, test, type Locator, type Page } from '@playwright/test';
import { applyPreset, currentRun, expectEvidence, identity, learning, ready, transport, utility, workspace } from './atelier-helpers';

const RECIPES = {
    regression: {
        title: 'Regression with No Hidden Layer', dataset: 'reg-plane', taskKind: 'regression',
        hiddenLayers: [], outputSize: 1, outputActivation: 'linear', objective: 'mean-squared-error',
        learningRate: 0.01, noise: 5,
        datasetSettings: 'Dataset settings: 300 samples, 5 noise, 50% train',
    },
    xor: {
        title: 'XOR Needs Hidden Layers', dataset: 'xor', taskKind: 'binary-classification',
        hiddenLayers: [4, 4], outputSize: 1, outputActivation: 'sigmoid', objective: 'binary-cross-entropy-with-logits',
        learningRate: 0.03, noise: 0,
        datasetSettings: 'Dataset settings: 300 samples, 0 noise, 50% train',
    },
    threeClass: {
        title: 'Three-Class Softmax Lab', dataset: 'three-class-clusters', taskKind: 'multiclass-classification',
        hiddenLayers: [6, 6], outputSize: 3, outputActivation: 'softmax', objective: 'categorical-cross-entropy-with-logits',
        learningRate: 0.03, noise: 0.05,
        datasetSettings: 'Dataset settings: 300 samples, 0.05 noise, 50% train',
    },
} as const;


const MEMORY_KEY = 'nn-playground-experiment-memory-v2';
const browserErrors = new WeakMap<Page, string[]>();
function collectErrors(page: Page) {
    const errors: string[] = [];
    browserErrors.set(page, errors);
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
}
test.beforeEach(({ page }) => collectErrors(page));
test.afterEach(({ page }) => expect(browserErrors.get(page) ?? []).toEqual([]));

async function load(page: Page) {
    await page.goto('./');
    await learning(page);
    await ready(page);
    await expectEvidence(page, 0);
}
async function converge(page: Page) {
    await expect.poll(async () => {
        const step = Number(await transport(page).getAttribute('data-model-step'));
        const evaluation = Number((await currentRun(page).innerText()).match(/Full evaluation \d+ at step ([\d,]+)/)?.[1]?.replaceAll(',', ''));
        return step >= 0 && step === evaluation;
    }).toBe(true);
    return Number(await transport(page).getAttribute('data-model-step'));
}
async function saved(page: Page) {
    await page.getByRole('button', { name: 'Saved runs', exact: true }).click();
    return page.locator('.saved-runs');
}
async function expectedRecipe(page: Page, recipe: typeof RECIPES[keyof typeof RECIPES]) {
    // Check the entire public recipe document, not a shortened UI summary or a test-only store.
    const payload = new URLSearchParams(new URL(page.url()).hash.slice(1)).get('r');
    expect(payload).not.toBeNull();
    const document = JSON.parse(Buffer.from(payload!, 'base64url').toString('utf8'));
    expect(document.schemaVersion).toBe(2);
    expect(document.recipe).toEqual({
        task: { kind: recipe.taskKind, dataset: recipe.dataset },
        data: { sampleCount: 300, trainFraction: 0.5, noise: recipe.noise, seed: 42 },
        inputs: { featureIds: ['x', 'y'] },
        model: { hiddenLayers: [...recipe.hiddenLayers], hiddenActivation: 'tanh', initialization: 'xavier', seed: 42 },
        training: { batchSize: 10, learningRate: recipe.learningRate, schedule: { kind: 'constant' }, optimizer: { kind: 'sgd' }, gradientClipping: { kind: 'none' } },
        objective: { dataLoss: { kind: recipe.objective }, penalty: { kind: 'none' }, reduction: 'mean-per-sample' },
    });

    await workspace(page, 'Setup');
    const sections = page.getByRole('navigation', { name: 'Setup sections' });
    await sections.getByRole('button', { name: 'Data', exact: true }).click();
    await expect(page.getByRole('textbox', { name: 'Samples', exact: true })).toHaveValue('300');
    await expect(page.getByRole('textbox', { name: 'Noise (%)', exact: true })).toHaveValue(String(recipe.noise));
    await sections.getByRole('button', { name: 'Training', exact: true }).click();
    await expect(page.getByRole('textbox', { name: 'Learning rate', exact: true })).toHaveValue(String(recipe.learningRate));
    await expect(page.getByRole('combobox', { name: 'Optimizer', exact: true })).toHaveValue('sgd');
    await expect(page.getByRole('textbox', { name: 'Batch size', exact: true })).toHaveValue('10');
    await sections.getByRole('button', { name: 'Inputs & layers', exact: true }).click();
    await expect(page.locator('.atelier-feature-choices input:checked')).toHaveCount(2);
    await expect(page.getByRole('combobox', { name: 'Hidden activation' })).toHaveValue('tanh');
    await learning(page);
    await expectEvidence(page, 0);
}

async function touchTarget(locator: Locator) {
    const box = await locator.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.width).toBeGreaterThanOrEqual(44);
    expect(box!.height).toBeGreaterThanOrEqual(44);
}

test('training can pause, single-step, preview and restore the initial checkpoint', async ({ page }) => {
    await load(page);
    await expect(transport(page).locator('xpath=ancestor-or-self::*[@aria-live or @role="status" or @role="alert"]')).toHaveCount(0);
    await page.getByRole('combobox', { name: 'Steps per frame' }).selectOption('1');
    await page.getByRole('button', { name: 'Start training', exact: true }).click();
    await expect(transport(page)).toHaveAttribute('data-status', 'running');
    await expect.poll(async () => Number(await transport(page).getAttribute('data-model-step'))).toBeGreaterThan(0);
    await page.getByRole('button', { name: 'Pause training', exact: true }).click();
    const paused = await converge(page);
    await expect(transport(page)).toHaveAttribute('data-status', 'paused');
    await page.getByRole('button', { name: 'Run one training step' }).click();
    await expectEvidence(page, paused + 1);
    const before = await identity(page);
    await utility(page, 'Session checkpoints');
    const slider = page.getByRole('slider', { name: 'Checkpoint timeline' });
    await slider.focus();
    await slider.press('Home');
    await expect(slider).toHaveAttribute('aria-valuetext', 'Step 0');
    expect(await identity(page)).toEqual(before);
    await page.getByRole('button', { name: 'Restore Step 0', exact: true }).click();
    await expect(page.getByText('Restored checkpoint', { exact: true })).toBeVisible();
    await page.keyboard.press('Escape');
    await expectEvidence(page, 0);
    await expect(transport(page)).toHaveAttribute('data-status', 'paused');
    await expect(transport(page)).not.toHaveAttribute('data-model-revision', before.revision!);
});

test('canonical presets reset complete recipes and expose all multiclass cells', async ({ page }) => {
    await load(page);
    for (const recipe of Object.values(RECIPES)) {
        await applyPreset(page, recipe.title);
        await expectedRecipe(page, recipe);
    }
    await workspace(page, 'Network');
    await expect(page.getByRole('region', { name: 'Architecture summary' })).toContainText('X₁, X₂ -> [6] -> [6] -> 3 outputs (softmax)');
    await workspace(page, 'Results');
    await page.getByRole('tab', { name: 'Errors & confusion', exact: true }).click();
    await expect(page.getByText('Multiclass Confusion Matrix (Full Test Split)', { exact: true })).toBeVisible();
    const cells = page.getByLabel(/^\d+ test samples with actual Class [012] predicted Class [012]$/);
    await expect(cells).toHaveCount(9);
    const total = await cells.evaluateAll(nodes => nodes.reduce((sum,node) => sum + Number(node.getAttribute('aria-label')!.match(/^\d+/)![0]),0));
    expect(total).toBe(150);
});

test('saved runs survive reload and explicitly reapply their complete recipe', async ({ page }) => {
    await load(page);
    await applyPreset(page, RECIPES.xor.title);
    const history = await saved(page);
    await history.getByRole('button', { name: 'Save current run', exact: true }).click();
    await expect(history.getByRole('article')).toHaveCount(1);
    const record = await page.evaluate(key => JSON.parse(localStorage.getItem(key)!).records[0], MEMORY_KEY);
    expect(record.snapshot.model.step).toBe(0);
    await page.reload();
    await expect(history.getByRole('article')).toHaveCount(1);
    expect(await page.evaluate(key => JSON.parse(localStorage.getItem(key)!).records[0], MEMORY_KEY)).toEqual(record);
    await applyPreset(page, RECIPES.threeClass.title);
    await saved(page);
    const row = history.locator('tbody tr').first();
    await row.locator('summary').click();
    await row.getByRole('button', { name: 'Apply saved recipe', exact: true }).click();
    const confirmation = page.getByRole('alertdialog', { name: 'Apply saved recipe', exact: true });
    await expect(confirmation).toContainText('fresh model');
    expect(new URL(page.url()).hash).not.toBe('');
    await confirmation.getByRole('button', { name: 'Confirm', exact: true }).click();
    await expect(confirmation).toBeHidden();
    await expectedRecipe(page, RECIPES.xor);
});

test('cross-tab saved-run memory follows native localStorage events', async ({ context, page }) => {
    await load(page);
    const peer = await context.newPage();
    collectErrors(peer);
    await load(peer);
    const peerHistory = await saved(peer);
    const owner = await saved(page);
    await owner.getByRole('button', { name: 'Save current run' }).click();
    await expect(owner.getByRole('article')).toHaveCount(1);
    await expect(peerHistory.getByRole('article')).toHaveCount(1);
    await page.evaluate(key => localStorage.removeItem(key), MEMORY_KEY);
    await expect(owner.getByRole('article')).toHaveCount(1);
    await expect(peerHistory.getByRole('article')).toHaveCount(0);
    expect(browserErrors.get(peer)).toEqual([]);
});

test('paused scientific evidence, checkpoints, records and code preference survive all navigation and guidance', async ({ page }) => {
    await load(page);
    await applyPreset(page, RECIPES.xor.title);
    await learning(page);
    await page.getByRole('combobox', { name: 'Steps per frame' }).selectOption('50');
    await page.getByRole('button', { name: /^(Start|Resume) training$/ }).click();
    await expect.poll(async () => Number(await transport(page).getAttribute('data-model-step'))).toBeGreaterThanOrEqual(50);
    await page.getByRole('button', { name: 'Pause training', exact: true }).click();
    const paused = await converge(page);
    await page.getByRole('button', { name: 'Run one training step' }).click();
    await expectEvidence(page, paused + 1);
    const history = await saved(page);
    await history.getByRole('button', { name: 'Save current run' }).click();
    await expect(history.getByRole('article')).toHaveCount(1);
    const records = await page.evaluate(key => localStorage.getItem(key), MEMORY_KEY);
    await learning(page);
    const before = await identity(page);
    await utility(page, 'Session checkpoints');
    const slider = page.getByRole('slider', { name: 'Checkpoint timeline' });
    const checkpoint = { max:await slider.getAttribute('max'), value:await slider.inputValue(), label:await slider.getAttribute('aria-valuetext') };
    expect(Number(checkpoint.max)).toBeGreaterThanOrEqual(1);
    await page.keyboard.press('Escape');
    await utility(page, 'Export / import');
    await page.getByRole('tab', { name: 'Code', exact: true }).click();
    await page.getByRole('tab', { name: 'NumPy', exact: true }).click();
    await page.keyboard.press('Escape');
    for (const mode of ['beginner', 'explore', 'lab']) {
        await utility(page, 'Guidance');
        await page.getByRole('combobox', { name: 'Explanation density' }).selectOption(mode);
        await page.keyboard.press('Escape');
        for (const view of ['Setup', 'Network', 'Results', 'Inspect'] as const) {
            await workspace(page, view);
            expect(await identity(page)).toEqual(before);
        }
        await utility(page, 'Export / import');
        await page.getByRole('tab', { name: 'Code', exact: true }).click();
        await expect(page.getByRole('tab', { name: 'NumPy', exact: true })).toHaveAttribute('aria-selected', 'true');
        await page.keyboard.press('Escape');
        await utility(page, 'Session checkpoints');
        expect({ max:await slider.getAttribute('max'), value:await slider.inputValue(), label:await slider.getAttribute('aria-valuetext') }).toEqual(checkpoint);
        await page.keyboard.press('Escape');
        await learning(page);
        await expectEvidence(page, paused+1);
        expect(await page.evaluate(key => localStorage.getItem(key), MEMORY_KEY)).toBe(records);
    }
});

for (const width of [1440,1401,1280,950,901,800,390,320]) {
    test(`global controls remain inside the header at ${width}px`, async ({ page }) => {
        await page.setViewportSize({ width, height: 844 });
        await load(page);
        const controls = await page.getByRole('banner').evaluate(header => {
            const box = header.getBoundingClientRect();
            return Array.from(header.querySelectorAll('button,select')).map(control => {
                const rect = control.getBoundingClientRect();
                return { name:control.getAttribute('aria-label') ?? control.textContent, contained:rect.left>=box.left-1 && rect.right<=box.right+1 && rect.top>=box.top-1 && rect.bottom<=box.bottom+1, inViewport:rect.left>=-1 && rect.right<=innerWidth+1, height:rect.height };
            });
        });
        expect(controls.length).toBe(6);
        expect(controls.filter(control=>!control.contained||!control.inViewport||control.height<44)).toEqual([]);
    });
}

test('reduced motion disables shell animation and transitions', async ({ page }) => {
    await page.emulateMedia({ reducedMotion:'reduce' });
    await load(page);
    await page.getByRole('button', { name:'Start training' }).click();
    await expect(transport(page)).toHaveAttribute('data-status','running');
    const motion = await page.locator('.atelier button, .atelier-tabs').evaluateAll(nodes => nodes.map(node => ({ animation:getComputedStyle(node).animationName, transition:getComputedStyle(node).transitionDuration })));
    expect(motion.every(value=>value.animation==='none' && value.transition.split(',').every(duration=>Number.parseFloat(duration)===0))).toBe(true);
    await page.getByRole('button', { name:'Pause training', exact:true }).click();
});

for (const width of [1280,390,320]) {
    test(`canonical concept help and keyboard shortcuts remain reachable at ${width}px`, async ({ page }) => {
        await page.setViewportSize({ width,height:844 });
        await load(page);
        const before = await identity(page);
        await utility(page,'Shortcuts & help');
        const dialog = page.getByRole('dialog',{name:'Shortcuts & help'});
        await expect(dialog.locator('kbd')).toHaveCount(3);
        const concepts = dialog.getByRole('region',{name:'Concept library'});
        await expect(concepts.locator('details')).toHaveCount(9);
        for (const term of ['Data loss','Checkpoint','Epoch','Train/test split','Learning rate']) {
            const summary = concepts.locator('summary').filter({hasText:new RegExp(`^${term}$`)});
            await summary.click();
            await touchTarget(summary);
            const details = summary.locator('..');
            await expect(details.locator('div')).toBeVisible();
            await summary.click();
            await expect(details.locator('div')).toBeHidden();
        }
        await page.keyboard.press('Escape');
        await expect(page.getByRole('button',{name:'Utilities',exact:true})).toBeFocused();
        expect(await identity(page)).toEqual(before);
    });
}

for (const width of [320,390]) test.describe(`${width}px touch controls`,()=>{
    test.use({viewport:{width,height:844},hasTouch:true,isMobile:true});
    test('graph, results, diagnostics, lessons and utilities remain reachable',async({page})=>{
        await load(page);
        const before = await identity(page);
        await workspace(page,'Network');
        for (const name of ['Zoom out graph','Zoom in graph','Fit graph to view','Show all edges','Show only strong edges','Show positive edges','Show negative edges']) {
            const button = page.getByRole('button',{name,exact:true});
            await button.scrollIntoViewIfNeeded();
            await touchTarget(button);
            await button.tap();
        }
        await workspace(page,'Results');
        await page.getByRole('tab',{name:'Prediction',exact:true}).tap();
        const overlays=page.getByLabel('Decision overlay controls',{exact:true});
        for(const name of ['Output','Uncertain','Errors','Split']) {
            const button=overlays.getByRole('button',{name,exact:true});
            await button.scrollIntoViewIfNeeded();
            await touchTarget(button);
            await button.tap();
            await expect(button).toHaveAttribute('aria-pressed','true');
        }
        await workspace(page,'Inspect');
        for(const name of ['Trace','Activations','Gradients']) {
            const tab=page.getByRole('tab',{name,exact:true});
            await touchTarget(tab);await tab.tap();
        }
        await utility(page,'Export / import');
        await page.getByRole('tab',{name:'Code',exact:true}).tap();
        const formats=page.getByRole('tablist',{name:'Code format'}).getByRole('tab');
        for(const format of await formats.all()){await touchTarget(format);await format.tap();}
        await page.getByRole('button',{name:'Close Export / import',exact:true}).tap();
        await page.getByRole('button',{name:'Lessons',exact:true}).tap();
        await expect(page.getByRole('heading',{name:'Learn by experimenting'})).toBeVisible();
        await workspace(page,'Network');
        expect(await identity(page)).toEqual(before);
        expect(await page.evaluate(()=>document.documentElement.scrollWidth-document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
    });
});
