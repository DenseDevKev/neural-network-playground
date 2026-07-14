import { expect, test, type Locator, type Page } from '@playwright/test';

interface RecipeExpectation {
    data: string;
    network: string;
    loss: string;
}

const RECIPES = {
    regression: {
        title: 'Regression with No Hidden Layer',
        data: 'Regression plane regression',
        network: '2 -> none -> 1, tanh',
        loss: 'mean squared error, batch 10',
    },
    xor: {
        title: 'XOR Needs Hidden Layers',
        data: 'XOR classification',
        network: '2 -> 4 x 4 -> 1, tanh',
        loss: 'binary cross entropy, batch 10',
    },
    threeClass: {
        title: 'Three-Class Softmax Lab',
        data: 'Three-class clusters classification',
        network: '2 -> 6 x 6 -> 3, tanh',
        loss: 'categorical cross entropy, batch 10',
    },
} as const;

const browserErrors = new WeakMap<Page, string[]>();

test.beforeEach(({ page }) => {
    const errors: string[] = [];
    browserErrors.set(page, errors);
    page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
    page.on('console', (message) => {
        if (message.type() === 'error') errors.push(`console.error: ${message.text()}`);
    });
});

test.afterEach(async ({ page }) => {
    const errors = browserErrors.get(page) ?? [];
    expect(errors, `Unexpected browser errors:\n${errors.join('\n')}`).toEqual([]);
});

function statusBar(page: Page): Locator {
    return page.getByRole('status', { name: 'Status bar' });
}

function timeline(page: Page): Locator {
    return page.getByRole('region', { name: 'Timeline strip' });
}

function currentRun(page: Page): Locator {
    return page.locator('section[role="region"][aria-label="Current run"]');
}

function parseNumber(value: string | undefined): number {
    if (!value) return -1;
    return Number(value.replaceAll(',', ''));
}

async function readStatusStep(page: Page): Promise<number> {
    const match = (await statusBar(page).innerText()).match(/\bSTEP\s+([\d,]+)/);
    return parseNumber(match?.[1]);
}

async function readFullEvaluationStep(page: Page): Promise<number> {
    const match = (await currentRun(page).innerText()).match(/Full evaluation \d+ at step ([\d,]+)/);
    return parseNumber(match?.[1]);
}

function formattedStep(step: number): string {
    return step.toLocaleString('en-US');
}

async function expectEvidenceAtStep(page: Page, step: number): Promise<void> {
    const value = formattedStep(step);
    const exactStep = `${value}(?![0-9,])`;

    await expect(statusBar(page)).toContainText(new RegExp(String.raw`STEP\s+${exactStep}`));
    await expect(timeline(page)).toContainText(new RegExp(String.raw`Step\s+${exactStep}`));
    await expect(currentRun(page)).toContainText(
        new RegExp(String.raw`Full evaluation \d+ at step ${exactStep}`),
    );
}

async function expectModelAndEvaluationToConverge(
    page: Page,
    expectedStep?: number,
): Promise<number> {
    await expect.poll(async () => {
        const modelStep = await readStatusStep(page);
        const evaluationStep = await readFullEvaluationStep(page);
        return modelStep >= 0
            && modelStep === evaluationStep
            && (expectedStep === undefined || modelStep === expectedStep);
    }).toBe(true);

    return readStatusStep(page);
}

async function ensureRunView(page: Page): Promise<void> {
    const runButton = page.getByRole('button', { name: 'run', exact: true });
    if (await runButton.getAttribute('aria-pressed') !== 'true') {
        await runButton.click();
    }
    await expect(currentRun(page)).toBeVisible();
}

async function loadPlayground(page: Page): Promise<void> {
    await page.goto('/');
    await expect(page.getByRole('main', { name: 'Neural network playground workspace' })).toBeVisible();
    await ensureRunView(page);
    await expect(statusBar(page)).toHaveAttribute('data-status', 'idle');
    await expectModelAndEvaluationToConverge(page, 0);
    await expect(page.getByRole('slider', { name: 'Checkpoint timeline' }))
        .toHaveAttribute('aria-valuetext', 'Step 0');
}

async function openDrawer(page: Page, name: 'Presets' | 'History'): Promise<Locator> {
    const dialog = page.getByRole('dialog', { name });
    if (!await dialog.isVisible()) {
        await page.getByRole('button', { name, exact: true }).click();
    }
    await expect(dialog).toBeVisible();
    return dialog;
}

async function expectRecipe(page: Page, recipe: RecipeExpectation): Promise<void> {
    const summary = page.getByRole('region', { name: 'Recipe summary' });
    await expect(summary).toBeVisible();
    await expect(summary.getByText(recipe.data, { exact: true })).toBeVisible();
    await expect(summary.getByText(recipe.network, { exact: true })).toBeVisible();
    await expect(summary.getByText(recipe.loss, { exact: true })).toBeVisible();
    await expect(summary.getByText('Ready', { exact: true })).toBeVisible();
}

async function applyPreset(
    page: Page,
    recipe: RecipeExpectation & { title: string },
): Promise<void> {
    const dialog = await openDrawer(page, 'Presets');
    await dialog.getByRole('button', { name: `Apply preset: ${recipe.title}` }).click();
    await expect(dialog).toBeHidden();
    await expect(statusBar(page)).toHaveAttribute('data-status', 'idle');
    await expectModelAndEvaluationToConverge(page, 0);
    await expectRecipe(page, recipe);
}

test('training can pause, single-step, and restore the initial checkpoint', async ({ page }) => {
    await loadPlayground(page);
    await expectEvidenceAtStep(page, 0);

    const transport = timeline(page);
    await transport.getByRole('button', { name: '1 step per frame' }).click();
    await transport.getByRole('button', { name: 'Start training' }).click();
    await expect(statusBar(page)).toHaveAttribute('data-status', 'running');
    await expect.poll(() => readStatusStep(page)).toBeGreaterThan(0);

    await transport.getByRole('button', { name: 'Pause training' }).click();
    await expect(statusBar(page)).toHaveAttribute('data-status', 'paused');
    const pausedStep = await expectModelAndEvaluationToConverge(page);
    expect(pausedStep).toBeGreaterThan(0);
    await expectEvidenceAtStep(page, pausedStep);

    await transport.getByRole('button', { name: 'Run one training step' }).click();
    const steppedTo = pausedStep + 1;
    await expectModelAndEvaluationToConverge(page, steppedTo);
    await expectEvidenceAtStep(page, steppedTo);
    await expect(statusBar(page)).toHaveAttribute('data-status', 'paused');

    const checkpointSlider = page.getByRole('slider', { name: 'Checkpoint timeline' });
    await checkpointSlider.focus();
    await checkpointSlider.press('Home');
    await expect(checkpointSlider).toHaveAttribute('aria-valuetext', 'Step 0');
    await transport.getByRole('button', { name: 'Restore checkpoint Step 0' }).click();

    await expect(statusBar(page)).toHaveAttribute('data-status', 'paused');
    await expectModelAndEvaluationToConverge(page, 0);
    await expectEvidenceAtStep(page, 0);
    await expect(checkpointSlider).toHaveAttribute('aria-valuetext', 'Step 0');
});

test('curated presets reset complete recipes and expose multiclass evidence', async ({ page }) => {
    await loadPlayground(page);

    await applyPreset(page, RECIPES.regression);
    await expectEvidenceAtStep(page, 0);

    await applyPreset(page, RECIPES.xor);
    await expectEvidenceAtStep(page, 0);

    await applyPreset(page, RECIPES.threeClass);
    await expectEvidenceAtStep(page, 0);
    await expect(page.getByLabel('Architecture summary')).toContainText(
        'X₁, X₂ -> [6] -> [6] -> 3 outputs (softmax)',
    );

    await page.getByRole('tab', { name: 'Confusion' }).click();
    const confusionPanel = page.getByRole('tabpanel', { name: 'Confusion' });
    await expect(confusionPanel.getByText('Multiclass Confusion Matrix (Full Test Split)', { exact: true }))
        .toBeVisible();
    await expect(confusionPanel.getByLabel(
        /^\d+ test samples with actual Class [012] predicted Class [012]$/,
    )).toHaveCount(9);
});

test('saved runs survive reload and reapply their complete recipe', async ({ page }) => {
    await loadPlayground(page);
    await applyPreset(page, RECIPES.xor);
    await expectEvidenceAtStep(page, 0);

    let history = await openDrawer(page, 'History');
    const saveButton = history.getByRole('button', { name: 'Save current run' });
    await expect(saveButton).toBeEnabled();
    await saveButton.click();
    await expect(history.getByRole('article')).toHaveCount(1);
    await expect(history.getByRole('article')).toContainText('Full evaluation at step 0;');

    await page.reload();
    await expect(page.getByRole('main', { name: 'Neural network playground workspace' })).toBeVisible();
    await ensureRunView(page);
    await expect(statusBar(page)).toHaveAttribute('data-status', 'idle');
    await expectModelAndEvaluationToConverge(page, 0);

    history = await openDrawer(page, 'History');
    await expect(history.getByRole('article')).toHaveCount(1);
    await expect(history.getByRole('article')).toContainText('Full evaluation at step 0;');

    await applyPreset(page, RECIPES.threeClass);
    await expectRecipe(page, RECIPES.threeClass);

    history = await openDrawer(page, 'History');
    const savedRun = history.getByRole('article');
    await expect(savedRun).toHaveCount(1);
    await savedRun.getByRole('button', { name: 'Apply saved recipe' }).click();

    await expect(statusBar(page)).toHaveAttribute('data-status', 'idle');
    await expectModelAndEvaluationToConverge(page, 0);
    await expectRecipe(page, RECIPES.xor);
    await expectEvidenceAtStep(page, 0);
});
