import { expect, test, type Locator, type Page } from '@playwright/test';

interface RecipeExpectation {
    data: string;
    network: string;
    training: string;
    loss: string;
    features: string;
    featureCount: string;
    datasetSettings: string;
}

interface PausedRunInvariant {
    hash: string;
    step: number;
    evaluationStep: number;
    modelGeneration: string;
    modelRevision: string;
    checkpointMax: string;
    checkpointValue: string;
    checkpointLabel: string;
    checkpointSummary: string;
    savedRunCount: number;
}

const RECIPES = {
    regression: {
        title: 'Regression with No Hidden Layer',
        data: 'Regression plane regression',
        network: '2 -> none -> 1, tanh',
        training: 'SGD, lr 0.01',
        loss: 'mean squared error, batch 10',
        features: 'x, y',
        featureCount: '2 features',
        datasetSettings: 'Dataset settings: 300 samples, 5 noise, 50% train',
    },
    xor: {
        title: 'XOR Needs Hidden Layers',
        data: 'XOR classification',
        network: '2 -> 4 x 4 -> 1, tanh',
        training: 'SGD, lr 0.03',
        loss: 'binary cross entropy, batch 10',
        features: 'x, y',
        featureCount: '2 features',
        datasetSettings: 'Dataset settings: 300 samples, 0 noise, 50% train',
    },
    threeClass: {
        title: 'Three-Class Softmax Lab',
        data: 'Three-class clusters classification',
        network: '2 -> 6 x 6 -> 3, tanh',
        training: 'SGD, lr 0.03',
        loss: 'categorical cross entropy, batch 10',
        features: 'x, y',
        featureCount: '2 features',
        datasetSettings: 'Dataset settings: 300 samples, 0.05 noise, 50% train',
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
    return page.getByRole('group', { name: 'Status bar' });
}

function timeline(page: Page): Locator {
    return page.getByRole('region', { name: 'Timeline strip' });
}

function currentRun(page: Page): Locator {
    return page.locator('section[role="region"][aria-label="Current run"]');
}

function audienceMode(page: Page): Locator {
    return page.getByRole('combobox', { name: 'Workspace profile' });
}

function advancedTools(page: Page): Locator {
    return page.getByRole('button', { name: 'Advanced Tools' });
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

async function closeDrawer(page: Page, name: 'Presets' | 'History'): Promise<void> {
    const dialog = page.getByRole('dialog', { name });
    await dialog.getByRole('button', { name: `Close ${name}` }).click();
    await expect(dialog).toBeHidden();
}

async function setAdvancedTools(page: Page, open: boolean): Promise<void> {
    const trigger = advancedTools(page);
    if ((await trigger.getAttribute('aria-expanded')) !== String(open)) {
        await trigger.click();
    }
    await expect(trigger).toHaveAttribute('aria-expanded', String(open));
}

async function readSavedRunCount(page: Page): Promise<number> {
    const history = await openDrawer(page, 'History');
    const count = await history.getByRole('article').count();
    await closeDrawer(page, 'History');
    return count;
}

async function readPausedRunInvariant(page: Page): Promise<PausedRunInvariant> {
    await expect(statusBar(page)).toHaveAttribute('data-status', 'paused');
    const run = currentRun(page);
    const checkpoint = page.getByRole('slider', { name: 'Checkpoint timeline' });
    const checkpointControls = page.getByLabel('Checkpoint timeline controls');

    return {
        hash: await page.evaluate(() => window.location.hash),
        step: await readStatusStep(page),
        evaluationStep: await readFullEvaluationStep(page),
        modelGeneration: (await run.getAttribute('data-model-generation')) ?? '',
        modelRevision: (await run.getAttribute('data-model-revision')) ?? '',
        checkpointMax: (await checkpoint.getAttribute('max')) ?? '',
        checkpointValue: await checkpoint.inputValue(),
        checkpointLabel: (await checkpoint.getAttribute('aria-valuetext')) ?? '',
        checkpointSummary: await checkpointControls.innerText(),
        savedRunCount: await readSavedRunCount(page),
    };
}

async function expectPausedRunInvariant(
    page: Page,
    expected: PausedRunInvariant,
): Promise<void> {
    expect(await readPausedRunInvariant(page)).toEqual(expected);
}

async function showCodeAndExpectNumPy(page: Page): Promise<void> {
    await ensureRunView(page);
    await setAdvancedTools(page, true);
    await page.getByRole('tab', { name: 'Code', exact: true }).click();
    await expect(page.getByRole('tabpanel', { name: 'Code' })).toBeVisible();
    await expect(page.getByRole('tablist', { name: 'Code format' })).toBeVisible();
    await expect(page.getByRole('tab', { name: 'NumPy', exact: true }))
        .toHaveAttribute('aria-selected', 'true');
}

async function expectFullyInViewport(page: Page, locator: Locator): Promise<void> {
    await expect(locator).toBeVisible();
    const box = await locator.boundingBox();
    const viewport = page.viewportSize();
    expect(box, 'expected a measurable element box').not.toBeNull();
    expect(viewport, 'expected a configured viewport').not.toBeNull();
    if (!box || !viewport) return;
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.y).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(viewport.width + 1);
    expect(box.y + box.height).toBeLessThanOrEqual(viewport.height + 1);
}

async function expectConceptHelpInViewport(page: Page, concept: 'Data loss' | 'Checkpoint') {
    const trigger = page.getByRole('button', { name: `Learn about ${concept}` }).first();
    await trigger.click();
    const panel = page.getByRole('region', { name: concept });
    await expectFullyInViewport(page, panel);
    await trigger.press('Escape');
    await expect(panel).toBeHidden();
    await expect(trigger).toBeFocused();
}

async function expectMinimumTouchTarget(locator: Locator, minimum = 44): Promise<void> {
    const count = await locator.count();
    expect(count).toBeGreaterThan(0);
    for (let index = 0; index < count; index++) {
        const target = locator.nth(index);
        await expect(target).toBeVisible();
        const box = await target.boundingBox();
        expect(box, 'expected a measurable touch target').not.toBeNull();
        if (!box) continue;
        expect(box.width).toBeGreaterThanOrEqual(minimum);
        expect(box.height).toBeGreaterThanOrEqual(minimum);
    }
}

function graphAndEvidenceTargets(page: Page): readonly Locator[] {
    const toolbar = page.getByRole('toolbar', { name: 'Network graph toolbar' });
    const modes = toolbar.getByRole('group', { name: 'Topology view mode' });
    const legend = page.getByLabel('Edge weight legend', { exact: true });
    const boundaryPanel = page.getByRole('tabpanel', { name: 'Boundary', exact: true });
    const overlays = boundaryPanel.getByLabel('Decision overlay controls', { exact: true });
    return [
        toolbar.getByRole('button', { name: 'Zoom out graph', exact: true }),
        toolbar.getByRole('button', { name: 'Zoom in graph', exact: true }),
        toolbar.getByRole('button', { name: 'Fit graph to view', exact: true }),
        modes.getByRole('button', { name: 'Weights', exact: true }),
        modes.getByRole('button', { name: 'Activations', exact: true }),
        legend.getByRole('button', { name: 'Show all edges', exact: true }),
        legend.getByRole('button', { name: 'Show only strong edges', exact: true }),
        legend.getByRole('button', { name: 'Show positive edges', exact: true }),
        legend.getByRole('button', { name: 'Show negative edges', exact: true }),
        overlays.getByRole('button', { name: 'Output', exact: true }),
        overlays.getByRole('button', { name: 'Uncertain', exact: true }),
        overlays.getByRole('button', { name: 'Errors', exact: true }),
        overlays.getByRole('button', { name: 'Split', exact: true }),
    ];
}

async function expectGraphAndEvidenceTargets(page: Page): Promise<void> {
    for (const target of graphAndEvidenceTargets(page)) {
        await expect(target).toHaveCount(1);
        await target.scrollIntoViewIfNeeded();
        await expectFullyInViewport(page, target);
        await expectMinimumTouchTarget(target);
    }
}

async function touchTap(page: Page, locator: Locator): Promise<void> {
    const box = await locator.boundingBox();
    expect(box, 'expected a measurable touch target').not.toBeNull();
    if (!box) return;
    await page.touchscreen.tap(box.x + box.width / 2, box.y + box.height / 2);
}

function cssTimeToMs(value: string): number {
    const trimmed = value.trim();
    if (trimmed.endsWith('ms')) return Number.parseFloat(trimmed);
    if (trimmed.endsWith('s')) return Number.parseFloat(trimmed) * 1000;
    return Number.NaN;
}

async function expectRecipe(page: Page, recipe: RecipeExpectation): Promise<void> {
    const summary = page.getByRole('region', { name: 'Recipe summary' });
    await expect(summary).toBeVisible();
    await expect(summary.getByText(recipe.data, { exact: true })).toBeVisible();
    await expect(summary.getByText(recipe.network, { exact: true })).toBeVisible();
    await expect(summary.getByText(recipe.training, { exact: true })).toBeVisible();
    await expect(summary.getByText(recipe.loss, { exact: true })).toBeVisible();
    await expect(summary.getByText(recipe.features, { exact: true })).toBeVisible();
    await expect(summary.getByText(recipe.featureCount, { exact: true })).toBeVisible();
    await expect(summary.getByText('Ready', { exact: true })).toBeVisible();

    const workspaceView = page.getByRole('group', { name: 'Workspace view' });
    await workspaceView.getByRole('button', { name: 'build', exact: true }).click();
    const dataConfiguration = page.getByRole('region', { name: 'Data', exact: true });
    await expect(dataConfiguration.getByLabel(recipe.datasetSettings, { exact: true })).toBeVisible();

    await workspaceView.getByRole('button', { name: 'run', exact: true }).click();
    await expect(currentRun(page)).toBeVisible();
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
    await expect(statusBar(page)).toBeVisible();
    await expect(statusBar(page).locator('xpath=ancestor-or-self::*[@aria-live or @role="status" or @role="alert"]')).toHaveCount(0);

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
    await history.getByRole('button', { name: 'Close History' }).click();
    await expect(history).toBeHidden();

    await expect(statusBar(page)).toHaveAttribute('data-status', 'idle');
    await expectModelAndEvaluationToConverge(page, 0);
    await expectRecipe(page, RECIPES.xor);
    await expectEvidenceAtStep(page, 0);
});

test('paused scientific state survives every audience profile and disclosure state', async ({ page }) => {
    await loadPlayground(page);
    await applyPreset(page, RECIPES.xor);

    const transport = timeline(page);
    await transport.getByRole('button', { name: '50 steps per frame' }).click();
    await transport.getByRole('button', { name: 'Start training' }).click();
    await expect(statusBar(page)).toHaveAttribute('data-status', 'running');
    await expect.poll(() => readStatusStep(page)).toBeGreaterThanOrEqual(50);
    await transport.getByRole('button', { name: 'Pause training' }).click();
    await expect(statusBar(page)).toHaveAttribute('data-status', 'paused');
    const pausedStep = await expectModelAndEvaluationToConverge(page);

    await transport.getByRole('button', { name: 'Run one training step' }).click();
    await expectModelAndEvaluationToConverge(page, pausedStep + 1);

    const history = await openDrawer(page, 'History');
    await history.getByRole('button', { name: 'Save current run' }).click();
    await expect(history.getByRole('article')).toHaveCount(1);
    await closeDrawer(page, 'History');

    await setAdvancedTools(page, true);
    await page.getByRole('tab', { name: 'Code', exact: true }).click();
    await page.getByRole('tab', { name: 'NumPy', exact: true }).click();
    await expect(page.getByRole('tab', { name: 'NumPy', exact: true }))
        .toHaveAttribute('aria-selected', 'true');

    const baseline = await readPausedRunInvariant(page);
    expect(baseline.hash.length).toBeGreaterThan(2);
    expect(baseline.step).toBeGreaterThanOrEqual(51);
    expect(baseline.evaluationStep).toBe(baseline.step);
    expect(baseline.modelGeneration).not.toBe('');
    expect(baseline.modelRevision).not.toBe('');
    expect(Number(baseline.checkpointMax)).toBeGreaterThanOrEqual(1);
    expect(baseline.savedRunCount).toBe(1);

    for (const mode of ['beginner', 'explore', 'lab'] as const) {
        await audienceMode(page).selectOption(mode);
        const defaultOpen = mode === 'lab';
        await expect(advancedTools(page)).toHaveAttribute('aria-expanded', String(defaultOpen));
        await expectPausedRunInvariant(page, baseline);

        await setAdvancedTools(page, !defaultOpen);
        await expectPausedRunInvariant(page, baseline);

        await showCodeAndExpectNumPy(page);
        await expectPausedRunInvariant(page, baseline);

        await setAdvancedTools(page, false);
        await expectPausedRunInvariant(page, baseline);
    }
});

test('concept help remains fully visible in the desktop Run workspace', async ({ page }) => {
    await loadPlayground(page);
    await expectConceptHelpInViewport(page, 'Data loss');
    await expectConceptHelpInViewport(page, 'Checkpoint');
});

test('reduced motion collapses shell animation and transition timing', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await loadPlayground(page);
    expect(await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches)).toBe(true);

    const transport = timeline(page);
    await transport.getByRole('button', { name: 'Start training' }).click();
    await expect(statusBar(page)).toHaveAttribute('data-status', 'running');

    const motion = await page.evaluate(() => {
        const statusDot = document.querySelector<HTMLElement>('.forge-statusbar__dot');
        const phaseButton = document.querySelector<HTMLElement>('.forge-phase__opt');
        if (!statusDot || !phaseButton) throw new Error('release motion targets are missing');
        const statusStyle = getComputedStyle(statusDot);
        const phaseStyle = getComputedStyle(phaseButton);
        return {
            animationDurations: statusStyle.animationDuration.split(','),
            animationIterations: statusStyle.animationIterationCount.split(','),
            transitionDurations: phaseStyle.transitionDuration.split(','),
        };
    });

    expect(motion.animationDurations.every((value) => cssTimeToMs(value) <= 1)).toBe(true);
    expect(motion.animationIterations.every((value) => value.trim() === '1')).toBe(true);
    expect(motion.transitionDurations.every((value) => cssTimeToMs(value) <= 1)).toBe(true);

    await transport.getByRole('button', { name: 'Pause training' }).click();
    await expect(statusBar(page)).toHaveAttribute('data-status', 'paused');
});

test.describe('800px compact shell', () => {
    test.use({ viewport: { width: 800, height: 844 } });

    test('keyboard shortcuts disclosure hides definitions when closed', async ({ page }) => {
        await page.route('https://fonts.googleapis.com/**', (route) => (
            route.fulfill({ contentType: 'text/css', body: '' })
        ));
        await loadPlayground(page);

        const details = timeline(page).getByRole('group', { name: 'Keyboard shortcuts' });
        const summary = details.locator('summary');
        const definitions = details.locator('dl');

        await expect(details).not.toHaveAttribute('open', '');
        await expect(definitions).toBeHidden();
        await expect(definitions).not.toHaveCSS('display', 'grid');

        await expect(summary).toBeVisible();
        await summary.click();
        await expect(details).toHaveAttribute('open', '');
        await expect(definitions).toBeVisible();

        await summary.click();
        await expect(details).not.toHaveAttribute('open', '');
        await expect(definitions).toBeHidden();
        await expect(definitions).not.toHaveCSS('display', 'grid');
    });
});

test.describe('390px graph and evidence targets', () => {
    test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });

    test('keeps every compact graph and evidence target reachable', async ({ page }) => {
        await loadPlayground(page);
        const boundaryTab = page.getByRole('tab', { name: 'Boundary', exact: true });
        await boundaryTab.click();
        await expect(boundaryTab).toHaveAttribute('aria-selected', 'true');
        await expect(page.getByRole('dialog')).toHaveCount(0);
        await expectGraphAndEvidenceTargets(page);

        const overflow = await page.evaluate(() => {
            const shell = document.querySelector<HTMLElement>('.forge-shell');
            if (!shell) throw new Error('forge shell is missing');
            return {
                document: document.documentElement.scrollWidth
                    <= document.documentElement.clientWidth + 1,
                shell: shell.scrollWidth <= shell.clientWidth + 1,
            };
        });
        expect(overflow).toEqual({ document: true, shell: true });

        const toolbar = page.getByRole('toolbar', { name: 'Network graph toolbar' });
        const summary = page.locator(
            '.forge-buildrun__topology-stage .network-graph-summary',
        );
        const toolbarBox = await toolbar.boundingBox();
        const summaryBox = await summary.boundingBox();
        expect(toolbarBox).not.toBeNull();
        expect(summaryBox).not.toBeNull();
        if (toolbarBox && summaryBox) {
            expect(summaryBox.y - (toolbarBox.y + toolbarBox.height))
                .toBeGreaterThanOrEqual(6);
        }
    });
});

test.describe('320px touch shell', () => {
    test.use({
        viewport: { width: 320, height: 844 },
        hasTouch: true,
        isMobile: true,
    });

    test('keeps critical controls reachable, sized, focused, and unclipped', async ({ page }) => {
        await loadPlayground(page);
        await expectConceptHelpInViewport(page, 'Data loss');
        await expectConceptHelpInViewport(page, 'Checkpoint');

        const workspaceView = page.getByRole('group', { name: 'Workspace view' });
        const criticalTargets = [
            workspaceView.getByRole('button', { name: 'build', exact: true }),
            workspaceView.getByRole('button', { name: 'run', exact: true }),
            audienceMode(page),
            page.getByRole('button', { name: 'Presets', exact: true }),
            page.getByRole('button', { name: 'Lessons', exact: true }),
            page.getByRole('button', { name: 'History', exact: true }),
            advancedTools(page),
            page.getByRole('button', { name: 'Start training' }),
            timeline(page).getByRole('button', { name: '10 steps per frame' }),
            timeline(page).getByRole('button', { name: 'Run one training step' }),
            timeline(page).getByRole('button', { name: 'Reset model and data' }),
            page.getByRole('tab', { name: 'Boundary', exact: true }),
            page.getByRole('tab', { name: 'Loss', exact: true }),
            page.getByRole('tab', { name: 'Confusion', exact: true }),
        ];
        for (const target of criticalTargets) await expectMinimumTouchTarget(target);

        const noGlobalOverflow = await page.evaluate(() => {
            const shell = document.querySelector<HTMLElement>('.forge-shell');
            if (!shell) throw new Error('forge shell is missing');
            return {
                document: document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1,
                shell: shell.scrollWidth <= shell.clientWidth + 1,
            };
        });
        expect(noGlobalOverflow).toEqual({ document: true, shell: true });

        await touchTap(page, advancedTools(page));
        await expect(advancedTools(page)).toHaveAttribute('aria-expanded', 'true');
        const codeTab = page.getByRole('tab', { name: 'Code', exact: true });
        await expectMinimumTouchTarget(codeTab);
        await codeTab.click();
        await expect(page.getByRole('tabpanel', { name: 'Code' })).toBeVisible();
        const codeFormatList = page.getByRole('tablist', { name: 'Code format' });
        await expect(codeFormatList).toBeVisible();
        const codeFormats = codeFormatList.getByRole('tab');
        await expectMinimumTouchTarget(codeFormats);
        await page.keyboard.press('Escape');
        await expect(advancedTools(page)).toHaveAttribute('aria-expanded', 'false');
        await expect(advancedTools(page)).toBeFocused();

        const historyTrigger = page.getByRole('button', { name: 'History', exact: true });
        await touchTap(page, historyTrigger);
        const history = page.getByRole('dialog', { name: 'History' });
        await expect(history).toBeVisible();
        const closeHistory = history.getByRole('button', { name: 'Close History' });
        await expectMinimumTouchTarget(closeHistory);
        await touchTap(page, closeHistory);
        await expect(history).toBeHidden();
        await expect(historyTrigger).toBeFocused();

        await touchTap(page, advancedTools(page));
        await expect(advancedTools(page)).toHaveAttribute('aria-expanded', 'true');
        await page.keyboard.press('Escape');
        await expect(advancedTools(page)).toHaveAttribute('aria-expanded', 'false');
        await expect(advancedTools(page)).toBeFocused();
        const focusStyle = await advancedTools(page).evaluate((element) => {
            const style = getComputedStyle(element);
            return { outlineStyle: style.outlineStyle, outlineWidth: style.outlineWidth, boxShadow: style.boxShadow };
        });
        expect(
            (
                focusStyle.outlineStyle !== 'none'
                && Number.parseFloat(focusStyle.outlineWidth) >= 2
            ) || focusStyle.boxShadow !== 'none',
        ).toBe(true);

    });
});
