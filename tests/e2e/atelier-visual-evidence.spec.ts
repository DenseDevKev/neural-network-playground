import { test, expect, type Page } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import process from 'node:process';
import { writeVisualGallery } from './atelier-visual-gallery';
import { ready, workspace, utility, applyPreset, learning, transport } from './atelier-helpers';

// Human-reviewed evidence only: opt in explicitly; never screenshot goldens.
const output = resolve(process.env.ATELIER_VISUAL_OUTPUT ?? '../../outputs/nn-forge-implementation');
const names = ['network', 'dataset', 'network-setup', 'training', 'neuron-inspect', 'learning', 'errors', 'saved', 'comparison', 'lessons', 'guided-lesson', 'trace-activations', 'gradients', 'export', 'multiclass', 'regression', 'save-failure', 'checkpoints', 'mobile-results', 'mobile-setup'];
test.skip(process.env.ATELIER_VISUAL_EVIDENCE !== '1', 'Opt-in real-engine visual evidence suite');
async function steps(page: Page, count: number) {
    for (let i = 0; i < count; i++) {
        const previous = Number(await transport(page).getAttribute('data-model-step'));
        await page.getByRole('button', { name: 'Run one training step', exact: true }).click();
        await expect(transport(page)).toHaveAttribute('data-model-step', String(previous + 1));
    }
}
for (const theme of ['light', 'dark'] as const) {
    test(`all twenty approved states, ${theme}`, async ({ page }) => {
        test.setTimeout(240_000);
        page.setDefaultTimeout(10_000);
        await mkdir(output, { recursive: true });
        await page.setViewportSize({ width: 1440, height: 1024 });
        await page.emulateMedia({ colorScheme: theme, reducedMotion: 'reduce' });
        await page.goto('./'); await ready(page);
        await page.getByLabel('Color theme').selectOption(theme);
        await applyPreset(page, 'Circle with One Hidden Layer');
        await learning(page); await steps(page, 40);
        const receipts: object[] = [];
        async function capture(id: number, fullPage = ![5, 14, 15, 16, 18].includes(id)) {
            await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
            await page.evaluate(() => document.fonts.ready);
            await page.evaluate(() => window.scrollTo(0, 0));
            await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => requestAnimationFrame(() => resolve())))));
            const stepBefore = await transport(page).getAttribute('data-model-step');
            if ([1, 5, 11, 15, 16].includes(id)) {
                const nodes = page.getByRole('group', { name: 'Select a neuron' }).getByRole('button', { name: /^(Hidden|Output)/ });
                for (const node of await nodes.all()) await expect(node).toHaveAttribute('data-grid-available', 'true');
                if (id === 15) await expect(page.getByRole('group', { name: 'Select a neuron' }).getByRole('button', { name: /^Output/ })).toHaveCount(3);
            }
            await expect(transport(page)).toHaveAttribute('data-model-step', stepBefore!);
            const dimensions = await page.evaluate(() => ({ scrollWidth: document.documentElement.scrollWidth, clientWidth: document.documentElement.clientWidth, scrollHeight: document.documentElement.scrollHeight }));
            expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth + 1);
            if ([5, 16].includes(id)) {
                await page.waitForTimeout(200); // Requested canvas/DOM synchronization diagnostic, not scientific readiness.
                const geometry = await page.evaluate(() => ({
                    canvases: [...document.querySelectorAll('canvas')].map((canvas) => { const r = canvas.getBoundingClientRect(); const t = canvas.getContext('2d')?.getTransform(); return { className: canvas.className, width: canvas.width, height: canvas.height, clientWidth: canvas.clientWidth, clientHeight: canvas.clientHeight, bounds: {x:r.x,y:r.y,width:r.width,height:r.height}, transform: t ? { a:t.a,b:t.b,c:t.c,d:t.d,e:t.e,f:t.f } : null }; }),
                    nodes: [...document.querySelectorAll('.network-node-target,.network-graph-heatmap-slot')].map((element) => { const r = element.getBoundingClientRect(); return {className:element.className,label:element.getAttribute('aria-label'),bounds:{x:r.x,y:r.y,width:r.width,height:r.height}}; }),
                }));
                await writeFile(resolve(output, `${String(id).padStart(2,'0')}-geometry-${theme}.json`), JSON.stringify(geometry,null,2));
            }
            if ([5, 11, 15, 16].includes(id)) await page.screenshot({ path: resolve(output, `${String(id).padStart(2, '0')}-${names[id - 1]}-${theme}-viewport.png`), animations: 'disabled' });
            await page.screenshot({ path: resolve(output, `${String(id).padStart(2, '0')}-${names[id - 1]}-${theme}.png`), fullPage, animations: 'disabled' });
            if ([2, 3, 4, 20].includes(id)) {
                const fields = page.locator('.atelier-setup-fields');
                await fields.evaluate((element) => { element.scrollTop = element.scrollHeight; });
                await page.screenshot({ path: resolve(output, `${String(id).padStart(2, '0')}-${names[id - 1]}-${theme}-controls.png`), animations: 'disabled' });
                await fields.evaluate((element) => { element.scrollTop = 0; });
            }
            receipts.push({ id, name: names[id - 1], theme, url: page.url(), step: await transport(page).getAttribute('data-model-step'), viewport: page.viewportSize(), dimensions, fullPage });
            await writeFile(resolve(output, `capture-receipts-${theme}.json`), JSON.stringify(receipts, null, 2));
        }
        await workspace(page, 'Network'); await capture(1);
        await workspace(page, 'Setup'); await capture(2);
        const setup = page.getByRole('navigation', { name: 'Setup sections' });
        await setup.getByRole('button', { name: 'Inputs & layers', exact: true }).click(); await capture(3);
        await setup.getByRole('button', { name: 'Training', exact: true }).click(); await capture(4);
        await workspace(page, 'Network');
        await page.getByRole('group', { name: 'Select a neuron', exact: true }).getByRole('button', { name: 'Hidden 1, Neuron 1', exact: true }).click();
        await expect(page.getByRole('region', { name: 'Selected neuron details' })).toBeVisible(); await capture(5);
        await learning(page); await capture(6);
        await page.getByRole('tab', { name: 'Prediction', exact: true }).click();
        await page.getByRole('button', { name: 'Errors', exact: true }).click();
        await page.getByRole('tab', { name: 'Errors & confusion', exact: true }).click(); await steps(page, 1); await capture(7);
        await page.getByRole('button', { name: 'Saved runs', exact: true }).click();
        await page.getByRole('button', { name: 'Save current run', exact: true }).click();
        await expect(page.locator('.saved-runs').getByRole('article')).toHaveCount(1);
        await learning(page); await steps(page, 10);
        await page.getByRole('button', { name: 'Saved runs', exact: true }).click();
        await page.getByRole('button', { name: 'Save current run', exact: true }).click();
        await expect(page.locator('.saved-runs').getByRole('article')).toHaveCount(2); await capture(8);
        const choices = page.getByRole('checkbox', { name: /^Compare / });
        await choices.nth(0).check(); await choices.nth(1).check();
        await page.getByRole('button', { name: 'Compare selected', exact: true }).click(); await capture(9);
        await page.getByRole('button', { name: '← Saved runs', exact: true }).click();
        await page.getByRole('button', { name: 'Lessons', exact: true }).click();
        await page.locator('.lesson-list').getByRole('button', { name: /Circle With One Hidden Layer/ }).click(); await capture(10);
        await page.getByRole('button', { name: 'Start lesson and reset', exact: true }).click(); await ready(page);
        await page.getByRole('button', { name: 'Next lesson step', exact: true }).click();
        await workspace(page, 'Network'); await capture(11);
        await page.getByRole('button', { name: 'Exit lesson', exact: true }).click();
        await workspace(page, 'Inspect'); await steps(page, 2);
        await page.getByRole('button', { name: 'Trace prediction', exact: true }).click();
        await expect(page.locator('.inspection__trace-result')).toBeVisible(); await capture(12);
        await page.getByRole('tab', { name: 'Gradients', exact: true }).click();
        await page.getByRole('button', { name: 'Preview backprop', exact: true }).click();
        await expect(page.getByRole('region', { name: 'Slow-motion backprop preview' })).toContainText('Preview from step 2');
        await page.getByRole('button', { name: 'Probe loss surface', exact: true }).click();
        await expect(page.getByRole('region', { name: 'Loss landscape probe', exact: true }).getByRole('img')).toBeVisible(); await capture(13);
        await utility(page, 'Export / import'); await page.getByRole('tab', { name: 'Code', exact: true }).click(); await expect(page.locator('.code-export__code')).toBeVisible(); await capture(14);
        await page.getByRole('button', { name: 'Close Export / import', exact: true }).click();
        await applyPreset(page, 'Three-Class Softmax Lab'); await steps(page, 15); await workspace(page, 'Network'); await capture(15);
        await applyPreset(page, 'Regression with No Hidden Layer'); await steps(page, 15); await workspace(page, 'Network'); await capture(16);
        await page.evaluate(() => {
            const original = Storage.prototype.setItem;
            Storage.prototype.setItem = function (key: string, value: string) {
                if (this === window.localStorage && key === 'nn-playground-experiment-memory-v2') throw new DOMException('Visual evidence quota failure', 'QuotaExceededError');
                return original.call(this, key, value);
            };
        });
        await page.getByRole('button', { name: 'Saved runs', exact: true }).click();
        await page.getByRole('button', { name: 'Save current run', exact: true }).click();
        await expect(page.getByRole('button', { name: 'Retry saving', exact: true })).toBeEnabled(); await capture(17);
        await workspace(page, 'Network'); await utility(page, 'Session checkpoints');
        await expect(page.getByLabel('Checkpoint timeline')).toBeVisible(); await capture(18);
        await page.getByRole('button', { name: 'Close Session checkpoints', exact: true }).click();
        await applyPreset(page, 'Circle with One Hidden Layer'); await steps(page, 15); await workspace(page, 'Results');
        await page.getByRole('tab', { name: 'Prediction', exact: true }).click();
        await page.setViewportSize({ width: 390, height: 844 }); await capture(19, true);
        await page.setViewportSize({ width: 360, height: 844 }); await workspace(page, 'Setup');
        await setup.getByRole('button', { name: 'Inputs & layers', exact: true }).click(); await capture(20, true);
    });
}

test.afterAll(async () => {
    if (process.env.ATELIER_VISUAL_EVIDENCE === '1') await writeVisualGallery(output, names);
});
