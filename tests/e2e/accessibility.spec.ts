import process from 'node:process';
import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';
import { ready, transport, workspace } from './atelier-helpers';

type AxeViolations = Awaited<ReturnType<AxeBuilder['analyze']>>['violations'];
const browserErrors = new WeakMap<Page, string[]>();

test.beforeEach(({ page }) => {
    const errors: string[] = [];
    browserErrors.set(page, errors);
    page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
    page.on('console', (message) => {
        if (message.type() === 'error') errors.push(`console.error: ${message.text()}`);
    });
});

test.afterEach(({ page }) => {
    const errors = browserErrors.get(page) ?? [];
    expect(errors, `Unexpected browser errors:\n${errors.join('\n')}`).toEqual([]);
});

function formatViolations(violations: AxeViolations): string {
    return violations.map((violation) => [
        `[${violation.impact?.toUpperCase() ?? 'UNKNOWN'}] ${violation.id}: ${violation.help}`,
        violation.helpUrl,
        ...violation.nodes.map((node, index) => (
            `  ${index + 1}. ${JSON.stringify(node.target)}`
            + `${node.failureSummary ? `\n     ${node.failureSummary}` : ''}`
        )),
    ].join('\n')).join('\n\n');
}

async function loadReadyPlayground(page: Page): Promise<void> {
    await page.goto('./');
    await expect(page.getByRole('main', {
        name: 'Neural network playground workspace',
    })).toBeVisible();
    await ready(page);
    await expect(transport(page)).toHaveAttribute('data-model-step', '0');
}

const SCAN_CASES = [
    { name: '1437px Signal Atelier accessibility', width: 1437, height: 742 },
    { name: '735px Signal Atelier accessibility', width: 735, height: 860 },
    { name: '320px Signal Atelier accessibility', width: 320, height: 844 },
    { name: 'desktop accessibility', width: 1280, height: 720 },
    { name: '390px compact accessibility', width: 390, height: 844 },
] as const;

for (const scanCase of SCAN_CASES) {
    test.describe(scanCase.name, () => {
        test.use({ viewport: { width: scanCase.width, height: scanCase.height } });
        test('has no serious or critical Axe violations', async ({ page }) => {
            await loadReadyPlayground(page);
            for (const theme of ['light', 'dark']) {
                await page.getByLabel('Color theme').selectOption(theme);
                await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
                const results = await new AxeBuilder({ page }).analyze();
                const blocking = results.violations.filter(
                    ({ impact }) => impact === 'serious' || impact === 'critical',
                );
                expect(blocking, `${theme}: ${formatViolations(blocking)}`).toEqual([]);
            }
        });
    });
}

test('320px neuron targets support keyboard selection without changing the model', async ({ page, browserName }) => {
    await page.setViewportSize({ width: 320, height: 844 });
    await loadReadyPlayground(page);
    await workspace(page, 'Network');
    const nodes = page.getByRole('group', { name: 'Select a neuron' }).getByRole('button');
    await nodes.first().focus();
    await page.keyboard.press('Enter');
    await expect(nodes.first()).toHaveAttribute('aria-pressed', 'true');
    // macOS WebKit uses Option-Tab to include native buttons in sequential navigation.
    // Enter preserves focus on Input 1 in both engines; traverse with the native full-control key.
    await expect(nodes.first()).toBeFocused();
    await page.keyboard.press(browserName === 'webkit' && process.platform === 'darwin' ? 'Alt+Tab' : 'Tab');
    await expect(nodes.nth(1)).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(nodes.nth(1)).toHaveAttribute('aria-pressed', 'true');
    await expect(transport(page)).toHaveAttribute('data-model-step', '0');
    await expect(transport(page)).toHaveAttribute('data-status', 'idle');
});

test('390px evidence regions support native keyboard focus and horizontal scrolling', async ({ page, browserName }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await loadReadyPlayground(page);
    const tab = browserName === 'webkit' && process.platform === 'darwin' ? 'Alt+Tab' : 'Tab';
    async function scrollEvidence(name: string) {
        const region = page.getByRole('region', { name, exact: true });
        await page.keyboard.press(tab);
        await expect(region).toBeFocused();
        expect(await region.evaluate((node) => node.scrollWidth > node.clientWidth)).toBe(true);
        const before = await region.evaluate((node) => node.scrollLeft);
        await page.keyboard.down('ArrowRight');
        await page.waitForTimeout(150);
        await page.keyboard.up('ArrowRight');
        await expect.poll(() => region.evaluate((node) => node.scrollLeft)).toBeGreaterThan(before);
        expect(await region.evaluate((node) => getComputedStyle(node).outlineStyle)).not.toBe('none');
    }
    await workspace(page, 'Inspect');
    await page.getByRole('tab', { name: 'Trace', exact: true }).click();
    const trace = page.getByRole('button', { name: 'Trace prediction', exact: true });
    await trace.click();
    await expect(page.getByRole('region', { name: 'Forward activation flow' })).toContainText('Output');
    await trace.focus();
    await scrollEvidence('Forward activation flow');
    await expect(transport(page)).toHaveAttribute('data-model-step', '0');
    await page.getByRole('button', { name: 'Saved runs', exact: true }).click();
    await page.getByRole('button', { name: 'Save current run', exact: true }).click();
    await expect(page.locator('.saved-runs').getByRole('article')).toHaveCount(1);
    await workspace(page, 'Results');
    await page.getByRole('button', { name: 'Run one training step' }).click();
    await expect(transport(page)).toHaveAttribute('data-model-step', '1');
    await page.getByRole('button', { name: 'Saved runs', exact: true }).click();
    await page.getByRole('button', { name: 'Save current run', exact: true }).click();
    await expect(page.locator('.saved-runs').getByRole('article')).toHaveCount(2);
    const choices = page.getByRole('checkbox', { name: /^Compare / });
    await choices.nth(0).check(); await choices.nth(1).check();
    await page.getByRole('button', { name: 'Compare selected', exact: true }).click();
    await page.getByRole('checkbox', { name: 'Only show differences' }).focus();
    await scrollEvidence('Configuration and evidence');
    await expect(transport(page)).toHaveAttribute('data-model-step', '1');
});

test('forced colors preserves distinct data legend colors and native control styling', async ({ page }, info) => {
    await page.emulateMedia({ forcedColors: 'active' });
    await loadReadyPlayground(page);
    test.skip(!await page.evaluate(() => matchMedia('(forced-colors: active)').matches), 'Browser does not emulate active forced colors');
    for (const theme of ['light', 'dark']) {
        await page.getByLabel('Color theme').selectOption(theme);
        const supportsAdjustment = await page.evaluate(() => CSS.supports('forced-color-adjust', 'none'));
        for (const selector of ['.decision-boundary__swatch', '.atelier-class-legend i', '.network-graph-legend__swatch']) {
            const marks = page.locator(selector);
            await expect(marks.first()).toBeVisible();
            const colors = await marks.evaluateAll((nodes) => nodes.map((node) => {
                const style = getComputedStyle(node);
                return { color: style.backgroundColor, adjustment: style.forcedColorAdjust };
            }));
            expect(new Set(colors.map(({ color }) => color)).size).toBeGreaterThan(1);
            if (supportsAdjustment) expect(colors.every(({ adjustment }) => adjustment === 'none')).toBe(true);
        }
        if (supportsAdjustment) expect(await page.getByLabel('Color theme').evaluate((node) => getComputedStyle(node).forcedColorAdjust)).toBe('auto');
        const graph = page.getByRole('img', { name: 'Neural network graph', exact: true });
        const drawing = await graph.evaluate((node) => {
            const style = getComputedStyle(node);
            const rgb = (value: string) => value.match(/[\d.]+/g)!.slice(0, 3).map(Number);
            const luminance = (value: string) => rgb(value).map((v) => v / 255).map((v) => v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4).reduce((sum, v, i) => sum + v * [.2126, .7152, .0722][i], 0);
            const a = luminance(style.color), b = luminance(style.backgroundColor);
            return { background: style.backgroundColor, contrast: (Math.max(a, b) + .05) / (Math.min(a, b) + .05) };
        });
        expect(drawing.background).toBe(theme === 'dark' ? 'rgb(23, 25, 27)' : 'rgb(247, 246, 242)');
        expect(drawing.contrast).toBeGreaterThanOrEqual(4.5);
        if (supportsAdjustment) {
            for (const selector of ['html', '.network-graph-toolbar', '.network-graph-legend']) {
                expect(await page.locator(selector).evaluate((node) => getComputedStyle(node).forcedColorAdjust)).toBe('auto');
            }
        }
        await page.locator('.network-graph-frame').screenshot({ path: `../../outputs/nn-forge-implementation/display-modes/${info.project.name}-${theme}-forced-colors-graph-fixed.png` });
        await info.attach(`forced-colors-${theme}-data-legends`, { body: await page.screenshot(), contentType: 'image/png' });
    }
});
