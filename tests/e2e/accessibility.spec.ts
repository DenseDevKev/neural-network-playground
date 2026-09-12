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
    await page.keyboard.press(browserName === 'webkit' ? 'Alt+Tab' : 'Tab');
    await expect(nodes.nth(1)).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(nodes.nth(1)).toHaveAttribute('aria-pressed', 'true');
    await expect(transport(page)).toHaveAttribute('data-model-step', '0');
    await expect(transport(page)).toHaveAttribute('data-status', 'idle');
});
