import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';

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
    const run = page.getByRole('button', { name: 'run', exact: true });
    if (await run.getAttribute('aria-pressed') !== 'true') await run.click();
    await expect(page.getByRole('group', { name: 'Status bar' }))
        .toHaveAttribute('data-status', 'idle');
    await expect(page.locator('section[role="region"][aria-label="Current run"]'))
        .toContainText(/Full evaluation \d+ at step 0(?![0-9,])/);
    await expect(page.getByRole('slider', { name: 'Checkpoint timeline' }))
        .toHaveAttribute('aria-valuetext', 'Step 0');
}

const SCAN_CASES = [
    { name: '1437px Precision Lab accessibility', width: 1437, height: 742 },
    { name: '735px Precision Lab accessibility', width: 735, height: 860 },
    { name: '320px Precision Lab accessibility', width: 320, height: 844 },
    { name: 'desktop accessibility', width: 1280, height: 720 },
    { name: '390px compact accessibility', width: 390, height: 844 },
] as const;

for (const scanCase of SCAN_CASES) {
    test.describe(scanCase.name, () => {
        test.use({ viewport: { width: scanCase.width, height: scanCase.height } });
        test('has no serious or critical Axe violations', async ({ page }) => {
            await loadReadyPlayground(page);
            const results = await new AxeBuilder({ page }).analyze();
            const blocking = results.violations.filter(
                ({ impact }) => impact === 'serious' || impact === 'critical',
            );
            expect(blocking, formatViolations(blocking)).toEqual([]);
        });
    });
}
