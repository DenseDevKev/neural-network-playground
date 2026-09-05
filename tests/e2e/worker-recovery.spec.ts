import { expect, test, type Page } from '@playwright/test';

function collectBrowserErrors(page: Page): string[] {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
    page.on('console', (message) => {
        if (message.type() === 'error') errors.push(`console.error: ${message.text()}`);
    });
    return errors;
}

async function expectEvidenceAtStep(page: Page, step: number): Promise<void> {
    const formatted = step.toLocaleString('en-US');
    const exact = `${formatted}(?![0-9,])`;
    await expect(page.getByRole('group', { name: 'Status bar' }))
        .toContainText(new RegExp(String.raw`STEP\s+${exact}`));
    await expect(page.getByRole('region', { name: 'Timeline strip' }))
        .toContainText(new RegExp(String.raw`Step\s+${exact}`));
    await expect(page.locator('section[role="region"][aria-label="Current run"]'))
        .toContainText(new RegExp(String.raw`Full evaluation \d+ at step ${exact}`));
}

async function expectEvidenceConvergence(page: Page, step: number): Promise<void> {
    await expect.poll(async () => {
        const statusText = await page.getByRole('group', { name: 'Status bar' }).innerText();
        const runText = await page.locator('section[role="region"][aria-label="Current run"]').innerText();
        return {
            status: Number(statusText.match(/\bSTEP\s+([\d,]+)/)?.[1]?.replaceAll(',', '') ?? -1),
            evaluation: Number(runText.match(/Full evaluation \d+ at step ([\d,]+)/)?.[1]?.replaceAll(',', '') ?? -1),
            checkpoint: await page.getByRole('slider', { name: 'Checkpoint timeline' })
                .getAttribute('aria-valuetext'),
        };
    }).toEqual({ status: step, evaluation: step, checkpoint: `Step ${step.toLocaleString('en-US')}` });
}

async function runOneStep(page: Page): Promise<void> {
    await page.getByRole('button', { name: 'Run one training step' }).click();
    await expectEvidenceAtStep(page, 1);
}

async function ensureRunView(page: Page): Promise<void> {
    const run = page.getByRole('button', { name: 'run', exact: true });
    if (await run.getAttribute('aria-pressed') !== 'true') await run.click();
    await expect(page.locator('section[role="region"][aria-label="Current run"]')).toBeVisible();
}

test('@fault-enabled recovers from an injected startup failure', async ({ page }) => {
    const errors = collectBrowserErrors(page);
    await page.goto('./?e2eWorkerFault=startup-once');

    // Fault injection only exists in bundles built with VITE_E2E_FAULTS=1
    // (test:e2e:recovery); the plain smoke build must skip this scenario.
    const faultsCompiledIn = await page.evaluate(() =>
        (window as { __nnpE2EFaultsEnabled?: boolean }).__nnpE2EFaultsEnabled === true);
    test.skip(!faultsCompiledIn, 'Bundle was built without VITE_E2E_FAULTS=1');

    const dialog = page.getByRole('alertdialog', { name: 'Worker connection lost' });
    await expect(dialog).toBeVisible();
    await expect(dialog).toHaveAttribute('aria-describedby', 'worker-error-description');
    await expect(dialog).toHaveAccessibleDescription(
        'Injected E2E worker startup failure. Refresh the page to restart the playground.',
    );
    await expect(dialog).toBeFocused();
    const shell = page.locator('.forge-shell');
    await expect(shell).toHaveAttribute('inert', '');
    await expect(shell).toHaveAttribute('aria-hidden', 'true');
    const refresh = dialog.getByRole('button', { name: 'Refresh page' });
    await refresh.focus();
    await page.keyboard.press('Tab');
    await expect(refresh).toBeFocused();
    await page.keyboard.press('Shift+Tab');
    await expect(refresh).toBeFocused();

    const reloaded = page.waitForNavigation({ waitUntil: 'domcontentloaded' });
    await refresh.click();
    await reloaded;
    await expect(dialog).toBeHidden();
    await expect(shell).not.toHaveAttribute('inert', '');
    await expect(shell).not.toHaveAttribute('aria-hidden', 'true');
    await ensureRunView(page);
    await expectEvidenceConvergence(page, 0);
    await runOneStep(page);
    expect(errors).toEqual([]);
});

test('@fault-disabled normal builds ignore the fault query', async ({ page }) => {
    const errors = collectBrowserErrors(page);
    await page.goto('./?e2eWorkerFault=startup-once');
    await expect(page.getByRole('alertdialog', { name: 'Worker connection lost' })).toHaveCount(0);
    await ensureRunView(page);
    await expectEvidenceConvergence(page, 0);
    await runOneStep(page);
    expect(errors).toEqual([]);
});
