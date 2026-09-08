import { Buffer } from 'node:buffer';
import { expect, test, type Page, type TestInfo } from '@playwright/test';

type Resource = { url: string; status: number; contentType: string };

function observeHosting(page: Page, baseURL: string) {
    const origin = new URL(baseURL).origin;
    const resources: Resource[] = [];
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
    page.on('console', (message) => {
        if (message.type() === 'error') errors.push(`console.error: ${message.text()}`);
    });
    page.on('requestfailed', (request) => {
        if (new URL(request.url()).origin === origin) {
            errors.push(`requestfailed: ${request.url()} ${request.failure()?.errorText}`);
        }
    });
    page.on('response', (response) => {
        if (new URL(response.url()).origin !== origin) return;
        resources.push({
            url: response.url(), status: response.status(),
            contentType: response.headers()['content-type'] ?? '',
        });
        if (response.status() >= 400) errors.push(`HTTP ${response.status()}: ${response.url()}`);
    });
    return { resources, errors };
}

async function attachHosting(info: TestInfo, baseURL: string, observed: ReturnType<typeof observeHosting>) {
    await info.attach('hosting-evidence', {
        body: Buffer.from(JSON.stringify({ baseURL, ...observed }, null, 2)),
        contentType: 'application/json',
    });
}

function expectChunk(resources: Resource[], baseURL: string, prefix: string) {
    const target = new URL(baseURL);
    const matches = resources.filter((resource) => {
        const url = new URL(resource.url);
        return url.origin === target.origin
            && url.pathname.startsWith(`${target.pathname}assets/${prefix}-`)
            && url.pathname.endsWith('.js');
    });
    expect(matches, `Missing successful ${prefix} JavaScript below ${target.pathname}`).not.toHaveLength(0);
    for (const resource of matches) {
        expect(resource.status).toBe(200);
        expect(resource.contentType).toMatch(/(?:java|ecma)script/i);
    }
}

async function readyAtZero(page: Page, url: string) {
    await page.goto(url);
    await expect(page.getByRole('main', { name: 'Neural network playground workspace' })).toBeVisible();
    const runButton = page.getByRole('button', { name: 'run', exact: true });
    if (await runButton.getAttribute('aria-pressed') !== 'true') await runButton.click();
    await expect(page.getByRole('group', { name: 'Status bar' })).toHaveAttribute('data-status', 'idle');
    await expect(page.locator('section[role="region"][aria-label="Current run"]'))
        .toContainText(/Full evaluation \d+ at step 0(?![0-9,])/);
    await expect(page.getByRole('slider', { name: 'Checkpoint timeline' }))
        .toHaveAttribute('aria-valuetext', 'Step 0');
}

function externalTarget(baseURL: string | undefined, info: TestInfo): string {
    test.skip(info.config.metadata.targetMode !== 'external', 'Hosting contracts require a non-isolated external target, not Vite preview');
    if (!baseURL) throw new Error('External hosting verification requires an explicit baseURL');
    return baseURL;
}

test('non-isolated project hosting loads the real worker and publishes a paired manual step', async ({ page, baseURL }, info) => {
    const target = externalTarget(baseURL, info);
    const observed = observeHosting(page, target);
    try {
        await readyAtZero(page, target);
        expect(await page.evaluate(() => ({
            secure: window.isSecureContext,
            isolated: window.crossOriginIsolated,
            faults: (window as { __nnpE2EFaultsEnabled?: boolean }).__nnpE2EFaultsEnabled === true,
        }))).toEqual({ secure: true, isolated: false, faults: false });
        expect(new URL(page.url()).pathname).toBe(new URL(target).pathname);
        expectChunk(observed.resources, target, 'training.worker');
        await page.getByRole('button', { name: 'Run one training step' }).click();
        await expect(page.locator('section[role="region"][aria-label="Current run"]'))
            .toContainText(/Full evaluation \d+ at step 1(?![0-9,])/);
        await expect(page.getByRole('group', { name: 'Status bar' })).toContainText(/STEP\s+1(?![0-9,])/);
        expect(observed.errors).toEqual([]);
    } finally {
        await attachHosting(info, target, observed);
    }
});

test('project hosting resolves lazy evidence and reopens an unchanged shared recipe', async ({ page, browser, baseURL }, info) => {
    const target = externalTarget(baseURL, info);
    const observed = observeHosting(page, target);
    try {
        await readyAtZero(page, target);
        await page.getByRole('button', { name: 'Presets', exact: true }).click();
        const presets = page.getByRole('dialog', { name: 'Presets' });
        await presets.getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers' }).click();
        await expect(presets).toBeHidden();
        await expect(page).toHaveURL(/#v=2&r=/);
        await page.getByRole('combobox', { name: 'Workspace profile' }).selectOption('lab');
        await expect(page.getByRole('button', { name: 'Advanced Tools' })).toHaveAttribute('aria-expanded', 'true');
        await page.getByRole('tab', { name: 'Inspect', exact: true }).click();
        await expect(page.getByRole('tab', { name: 'Inspect', exact: true })).toHaveAttribute('aria-selected', 'true');
        await expect(page.getByRole('tabpanel', { name: 'Inspect', exact: true })).toBeVisible();
        await expect(page.locator('.inspection-panel')).toBeVisible();
        await page.getByRole('tab', { name: 'Code', exact: true }).click();
        await expect(page.getByRole('tab', { name: 'Code', exact: true })).toHaveAttribute('aria-selected', 'true');
        await page.getByRole('tab', { name: 'NumPy', exact: true }).click();
        await expect(page.getByRole('tabpanel', { name: 'NumPy', exact: true })).toContainText('import numpy as np');
        await page.getByRole('button', { name: 'History', exact: true }).click();
        const history = page.getByRole('dialog', { name: 'History' });
        await expect(history).toContainText(/do not\s+contain trained parameters/);
        await history.getByRole('button', { name: 'Close History' }).click();
        await page.getByRole('button', { name: 'build', exact: true }).click();
        await page.getByRole('navigation', { name: 'Build tools' }).getByRole('button', { name: 'Configuration', exact: true }).click();
        await expect(page.getByRole('region', { name: 'Configuration context', exact: true })).toBeVisible();
        await expect(page.getByRole('region', { name: 'Configuration context', exact: true })
            .getByRole('button', { name: /Export JSON/ })).toBeVisible();
        for (const prefix of ['InspectionPanel', 'CodeExportPanel', 'RunHistoryPanel', 'ConfigPanel']) {
            expectChunk(observed.resources, target, prefix);
        }
        const recipe = await page.getByRole('region', { name: 'Recipe summary', exact: true }).innerText();
        const architecture = await page.getByLabel('Architecture summary', { exact: true }).textContent();
        const sharedURL = page.url();
        expect(new URL(sharedURL).hash).toMatch(/^#v=2&r=/);
        expect(new URL(sharedURL).origin).toBe(new URL(target).origin);
        expect(new URL(sharedURL).pathname).toBe(new URL(target).pathname);
        const fresh = await browser.newContext({ locale: 'en-US' });
        const peer = await fresh.newPage();
        const peerObserved = observeHosting(peer, target);
        try {
            await readyAtZero(peer, sharedURL);
            await peer.getByRole('combobox', { name: 'Workspace profile' }).selectOption('lab');
            await peer.getByRole('button', { name: 'build', exact: true }).click();
            await expect(peer.getByRole('region', { name: 'Recipe summary', exact: true })).toHaveText(recipe, { useInnerText: true });
            await expect(peer.getByLabel('Architecture summary', { exact: true })).toHaveText(architecture ?? '');
            expect(peer.url()).toBe(sharedURL);
            expect(peerObserved.errors).toEqual([]);
        } finally {
            await attachHosting(info, target, peerObserved);
            await fresh.close();
        }
        expect(observed.errors).toEqual([]);
    } finally {
        await attachHosting(info, target, observed);
    }
});
