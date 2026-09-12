import { Buffer } from 'node:buffer';
import { expect, test, type Page, type TestInfo } from '@playwright/test';
import { applyPreset, expectEvidence, learning, ready, transport, utility, workspace } from './atelier-helpers';

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
    await learning(page);
    await ready(page);
    await expect(transport(page)).toHaveAttribute('data-status', 'idle');
    await expectEvidence(page, 0);
    await utility(page, 'Session checkpoints');
    await expect(page.getByRole('slider', { name: 'Checkpoint timeline' })).toHaveAttribute('aria-valuetext', 'Step 0');
    await page.keyboard.press('Escape');
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
        await expect(transport(page)).toHaveAttribute('data-model-step', '1');
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
        await applyPreset(page, 'XOR Needs Hidden Layers');
        await expect(page).toHaveURL(/#v=2&r=/);
        await workspace(page, 'Inspect');
        await expect(page.locator('.inspection-panel')).toBeVisible();
        // Saved/export utilities stay off initial and diagnostics paths.
        expect(observed.resources.some(({ url }) => /\/WorkspaceUtilities-[^/]+\.js$/.test(new URL(url).pathname))).toBe(false);
        await utility(page, 'Export / import');
        await expect(page.getByRole('button', { name:'Export JSON setup',exact:true })).toBeVisible();
        await page.getByRole('tab', { name:'Code',exact:true }).click();
        await page.getByRole('tab', { name:'NumPy',exact:true }).click();
        await expect(page.getByRole('tabpanel', { name:'NumPy',exact:true })).toContainText('import numpy as np');
        await page.keyboard.press('Escape');
        await page.getByRole('button', { name:'Saved runs',exact:true }).click();
        await expect(page.locator('.saved-runs')).toContainText('Saved evidence includes a recipe and full evaluation, without trained parameters. Applying a recipe starts a fresh model.');
        for (const prefix of ['InspectionPanel', 'CodeExportPanel', 'WorkspaceUtilities']) expectChunk(observed.resources, target, prefix);
        await workspace(page, 'Network');
        const architecture = await page.getByLabel('Architecture summary', { exact:true }).textContent();
        const sharedURL = page.url();
        expect(new URL(sharedURL).hash).toMatch(/^#v=2&r=/);
        expect(new URL(sharedURL).origin).toBe(new URL(target).origin);
        expect(new URL(sharedURL).pathname).toBe(new URL(target).pathname);
        const fresh = await browser.newContext({ locale: 'en-US' });
        const peer = await fresh.newPage();
        const peerObserved = observeHosting(peer, target);
        try {
            await readyAtZero(peer, sharedURL);
            await workspace(peer, 'Network');
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
