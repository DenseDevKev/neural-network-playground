import { Buffer } from 'node:buffer';
import { expect, test, type Page } from '@playwright/test';

const MEMORY_KEY = 'nn-playground-experiment-memory-v2';
const XOR = 'Apply preset: XOR Needs Hidden Layers';
const THREE_CLASS = 'Apply preset: Three-Class Softmax Lab';
const timeline = (page: Page) => page.getByRole('region', { name: 'Timeline strip', exact: true });
const currentRun = (page: Page) => page.getByRole('region', { name: 'Current run', exact: true });

async function applyPreset(page: Page, name: string) {
    await page.getByRole('button', { name: 'Presets', exact: true }).click();
    await page.getByRole('dialog', { name: 'Presets', exact: true }).getByRole('button', { name, exact: true }).click();
    await page.getByRole('group', { name: 'Workspace view', exact: true }).getByRole('button', { name: 'run', exact: true }).click();
    await expect(currentRun(page)).toHaveAttribute('data-model-step', '0');
    await expect(currentRun(page)).toContainText(/Full evaluation \d+ at step 0(?!\d)/);
}

async function scientificIdentity(page: Page) {
    return {
        url: page.url(),
        generation: await currentRun(page).getAttribute('data-model-generation'),
        revision: await currentRun(page).getAttribute('data-model-revision'),
        step: await currentRun(page).getAttribute('data-model-step'),
        checkpoint: await timeline(page).getByRole('slider', { name: 'Checkpoint timeline', exact: true }).getAttribute('aria-valuetext'),
    };
}

for (const viewport of [{ width: 1437, height: 742 }, { width: 735, height: 860 }, { width: 320, height: 844 }]) {
    test(`Precision neuron selection preserves scientific state at ${viewport.width}px`, async ({ page }, info) => {
        await page.setViewportSize(viewport);
        const errors: string[] = [];
        page.on('pageerror', (error) => errors.push(error.message));
        page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
        await page.goto('./');
        await applyPreset(page, XOR);
        await page.getByRole('combobox', { name: 'Workspace profile' }).selectOption('lab');
        await timeline(page).getByRole('button', { name: 'Run one training step', exact: true }).click();
        await expect(currentRun(page)).toHaveAttribute('data-model-step', '1');
        await expect(currentRun(page)).toContainText(/Full evaluation \d+ at step 1(?!\d)/);
        const before = await scientificIdentity(page);
        expect(before.generation).not.toBeNull();
        const liveCanvas = page.locator('[data-decision-boundary-canvas]');
        await expect(liveCanvas).toHaveCount(1);
        const originalCanvas = await liveCanvas.elementHandle();
        expect(originalCanvas).not.toBeNull();
        const targets = page.getByRole('group', { name: 'Select a neuron', exact: true });
        const first = targets.getByRole('button', { name: 'Hidden 1, Neuron 1', exact: true });
        const second = targets.getByRole('button', { name: 'Hidden 1, Neuron 2', exact: true });
        await first.click();
        await expect(first).toHaveAttribute('aria-pressed', 'true');
        const details = page.getByRole('region', { name: 'Selected neuron details', exact: true });
        await expect(details).toContainText('Hidden 1 · neuron 1');
        await expect(details).toContainText('Weights at step 1;');
        await expect(details.getByRole('heading', { name: 'Strongest inputs', exact: true })).toBeVisible();
        await second.focus();
        await second.press('Space');
        await expect(second).toHaveAttribute('aria-pressed', 'true');
        await expect(first).toHaveAttribute('aria-pressed', 'false');
        await expect(details).toContainText('Hidden 1 · neuron 2');
        await expect(page.getByRole('group', { name: 'Status bar', exact: true })).toHaveAttribute('data-status', 'paused');
        for (const profile of ['beginner', 'explore', 'lab']) {
            await page.getByRole('combobox', { name: 'Workspace profile' }).selectOption(profile);
            await expect(second).toHaveAttribute('aria-pressed', 'true');
            expect(await scientificIdentity(page)).toEqual(before);
        }
        const views = page.getByRole('group', { name: 'Workspace view', exact: true });
        await views.getByRole('button', { name: 'build', exact: true }).click();
        await expect(second).toHaveAttribute('aria-pressed', 'true');
        await views.getByRole('button', { name: 'run', exact: true }).click();
        const boundaryTab = page.getByRole('tab', { name: 'Boundary', exact: true });
        await boundaryTab.focus();
        await boundaryTab.press('ArrowRight');
        await expect(page.getByRole('tab', { name: 'Loss', exact: true })).toBeFocused();
        await page.keyboard.press('End');
        await expect(page.getByRole('tab', { name: 'Code', exact: true })).toBeFocused();
        await page.keyboard.press('Home');
        await expect(boundaryTab).toBeFocused();
        await expect(second).toHaveAttribute('aria-pressed', 'true');
        expect(await scientificIdentity(page)).toEqual(before);
        await expect(liveCanvas).toHaveCount(1);
        expect(await originalCanvas!.evaluate((element) => element.isConnected)).toBe(true);
        await details.getByRole('button', { name: 'Clear selection', exact: true }).click();
        await expect(details).toHaveCount(0);
        expect(await scientificIdentity(page)).toEqual(before);
        await first.focus();
        await first.press('Enter');
        await expect(first).toHaveAttribute('aria-pressed', 'true');
        await timeline(page).getByRole('button', { name: 'Reset training', exact: true }).click();
        await expect(currentRun(page)).not.toHaveAttribute('data-model-generation', before.generation!);
        await expect(details).toHaveCount(0);
        await expect(currentRun(page)).toHaveAttribute('data-model-step', '0');
        await first.click();
        await expect(details).toHaveCount(1);
        await applyPreset(page, THREE_CLASS);
        await expect(details).toHaveCount(0);
        await expect(targets.locator('[aria-pressed="true"]')).toHaveCount(0);
        await info.attach('selection-identity', { body: Buffer.from(JSON.stringify({ viewport, before, after: await scientificIdentity(page) }, null, 2)), contentType: 'application/json' });
        expect(errors).toEqual([]);
    });
}

interface PersistenceProbe { reject: boolean; writes: string[] }
type ProbeWindow = Window & { __precisionPersistenceProbe: PersistenceProbe };

for (const width of [1437, 320]) {
    test(`Precision save retry retains exact artifact across recipe change and History reopen at ${width}px`, async ({ page }, info) => {
        await page.setViewportSize({ width, height: 844 });
        // Deliberately fail the actual storage boundary, not the worker or capture.
        await page.addInitScript((key) => {
            const probe: PersistenceProbe = { reject: true, writes: [] };
            (window as ProbeWindow).__precisionPersistenceProbe = probe;
            const original = Storage.prototype.setItem;
            Storage.prototype.setItem = function (name: string, value: string) {
                if (this === window.localStorage && name === key) {
                    probe.writes.push(value);
                    if (probe.reject) throw new DOMException('Injected quota failure', 'QuotaExceededError');
                }
                return original.call(this, name, value);
            };
        }, MEMORY_KEY);
        const writes = () => page.evaluate(() => (window as ProbeWindow).__precisionPersistenceProbe.writes);
        await page.goto('./');
        await applyPreset(page, XOR);
        await timeline(page).getByRole('button', { name: 'Run one training step', exact: true }).click();
        await expect(currentRun(page)).toHaveAttribute('data-model-step', '1');
        await timeline(page).getByRole('button', { name: 'Save run', exact: true }).click();
        await expect(timeline(page).getByRole('button', { name: 'Retry pending artifact', exact: true })).toBeEnabled();
        await expect.poll(async () => (await writes()).length).toBe(1);
        const intended = (await writes())[0];
        expect(JSON.parse(intended).records).toHaveLength(1);
        expect(await page.evaluate((key) => localStorage.getItem(key), MEMORY_KEY)).toBeNull();
        await page.getByRole('button', { name: 'History', exact: true }).click();
        const history = page.getByRole('dialog', { name: 'History', exact: true });
        await expect(history.getByRole('alert')).toContainText('Storage quota was exceeded');
        await history.getByRole('button', { name: 'Retry saving', exact: true }).click();
        await expect.poll(async () => (await writes()).length).toBe(2);
        expect((await writes())[1]).toBe(intended);
        await expect(history.getByRole('button', { name: 'Retry saving', exact: true })).toBeEnabled();
        await history.getByRole('button', { name: 'Close History', exact: true }).click();
        await applyPreset(page, THREE_CLASS);
        await expect(timeline(page).getByRole('button', { name: 'Save run', exact: true })).toBeDisabled();
        await page.getByRole('button', { name: 'History', exact: true }).click();
        await expect(history.getByRole('button', { name: 'Save current run', exact: true })).toBeDisabled();
        await page.evaluate(() => { (window as ProbeWindow).__precisionPersistenceProbe.reject = false; });
        await history.getByRole('button', { name: 'Retry saving', exact: true }).click();
        await expect(history.getByRole('button', { name: 'Retry saving', exact: true })).toHaveCount(0);
        await expect(history.getByRole('article')).toHaveCount(1);
        const stored = await page.evaluate((key) => localStorage.getItem(key), MEMORY_KEY);
        expect(stored).toBe(intended);
        expect(await writes()).toEqual([intended, intended, intended]);
        await info.attach('exact-retry-artifact', { body: Buffer.from(JSON.stringify({ intended: JSON.parse(intended), persisted: JSON.parse(stored!), attempts: (await writes()).length }, null, 2)), contentType: 'application/json' });
        await page.reload();
        await page.getByRole('button', { name: 'History', exact: true }).click();
        await expect(history.getByRole('article')).toHaveCount(1);
        await expect(history.getByRole('article')).toContainText('Full evaluation at step 1;');
    });
}
