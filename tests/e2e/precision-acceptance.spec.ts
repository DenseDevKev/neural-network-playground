import { Buffer } from 'node:buffer';
import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { applyPreset, expectEvidence, identity, learning, ready, transport, utility, workspace } from './atelier-helpers';

const MEMORY_KEY = 'nn-playground-experiment-memory-v2';
const XOR = 'XOR Needs Hidden Layers';
const THREE_CLASS = 'Three-Class Softmax Lab';

for (const viewport of [{ width: 1437, height: 742 }, { width: 735, height: 860 }, { width: 320, height: 844 }]) {
    test(`Neuron selection preserves scientific state at ${viewport.width}px`, async ({ page }, info) => {
        await page.setViewportSize(viewport);
        const errors: string[] = [];
        page.on('pageerror', (error) => errors.push(error.message));
        page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
        await page.goto('./');
        await ready(page);
        await applyPreset(page, XOR);
        await learning(page);
        await page.getByRole('button', { name: 'Run one training step', exact: true }).click();
        await expectEvidence(page, 1);
        await workspace(page, 'Network');
        const before = await identity(page);
        expect(before.generation).not.toBeNull();
        const targets = page.getByRole('group', { name: 'Select a neuron', exact: true });
        const first = targets.getByRole('button', { name: 'Hidden 1, Neuron 1', exact: true });
        const second = targets.getByRole('button', { name: 'Hidden 1, Neuron 2', exact: true });
        await first.focus();
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
        for (const guidance of ['beginner', 'explore', 'lab']) {
            await utility(page, 'Guidance');
            await page.getByRole('combobox', { name: 'Explanation density' }).selectOption(guidance);
            await page.keyboard.press('Escape');
            await expect(second).toHaveAttribute('aria-pressed', 'true');
            expect(await identity(page)).toEqual(before);
        }
        await workspace(page, 'Setup');
        await expect(targets).toHaveCount(0);
        await workspace(page, 'Network');
        await expect(second).toHaveAttribute('aria-pressed', 'true');
        const tabs = page.getByRole('tablist', { name: 'Experiment workspace' });
        await tabs.getByRole('tab', { name: 'Network', exact: true }).focus();
        await page.keyboard.press('ArrowRight');
        await expect(tabs.getByRole('tab', { name: 'Results', exact: true })).toBeFocused();
        await page.keyboard.press('End');
        await expect(tabs.getByRole('tab', { name: 'Inspect', exact: true })).toBeFocused();
        await page.keyboard.press('Home');
        await expect(tabs.getByRole('tab', { name: 'Setup', exact: true })).toBeFocused();
        await workspace(page, 'Network');
        await expect(second).toHaveAttribute('aria-pressed', 'true');
        expect(await identity(page)).toEqual(before);
        await details.getByRole('button', { name: 'Clear selection', exact: true }).click();
        await expect(details).toHaveCount(0);
        await expect(second).toBeFocused();
        expect(await identity(page)).toEqual(before);
        await first.focus();
        await first.press('Enter');
        await page.getByRole('button', { name: 'Reset training', exact: true }).click();
        await expect(transport(page)).not.toHaveAttribute('data-model-generation', before.generation!);
        await expect(details).toHaveCount(0);
        await expect(transport(page)).toHaveAttribute('data-model-step', '0');
        await first.focus();
        await first.click();
        await expect(details).toHaveCount(1);
        await applyPreset(page, THREE_CLASS);
        await workspace(page, 'Network');
        await expect(details).toHaveCount(0);
        await expect(targets.locator('[aria-pressed="true"]')).toHaveCount(0);
        await info.attach('selection-identity', { body: Buffer.from(JSON.stringify({ viewport, before, after: await identity(page) }, null, 2)), contentType: 'application/json' });
        expect(errors).toEqual([]);
    });
}

interface PersistenceProbe { reject: boolean; writes: string[] }
type ProbeWindow = Window & { __precisionPersistenceProbe: PersistenceProbe };

for (const width of [1437, 320]) {
    test(`Save retry and download retain the exact artifact across recipe change and navigation at ${width}px`, async ({ page }, info) => {
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
        await learning(page);
        await page.getByRole('button', { name: 'Run one training step', exact: true }).click();
        await expectEvidence(page, 1);
        await page.getByRole('button', { name: 'Saved runs', exact: true }).click();
        const history = page.locator('.saved-runs');
        await history.getByRole('button', { name: 'Save current run', exact: true }).click();
        await expect(history.getByRole('button', { name: 'Retry saving', exact: true })).toBeEnabled();
        await expect.poll(async () => (await writes()).length).toBe(1);
        const intended = (await writes())[0];
        expect(JSON.parse(intended).records).toHaveLength(1);
        expect(await page.evaluate((key) => localStorage.getItem(key), MEMORY_KEY)).toBeNull();
        const pendingDownload = page.waitForEvent('download');
        await history.getByRole('button', { name: 'Download pending evidence' }).click();
        const downloaded = await pendingDownload;
        expect(JSON.parse(await readFile((await downloaded.path())!, 'utf8'))).toEqual(JSON.parse(intended).records[0]);
        expect(await writes()).toEqual([intended]);
        await expect(history.getByRole('alert')).toContainText('Storage quota was exceeded');
        await history.getByRole('button', { name: 'Retry saving', exact: true }).click();
        await expect.poll(async () => (await writes()).length).toBe(2);
        expect((await writes())[1]).toBe(intended);
        await expect(history.getByRole('button', { name: 'Retry saving', exact: true })).toBeEnabled();
        await applyPreset(page, THREE_CLASS);
        await page.getByRole('button', { name: 'Saved runs', exact: true }).click();
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
        await page.getByRole('button', { name: 'Saved runs', exact: true }).click();
        await expect(history.getByRole('article')).toHaveCount(1);
        const reloaded = await page.evaluate((key) => JSON.parse(localStorage.getItem(key)!).records, MEMORY_KEY);
        expect(reloaded[0].snapshot.model.step).toBe(1);
        expect(reloaded[0]).toEqual(JSON.parse(intended).records[0]);
    });
}
