import { expect, test, type Page } from '@playwright/test';

const MEMORY_KEY = 'nn-playground-experiment-memory-v2';
type GatedWindow = Window & { releaseDigest?: () => void; digestBlocked?: boolean; gateDigest?: boolean };

async function openHistory(page: Page) {
    await page.goto('./');
    await page.getByRole('button', { name: 'Saved runs', exact: true }).click();
    const history = page.locator('.saved-runs');
    await expect(history.getByRole('button', { name: 'Save current run', exact: true })).toBeEnabled();
    return history;
}

test('two tabs preserve both saves when a stale save is delayed during validation', async ({ context, page }) => {
    const second = await context.newPage();
    // Delay storage notifications and one real digest to control the stale-tab
    // interleaving. Neither persistence nor worker results are replaced.
    await second.addInitScript(() => {
        window.addEventListener('storage', (event) => event.stopImmediatePropagation());
        const target = window as GatedWindow;
        const digest = crypto.subtle.digest.bind(crypto.subtle);
        crypto.subtle.digest = async (algorithm, data) => {
            if (target.gateDigest) {
                target.gateDigest = false;
                target.digestBlocked = true;
                await new Promise<void>((resolve) => { target.releaseDigest = resolve; });
            }
            return digest(algorithm, data);
        };
    });
    const firstHistory = await openHistory(page);
    const secondHistory = await openHistory(second);
    await second.evaluate(() => { (window as GatedWindow).gateDigest = true; });
    await secondHistory.getByRole('button', { name: 'Save current run', exact: true }).click();
    await expect.poll(() => second.evaluate(() => (window as GatedWindow).digestBlocked)).toBe(true);
    await firstHistory.getByRole('button', { name: 'Save current run', exact: true }).click();
    await expect(firstHistory.getByRole('article')).toHaveCount(1);
    const first = await page.evaluate((key) => JSON.parse(localStorage.getItem(key)!).records[0], MEMORY_KEY);
    await second.evaluate(() => (window as GatedWindow).releaseDigest?.());
    await expect(secondHistory.getByRole('button', { name: 'Save current run', exact: true })).toBeEnabled();
    await expect(secondHistory.getByRole('article')).toHaveCount(2);
    const stored = await page.evaluate((key) => JSON.parse(localStorage.getItem(key)!).records, MEMORY_KEY);
    expect(stored).toHaveLength(2);
    expect(stored.find((record: { id: string }) => record.id === first.id)).toEqual(first);
    expect(new Set(stored.map((record: { id: string }) => record.id)).size).toBe(2);
    await second.reload();
    await second.getByRole('button', { name: 'Saved runs', exact: true }).click();
    await expect(secondHistory.getByRole('article')).toHaveCount(2);
});

test('a second tab waits while the first writer validates its envelope under the origin lock', async ({ context, page }) => {
    await page.addInitScript((key) => {
        const target = window as GatedWindow & { gateOnMemoryRead?: boolean };
        const getItem = Storage.prototype.getItem;
        Storage.prototype.getItem = function (name) {
            if (name === key && target.gateOnMemoryRead) {
                target.gateOnMemoryRead = false;
                target.gateDigest = true;
            }
            return getItem.call(this, name);
        };
        const digest = crypto.subtle.digest.bind(crypto.subtle);
        crypto.subtle.digest = async (algorithm, data) => {
            if (target.gateDigest) {
                target.gateDigest = false;
                target.digestBlocked = true;
                await new Promise<void>((resolve) => { target.releaseDigest = resolve; });
            }
            return digest(algorithm, data);
        };
    }, MEMORY_KEY);
    const firstHistory = await openHistory(page);
    const second = await context.newPage();
    const secondHistory = await openHistory(second);
    await page.evaluate(() => { (window as GatedWindow & { gateOnMemoryRead?: boolean }).gateOnMemoryRead = true; });
    await firstHistory.getByRole('button', { name: 'Save current run', exact: true }).click();
    await expect.poll(() => page.evaluate(() => (window as GatedWindow).digestBlocked)).toBe(true);
    await secondHistory.getByRole('button', { name: 'Save current run', exact: true }).click();
    await expect.poll(() => second.evaluate(async (key) => {
        const locks = await navigator.locks.query();
        return {
            held: locks.held?.filter((lock) => lock.name === key).length,
            pending: locks.pending?.filter((lock) => lock.name === key).length,
        };
    }, MEMORY_KEY)).toEqual({ held: 1, pending: 1 });
    expect(await second.evaluate((key) => localStorage.getItem(key), MEMORY_KEY)).toBeNull();
    await page.evaluate(() => (window as GatedWindow).releaseDigest?.());
    await expect(firstHistory.getByRole('article')).toHaveCount(2);
    await expect(secondHistory.getByRole('article')).toHaveCount(2);
    const ids = await second.evaluate((key) => JSON.parse(localStorage.getItem(key)!).records.map((record: { id: string }) => record.id), MEMORY_KEY);
    expect(new Set(ids).size).toBe(2);
});
