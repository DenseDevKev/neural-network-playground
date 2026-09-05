import { expect, test } from '@playwright/test';

const FONT_REQUESTS = [
    '400 16px Inter',
    '500 16px Inter',
    '600 16px Inter',
    '700 16px Inter',
    '400 16px "Space Grotesk"',
    '500 16px "Space Grotesk"',
    '600 16px "Space Grotesk"',
];

// Exercise real font bytes and native reloads. Request routing is deliberately
// absent: it changes browser caching and could hide the recovery regression.
test('bundled fonts load from the application origin before and after native reload', async ({ page }, testInfo) => {
    const errors: string[] = [];
    const fonts: string[] = [];
    const remoteFontRequests: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('console', (message) => {
        if (message.type() === 'error') errors.push(message.text());
    });
    page.on('request', (request) => {
        const url = new URL(request.url());
        if (url.hostname === 'fonts.googleapis.com' || url.hostname === 'fonts.gstatic.com') {
            remoteFontRequests.push(request.url());
        }
    });
    page.on('response', (response) => {
        if (/\.woff2?(?:\?|$)/.test(response.url())) {
            fonts.push(response.url());
            if (!response.ok()) errors.push(`Font HTTP ${response.status()}: ${response.url()}`);
        }
    });
    page.on('requestfailed', (request) => {
        if (/\.woff2?(?:\?|$)/.test(request.url())) {
            errors.push(`Font request failed: ${request.url()} ${request.failure()?.errorText}`);
        }
    });

    const response = await page.goto('./');
    expect(response?.ok()).toBe(true);
    const initialURL = new URL(page.url());
    try {
        for (let visit = 0; visit < 3; visit++) {
            if (visit > 0) await page.reload();
            await expect(page.getByRole('main', { name: 'Neural network playground workspace' })).toBeVisible();
            const loaded = await page.evaluate(async (requests) => {
                const results = await Promise.all(requests.map(async (request) => {
                    const faces = await document.fonts.load(request, 'NN.FORGE café mañana');
                    return { request, count: faces.length, loaded: faces.every((face) => face.status === 'loaded') };
                }));
                await document.fonts.ready;
                return results;
            }, FONT_REQUESTS);
            for (const result of loaded) {
                expect(result.count, result.request).toBeGreaterThan(0);
                expect(result.loaded, result.request).toBe(true);
            }
            expect(new URL(page.url()).origin).toBe(initialURL.origin);
            expect(new URL(page.url()).pathname).toBe(initialURL.pathname);
            expect(remoteFontRequests).toEqual([]);
            expect(fonts.length).toBeGreaterThan(0);
            expect(fonts.every((url) => {
                const font = new URL(url);
                return font.origin === initialURL.origin && font.pathname.startsWith(initialURL.pathname);
            })).toBe(true);
            expect(errors).toEqual([]);
        }
    } finally {
        await testInfo.attach('font-delivery.json', {
            body: JSON.stringify({ url: initialURL.href, fonts, remoteFontRequests, errors }, null, 2),
            contentType: 'application/json',
        });
    }
});
