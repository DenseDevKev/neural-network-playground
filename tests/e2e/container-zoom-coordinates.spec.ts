import { Buffer } from 'node:buffer';
import { expect, test } from '@playwright/test';

// Isolated CSS diagnostic for the WebKit/Chromium disagreement observed in
// run 34535418504. The separate application zoom assertions remain unchanged.
test('record container-query coordinate spaces before and after document zoom', async ({ page }, info) => {
    await page.setViewportSize({ width: 735, height: 860 });
    await page.setContent(`<!doctype html><style>
        html, body, #root { height: 100%; margin: 0; font-size: 14px; }
        #root { container: probe / size; }
        .sample { width: 10px; height: 10px; --compact: no; }
        @container probe (max-width: 679px) { #px { --compact: yes; } }
        @container probe (max-width: 48.5em) { #em { --compact: yes; } }
        @container probe (max-width: 48.5rem) { #rem { --compact: yes; } }
        @container probe (max-inline-size: 679px) { #inline { --compact: yes; } }
        @container probe (max-height: 600px) { #height { --compact: yes; } }
        @container probe (max-height: calc(600em / 14)) { #heightCalc { --compact: yes; } }
        @container probe (max-height: 42.8571428571em) { #heightEm { --compact: yes; } }
    </style><div id="root"><div class="sample" id="px"></div><div class="sample" id="em"></div><div class="sample" id="rem"></div><div class="sample" id="inline"></div><div class="sample" id="height"></div><div class="sample" id="heightCalc"></div><div class="sample" id="heightEm"></div></div>`);
    const read = () => page.evaluate(() => {
        const root = document.getElementById('root')!;
        const style = getComputedStyle(root);
        return { zoom: getComputedStyle(document.documentElement).zoom,
            rect: root.getBoundingClientRect().toJSON(), clientWidth: root.clientWidth,
            cssWidth: style.width, cssHeight: style.height, fontSize: style.fontSize,
            matches: Object.fromEntries(['px', 'em', 'rem', 'inline', 'height', 'heightCalc', 'heightEm'].map((id) =>
                [id, getComputedStyle(document.getElementById(id)!).getPropertyValue('--compact').trim()])),
        };
    });
    // Validate the diagnostic cascade before interpreting any unknown zoom result.
    await page.setViewportSize({ width: 500, height: 500 });
    const positive = await read();
    expect(Object.values(positive.matches)).toEqual(Array(7).fill('yes'));
    await page.setViewportSize({ width: 735, height: 860 });
    const before = await read();
    expect(Object.values(before.matches)).toEqual(Array(7).fill('no'));
    await page.evaluate(() => { document.documentElement.style.zoom = '2'; });
    const zoomed = await read();
    // Detect stale query caches independently of the application's layout.
    await page.evaluate(() => { document.getElementById('root')!.style.containerName = 'unmatched'; });
    await read();
    await page.evaluate(() => { document.getElementById('root')!.style.containerName = 'probe'; });
    const reselected = await read();
    await info.attach('container-query-coordinates', {
        body: Buffer.from(JSON.stringify({ positive, before, zoomed, reselected }, null, 2)), contentType: 'application/json',
    });
    expect(before.rect.width).toBe(735);
    expect(zoomed.rect.width).toBeGreaterThan(0);
    expect(reselected.rect.width).toBeGreaterThan(0);
    // The literal production bound must agree in both engines after magnification.
    expect(zoomed.matches.heightEm).toBe('yes');
    expect(reselected.matches.heightEm).toBe('yes');
});
