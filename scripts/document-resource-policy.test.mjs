import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { URL } from 'node:url';

test('external font stylesheets opt into anonymous CORS for isolated reloads', () => {
    const html = readFileSync(new URL('../apps/web/index.html', import.meta.url), 'utf8');
    const links = (html.match(/<link\b[^>]*>/giu) ?? []).filter((tag) => (
        /rel\s*=\s*["']stylesheet["']/iu.test(tag)
        && /https:\/\/fonts\.googleapis\.com\//u.test(tag)
    ));
    assert.ok(links.length > 0, 'expected the existing production font stylesheets');
    for (const link of links) {
        assert.match(link, /crossorigin\s*=\s*["']anonymous["']/iu);
    }
});
