import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { mkdtemp, mkdir, writeFile, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { once } from 'node:events';
import { createReleaseFixtureServer } from './serve-release-fixture.mjs';

let root, server, origin;
before(async () => {
    root = await mkdtemp(join(tmpdir(), 'nn-release-'));
    await mkdir(join(root, 'dist', 'assets'), { recursive: true });
    await writeFile(join(root, 'dist', 'index.html'), '<main>real fixture</main>');
    await writeFile(join(root, 'dist', 'assets', 'worker.js'), 'postMessage("ready")');
    await writeFile(join(root, 'outside.txt'), 'not public');
    await symlink(join(root, 'outside.txt'), join(root, 'dist', 'escape.txt'));
    server = await createReleaseFixtureServer(join(root, 'dist'));
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    origin = `http://127.0.0.1:${server.address().port}`;
});
after(async () => {
    if (server) await new Promise((resolve) => server.close(resolve));
    if (root) await rm(root, { recursive: true, force: true });
});
test('serves the real project document without isolation headers', async () => {
    const response = await globalThis.fetch(`${origin}/neural-network-playground/`);
    assert.equal(response.status, 200);
    assert.match(response.headers.get('content-type'), /text\/html/);
    assert.equal(response.headers.get('cross-origin-opener-policy'), null);
    assert.equal(response.headers.get('cross-origin-embedder-policy'), null);
    assert.equal(response.headers.get('cache-control'), 'no-store');
    assert.equal(await response.text(), '<main>real fixture</main>');
});
test('serves JavaScript assets with the correct MIME type', async () => {
    const response = await globalThis.fetch(`${origin}/neural-network-playground/assets/worker.js`);
    assert.equal(response.status, 200);
    assert.match(response.headers.get('content-type'), /javascript/);
    assert.equal(await response.text(), 'postMessage("ready")');
});
test('preserves a query on the canonical trailing-slash redirect', async () => {
    const response = await globalThis.fetch(`${origin}/neural-network-playground?x=1`, { redirect: 'manual' });
    assert.equal(response.status, 308);
    assert.equal(response.headers.get('location'), '/neural-network-playground/?x=1');
});
test('does not turn missing assets or the wrong project path into HTML success', async () => {
    for (const path of ['/', '/other/', '/neural-network-playground/missing.js']) {
        assert.equal((await globalThis.fetch(origin + path)).status, 404);
    }
});
test('does not serve files outside the dist directory through traversal or symlinks', async () => {
    for (const path of ['/neural-network-playground/%2e%2e/outside.txt', '/neural-network-playground/escape.txt']) {
        const response = await globalThis.fetch(origin + path);
        assert.ok([403, 404].includes(response.status));
        assert.notEqual(await response.text(), 'not public');
    }
});
test('HEAD returns metadata without a response body', async () => {
    const response = await globalThis.fetch(`${origin}/neural-network-playground/`, { method: 'HEAD' });
    assert.equal(response.status, 200);
    assert.equal(await response.text(), '');
});
test('rejects mutation requests', async () => {
    const response = await globalThis.fetch(`${origin}/neural-network-playground/`, { method: 'POST' });
    assert.equal(response.status, 405);
    assert.equal(response.headers.get('allow'), 'GET, HEAD');
});
test('rejects a missing dist directory before listening', async () => {
    await assert.rejects(() => createReleaseFixtureServer(join(root, 'missing')), /ENOENT/);
});
