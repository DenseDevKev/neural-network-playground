import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { spawnSync } from 'node:child_process';
import process from 'node:process';
import { gzipSync } from 'node:zlib';
import { fileURLToPath, URL } from 'node:url';
import { BUNDLE_LIMITS, assertBundleWithinLimits, measureWebBundle } from './check-web-bundle-gzip.mjs';

const roots = [];
afterEach(() => {
    for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});
function fixture(overrides = {}) {
    const root = mkdtempSync(join(tmpdir(), 'nn-forge-bundle-'));
    roots.push(root);
    const files = {
        'index.html': '<script crossorigin src="./assets/index-a.js" type="module"></script>',
        'assets/index-a.js': 'export const entry = 1;',
        'assets/InspectionPanel-b.js': 'export const inspect = 1;',
        'assets/nested/worker-c.js': 'export const worker = 1;',
        'assets/style.css': 'body { margin: 0; }',
        ...overrides,
    };
    for (const [path, content] of Object.entries(files)) {
        if (content === null) continue;
        const absolute = join(root, path);
        mkdirSync(dirname(absolute), { recursive: true });
        writeFileSync(absolute, content);
    }
    return root;
}
const measurement = () => ({
    entry: { file: 'assets/index-a.js', gzipBytes: BUNDLE_LIMITS.entry },
    inspection: { file: 'assets/InspectionPanel-b.js', gzipBytes: BUNDLE_LIMITS.inspection },
    totalJavaScript: { files: 3, gzipBytes: BUNDLE_LIMITS.totalJavaScript },
});

test('retains exactly the reviewed immutable limits', () => {
    assert.deepEqual(BUNDLE_LIMITS, { entry: 152245, inspection: 7373, totalJavaScript: 234161 });
    assert.ok(Object.isFrozen(BUNDLE_LIMITS));
});
test('measures exact gzip bytes recursively and excludes non-JavaScript', () => {
    const result = measureWebBundle(fixture());
    assert.equal(result.entry.file, 'assets/index-a.js');
    assert.equal(result.inspection.file, 'assets/InspectionPanel-b.js');
    assert.equal(result.entry.gzipBytes, gzipSync('export const entry = 1;').length);
    assert.equal(result.totalJavaScript.files, 3);
    assert.equal(result.totalJavaScript.gzipBytes, [
        'export const entry = 1;', 'export const inspect = 1;', 'export const worker = 1;',
    ].reduce((sum, value) => sum + gzipSync(value).length, 0));
});
for (const src of ['assets/index-a.js', './assets/index-a.js', '/assets/index-a.js', '/neural-network-playground/assets/index-a.js']) {
    test(`resolves the actual module entry for ${src}`, () => {
        const result = measureWebBundle(fixture({ 'index.html': `<script type='module' src='${src}'></script>` }));
        assert.equal(result.entry.file, 'assets/index-a.js');
    });
}
for (const [name, html] of [
    ['missing', '<script src="assets/index-a.js"></script>'],
    ['ambiguous', '<script type="module" src="assets/index-a.js"></script><script type="module" src="assets/nested/worker-c.js"></script>'],
    ['external', '<script type="module" src="https://other.test/assets/index-a.js"></script>'],
    ['traversal', '<script type="module" src="../assets/index-a.js"></script>'],
    ['missing asset', '<script type="module" src="assets/absent.js"></script>'],
]) {
    test(`rejects ${name} module entry instead of guessing`, () => {
        assert.throws(() => measureWebBundle(fixture({ 'index.html': html })), /entry/i);
    });
}
test('rejects missing or ambiguous InspectionPanel chunks', () => {
    assert.throws(() => measureWebBundle(fixture({ 'assets/InspectionPanel-b.js': null })), /InspectionPanel/i);
    assert.throws(() => measureWebBundle(fixture({ 'assets/InspectionPanel-extra.js': 'export {};' })), /InspectionPanel/i);
});
test('rejects missing index and symlinks rather than measuring outside the build', () => {
    assert.throws(() => measureWebBundle(fixture({ 'index.html': null })), /index.html/);
    const dist = fixture();
    symlinkSync(join(dist, 'index.html'), join(dist, 'assets', 'linked.js'));
    assert.throws(() => measureWebBundle(dist), /symbolic link/i);
});
test('accepts equality at every cap and returns the measurement', () => {
    const value = measurement();
    assert.equal(assertBundleWithinLimits(value), value);
});
for (const dimension of ['entry', 'inspection', 'totalJavaScript']) {
    test(`rejects a one-byte ${dimension} overrun with measured and allowed values`, () => {
        const value = measurement();
        value[dimension].gzipBytes++;
        assert.throws(() => assertBundleWithinLimits(value), (error) => {
            assert.match(error.message, new RegExp(dimension, 'i'));
            assert.ok(error.message.includes(String(BUNDLE_LIMITS[dimension] + 1)));
            assert.ok(error.message.includes(String(BUNDLE_LIMITS[dimension])));
            return true;
        });
    });
}
test('reports all failed dimensions together', () => {
    const value = measurement();
    for (const dimension of Object.keys(BUNDLE_LIMITS)) value[dimension].gzipBytes++;
    assert.throws(() => assertBundleWithinLimits(value), /entry[\s\S]*inspection[\s\S]*totalJavaScript/);
});
for (const invalid of [NaN, Infinity, -1, 1.5, '1']) {
    test(`rejects invalid measurements and limits: ${String(invalid)}`, () => {
        const value = measurement();
        value.entry.gzipBytes = invalid;
        assert.throws(() => assertBundleWithinLimits(value), /entry/i);
        assert.throws(() => assertBundleWithinLimits(measurement(), { ...BUNDLE_LIMITS, entry: invalid }), /entry/i);
    });
}
test('CLI emits each measured dimension on success and fails on an absent build', () => {
    const script = fileURLToPath(new URL('./check-web-bundle-gzip.mjs', import.meta.url));
    const good = spawnSync(process.execPath, [script, fixture()], { encoding: 'utf8' });
    assert.equal(good.status, 0, good.stderr);
    assert.match(good.stdout, /entry gzip.*152245/);
    assert.match(good.stdout, /InspectionPanel gzip.*7373/);
    assert.match(good.stdout, /total JavaScript gzip.*234161/);
    const bad = spawnSync(process.execPath, [script, join(tmpdir(), 'absent-nn-forge-build')], { encoding: 'utf8' });
    assert.equal(bad.status, 1);
    assert.match(bad.stderr, /index.html|ENOENT/);
});
