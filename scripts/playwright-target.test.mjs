import assert from 'node:assert/strict';
import { test } from 'node:test';
import { URL } from 'node:url';
import { resolvePlaywrightTarget } from './playwright-target.mjs';

test('defaults to the existing local preview and port', () => {
    assert.deepEqual(resolvePlaywrightTarget({}), {
        mode: 'local', baseURL: 'http://127.0.0.1:4173/', port: 4173,
    });
});

for (const port of ['1', '65535']) {
    test(`accepts local port ${port}`, () => {
        assert.equal(resolvePlaywrightTarget({ PLAYWRIGHT_PORT: port }).port, Number(port));
    });
}
for (const port of ['', 'abc', '0', '65536', '-1', '1.5', '1e3', ' 4173', '99999999999999999']) {
    test(`rejects invalid local port ${JSON.stringify(port)}`, () => {
        assert.throws(() => resolvePlaywrightTarget({ PLAYWRIGHT_PORT: port }), /PLAYWRIGHT_PORT.*1.*65535/);
    });
}
for (const url of ['https://example.test/project', 'https://example.test/project/']) {
    test(`preserves the external project path ${url}`, () => {
        const target = resolvePlaywrightTarget({ PLAYWRIGHT_BASE_URL: url });
        assert.deepEqual(target, { mode: 'external', baseURL: 'https://example.test/project/' });
        assert.equal(new URL('./', target.baseURL).pathname, '/project/');
        assert.equal(new URL('./?e2eWorkerFault=startup-once', target.baseURL).pathname, '/project/');
    });
}
for (const host of ['127.0.0.1', 'localhost', '[::1]']) {
    test(`allows HTTP loopback fixture ${host}`, () => {
        const target = resolvePlaywrightTarget({ PLAYWRIGHT_BASE_URL: `http://${host}:4174/project/` });
        assert.equal(target.mode, 'external');
        assert.equal('port' in target, false);
    });
}
for (const url of ['', '/project', 'not a url', ' https://example.test/', 'https://example.test/ ', 'https://example.test/a\\b']) {
    test(`rejects malformed external URL ${JSON.stringify(url)}`, () => {
        assert.throws(() => resolvePlaywrightTarget({ PLAYWRIGHT_BASE_URL: url }), /PLAYWRIGHT_BASE_URL/);
    });
}
for (const url of ['http://example.test/project/', 'file:///tmp/project/', 'ftp://example.test/']) {
    test(`rejects insecure or non-HTTP external URL ${url}`, () => {
        assert.throws(() => resolvePlaywrightTarget({ PLAYWRIGHT_BASE_URL: url }), /HTTPS.*loopback/);
    });
}
for (const suffix of ['?a=1', '?', '#a', '#']) {
    test(`rejects URL query or fragment ${suffix}`, () => {
        assert.throws(() => resolvePlaywrightTarget({ PLAYWRIGHT_BASE_URL: `https://example.test/${suffix}` }), /query.*fragment/);
    });
}
for (const url of ['https://user@example.test/', 'https://user:password@example.test/']) {
    test('rejects URL credentials', () => {
        assert.throws(() => resolvePlaywrightTarget({ PLAYWRIGHT_BASE_URL: url }), /credentials/);
    });
}
test('rejects conflicting explicit target settings rather than silently choosing one', () => {
    assert.throws(() => resolvePlaywrightTarget({
        PLAYWRIGHT_BASE_URL: 'https://example.test/project/', PLAYWRIGHT_PORT: '4173',
    }), /PLAYWRIGHT_PORT.*PLAYWRIGHT_BASE_URL/);
});
test('does not mutate the supplied environment', () => {
    const env = Object.freeze({ PLAYWRIGHT_BASE_URL: 'https://example.test/project' });
    resolvePlaywrightTarget(env);
    assert.equal(env.PLAYWRIGHT_BASE_URL, 'https://example.test/project');
});
