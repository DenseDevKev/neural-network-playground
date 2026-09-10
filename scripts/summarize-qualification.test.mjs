import { execFileSync, spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import process from 'node:process';
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { summarizePlaywright, summarizeStep, readPlaywrightReport } from './summarize-qualification.mjs';

function report(statuses = ['expected']) {
    const stats = { expected: 0, unexpected: 0, flaky: 0, skipped: 0 };
    statuses.forEach((status) => { stats[status]++; });
    return { stats, errors: [], suites: [{ title: 'suite', specs: statuses.map((status, index) => ({
        title: `case ${index}`, tests: [{ projectName: 'chromium', status, results: [{
            status: status === 'expected' ? 'passed' : status === 'skipped' ? 'skipped' : 'failed',
            errors: status === 'unexpected' ? [{ message: '\u001b[31mwrong target\u001b[0m' }] : [],
        }] }],
    })) }] };
}

test('summarizes executed passing tests, retaining intentional skips separately', () => {
    const summary = summarizePlaywright(report(['expected', 'skipped']));
    assert.equal(summary.status, 'passed');
    assert.deepEqual(summary.counts, { expected: 1, unexpected: 0, flaky: 0, skipped: 1 });
});
test('does not call an all-skipped or empty suite passed', () => {
    assert.equal(summarizePlaywright(report(['skipped'])).status, 'skipped');
    assert.equal(summarizePlaywright(report([])).status, 'incomplete');
});
test('reports test and global failures with bounded, ANSI-free errors', () => {
    const input = report(['unexpected']);
    input.errors.push({ message: 'global failure' });
    const summary = summarizePlaywright(input);
    assert.equal(summary.status, 'failed');
    assert.equal(summary.failures[0].error, 'wrong target');
    assert.deepEqual(summary.errors, ['global failure']);
});
test('never converts a flaky result to a clean pass', () => {
    assert.equal(summarizePlaywright(report(['flaky'])).status, 'failed');
});
test('reports interrupted execution rather than claiming ordinary completion', () => {
    const input = report(['unexpected']);
    input.suites[0].specs[0].tests[0].results[0].status = 'interrupted';
    assert.equal(summarizePlaywright(input).status, 'interrupted');
});
test('rejects malformed and inconsistent result counts', () => {
    for (const input of [null, {}, { suites: [] }, { ...report(), stats: { expected: -1 } }]) {
        assert.equal(summarizePlaywright(input).status, 'invalid');
    }
    const input = report();
    input.stats.expected = 42;
    assert.equal(summarizePlaywright(input).status, 'invalid');
});
test('bounds failure evidence without losing total failure count', () => {
    const input = report(Array(30).fill('unexpected'));
    input.suites[0].specs.forEach((spec) => { spec.tests[0].results[0].errors[0].message = 'x'.repeat(20000); });
    const summary = summarizePlaywright(input);
    assert.equal(summary.counts.unexpected, 30);
    assert.equal(summary.failures.length, 8);
    assert.equal(summary.omittedFailures, 22);
    assert.ok(summary.failures[0].error.length <= 600);
    assert.ok(JSON.stringify(summary).length < 10000);
});
test('distinguishes missing report from invalid report', (t) => {
    const dir = mkdtempSync(join(tmpdir(), 'nn-summary-'));
    t.after(() => rmSync(dir, { recursive: true, force: true }));
    const path = join(dir, 'results.json');
    assert.equal(readPlaywrightReport(path).status, 'missing');
    writeFileSync(path, '{broken');
    assert.equal(readPlaywrightReport(path).status, 'invalid');
    writeFileSync(path, JSON.stringify(report()));
    assert.equal(readPlaywrightReport(path).status, 'passed');
    assert.ok(readFileSync(path).length > 0);
});

const receipt = () => ({ schemaVersion: 1, name: 'build', status: 'passed', exitCode: 0, signal: null,
    command: ['pnpm', 'build'], source: { sha: 'a'.repeat(40), tree: 'b'.repeat(40), trackedChanges: false },
    startedAt: '2026-09-10T00:00:00.000Z', completedAt: '2026-09-10T00:00:01.000Z' });
test('validates step receipts against the exact source rather than a green-looking label', () => {
    assert.equal(summarizeStep(receipt(), 'a'.repeat(40)).status, 'passed');
    assert.equal(summarizeStep(receipt(), 'c'.repeat(40)).status, 'invalid');
    assert.equal(summarizeStep({ ...receipt(), exitCode: 7 }, 'a'.repeat(40)).status, 'invalid');
    assert.equal(summarizeStep({ ...receipt(), source: { ...receipt().source, trackedChanges: true } }, 'a'.repeat(40)).status, 'invalid');
});
test('missing and incomplete step receipts cannot pass', () => {
    assert.equal(summarizeStep(null, 'a'.repeat(40)).status, 'missing');
    assert.equal(summarizeStep({ ...receipt(), status: 'running', completedAt: null, exitCode: null }, 'a'.repeat(40)).status, 'incomplete');
});
test('expected without executed results and contradictory results cannot pass', () => {
    const missing = report();
    missing.suites[0].specs[0].tests[0].results = [];
    assert.equal(summarizePlaywright(missing).status, 'invalid');
    const contradictory = report();
    contradictory.suites[0].specs[0].tests[0].results[0].status = 'failed';
    assert.equal(summarizePlaywright(contradictory).status, 'invalid');
});
test('expected failure tests remain expected, not reinterpreted as application defects', () => {
    const input = report();
    input.suites[0].specs[0].tests[0].expectedStatus = 'failed';
    input.suites[0].specs[0].tests[0].results[0].status = 'failed';
    assert.equal(summarizePlaywright(input).status, 'passed');
});
test('invalid source, signal, error and timestamp receipts cannot look successful', () => {
    for (const change of [
        { source: { ...receipt().source, tree: null } }, { signal: 'SIGTERM' },
        { error: 'spawn failed' }, { completedAt: '2026-09-09T00:00:00.000Z' },
        { exitCode: 256 }, { command: [null] },
    ]) assert.equal(summarizeStep({ ...receipt(), ...change }, 'a'.repeat(40)).status, 'invalid');
});

test('reads the installed Playwright JSON reporter, including an expected failure', (t) => {
    const require = createRequire(import.meta.url);
    const directory = mkdtempSync(join(tmpdir(), 'nn-real-reporter-'));
    t.after(() => rmSync(directory, { recursive: true, force: true }));
    const output = join(directory, 'results.json');
    const config = join(directory, 'playwright.config.cjs');
    writeFileSync(config, `module.exports = ${JSON.stringify({ testDir: directory, testMatch: 'probe.spec.cjs', outputDir: join(directory, 'output'), workers: 1, retries: 0, reporter: [['json', { outputFile: output }]] })};`);
    writeFileSync(join(directory, 'probe.spec.cjs'), `
        const { test, expect } = require(${JSON.stringify(require.resolve('@playwright/test'))});
        test('passing probe', () => expect(1).toBe(1));
        test('failing probe', () => expect(1).toBe(2));
        test.skip('intentional skip', () => {});
        test('expected failure', () => { test.fail(); expect(1).toBe(2); });
    `);
    const result = spawnSync(process.execPath, [require.resolve('@playwright/test/cli'), 'test', '--config', config], { encoding: 'utf8', timeout: 20000 });
    assert.equal(result.status, 1, result.stderr);
    const summary = readPlaywrightReport(output);
    assert.equal(summary.status, 'failed', JSON.stringify(summary));
    assert.deepEqual(summary.counts, { expected: 2, unexpected: 1, flaky: 0, skipped: 1 });
    assert.match(summary.failures[0].test, /failing probe/);
});

test('invalid expected and actual result status values cannot pass', () => {
    const input = report();
    input.suites[0].specs[0].tests[0].expectedStatus = 'invented';
    input.suites[0].specs[0].tests[0].results[0].status = 'invented';
    assert.equal(summarizePlaywright(input).status, 'invalid');
});

test('CLI refuses passing receipts when tracked source changes after the command', (t) => {
    const directory = mkdtempSync(join(tmpdir(), 'nn-summary-source-'));
    t.after(() => rmSync(directory, { recursive: true, force: true }));
    const git = (...args) => execFileSync('git', args, { cwd: directory, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
    git('init');
    writeFileSync(join(directory, 'source.txt'), 'before');
    git('add', 'source.txt');
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'fixture');
    const sha = git('rev-parse', 'HEAD');
    const tree = git('rev-parse', 'HEAD^{tree}');
    const evidence = join(directory, 'qualification-evidence');
    mkdirSync(evidence);
    for (const name of ['build', 'preview']) {
        writeFileSync(join(evidence, `${name}.json`), JSON.stringify({ ...receipt(), name, source: { sha, tree, trackedChanges: false } }));
    }
    writeFileSync(join(directory, 'playwright-results.json'), JSON.stringify(report()));
    const invoke = () => spawnSync(process.execPath, [resolve('scripts/summarize-qualification.mjs'), 'preview'], {
        cwd: directory, encoding: 'utf8', timeout: 10000,
        env: { ...process.env, NN_FORGE_EVIDENCE_DIR: evidence },
    });
    const clean = invoke();
    assert.equal(clean.status, 0, clean.stderr);
    writeFileSync(join(directory, 'source.txt'), 'after');
    const dirty = invoke();
    assert.equal(dirty.status, 1, dirty.stdout);
    assert.equal(JSON.parse(dirty.stdout).status, 'incomplete');
    assert.equal(JSON.parse(dirty.stdout).trackedChanges, true);
});


test('a syntactically valid receipt tree must match the qualified source tree', () => {
    assert.equal(summarizeStep(receipt(), 'a'.repeat(40), 'build', 'c'.repeat(40)).status, 'invalid');
    assert.equal(summarizeStep(receipt(), 'a'.repeat(40), 'build', 'b'.repeat(40)).status, 'passed');
});
