import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { stripVTControlCharacters } from 'node:util';
import process from 'node:process';
import { measureWebBundle, assertBundleWithinLimits, BUNDLE_LIMITS } from './check-web-bundle-gzip.mjs';
import { loadPerformanceReference } from './compare-performance-reference.mjs';

const limit = 8;
const text = (value) => stripVTControlCharacters(String(value ?? '')).slice(0, 600);
const invalid = (error) => ({ status: 'invalid', counts: null, error: text(error) });
const readJson = (path) => JSON.parse(readFileSync(path, 'utf8'));

/** A summary never upgrades skipped, interrupted, absent or inconsistent evidence. */
export function summarizePlaywright(report) {
    try {
        if (!Array.isArray(report?.suites) || !Array.isArray(report.errors)) return invalid('Missing suites/errors');
        const counts = { expected: 0, unexpected: 0, flaky: 0, skipped: 0 };
        for (const name of Object.keys(counts)) {
            if (!Number.isSafeInteger(report.stats?.[name]) || report.stats[name] < 0) return invalid('Invalid report counts');
        }
        const failures = [];
        const resultStatuses = new Set(['passed', 'failed', 'timedOut', 'skipped', 'interrupted']);
        let interrupted = false;
        function visit(suites, titles = []) {
            for (const suite of suites) {
                for (const spec of suite.specs ?? []) {
                    if (!Array.isArray(spec.tests)) throw new Error('Missing test results');
                    for (const test of spec.tests) {
                        if (!Object.hasOwn(counts, test.status) || !Array.isArray(test.results)
                            || !resultStatuses.has(test.expectedStatus ?? 'passed')
                            || test.results.some((result) => !resultStatuses.has(result?.status))) throw new Error('Invalid test status/results');
                        if (test.status === 'expected' && test.results.at(-1)?.status !== (test.expectedStatus ?? 'passed')) throw new Error('Expected test lacks a matching completed result');
                        counts[test.status]++;
                        interrupted ||= test.results.some((result) => result.status === 'interrupted');
                        if (test.status === 'unexpected' || test.status === 'flaky') {
                            const error = test.results.flatMap((result) => result.errors ?? []).map((error) => error.message ?? error.value).join('\n');
                            failures.push({ test: text([...titles, suite.title, spec.title].join(' > ')), project: text(test.projectName), error: text(error) });
                        }
                    }
                }
                if (suite.suites) visit(suite.suites, [...titles, suite.title]);
            }
        }
        visit(report.suites);
        for (const name of Object.keys(counts)) if (counts[name] !== report.stats[name]) return invalid('Counts do not match enumerated tests');
        const errors = report.errors.map((error) => text(error.message ?? error.value));
        const status = interrupted ? 'interrupted' : errors.length || counts.unexpected || counts.flaky ? 'failed'
            : counts.expected ? 'passed' : counts.skipped ? 'skipped' : 'incomplete';
        return { status, counts, failures: failures.slice(0, limit), omittedFailures: Math.max(0, failures.length - limit), errors: errors.slice(0, limit), omittedErrors: Math.max(0, errors.length - limit) };
    } catch (error) { return invalid(error.message); }
}

export function readPlaywrightReport(path) {
    try { return summarizePlaywright(readJson(path)); }
    catch (error) { return error.code === 'ENOENT' ? { status: 'missing', counts: null, error: 'Report not produced' } : invalid(error.message); }
}

export function summarizeStep(receipt, sourceSha, expectedName = receipt?.name) {
    if (!receipt) return { status: 'missing' };
    if (receipt.schemaVersion !== 1 || receipt.name !== expectedName || !/^[a-z][a-z0-9-]{0,63}$/.test(receipt.name)
        || !Array.isArray(receipt.command) || !receipt.command.length || !receipt.command.every((arg) => typeof arg === 'string')
        || !/^[a-f0-9]{40}$/.test(sourceSha) || !/^[a-f0-9]{40}$/.test(receipt.source?.tree)
        || !sourceSha || receipt.source?.sha !== sourceSha || receipt.source?.trackedChanges !== false) return invalid('Invalid or mismatched source receipt');
    if (receipt.status === 'running' && receipt.exitCode === null && receipt.completedAt === null) return { status: 'incomplete' };
    if (!Number.isInteger(receipt.exitCode) || receipt.exitCode < 0 || receipt.exitCode > 255
        || !Number.isFinite(Date.parse(receipt.startedAt)) || !Number.isFinite(Date.parse(receipt.completedAt)) || Date.parse(receipt.completedAt) < Date.parse(receipt.startedAt)
        || (receipt.status === 'passed' && (receipt.signal || receipt.error))
        || (receipt.status === 'interrupted' && !receipt.signal)
        || !receipt.completedAt
        || !['passed', 'failed', 'interrupted'].includes(receipt.status)
        || (receipt.status === 'passed') !== (receipt.exitCode === 0)) return invalid('Inconsistent completion receipt');
    return { status: receipt.status, command: text(receipt.command.join(' ')), exitCode: receipt.exitCode, signal: receipt.signal, error: receipt.error ? text(receipt.error) : null };
}

const modes = {
    focused: { steps: ['changed-tests', 'helpers', 'tests', 'lint', 'types', 'build', 'bundle'], reports: [] },
    preview: { steps: ['build', 'preview'], reports: ['playwright-results.json'] },
    subpath: { steps: ['build', 'subpath'], reports: ['playwright-results.json'] },
    recovery: { steps: ['build', 'recovery', 'recovery-rebuild', 'fault-disabled'], reports: ['recovery-evidence/playwright-results.json', 'playwright-results.json'] },
    performance: { steps: ['performance'], reports: [] },
};

export async function summarizeQualification(mode, directory = 'qualification-evidence') {
    if (!Object.hasOwn(modes, mode)) throw new Error('Unknown qualification mode');
    let sourceSha = null;
    let sourceTree = null;
    let trackedChanges = null;
    try {
        sourceSha = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
        sourceTree = execFileSync('git', ['rev-parse', 'HEAD^{tree}'], { encoding: 'utf8' }).trim();
        trackedChanges = execFileSync('git', ['status', '--porcelain', '--untracked-files=no'], { encoding: 'utf8' }).trim() !== '';
    } catch { /* missing provenance cannot pass */ }
    const summary = { schemaVersion: 1, mode, sourceSha, sourceTree, trackedChanges, runId: process.env.GITHUB_RUN_ID ?? null, attempt: process.env.GITHUB_RUN_ATTEMPT ?? null, steps: {}, reports: {} };
    for (const name of modes[mode].steps) {
        try { summary.steps[name] = summarizeStep(readJson(join(directory, `${name}.json`)), sourceSha, name); }
        catch (error) { summary.steps[name] = error.code === 'ENOENT' ? { status: 'missing' } : invalid(error.message); }
    }
    for (const path of modes[mode].reports) summary.reports[path] = readPlaywrightReport(path);
    if (mode === 'focused') {
        try {
            const measurement = measureWebBundle('apps/web/dist');
            let status = 'passed';
            try { assertBundleWithinLimits(measurement); } catch { status = 'failed'; }
            summary.bundle = { status, measurement, limits: BUNDLE_LIMITS };
        } catch (error) { summary.bundle = { status: 'missing', error: text(error.message), limits: BUNDLE_LIMITS }; }
        try {
            summary.testSummaryLines = stripVTControlCharacters(readFileSync(join(directory, 'tests.log'), 'utf8'))
                .split('\n').filter((line) => /\b(?:Tests|Test Files)\s+\d/.test(line)).slice(-12).map(text);
        } catch { summary.testSummaryLines = []; }
    }
    if (mode === 'performance') {
        try {
            const evidence = await loadPerformanceReference('performance-evidence', {
                expectedBaselineSha: 'ae09b9863ae90f8fb2f62545834fcc138755ba9a', expectedCandidateSha: sourceSha,
            });
            summary.performance = { status: evidence.assessment.passed ? 'passed' : 'failed', assessment: evidence.assessment };
        } catch (error) { summary.performance = invalid(error.message); }
    }
    const statuses = [...Object.values(summary.steps), ...Object.values(summary.reports), ...[summary.bundle, summary.performance].filter(Boolean)].map((item) => item.status);
    summary.status = trackedChanges === false && /^[a-f0-9]{40}$/.test(sourceTree) && statuses.every((status) => status === 'passed') ? 'passed' : statuses.some((status) => status === 'failed' || status === 'interrupted') ? 'failed' : 'incomplete';
    mkdirSync(directory, { recursive: true });
    writeFileSync(join(directory, 'summary.json'), `${JSON.stringify(summary, null, 2)}\n`);
    return summary;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
    try {
        const summary = await summarizeQualification(process.argv[2], process.env.NN_FORGE_EVIDENCE_DIR ?? 'qualification-evidence');
        process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
        process.exitCode = summary.status === 'passed' ? 0 : 1;
    } catch (error) { process.stderr.write(`Qualification summary failed: ${text(error.message)}\n`); process.exitCode = 1; }
}
