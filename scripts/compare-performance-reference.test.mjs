import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const BASELINE_SHA = 'ae09b9863ae90f8fb2f62545834fcc138755ba9a';
const CANDIDATE_SHA = '64ffa81720f1c562cf39ebd4f4f774f5f821346c';

function logFor({
    predictGrid = 10,
    predictGridInto = 10,
    predictGridWithNeurons = 12,
    predictGridWithNeuronsInto = 11,
    adam = 4,
    sgd = 1.4,
    forcedPair = 8,
    saveCapture = 10,
} = {}) {
    return [
        `predictGrid: ${predictGrid.toFixed(4)}ms median per iteration; 200.0000ms median round (7 rounds x 20 iterations)`,
        `predictGridInto: ${predictGridInto.toFixed(4)}ms median per iteration; 200.0000ms median round (7 rounds x 20 iterations)`,
        `predictGridWithNeurons: ${predictGridWithNeurons.toFixed(4)}ms median per iteration; 120.0000ms median round (7 rounds x 10 iterations)`,
        `predictGridWithNeuronsInto: ${predictGridWithNeuronsInto.toFixed(4)}ms median per iteration; 110.0000ms median round (7 rounds x 10 iterations)`,
        `Average applyGradients time (Adam, L2, Clip): ${adam.toFixed(4)}ms`,
        `Average applyGradients time (SGD zero-gradient adapter): ${sgd.toFixed(4)}ms`,
        `Scientific trust forced pair median: ${forcedPair.toFixed(4)}ms`,
        `Scientific trust save capture median: ${saveCapture.toFixed(4)}ms`,
        '[perf] engine: FAIL (exit 1)',
        '[perf] web: PASS',
    ].join('\n');
}

async function importSubject() {
    return import('./compare-performance-reference.mjs');
}

test('retains the original same-machine 20% regression rule and worker budgets', async () => {
    const { PERFORMANCE_REFERENCE_POLICY } = await importSubject();
    assert.deepEqual(PERFORMANCE_REFERENCE_POLICY, {
        runCount: 5,
        maximumRegressionFactor: 1.2,
        forcedPairMaximumMs: 250,
        saveCaptureMaximumMs: 500,
    });
    assert.equal(Object.isFrozen(PERFORMANCE_REFERENCE_POLICY), true);
});

test('parses robust benchmark output and all worker medians', async () => {
    const { parsePerformanceLog } = await importSubject();
    assert.deepEqual(parsePerformanceLog(logFor()), {
        predictGrid: 10,
        predictGridInto: 10,
        predictGridWithNeurons: 12,
        predictGridWithNeuronsInto: 11,
        adam: 4,
        sgd: 1.4,
        forcedPair: 8,
        saveCapture: 10,
    });
});

test('parses the historical total-for-iterations grid format without changing units', async () => {
    const { parsePerformanceLog } = await importSubject();
    const legacy = [
        'predictGrid: 1000.0000ms total for 100 iterations',
        'predictGridInto: 1100.0000ms total for 100 iterations',
        'predictGridWithNeurons: 600.0000ms total for 50 iterations',
        'predictGridWithNeuronsInto: 550.0000ms total for 50 iterations',
        'Average applyGradients time (Adam, L2, Clip): 4.0000ms',
        'Average applyGradients time (SGD): 1.5000ms',
        'Scientific trust forced pair median: 9.0000ms',
        'Scientific trust save capture median: 12.0000ms',
        '[perf] web: PASS',
    ].join('\n');
    assert.deepEqual(parsePerformanceLog(legacy), {
        predictGrid: 10,
        predictGridInto: 11,
        predictGridWithNeurons: 12,
        predictGridWithNeuronsInto: 11,
        adam: 4,
        sgd: 1.5,
        forcedPair: 9,
        saveCapture: 12,
    });
});

test('fails closed for missing or duplicate measurements', async () => {
    const { parsePerformanceLog } = await importSubject();
    assert.throws(() => parsePerformanceLog('predictGrid: 10.0000ms median per iteration'), /missing.*predictGridInto/i);
    assert.throws(() => parsePerformanceLog(`${logFor()}\npredictGrid: 9.0000ms median per iteration`), /duplicate.*predictGrid/i);
});

test('requires the web scientific-trust gate to have completed successfully', async () => {
    const { parsePerformanceLog } = await importSubject();
    assert.throws(
        () => parsePerformanceLog(logFor().replace('[perf] web: PASS', '[perf] web: FAIL (exit 1)')),
        /web.*PASS/i,
    );
});

test('compares five candidate medians to five same-runner baseline medians', async () => {
    const { assessPerformanceReference } = await importSubject();
    const baselineRuns = Array.from({ length: 5 }, (_, index) =>
        ({ ...awaitableIdentity(logFor({ predictGrid: 10 + index * 0.1 })) })
    );
    const candidateRuns = Array.from({ length: 5 }, (_, index) =>
        ({ ...awaitableIdentity(logFor({ predictGrid: 11 + index * 0.1 })) })
    );
    const { parsePerformanceLog } = await importSubject();
    const assessment = assessPerformanceReference({
        baselineRuns: baselineRuns.map(({ text }) => parsePerformanceLog(text)),
        candidateRuns: candidateRuns.map(({ text }) => parsePerformanceLog(text)),
    });
    assert.equal(assessment.passed, true);
    const grid = assessment.measurements.find((entry) => entry.name === 'predictGrid');
    assert.equal(grid.baselineMedian, 10.2);
    assert.equal(grid.candidateMedian, 11.2);
    assert.equal(grid.maximumCandidate, 12.24);
    assert.equal(grid.passed, true);
});

function awaitableIdentity(text) {
    return { text };
}

test('rejects an engine median beyond 120% while reporting every failed dimension', async () => {
    const { assessPerformanceReference, parsePerformanceLog, assertPerformanceReference } = await importSubject();
    const baselineRuns = Array.from({ length: 5 }, () => parsePerformanceLog(logFor()));
    const candidateRuns = Array.from({ length: 5 }, () => parsePerformanceLog(logFor({
        predictGrid: 12.01,
        predictGridInto: 12.5,
    })));
    const assessment = assessPerformanceReference({ baselineRuns, candidateRuns });
    assert.equal(assessment.passed, false);
    assert.throws(
        () => assertPerformanceReference(assessment),
        /predictGrid[\s\S]*predictGridInto/,
    );
});

test('keeps the 250ms forced-pair and 500ms save-capture budgets absolute', async () => {
    const { assessPerformanceReference, parsePerformanceLog, assertPerformanceReference } = await importSubject();
    const baselineRuns = Array.from({ length: 5 }, () => parsePerformanceLog(logFor()));
    const candidateRuns = Array.from({ length: 5 }, () => parsePerformanceLog(logFor({
        forcedPair: 250.01,
        saveCapture: 500.01,
    })));
    const assessment = assessPerformanceReference({ baselineRuns, candidateRuns });
    assert.equal(assessment.passed, false);
    assert.throws(() => assertPerformanceReference(assessment), /forcedPair[\s\S]*saveCapture/);
});

test('requires exactly five complete runs per revision', async () => {
    const { assessPerformanceReference, parsePerformanceLog } = await importSubject();
    const run = parsePerformanceLog(logFor());
    assert.throws(() => assessPerformanceReference({
        baselineRuns: [run, run, run, run],
        candidateRuns: [run, run, run, run, run],
    }), /exactly 5 baseline/i);
    assert.throws(() => assessPerformanceReference({
        baselineRuns: [run, run, run, run, run],
        candidateRuns: [run, run, run, run],
    }), /exactly 5 candidate/i);
});

test('loads exact SHAs, all ten logs, and the recorded exits from an evidence directory', async () => {
    const { loadPerformanceReference } = await importSubject();
    const root = await mkdtemp(join(tmpdir(), 'nn-forge-perf-reference-'));
    await writeFile(join(root, 'baseline-sha.txt'), `${BASELINE_SHA}\n`);
    await writeFile(join(root, 'candidate-sha.txt'), `${CANDIDATE_SHA}\n`);
    await writeFile(join(root, 'exits.csv'), [
        'pass,revision,exit_code',
        ...Array.from({ length: 5 }, (_, index) => [
            `${index + 1},baseline,1`,
            `${index + 1},candidate,1`,
        ]).flat(),
    ].join('\n'));
    for (let pass = 1; pass <= 5; pass++) {
        await writeFile(join(root, `${pass}-baseline.log`), logFor());
        await writeFile(join(root, `${pass}-candidate.log`), logFor({ predictGrid: 11 }));
    }
    const loaded = await loadPerformanceReference(root, {
        expectedBaselineSha: BASELINE_SHA,
        expectedCandidateSha: CANDIDATE_SHA,
    });
    assert.equal(loaded.baselineRuns.length, 5);
    assert.equal(loaded.candidateRuns.length, 5);
    assert.equal(loaded.exits.length, 10);
    assert.equal(loaded.assessment.passed, true);
});

test('rejects wrong SHAs, missing pass/revision pairs, and unsupported exit codes', async () => {
    const { loadPerformanceReference } = await importSubject();
    const root = await mkdtemp(join(tmpdir(), 'nn-forge-perf-reference-invalid-'));
    await writeFile(join(root, 'baseline-sha.txt'), `${BASELINE_SHA}\n`);
    await writeFile(join(root, 'candidate-sha.txt'), `${CANDIDATE_SHA}\n`);
    await writeFile(join(root, 'exits.csv'), 'pass,revision,exit_code\n1,baseline,3\n');
    await assert.rejects(
        loadPerformanceReference(root, {
            expectedBaselineSha: '0000000000000000000000000000000000000000',
            expectedCandidateSha: CANDIDATE_SHA,
        }),
        /baseline SHA/i,
    );
    await assert.rejects(
        loadPerformanceReference(root, {
            expectedBaselineSha: BASELINE_SHA,
            expectedCandidateSha: CANDIDATE_SHA,
        }),
        /(unsupported exit code|missing.*pass)/i,
    );
});
