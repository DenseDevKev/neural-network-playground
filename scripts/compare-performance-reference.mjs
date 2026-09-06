import process from 'node:process';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

export const PERFORMANCE_REFERENCE_POLICY = Object.freeze({
    runCount: 5,
    maximumRegressionFactor: 1.2,
    forcedPairMaximumMs: 250,
    saveCaptureMaximumMs: 500,
});

const ENGINE_METRICS = Object.freeze([
    'predictGrid',
    'predictGridInto',
    'predictGridWithNeurons',
    'predictGridWithNeuronsInto',
    'adam',
    'sgd',
]);

const ALL_METRICS = Object.freeze([
    ...ENGINE_METRICS,
    'forcedPair',
    'saveCapture',
]);

function numericMatches(text, pattern, transform = (value) => value) {
    const matches = [];
    for (const match of text.matchAll(pattern)) {
        const value = Number(match[1]);
        const transformed = transform(value, match);
        if (!Number.isFinite(transformed) || transformed < 0) {
            throw new Error(`Invalid performance measurement: ${match[0]}`);
        }
        matches.push(transformed);
    }
    return matches;
}

function readUniqueMetric(name, candidates) {
    const values = candidates.flat();
    if (values.length === 0) {
        throw new Error(`Missing performance measurement: ${name}`);
    }
    if (values.length > 1) {
        throw new Error(`Duplicate performance measurement: ${name}`);
    }
    return values[0];
}

function parseGridMetric(text, name) {
    const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const modern = numericMatches(
        text,
        new RegExp(`^${escaped}:\\s*([0-9]+(?:\\.[0-9]+)?)ms median per iteration\\b`, 'gm'),
    );
    const legacy = numericMatches(
        text,
        new RegExp(`^${escaped}:\\s*([0-9]+(?:\\.[0-9]+)?)ms total for ([0-9]+) iterations\\b`, 'gm'),
        (total, match) => {
            const iterations = Number(match[2]);
            if (!Number.isInteger(iterations) || iterations <= 0) {
                throw new Error(`Invalid iteration count for ${name}: ${match[2]}`);
            }
            return total / iterations;
        },
    );
    return readUniqueMetric(name, [modern, legacy]);
}

export function parsePerformanceLog(text) {
    if (typeof text !== 'string') {
        throw new TypeError('Performance log must be text');
    }
    if (!/^\[perf\] web: PASS\s*$/m.test(text)) {
        throw new Error('Performance log must contain a successful [perf] web: PASS summary');
    }

    const measurements = {
        predictGrid: parseGridMetric(text, 'predictGrid'),
        predictGridInto: parseGridMetric(text, 'predictGridInto'),
        predictGridWithNeurons: parseGridMetric(text, 'predictGridWithNeurons'),
        predictGridWithNeuronsInto: parseGridMetric(text, 'predictGridWithNeuronsInto'),
        adam: readUniqueMetric('adam', [numericMatches(
            text,
            /^Average applyGradients time \(Adam, L2, Clip\):\s*([0-9]+(?:\.[0-9]+)?)ms\s*$/gm,
        )]),
        sgd: readUniqueMetric('sgd', [numericMatches(
            text,
            /^Average applyGradients time \(SGD(?: zero-gradient adapter)?\):\s*([0-9]+(?:\.[0-9]+)?)ms\s*$/gm,
        )]),
        forcedPair: readUniqueMetric('forcedPair', [numericMatches(
            text,
            /^Scientific trust forced pair median:\s*([0-9]+(?:\.[0-9]+)?)ms\s*$/gm,
        )]),
        saveCapture: readUniqueMetric('saveCapture', [numericMatches(
            text,
            /^Scientific trust save capture median:\s*([0-9]+(?:\.[0-9]+)?)ms\s*$/gm,
        )]),
    };

    return Object.freeze(measurements);
}

export function median(values) {
    if (!Array.isArray(values) || values.length === 0) {
        throw new Error('Median requires at least one measurement');
    }
    const sorted = values.map((value) => {
        if (!Number.isFinite(value) || value < 0) {
            throw new Error(`Invalid median measurement: ${value}`);
        }
        return value;
    }).sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 1
        ? sorted[middle]
        : (sorted[middle - 1] + sorted[middle]) / 2;
}

function round(value, digits = 6) {
    const factor = 10 ** digits;
    return Math.round((value + Number.EPSILON) * factor) / factor;
}

function validateRuns(label, runs, runCount) {
    if (!Array.isArray(runs) || runs.length !== runCount) {
        throw new Error(`Expected exactly ${runCount} ${label} runs`);
    }
    for (const [index, run] of runs.entries()) {
        if (!run || typeof run !== 'object') {
            throw new Error(`Invalid ${label} run ${index + 1}`);
        }
        for (const name of ALL_METRICS) {
            const value = run[name];
            if (!Number.isFinite(value) || value < 0) {
                throw new Error(`Invalid ${label} ${name} measurement in run ${index + 1}`);
            }
        }
    }
}

export function assessPerformanceReference(
    { baselineRuns, candidateRuns },
    policy = PERFORMANCE_REFERENCE_POLICY,
) {
    validateRuns('baseline', baselineRuns, policy.runCount);
    validateRuns('candidate', candidateRuns, policy.runCount);

    const measurements = [];
    for (const name of ENGINE_METRICS) {
        const baselineMedian = median(baselineRuns.map((run) => run[name]));
        const candidateMedian = median(candidateRuns.map((run) => run[name]));
        if (baselineMedian <= 0) {
            throw new Error(`Baseline median must be greater than zero for ${name}`);
        }
        const maximumCandidateRaw = baselineMedian * policy.maximumRegressionFactor;
        measurements.push(Object.freeze({
            name,
            basis: 'relative',
            baselineMedian,
            candidateMedian,
            maximumCandidate: round(maximumCandidateRaw),
            changePercent: round(((candidateMedian / baselineMedian) - 1) * 100, 4),
            passed: candidateMedian <= maximumCandidateRaw,
        }));
    }

    for (const [name, maximumCandidate] of [
        ['forcedPair', policy.forcedPairMaximumMs],
        ['saveCapture', policy.saveCaptureMaximumMs],
    ]) {
        const baselineMedian = median(baselineRuns.map((run) => run[name]));
        const candidateMedian = median(candidateRuns.map((run) => run[name]));
        measurements.push(Object.freeze({
            name,
            basis: 'absolute',
            baselineMedian,
            candidateMedian,
            maximumCandidate,
            changePercent: baselineMedian === 0
                ? null
                : round(((candidateMedian / baselineMedian) - 1) * 100, 4),
            passed: baselineMedian <= maximumCandidate && candidateMedian <= maximumCandidate,
        }));
    }

    return Object.freeze({
        policy,
        measurements: Object.freeze(measurements),
        passed: measurements.every((measurement) => measurement.passed),
    });
}

export function assertPerformanceReference(assessment) {
    const failures = assessment.measurements.filter((measurement) => !measurement.passed);
    if (failures.length === 0) {
        return assessment;
    }
    const details = failures.map((failure) =>
        `- ${failure.name}: candidate ${failure.candidateMedian}ms, baseline ${failure.baselineMedian}ms, maximum ${failure.maximumCandidate}ms (${failure.basis})`
    );
    throw new Error(`Performance reference gate failed:\n${details.join('\n')}`);
}

function parseExitsCsv(text, runCount) {
    const lines = text.trim().split(/\r?\n/);
    if (lines.shift() !== 'pass,revision,exit_code') {
        throw new Error('Invalid exits.csv header');
    }
    const seen = new Set();
    const exits = [];
    for (const line of lines) {
        if (!line.trim()) continue;
        const match = /^(\d+),(baseline|candidate),(-?\d+)$/.exec(line.trim());
        if (!match) {
            throw new Error(`Invalid exits.csv row: ${line}`);
        }
        const pass = Number(match[1]);
        const revision = match[2];
        const exitCode = Number(match[3]);
        if (!Number.isInteger(pass) || pass < 1 || pass > runCount) {
            throw new Error(`Invalid performance pass: ${match[1]}`);
        }
        if (exitCode !== 0 && exitCode !== 1) {
            throw new Error(`Unsupported exit code ${exitCode} for pass ${pass} ${revision}`);
        }
        const key = `${pass}:${revision}`;
        if (seen.has(key)) {
            throw new Error(`Duplicate performance pass: ${key}`);
        }
        seen.add(key);
        exits.push(Object.freeze({ pass, revision, exitCode }));
    }
    for (let pass = 1; pass <= runCount; pass++) {
        for (const revision of ['baseline', 'candidate']) {
            if (!seen.has(`${pass}:${revision}`)) {
                throw new Error(`Missing performance pass ${pass} ${revision}`);
            }
        }
    }
    return Object.freeze(exits);
}

async function readTrimmed(path) {
    return (await readFile(path, 'utf8')).trim();
}

export async function loadPerformanceReference(
    directory,
    {
        expectedBaselineSha,
        expectedCandidateSha,
        policy = PERFORMANCE_REFERENCE_POLICY,
    } = {},
) {
    const root = resolve(directory);
    const baselineSha = await readTrimmed(resolve(root, 'baseline-sha.txt'));
    const candidateSha = await readTrimmed(resolve(root, 'candidate-sha.txt'));
    if (expectedBaselineSha && baselineSha !== expectedBaselineSha) {
        throw new Error(`Unexpected baseline SHA: ${baselineSha}`);
    }
    if (expectedCandidateSha && candidateSha !== expectedCandidateSha) {
        throw new Error(`Unexpected candidate SHA: ${candidateSha}`);
    }

    const exits = parseExitsCsv(
        await readFile(resolve(root, 'exits.csv'), 'utf8'),
        policy.runCount,
    );
    const baselineRuns = [];
    const candidateRuns = [];
    for (let pass = 1; pass <= policy.runCount; pass++) {
        baselineRuns.push(parsePerformanceLog(
            await readFile(resolve(root, `${pass}-baseline.log`), 'utf8'),
        ));
        candidateRuns.push(parsePerformanceLog(
            await readFile(resolve(root, `${pass}-candidate.log`), 'utf8'),
        ));
    }
    const assessment = assessPerformanceReference({ baselineRuns, candidateRuns }, policy);
    return Object.freeze({
        baselineSha,
        candidateSha,
        exits,
        baselineRuns: Object.freeze(baselineRuns),
        candidateRuns: Object.freeze(candidateRuns),
        assessment,
    });
}

function formatMeasurement(measurement) {
    const change = measurement.changePercent === null
        ? 'n/a'
        : `${measurement.changePercent >= 0 ? '+' : ''}${measurement.changePercent}%`;
    return [
        measurement.name,
        `baseline=${measurement.baselineMedian}ms`,
        `candidate=${measurement.candidateMedian}ms`,
        `change=${change}`,
        `maximum=${measurement.maximumCandidate}ms`,
        measurement.passed ? 'PASS' : 'FAIL',
    ].join(' ');
}

export async function main({
    argv = process.argv.slice(2),
    writeLine = (line) => console.log(line),
} = {}) {
    const [directory, expectedBaselineSha, expectedCandidateSha] = argv;
    if (!directory || !expectedBaselineSha || !expectedCandidateSha) {
        throw new Error('Usage: node scripts/compare-performance-reference.mjs <evidence-dir> <baseline-sha> <candidate-sha>');
    }
    const loaded = await loadPerformanceReference(directory, {
        expectedBaselineSha,
        expectedCandidateSha,
    });
    writeLine(`performance baseline ${loaded.baselineSha}`);
    writeLine(`performance candidate ${loaded.candidateSha}`);
    for (const measurement of loaded.assessment.measurements) {
        writeLine(formatMeasurement(measurement));
    }
    assertPerformanceReference(loaded.assessment);
    return 0;
}

const entry = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : null;
if (entry === import.meta.url) {
    main().catch((error) => {
        console.error(error instanceof Error ? error.message : String(error));
        process.exitCode = 1;
    });
}
