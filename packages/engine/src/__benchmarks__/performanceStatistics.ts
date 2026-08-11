import { performance } from 'node:perf_hooks';

const MIN_MEASURED_ROUNDS = 5;

export interface PerformanceMeasurementOptions {
    warmupIterations: number;
    measuredRounds: number;
    iterationsPerRound: number;
    now?: () => number;
}

export interface PerformanceBudgetResult {
    name: string;
    measured: number;
    limit: number;
}

export function median(values: readonly number[]): number {
    if (values.length === 0) {
        throw new Error('median requires at least one value');
    }
    if (values.some((value) => !Number.isFinite(value))) {
        throw new Error('median requires finite values');
    }

    const sorted = [...values].sort((left, right) => left - right);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
        ? (sorted[middle - 1] + sorted[middle]) / 2
        : sorted[middle];
}

export function measureMedianMsPerIteration(
    run: () => void,
    options: PerformanceMeasurementOptions,
): number {
    assertNonNegativeInteger(options.warmupIterations, 'warmupIterations');
    assertPositiveInteger(options.iterationsPerRound, 'iterationsPerRound');
    if (!Number.isInteger(options.measuredRounds) || options.measuredRounds < MIN_MEASURED_ROUNDS) {
        throw new Error(`measuredRounds must be an integer of at least ${MIN_MEASURED_ROUNDS}`);
    }

    for (let iteration = 0; iteration < options.warmupIterations; iteration++) {
        run();
    }

    const now = options.now ?? (() => performance.now());
    const samples: number[] = [];
    for (let round = 0; round < options.measuredRounds; round++) {
        const start = now();
        for (let iteration = 0; iteration < options.iterationsPerRound; iteration++) {
            run();
        }
        const elapsed = now() - start;
        if (!Number.isFinite(elapsed) || elapsed < 0) {
            throw new Error('measurement requires a finite non-negative elapsed time');
        }
        samples.push(elapsed / options.iterationsPerRound);
    }

    return median(samples);
}

export function assertPerformanceBudgets(results: readonly PerformanceBudgetResult[]): void {
    if (results.length === 0) {
        throw new Error('performance budgets require at least one result');
    }

    const names = new Set<string>();
    for (const result of results) {
        if (names.has(result.name)) {
            throw new Error(`duplicate performance budget result: ${result.name}`);
        }
        names.add(result.name);
        if (!Number.isFinite(result.limit) || result.limit < 0) {
            throw new Error(`performance budget limit for ${result.name} must be finite and non-negative`);
        }
    }

    const failures = results.filter((result) => (
        !Number.isFinite(result.measured)
        || result.measured < 0
        || result.measured > result.limit
    ));
    if (failures.length === 0) return;

    const details = failures.map((result) => (
        `- ${result.name}: ${formatMilliseconds(result.measured)} ms/iteration `
        + `(limit ${formatMilliseconds(result.limit)} ms/iteration)`
    ));
    throw new Error(`Performance budgets exceeded:\n${details.join('\n')}`);
}

function assertNonNegativeInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value < 0) {
        throw new Error(`${name} must be a non-negative integer`);
    }
}

function assertPositiveInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new Error(`${name} must be a positive integer`);
    }
}

function formatMilliseconds(value: number): string {
    return Number.isFinite(value) ? value.toFixed(4) : String(value);
}
