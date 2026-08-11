import { spawn } from 'node:child_process';
import nodeConsole from 'node:console';
import { resolve } from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';

export const PERFORMANCE_GATES = Object.freeze([
    Object.freeze({
        name: 'engine',
        args: Object.freeze(['--filter', '@nn-playground/engine', 'test:perf']),
    }),
    Object.freeze({
        name: 'web',
        args: Object.freeze(['--filter', '@nn-playground/web', 'test:perf']),
    }),
]);

function normalizeError(error) {
    if (error === null || error === undefined) {
        return null;
    }

    const message = error instanceof Error ? error.message : String(error);
    return message.replace(/\s+/gu, ' ').trim();
}

function normalizeResult(gate, outcome) {
    return {
        name: gate.name,
        code: typeof outcome?.code === 'number' ? outcome.code : null,
        signal: typeof outcome?.signal === 'string' ? outcome.signal : null,
        error: normalizeError(outcome?.error),
    };
}

function normalizeThrownResult(gate, error) {
    return {
        name: gate.name,
        code: null,
        signal: null,
        error: normalizeError(error),
    };
}

function passed(result) {
    return result.error === null && result.code === 0 && result.signal === null;
}

export function resolvePnpmInvocation(
    gate,
    platform = process.platform,
    env = process.env,
) {
    if (platform === 'win32') {
        return {
            command: env.ComSpec ?? 'cmd.exe',
            args: ['/d', '/s', '/c', 'pnpm.cmd', ...gate.args],
        };
    }

    return {
        command: 'pnpm',
        args: [...gate.args],
    };
}

function runPnpmGate(gate) {
    const { command, args } = resolvePnpmInvocation(gate);

    return new Promise((resolveGate, rejectGate) => {
        let settled = false;
        const settle = (callback, value) => {
            if (settled) {
                return;
            }
            settled = true;
            callback(value);
        };

        let child;
        try {
            child = spawn(command, args, {
                shell: false,
                stdio: 'inherit',
            });
        } catch (error) {
            settle(rejectGate, error);
            return;
        }

        child.once('error', (error) => settle(rejectGate, error));
        child.once('close', (code, signal) => {
            settle(resolveGate, { code, signal });
        });
    });
}

export async function runPerformanceGates(runCommand) {
    const results = [];

    for (const gate of PERFORMANCE_GATES) {
        try {
            const outcome = await runCommand(gate);
            results.push(normalizeResult(gate, outcome));
        } catch (error) {
            results.push(normalizeThrownResult(gate, error));
        }
    }

    return {
        exitCode: results.every(passed) ? 0 : 1,
        results,
    };
}

function failureReason(result) {
    if (result.error !== null) {
        return `spawn error: ${result.error}`;
    }
    if (result.signal !== null) {
        return `signal ${result.signal}`;
    }
    if (result.code !== null) {
        return `exit ${result.code}`;
    }
    return 'no exit status';
}

function formatSummary(result) {
    if (passed(result)) {
        return `[perf] ${result.name}: PASS`;
    }
    return `[perf] ${result.name}: FAIL (${failureReason(result)})`;
}

export async function main({
    runCommand = runPnpmGate,
    writeLine = nodeConsole.log,
} = {}) {
    const aggregate = await runPerformanceGates(runCommand);

    for (const result of aggregate.results) {
        writeLine(formatSummary(result));
    }

    return aggregate.exitCode;
}

const isDirectRun =
    process.argv[1] !== undefined &&
    import.meta.url === pathToFileURL(resolve(process.argv[1])).href;

if (isDirectRun) {
    process.exitCode = await main();
}
