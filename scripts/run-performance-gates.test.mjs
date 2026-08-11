import assert from 'node:assert/strict';
import nodeConsole from 'node:console';
import process from 'node:process';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { URL } from 'node:url';

test('root test:perf delegates exactly to the aggregate runner', async () => {
    const packageJson = JSON.parse(
        await readFile(new URL('../package.json', import.meta.url), 'utf8'),
    );

    assert.equal(
        packageJson.scripts['test:perf'],
        'node scripts/run-performance-gates.mjs',
    );
    assert.equal(packageJson.scripts['test:perf'].includes('&&'), false);
});

test('performance gates and their argv are deeply frozen', async () => {
    const { PERFORMANCE_GATES } = await import('./run-performance-gates.mjs');

    assert.deepEqual(PERFORMANCE_GATES, [
        {
            name: 'engine',
            args: ['--filter', '@nn-playground/engine', 'test:perf'],
        },
        {
            name: 'web',
            args: ['--filter', '@nn-playground/web', 'test:perf'],
        },
    ]);
    assert.equal(Object.isFrozen(PERFORMANCE_GATES), true);
    for (const gate of PERFORMANCE_GATES) {
        assert.equal(Object.isFrozen(gate), true);
        assert.equal(Object.isFrozen(gate.args), true);
    }
});

test('aggregate result covers every pass and failure combination', async () => {
    const { runPerformanceGates } = await import('./run-performance-gates.mjs');
    const scenarios = [
        { codes: [0, 0], exitCode: 0 },
        { codes: [3, 0], exitCode: 1 },
        { codes: [0, 4], exitCode: 1 },
        { codes: [2, 5], exitCode: 1 },
    ];

    for (const scenario of scenarios) {
        const calls = [];
        const result = await runPerformanceGates(async (gate) => {
            calls.push(gate.name);
            const index = gate.name === 'engine' ? 0 : 1;
            return { code: scenario.codes[index], signal: null };
        });

        assert.deepEqual(calls, ['engine', 'web']);
        assert.equal(result.exitCode, scenario.exitCode);
        assert.deepEqual(
            result.results,
            [
                {
                    name: 'engine',
                    code: scenario.codes[0],
                    signal: null,
                    error: null,
                },
                {
                    name: 'web',
                    code: scenario.codes[1],
                    signal: null,
                    error: null,
                },
            ],
        );
    }
});

test('gates execute strictly in order without overlapping', async () => {
    const { runPerformanceGates } = await import('./run-performance-gates.mjs');
    const events = [];
    let finishEngine;
    const engine = new Promise((resolve) => {
        finishEngine = resolve;
    });

    const aggregate = runPerformanceGates(async (gate) => {
        events.push(`start:${gate.name}`);
        if (gate.name === 'engine') {
            await engine;
        }
        events.push(`finish:${gate.name}`);
        return { code: 0, signal: null };
    });

    await Promise.resolve();
    assert.deepEqual(events, ['start:engine']);
    finishEngine();
    const result = await aggregate;

    assert.deepEqual(events, [
        'start:engine',
        'finish:engine',
        'start:web',
        'finish:web',
    ]);
    assert.equal(result.exitCode, 0);
});

test('a synchronous throw or rejected engine gate never skips web', async () => {
    const { runPerformanceGates } = await import('./run-performance-gates.mjs');
    const failures = [
        () => {
            throw new Error('sync\nengine failure');
        },
        () => Promise.reject(new Error('async\nengine failure')),
    ];

    for (const failEngine of failures) {
        const calls = [];
        const result = await runPerformanceGates((gate) => {
            calls.push(gate.name);
            if (gate.name === 'engine') {
                return failEngine();
            }
            return Promise.resolve({ code: 0, signal: null });
        });

        assert.deepEqual(calls, ['engine', 'web']);
        assert.equal(result.exitCode, 1);
        assert.equal(result.results[0].code, null);
        assert.equal(result.results[0].signal, null);
        assert.match(result.results[0].error, /^(sync|async) engine failure$/);
        assert.deepEqual(result.results[1], {
            name: 'web',
            code: 0,
            signal: null,
            error: null,
        });
    }
});

test('signals, missing statuses, and injected errors normalize to failures', async () => {
    const { runPerformanceGates } = await import('./run-performance-gates.mjs');
    const result = await runPerformanceGates(async (gate) => {
        if (gate.name === 'engine') {
            return { code: null, signal: 'SIGTERM' };
        }
        return {
            code: null,
            signal: null,
            error: new Error('  spawn\n\n exploded  '),
        };
    });

    assert.equal(result.exitCode, 1);
    assert.deepEqual(result.results, [
        {
            name: 'engine',
            code: null,
            signal: 'SIGTERM',
            error: null,
        },
        {
            name: 'web',
            code: null,
            signal: null,
            error: 'spawn exploded',
        },
    ]);

    const noStatus = await runPerformanceGates(async () => ({
        code: null,
        signal: null,
    }));
    assert.equal(noStatus.exitCode, 1);
});

test('resolver returns exact tokenized POSIX and Windows invocations', async () => {
    const { PERFORMANCE_GATES, resolvePnpmInvocation } = await import(
        './run-performance-gates.mjs'
    );
    const gate = PERFORMANCE_GATES[0];

    assert.deepEqual(resolvePnpmInvocation(gate, 'darwin', {}), {
        command: 'pnpm',
        args: ['--filter', '@nn-playground/engine', 'test:perf'],
    });
    assert.deepEqual(resolvePnpmInvocation(gate, 'linux', {}), {
        command: 'pnpm',
        args: ['--filter', '@nn-playground/engine', 'test:perf'],
    });
    assert.deepEqual(resolvePnpmInvocation(gate, 'win32', {}), {
        command: 'cmd.exe',
        args: [
            '/d',
            '/s',
            '/c',
            'pnpm.cmd',
            '--filter',
            '@nn-playground/engine',
            'test:perf',
        ],
    });
    assert.deepEqual(
        resolvePnpmInvocation(gate, 'win32', { ComSpec: 'C:\\Windows\\cmd.exe' }),
        {
            command: 'C:\\Windows\\cmd.exe',
            args: [
                '/d',
                '/s',
                '/c',
                'pnpm.cmd',
                '--filter',
                '@nn-playground/engine',
                'test:perf',
            ],
        },
    );
});

test('main writes exactly two ordered summaries for successful gates', async () => {
    const { main } = await import('./run-performance-gates.mjs');
    const lines = [];

    const exitCode = await main({
        runCommand: async () => ({ code: 0, signal: null }),
        writeLine: (line) => lines.push(line),
    });

    assert.equal(exitCode, 0);
    assert.deepEqual(lines, ['[perf] engine: PASS', '[perf] web: PASS']);
});

test('summary failure reasons are exact and use the documented precedence', async () => {
    const { main } = await import('./run-performance-gates.mjs');

    const exitAndSignal = [];
    assert.equal(
        await main({
            runCommand: async (gate) =>
                gate.name === 'engine'
                    ? { code: 2, signal: null }
                    : { code: null, signal: 'SIGTERM' },
            writeLine: (line) => exitAndSignal.push(line),
        }),
        1,
    );
    assert.deepEqual(exitAndSignal, [
        '[perf] engine: FAIL (exit 2)',
        '[perf] web: FAIL (signal SIGTERM)',
    ]);

    const spawnAndNoStatus = [];
    assert.equal(
        await main({
            runCommand: async (gate) => {
                if (gate.name === 'engine') {
                    throw new Error('spawn\nerror');
                }
                return { code: null, signal: null };
            },
            writeLine: (line) => spawnAndNoStatus.push(line),
        }),
        1,
    );
    assert.deepEqual(spawnAndNoStatus, [
        '[perf] engine: FAIL (spawn error: spawn error)',
        '[perf] web: FAIL (no exit status)',
    ]);

    const precedence = [];
    await main({
        runCommand: async () => ({
            code: 7,
            signal: 'SIGKILL',
            error: new Error('first\nreason'),
        }),
        writeLine: (line) => precedence.push(line),
    });
    assert.deepEqual(precedence, [
        '[perf] engine: FAIL (spawn error: first reason)',
        '[perf] web: FAIL (spawn error: first reason)',
    ]);
});

test('default spawn errors settle once, continue, and use inherited runner summaries', async () => {
    if (process.platform === 'win32') {
        return;
    }

    const { main } = await import('./run-performance-gates.mjs');
    const originalPath = process.env.PATH;
    const lines = [];
    process.env.PATH = '/path/that/does/not/contain/pnpm';

    try {
        const exitCode = await main({ writeLine: (line) => lines.push(line) });
        assert.equal(exitCode, 1);
        assert.equal(lines.length, 2);
        assert.match(lines[0], /^\[perf\] engine: FAIL \(spawn error: .+\)$/);
        assert.match(lines[1], /^\[perf\] web: FAIL \(spawn error: .+\)$/);
    } finally {
        if (originalPath === undefined) {
            delete process.env.PATH;
        } else {
            process.env.PATH = originalPath;
        }
    }
});

test('import with no argv entry never starts the default runner or throws', async () => {
    const originalArgv = [...process.argv];
    const originalExitCode = process.exitCode;
    const originalLog = nodeConsole.log;
    const lines = [];
    process.argv.splice(0, process.argv.length, originalArgv[0]);
    nodeConsole.log = (line) => lines.push(line);

    try {
        await import('./run-performance-gates.mjs?no-argv-entry');
        assert.deepEqual(lines, []);
        assert.equal(process.exitCode, originalExitCode);
    } finally {
        process.argv.splice(0, process.argv.length, ...originalArgv);
        nodeConsole.log = originalLog;
        process.exitCode = originalExitCode;
    }
});
