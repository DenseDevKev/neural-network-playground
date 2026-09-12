import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import process from 'node:process';
import { test } from 'node:test';

const wrapper = resolve('scripts/run-with-evidence.mjs');
function run(t, command, args = [], name = 'probe') {
    const directory = mkdtempSync(join(tmpdir(), 'nn-evidence-'));
    t.after(() => rmSync(directory, { recursive: true, force: true }));
    const result = spawnSync(process.execPath, [wrapper, name, '--', command, ...args], {
        encoding: 'utf8', timeout: 10000,
        env: { ...process.env, NN_FORGE_EVIDENCE_DIR: directory },
    });
    return { result, directory, receipt: () => JSON.parse(readFileSync(join(directory, `${name}.json`), 'utf8')) };
}

test('records a successful command, source identity and both raw streams', (t) => {
    const { result, directory, receipt } = run(t, process.execPath, ['-e', "console.log('out'); console.error('err');"]);
    assert.equal(result.status, 0, result.stderr);
    const record = receipt();
    assert.equal(record.status, 'passed');
    assert.equal(record.exitCode, 0);
    assert.equal(record.schemaVersion, 1);
    assert.match(record.source.sha, /^[a-f0-9]{40}$/);
    assert.match(record.source.tree, /^[a-f0-9]{40}$/);
    assert.ok(record.completedAt >= record.startedAt);
    assert.equal(record.nodeVersion, process.version);
    assert.match(readFileSync(join(directory, 'probe.log'), 'utf8'), /out/);
    assert.match(readFileSync(join(directory, 'probe.log'), 'utf8'), /err/);
});
test('preserves a failing child exit code rather than turning reporting into success', (t) => {
    const { result, receipt } = run(t, process.execPath, ['-e', 'process.exit(7)']);
    assert.equal(result.status, 7);
    assert.equal(receipt().status, 'failed');
    assert.equal(receipt().exitCode, 7);
});
test('records missing executable as a failed launch', (t) => {
    const { result, receipt } = run(t, 'nn-forge-command-that-does-not-exist');
    assert.equal(result.status, 127);
    assert.equal(receipt().status, 'failed');
    assert.match(receipt().error, /ENOENT/);
});
test('passes argument boundaries without shell expansion', (t) => {
    const args = ['one two', '$(echo injected)', '; exit 9', '*.ts'];
    const { result, receipt } = run(t, process.execPath, ['-e', 'console.log(JSON.stringify(process.argv.slice(1)))', ...args]);
    assert.equal(result.status, 0);
    assert.deepEqual(JSON.parse(result.stdout.trim()), args);
    assert.deepEqual(receipt().command.slice(-4), args);
});
test('retains larger streamed output without truncating its raw log', (t) => {
    const { result, directory } = run(t, process.execPath, ['-e', "process.stdout.write('x'.repeat(100000));"]);
    assert.equal(result.status, 0);
    assert.equal(readFileSync(join(directory, 'probe.log')).length, 100000);
});
test('records a signalled child as interrupted, never passed', (t) => {
    const { result, receipt } = run(t, process.execPath, ['-e', "process.kill(process.pid, 'SIGTERM');"]);
    assert.equal(result.status, 143);
    assert.equal(receipt().status, 'interrupted');
    assert.equal(receipt().signal, 'SIGTERM');
});
test('rejects traversal in evidence names before launching a command', (t) => {
    const { result, directory } = run(t, process.execPath, ['-e', 'process.exit(0)'], '../escape');
    assert.equal(result.status, 2);
    assert.deepEqual(readdirSync(directory), []);
});
test('missing command fails with usage, not a passing receipt', () => {
    const result = spawnSync(process.execPath, [wrapper, 'probe', '--'], { encoding: 'utf8' });
    assert.equal(result.status, 2);
    assert.match(result.stderr, /Usage:/);
});

test('parent cancellation remains interrupted even when the child exits gracefully', { timeout: 10000 }, async (t) => {
    const directory = mkdtempSync(join(tmpdir(), 'nn-evidence-cancel-'));
    t.after(() => rmSync(directory, { recursive: true, force: true }));
    const child = spawn(process.execPath, [wrapper, 'cancel', '--', process.execPath, '-e',
        "process.on('SIGTERM', () => process.exit(0)); console.log('ready'); setInterval(() => {}, 1000);"], {
        env: { ...process.env, NN_FORGE_EVIDENCE_DIR: directory },
        stdio: ['ignore', 'pipe', 'pipe'],
    });
    t.after(() => { if (child.exitCode === null) child.kill('SIGKILL'); });
    let output = '';
    let sent = false;
    child.stdout.on('data', (data) => {
        output += data;
        if (!sent && output.includes('ready')) { sent = true; child.kill('SIGTERM'); }
    });
    const code = await new Promise((resolve, reject) => {
        child.on('error', reject);
        child.on('close', resolve);
    });
    assert.equal(sent, true, 'the child installed its graceful termination handler');
    assert.equal(code, 143);
    const receipt = JSON.parse(readFileSync(join(directory, 'cancel.json'), 'utf8'));
    assert.equal(receipt.status, 'interrupted');
    assert.equal(receipt.signal, 'SIGTERM');
    assert.equal(receipt.exitCode, 143);
});
