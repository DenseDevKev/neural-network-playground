import { Buffer } from 'node:buffer';
import { spawn, execFileSync } from 'node:child_process';
import { mkdirSync, openSync, writeSync, closeSync, writeFileSync, renameSync } from 'node:fs';
import { constants } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import process from 'node:process';

function git(...args) {
    try { return execFileSync('git', args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim(); }
    catch { return null; }
}

/** Stream logs to disk; command arguments are never evaluated by a shell. */
export async function runWithEvidence(name, command, args, directory = 'qualification-evidence') {
    if (!/^[a-z][a-z0-9-]{0,63}$/.test(name) || !command) throw new Error('Invalid receipt name or command');
    mkdirSync(directory, { recursive: true });
    const destination = join(directory, `${name}.json`);
    const log = openSync(join(directory, `${name}.log`), 'w');
    const dirty = git('status', '--porcelain', '--untracked-files=no');
    const receipt = {
        schemaVersion: 1, name, command: [command, ...args],
        source: { sha: git('rev-parse', 'HEAD'), tree: git('rev-parse', 'HEAD^{tree}'), trackedChanges: dirty === null ? null : dirty !== '' },
        nodeVersion: process.version,
        startedAt: new Date().toISOString(), completedAt: null,
        status: 'running', exitCode: null, signal: null, error: null,
    };
    const persist = () => {
        writeFileSync(`${destination}.tmp`, `${JSON.stringify(receipt, null, 2)}\n`);
        renameSync(`${destination}.tmp`, destination);
    };
    try { persist(); } catch (error) { closeSync(log); throw error; }
    return new Promise((resolvePromise, reject) => {
        const child = spawn(command, args, { stdio: ['inherit', 'pipe', 'pipe'], detached: process.platform !== 'win32' });
        let ioError = null;
        let spawnError = null;
        let requestedSignal = null;
        const forward = (signal) => {
            try {
                if (process.platform !== 'win32' && child.pid) process.kill(-child.pid, signal);
                else child.kill(signal);
            } catch (error) { if (error.code !== 'ESRCH') ioError = error; }
        };
        const terminate = () => { requestedSignal ??= 'SIGTERM'; forward('SIGTERM'); };
        const interrupt = () => { requestedSignal ??= 'SIGINT'; forward('SIGINT'); };
        process.on('SIGTERM', terminate);
        process.on('SIGINT', interrupt);
        const stream = (chunk, output) => {
            try { writeSync(log, chunk); output.write(chunk); }
            catch (error) { ioError = error; forward('SIGTERM'); }
        };
        child.stdout.on('data', (chunk) => stream(chunk, process.stdout));
        child.stderr.on('data', (chunk) => stream(chunk, process.stderr));
        child.on('error', (error) => { spawnError = error; stream(Buffer.from(`${error.message}\n`), process.stderr); });
        child.on('close', (code, signal) => {
            process.off('SIGTERM', terminate);
            process.off('SIGINT', interrupt);
            const interruption = requestedSignal ?? signal;
            const exitCode = interruption ? 128 + (constants.signals[interruption] ?? 0) : spawnError ? 127 : code ?? 1;
            Object.assign(receipt, {
                completedAt: new Date().toISOString(), signal: interruption,
                exitCode: ioError && exitCode === 0 ? 1 : exitCode,
                status: interruption ? 'interrupted' : exitCode === 0 && !ioError ? 'passed' : 'failed',
                error: (spawnError ?? ioError)?.message ?? null,
            });
            try { closeSync(log); persist(); resolvePromise(receipt.exitCode); }
            catch (error) { reject(error); }
        });
    });
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
    const [name, separator, command, ...args] = process.argv.slice(2);
    if (!name || separator !== '--' || !command || !/^[a-z][a-z0-9-]{0,63}$/.test(name)) {
        process.stderr.write('Usage: node scripts/run-with-evidence.mjs <safe-name> -- <command> [arguments...]\n');
        process.exitCode = 2;
    } else {
        try { process.exitCode = await runWithEvidence(name, command, args, process.env.NN_FORGE_EVIDENCE_DIR ?? 'qualification-evidence'); }
        catch (error) { process.stderr.write(`Evidence recording failed: ${error.message}\n`); process.exitCode = 1; }
    }
}
