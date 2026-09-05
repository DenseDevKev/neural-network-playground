import { lstatSync, readFileSync, readdirSync } from 'node:fs';
import { basename, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
import process from 'node:process';

export const BUNDLE_LIMITS = Object.freeze({
    entry: 152_245,
    inspection: 7_373,
    totalJavaScript: 234_161,
});

function javascriptFiles(root, directory = root) {
    const result = [];
    for (const name of readdirSync(directory).sort()) {
        const path = join(directory, name);
        const stat = lstatSync(path);
        if (stat.isSymbolicLink()) throw new Error(`Bundle contains a symbolic link: ${path}`);
        if (stat.isDirectory()) result.push(...javascriptFiles(root, path));
        else if (stat.isFile() && name.endsWith('.js')) {
            result.push(relative(root, path).split(sep).join('/'));
        }
    }
    return result;
}

function entrySource(html) {
    // Parse attributes independently so Vite's ordering and quote style do not
    // matter. Inline modules and ambiguous entry scripts are rejected.
    const entries = [];
    for (const script of html.matchAll(/<script\b([^>]*)>/gi)) {
        const attrs = new Map();
        for (const attr of script[1].matchAll(/([^\s"'<>/=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g)) {
            const name = attr[1].toLowerCase();
            if (attrs.has(name)) throw new Error(`Duplicate module entry attribute: ${name}`);
            attrs.set(name, attr[2] ?? attr[3] ?? attr[4] ?? '');
        }
        if (attrs.get('type')?.toLowerCase() === 'module') entries.push(attrs.get('src'));
    }
    if (entries.length !== 1 || !entries[0]) {
        throw new Error(`Expected exactly one external module entry in index.html; found ${entries.length}`);
    }
    return entries[0];
}

export function measureWebBundle(distDir) {
    const root = resolve(distDir);
    const index = join(root, 'index.html');
    if (!lstatSync(index).isFile()) throw new Error('index.html must be a regular file');
    const src = entrySource(readFileSync(index, 'utf8'));
    const files = javascriptFiles(root);
    const path = src.replace(/^\.\//, '').replace(/^\//, '');
    if (/^[a-z][a-z\d+.-]*:/i.test(src) || src.startsWith('//')
        || /[\\?#%]/.test(src) || path.split('/').some((part) => part === '.' || part === '..')) {
        throw new Error(`Module entry must name a local build asset: ${src}`);
    }
    const entries = files.filter((file) => path === file || (src.startsWith('/') && path.endsWith(`/${file}`)));
    if (entries.length !== 1) throw new Error(`Missing or ambiguous module entry asset: ${src}`);
    const inspections = files.filter((file) => /^InspectionPanel-[^.]+\.js$/.test(basename(file)));
    if (inspections.length !== 1) throw new Error(`Expected one InspectionPanel chunk; found ${inspections.length}`);
    if (entries[0] === inspections[0]) throw new Error('Module entry and InspectionPanel chunk must differ');
    const sizes = new Map(files.map((file) => [file, gzipSync(readFileSync(join(root, file))).length]));
    return {
        entry: { file: entries[0], gzipBytes: sizes.get(entries[0]) },
        inspection: { file: inspections[0], gzipBytes: sizes.get(inspections[0]) },
        totalJavaScript: { files: files.length, gzipBytes: [...sizes.values()].reduce((sum, value) => sum + value, 0) },
    };
}

export function assertBundleWithinLimits(measurement, limits = BUNDLE_LIMITS) {
    const failures = [];
    for (const dimension of Object.keys(BUNDLE_LIMITS)) {
        const measured = measurement?.[dimension]?.gzipBytes;
        const allowed = limits?.[dimension];
        if (!Number.isSafeInteger(allowed) || allowed < 0) {
            failures.push(`${dimension}: invalid byte limit ${String(allowed)}`);
        } else if (!Number.isSafeInteger(measured) || measured < 0 || measured > allowed) {
            failures.push(`${dimension}: measured ${String(measured)} bytes; allowed ${allowed} bytes`);
        }
    }
    if (failures.length) throw new Error(`JavaScript gzip budgets failed:\n${failures.join('\n')}`);
    return measurement;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
    try {
        if (process.argv.length > 3) throw new Error('Usage: node scripts/check-web-bundle-gzip.mjs [dist-directory]');
        const measurement = measureWebBundle(process.argv[2] ?? 'apps/web/dist');
        process.stdout.write(`entry gzip ${measurement.entry.gzipBytes} / ${BUNDLE_LIMITS.entry}\n`);
        process.stdout.write(`InspectionPanel gzip ${measurement.inspection.gzipBytes} / ${BUNDLE_LIMITS.inspection}\n`);
        process.stdout.write(`total JavaScript gzip ${measurement.totalJavaScript.gzipBytes} / ${BUNDLE_LIMITS.totalJavaScript}\n`);
        assertBundleWithinLimits(measurement);
    } catch (error) {
        process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
        process.exitCode = 1;
    }
}
