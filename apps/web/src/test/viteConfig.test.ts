import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { describe, expect, it } from 'vitest';

const execFileAsync = promisify(execFile);

describe('Vite response headers', () => {
    it('keeps cross-origin isolation on uncached dev and preview responses', async () => {
        const inspectConfig = `
            import { createRequire } from 'node:module';
            import { resolve } from 'node:path';
            import { pathToFileURL } from 'node:url';

            const configPath = resolve(process.cwd(), 'vite.config.ts');
            const require = createRequire(configPath);
            const vite = await import(pathToFileURL(require.resolve('vite')).href);
            const loaded = await vite.loadConfigFromFile(
                { command: 'serve', mode: 'test' },
                configPath,
            );

            if (!loaded) {
                throw new Error('Vite config could not be loaded');
            }

            process.stdout.write(JSON.stringify({
                server: loaded.config.server?.headers,
                preview: loaded.config.preview?.headers,
            }));
        `;
        const { stdout } = await execFileAsync(
            process.execPath,
            ['--input-type=module', '--eval', inspectConfig],
            { cwd: process.cwd() },
        );
        const config = JSON.parse(stdout) as {
            server?: Record<string, string>;
            preview?: Record<string, string>;
        };
        const expectedHeaders = {
            'Cross-Origin-Opener-Policy': 'same-origin',
            'Cross-Origin-Embedder-Policy': 'require-corp',
            'Cache-Control': 'no-store',
        };

        expect(config.server).toEqual(expectedHeaders);
        expect(config.preview).toEqual(expectedHeaders);
    });
});

describe('Vite production output', () => {
    it('uses the size-focused minifier for the release bundle', async () => {
        const inspectConfig = `
            import { createRequire } from 'node:module';
            import { resolve } from 'node:path';
            import { pathToFileURL } from 'node:url';

            const configPath = resolve(process.cwd(), 'vite.config.ts');
            const require = createRequire(configPath);
            const vite = await import(pathToFileURL(require.resolve('vite')).href);
            const loaded = await vite.loadConfigFromFile(
                { command: 'build', mode: 'production' },
                configPath,
            );

            if (!loaded) {
                throw new Error('Vite config could not be loaded');
            }

            process.stdout.write(JSON.stringify({
                minify: loaded.config.build?.minify,
            }));
        `;
        const { stdout } = await execFileAsync(
            process.execPath,
            ['--input-type=module', '--eval', inspectConfig],
            { cwd: process.cwd() },
        );
        const config = JSON.parse(stdout) as { minify?: string | boolean };

        expect(config.minify).toBe('terser');
    });
});
