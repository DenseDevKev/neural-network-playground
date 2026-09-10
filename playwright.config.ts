import { defineConfig, devices } from '@playwright/test';
import process from 'node:process';
import { resolvePlaywrightTarget } from './scripts/playwright-target.mjs';

const isCI = Boolean(process.env.CI);
const target = resolvePlaywrightTarget(process.env);
const baseURL = target.baseURL;

export default defineConfig({
    testDir: './tests/e2e',
    metadata: { targetMode: target.mode, baseURL },
    timeout: 45_000,
    expect: {
        timeout: 10_000,
    },
    fullyParallel: false,
    forbidOnly: isCI,
    retries: 0,
    workers: isCI ? 1 : undefined,
    reporter: [
        ['list'],
        ['json', { outputFile: 'playwright-results.json' }],
        ['html', { outputFolder: 'playwright-report', open: 'never' }],
    ],
    use: {
        baseURL,
        locale: 'en-US',
        trace: 'retain-on-failure',
        screenshot: 'only-on-failure',
        video: 'off',
    },
    projects: [
        {
            name: 'chromium',
            use: { ...devices['Desktop Chrome'] },
        },
        {
            name: 'webkit',
            use: { ...devices['Desktop Safari'] },
        },
    ],
    webServer: target.mode === 'local' ? {
        command: `pnpm --filter @nn-playground/web preview --host 127.0.0.1 --port ${target.port} --strictPort`,
        url: baseURL,
        reuseExistingServer: false,
        timeout: 120_000,
    } : undefined,
});
