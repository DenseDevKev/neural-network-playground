import { defineConfig, devices } from '@playwright/test';
import process from 'node:process';

const isCI = Boolean(process.env.CI);
const requestedPort = process.env.PLAYWRIGHT_PORT ?? '4173';
if (!/^\d+$/.test(requestedPort)) {
    throw new Error('PLAYWRIGHT_PORT must be an integer between 1 and 65535');
}
const port = Number(requestedPort);
if (!Number.isSafeInteger(port) || port < 1 || port > 65_535) {
    throw new Error('PLAYWRIGHT_PORT must be an integer between 1 and 65535');
}
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
    testDir: './tests/e2e',
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
    webServer: {
        command: `pnpm --filter @nn-playground/web preview --host 127.0.0.1 --port ${port} --strictPort`,
        url: baseURL,
        reuseExistingServer: false,
        timeout: 120_000,
    },
});
