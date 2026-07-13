import { defineConfig } from 'vitest/config';

export default defineConfig({
    test: {
        environment: 'jsdom',
        include: ['src/worker/scientificTrust.performance.test.ts'],
        fileParallelism: false,
        maxWorkers: 1,
        minWorkers: 1,
    },
});
