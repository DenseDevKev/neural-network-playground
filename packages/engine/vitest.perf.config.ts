import { defineConfig } from 'vitest/config';

export default defineConfig({
    test: {
        include: ['src/__benchmarks__/**/*.bench.ts'],
        fileParallelism: false,
        maxWorkers: 1,
        minWorkers: 1,
    },
});
