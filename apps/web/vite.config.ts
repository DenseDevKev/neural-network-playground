import { defineConfig } from 'vite';
import { configDefaults } from 'vitest/config';
import react from '@vitejs/plugin-react';
import process from 'node:process';

// Cross-origin isolation headers are required for SharedArrayBuffer to be
// available to the app (AS-3 snapshot fast path). When the headers are
// absent, the app still works — the worker falls back to postMessage
// transferables. Applied to both dev server and preview; production
// hosting must set these headers separately (GH Pages cannot — SAB path
// will transparently disable itself there).
const crossOriginIsolationHeaders = {
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cache-Control': 'no-store',
};

export default defineConfig(({ mode }) => ({
    plugins: [react()],
    // Use hash-based routing, so base can be set for GH Pages subdirectory
    base: process.env.VITE_BASE ?? './',
    worker: {
        format: 'es',
    },
    optimizeDeps: {
        include: ['comlink'],
    },
    server: {
        headers: crossOriginIsolationHeaders,
    },
    preview: {
        headers: crossOriginIsolationHeaders,
    },
    build: {
        target: 'es2022',
        minify: 'terser',
        // Two safe compression passes reduce repeated expressions without unsafe rewrites.
        terserOptions: { compress: { passes: 2, inline: 1 } },
        sourcemap: mode === 'development',
        rollupOptions: {
            output: {
                // Keep all initial capabilities in the capped entry. Merge tiny
                // shared helpers to avoid separate gzip overhead; feature lazies stay split.
                experimentalMinChunkSize: 10000,
                chunkFileNames: 'assets/[name]-[hash].js',
                entryFileNames: 'assets/[name]-[hash].js',
                assetFileNames: 'assets/[name]-[hash].[ext]',
            },
        },
        chunkSizeWarningLimit: 200,
    },
    test: {
        globals: true,
        environment: 'jsdom',
        setupFiles: './src/test/setup.ts',
        testTimeout: 60000,
        exclude: [
            ...configDefaults.exclude,
            'src/worker/scientificTrust.performance.test.ts',
        ],
    },
}));
