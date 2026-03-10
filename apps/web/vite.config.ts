import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => ({
    plugins: [react()],
    // Use hash-based routing, so base can be set for GH Pages subdirectory
    base: './',
    worker: {
        format: 'es',
    },
    optimizeDeps: {
        include: ['comlink'],
    },
    build: {
        target: 'es2022',
        sourcemap: mode === 'development',
        rollupOptions: {
            output: {
                manualChunks: {
                    react: ['react', 'react-dom'],
                    engine: ['@nn-playground/engine'],
                },
                chunkFileNames: 'assets/[name]-[hash].js',
                entryFileNames: 'assets/[name]-[hash].js',
                assetFileNames: 'assets/[name]-[hash].[ext]',
            },
        },
        chunkSizeWarningLimit: 200,
    },
}));
