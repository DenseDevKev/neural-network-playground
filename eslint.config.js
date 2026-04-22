// ESLint flat config (v9+)
import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import react from 'eslint-plugin-react';
import reactHooks from 'eslint-plugin-react-hooks';
import globals from 'globals';

export default [
    // ── Global ignores ──
    {
        ignores: [
            '**/dist/',
            '**/node_modules/',
        ],
        linterOptions: {
            // Don't error on eslint-disable comments that may become "unused"
            // as rules shift between warn/error levels during initial rollout.
            reportUnusedDisableDirectives: 'warn',
        },
    },

    // ── Base JS recommended (all files) ──
    js.configs.recommended,

    // ── TypeScript recommended (all TS/TSX files) ──
    ...tseslint.configs.recommended.map((config) => ({
        ...config,
        files: ['**/*.{ts,tsx}'],
    })),

    // ── React rules scoped to web app ──
    {
        files: ['apps/web/**/*.{ts,tsx}'],
        plugins: {
            react,
            'react-hooks': reactHooks,
        },
        languageOptions: {
            globals: {
                ...globals.browser,
            },
        },
        settings: {
            react: {
                version: 'detect',
            },
        },
        rules: {
            // React 19 uses the automatic JSX runtime — no React import needed
            ...react.configs['jsx-runtime'].rules,
            ...reactHooks.configs.recommended.rules,

            // Warn on missing deps rather than error — existing suppression at
            // useTraining.ts:273 must remain meaningful.
            'react-hooks/exhaustive-deps': 'warn',
        },
    },

    // ── Project-wide TypeScript rule overrides ──
    {
        files: ['**/*.{ts,tsx}'],
        rules: {
            // Intentional `as any` casts exist in tests — keep off globally.
            '@typescript-eslint/no-explicit-any': 'off',
            // Allow underscore-prefixed identifiers to indicate intentionally
            // unused parameters (common in stubs and callbacks).
            '@typescript-eslint/no-unused-vars': [
                'error',
                {
                    argsIgnorePattern: '^_',
                    varsIgnorePattern: '^_',
                    caughtErrorsIgnorePattern: '^_',
                },
            ],
        },
    },
];
