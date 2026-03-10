// eslint.config.js
/** @type {import('eslint').Linter.Config} */
export default [
  {
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: {
        window: 'readonly',
        document: 'readonly',
        d3: 'readonly',
        tf: 'readonly',
      },
    },
    rules: {
      'no-unused-vars': ['warn'],
      'no-console': ['off'],
      'semi': ['error', 'always'],
      'quotes': ['error', 'single'],
      'indent': ['error', 2],
      'comma-dangle': ['error', 'always-multiline'],
    },
  },
];
