/// <reference types="vite/client" />

interface ImportMetaEnv {
    readonly VITE_E2E_FAULTS?: string;
}

interface ImportMeta {
    readonly env: ImportMetaEnv;
}
