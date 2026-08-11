declare module 'jest-axe' {
    export function axe(...args: unknown[]): Promise<{
        violations: readonly {
            readonly id: string;
            readonly nodes: readonly { readonly html: string }[];
        }[];
    }>;
}
