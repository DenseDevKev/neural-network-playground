declare module 'jest-axe' {
    export function axe(...args: any[]): Promise<{
        violations: unknown[];
    }>;
}
