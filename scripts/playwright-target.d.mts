export type PlaywrightTarget =
    | { mode: 'local'; baseURL: string; port: number }
    | { mode: 'external'; baseURL: string };

export function resolvePlaywrightTarget(env: {
    PLAYWRIGHT_BASE_URL?: string;
    PLAYWRIGHT_PORT?: string;
}): PlaywrightTarget;
