import { URL } from 'node:url';

/**
 * Select a browser-test destination without falling back after bad external input.
 * @param {{ PLAYWRIGHT_BASE_URL?: string, PLAYWRIGHT_PORT?: string }} env
 * @returns {{ mode: 'local', baseURL: string, port: number } | { mode: 'external', baseURL: string }}
 */
export function resolvePlaywrightTarget(env) {
    if (env.PLAYWRIGHT_BASE_URL === undefined) {
        const requestedPort = env.PLAYWRIGHT_PORT ?? '4173';
        const port = Number(requestedPort);
        if (!/^\d+$/.test(requestedPort) || !Number.isSafeInteger(port) || port < 1 || port > 65_535) {
            throw new Error('PLAYWRIGHT_PORT must be an integer between 1 and 65535');
        }
        return { mode: 'local', baseURL: `http://127.0.0.1:${port}/`, port };
    }
    if (env.PLAYWRIGHT_PORT !== undefined) {
        throw new Error('PLAYWRIGHT_PORT cannot be combined with PLAYWRIGHT_BASE_URL');
    }
    const raw = env.PLAYWRIGHT_BASE_URL;
    if (raw.length === 0 || /\s|\\/.test(raw)) {
        throw new Error('PLAYWRIGHT_BASE_URL must be an absolute URL without whitespace or backslashes');
    }
    let url;
    try {
        url = new URL(raw);
    } catch {
        throw new Error('PLAYWRIGHT_BASE_URL must be an absolute URL');
    }
    if (url.username || url.password) {
        throw new Error('PLAYWRIGHT_BASE_URL must not contain credentials');
    }
    if (raw.includes('?') || raw.includes('#')) {
        throw new Error('PLAYWRIGHT_BASE_URL must not contain a query or fragment');
    }
    const loopback = ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname);
    if (url.protocol !== 'https:' && !(url.protocol === 'http:' && loopback)) {
        throw new Error('PLAYWRIGHT_BASE_URL requires HTTPS except for an HTTP loopback fixture');
    }
    if (!url.pathname.endsWith('/')) url.pathname += '/';
    return { mode: 'external', baseURL: url.href };
}
