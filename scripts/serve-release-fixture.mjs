import { createServer } from 'node:http';
import { readFile, realpath, stat } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';
import { URL, pathToFileURL } from 'node:url';
import process from 'node:process';

const BASE_PATH = '/neural-network-playground/';
const MIME = {
    '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
    '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
    '.json': 'application/json', '.map': 'application/json', '.svg': 'image/svg+xml',
    '.png': 'image/png', '.jpg': 'image/jpeg', '.ico': 'image/x-icon',
    '.woff': 'font/woff', '.woff2': 'font/woff2',
};

/** Serve actual build bytes at the Pages project path, deliberately without COOP/COEP. */
export async function createReleaseFixtureServer(distDirectory) {
    const root = await realpath(distDirectory);
    if (!(await stat(root)).isDirectory()) throw new Error('Release fixture requires a dist directory');
    return createServer(async (request, response) => {
        response.setHeader('Cache-Control', 'no-store');
        response.setHeader('X-Content-Type-Options', 'nosniff');
        if (request.method !== 'GET' && request.method !== 'HEAD') {
            response.writeHead(405, { Allow: 'GET, HEAD' }).end();
            return;
        }
        try {
            const url = new URL(request.url ?? '/', 'http://127.0.0.1');
            if (url.pathname === BASE_PATH.slice(0, -1)) {
                response.writeHead(308, { Location: BASE_PATH + url.search }).end();
                return;
            }
            if (!url.pathname.startsWith(BASE_PATH)) {
                response.writeHead(404).end();
                return;
            }
            const relative = decodeURIComponent(url.pathname.slice(BASE_PATH.length)) || 'index.html';
            const requested = resolve(root, relative);
            if (!requested.startsWith(root + sep)) {
                response.writeHead(403).end();
                return;
            }
            const file = await realpath(requested);
            if (!file.startsWith(root + sep)) {
                response.writeHead(403).end();
                return;
            }
            if (!(await stat(file)).isFile()) {
                response.writeHead(404).end();
                return;
            }
            const bytes = await readFile(file);
            response.writeHead(200, {
                'Content-Type': MIME[extname(file)] ?? 'application/octet-stream',
                'Content-Length': bytes.length,
            });
            response.end(request.method === 'HEAD' ? undefined : bytes);
        } catch (error) {
            const status = error instanceof URIError ? 400
                : error?.code === 'ENOENT' || error?.code === 'ENOTDIR' ? 404 : 500;
            response.writeHead(status).end();
        }
    });
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
    const directory = process.argv[2] ?? 'apps/web/dist';
    const rawPort = process.argv[3] ?? '4174';
    const port = Number(rawPort);
    if (!/^\d+$/.test(rawPort) || !Number.isSafeInteger(port) || port < 1 || port > 65_535) {
        throw new Error('Release fixture port must be an integer between 1 and 65535');
    }
    const server = await createReleaseFixtureServer(directory);
    server.on('error', (error) => {
        process.stderr.write(`${error.message}\n`);
        process.exitCode = 1;
    });
    server.listen(port, '127.0.0.1', () => {
        process.stdout.write(`Release fixture: http://127.0.0.1:${port}${BASE_PATH}\n`);
    });
    for (const signal of ['SIGINT', 'SIGTERM']) {
        process.once(signal, () => server.close());
    }
}
