import { getWorkerApi } from '../worker/workerBridge.ts';
import { createDefaultRunTitle } from '../components/controls/runTitle.ts';

function createUuid(): string {
    if (typeof globalThis.crypto?.randomUUID === 'function') {
        return globalThis.crypto.randomUUID();
    }
    const bytes = new Uint8Array(16);
    globalThis.crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
    return [hex.slice(0, 8), hex.slice(8, 12), hex.slice(12, 16), hex.slice(16, 20), hex.slice(20)].join('-');
}

/** Metadata only: the worker owns the recipe, snapshot, and evaluation capture. */
export async function captureCurrentRun(title: string, timestamp: string) {
    const api = await getWorkerApi();
    const record = await api.captureRunArtifact({
        id: createUuid(), createdAt: timestamp, updatedAt: timestamp,
        ...(title ? { title } : {}),
    });
    return record.title === undefined
        ? { ...record, title: createDefaultRunTitle(record.recipe, record.snapshot) }
        : record;
}
