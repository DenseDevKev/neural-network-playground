export const WORKER_READY_MESSAGE = Object.freeze({
    type: 'nn-playground:worker-ready',
    protocolVersion: 1,
} as const);

export type WorkerReadyMessage = typeof WORKER_READY_MESSAGE;

export function isWorkerReadyMessage(value: unknown): value is WorkerReadyMessage {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
    const candidate = value as Record<string, unknown>;
    const keys = Object.keys(candidate);
    return keys.length === 2
        && Object.prototype.hasOwnProperty.call(candidate, 'type')
        && Object.prototype.hasOwnProperty.call(candidate, 'protocolVersion')
        && candidate.type === WORKER_READY_MESSAGE.type
        && candidate.protocolVersion === WORKER_READY_MESSAGE.protocolVersion;
}

export function exposeWorkerApiAndAnnounceReady(
    expose: () => void,
    target: { postMessage(message: unknown): void },
): void {
    expose();
    target.postMessage(WORKER_READY_MESSAGE);
}
