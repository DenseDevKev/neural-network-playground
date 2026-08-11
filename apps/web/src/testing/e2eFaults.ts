export const E2E_WORKER_FAULT_MESSAGE = 'Injected E2E worker startup failure.';

const STARTUP_ONCE_MARKER = 'nn-playground:e2e-worker-fault:startup-once';

export type E2EWorkerFault = 'startup-once';

export function consumeE2EWorkerFault(
    url: URL,
    storage: Pick<Storage, 'getItem' | 'setItem'>,
    enabled: boolean,
): E2EWorkerFault | null {
    if (!enabled || url.searchParams.get('e2eWorkerFault') !== 'startup-once') return null;
    try {
        if (storage.getItem(STARTUP_ONCE_MARKER) === 'consumed') return null;
        storage.setItem(STARTUP_ONCE_MARKER, 'consumed');
        return 'startup-once';
    } catch {
        return null;
    }
}
