export const E2E_WORKER_FAULT_MESSAGE = 'Injected E2E worker startup failure.';

/** True only in bundles built with VITE_E2E_FAULTS=1 (see test:e2e:recovery). */
export const E2E_FAULTS_ENABLED = import.meta.env.VITE_E2E_FAULTS === '1';

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
