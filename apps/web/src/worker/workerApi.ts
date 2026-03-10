// ── Worker API wrapper — typed Comlink proxy ──
import * as Comlink from 'comlink';
import type { TrainingWorkerApi } from './training.worker.ts';

let worker: Worker | null = null;
let api: Comlink.Remote<TrainingWorkerApi> | null = null;

export function getWorkerApi(): Comlink.Remote<TrainingWorkerApi> {
    if (!api) {
        worker = new Worker(
            new URL('./training.worker.ts', import.meta.url),
            { type: 'module' },
        );
        api = Comlink.wrap<TrainingWorkerApi>(worker);
    }
    return api;
}

export function terminateWorker(): void {
    if (worker) {
        worker.terminate();
        worker = null;
        api = null;
    }
}
