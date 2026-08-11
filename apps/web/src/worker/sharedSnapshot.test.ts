import { afterEach, describe, expect, it, vi } from 'vitest';
import {
    CTL_FLAGS,
    CTL_SEQ_END,
    CTL_SEQ_START,
    FLAG_NEURON_GRIDS,
    FLAG_OUTPUT_GRID,
    allocSharedSnapshotViews,
    attachSharedSnapshotViews,
    canUseSharedBuffers,
    publishSharedSnapshot,
    readSharedSnapshot,
} from './sharedSnapshot.ts';

type NumberAtomicsArray =
    | Int8Array
    | Uint8Array
    | Int16Array
    | Uint16Array
    | Int32Array
    | Uint32Array;
type BigIntAtomicsArray = BigInt64Array | BigUint64Array;

function atomicsLoadMock(
    loadNumber: (array: NumberAtomicsArray, index: number, actual: number) => number,
): typeof Atomics.load {
    const realLoad = Atomics.load;

    function load(array: NumberAtomicsArray, index: number): number;
    function load(array: BigIntAtomicsArray, index: number): bigint;
    function load(
        array: NumberAtomicsArray | BigIntAtomicsArray,
        index: number,
    ): number | bigint {
        if (array instanceof BigInt64Array || array instanceof BigUint64Array) {
            return realLoad(array, index);
        }
        const actual = realLoad(array, index);
        return loadNumber(array, index, actual);
    }

    return load;
}

describe('sharedSnapshot', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('disables shared buffers when cross-origin isolation is explicitly false', () => {
        const original = Object.getOwnPropertyDescriptor(globalThis, 'crossOriginIsolated');
        Object.defineProperty(globalThis, 'crossOriginIsolated', {
            configurable: true,
            value: false,
        });

        try {
            expect(canUseSharedBuffers()).toBe(false);
        } finally {
            if (original) {
                Object.defineProperty(globalThis, 'crossOriginIsolated', original);
            } else {
                delete (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated;
            }
        }
    });

    it('disables shared buffers when SharedArrayBuffer is unavailable', () => {
        const original = Object.getOwnPropertyDescriptor(globalThis, 'SharedArrayBuffer');
        Object.defineProperty(globalThis, 'SharedArrayBuffer', {
            configurable: true,
            value: undefined,
        });

        try {
            expect(canUseSharedBuffers()).toBe(false);
        } finally {
            if (original) {
                Object.defineProperty(globalThis, 'SharedArrayBuffer', original);
            } else {
                delete (globalThis as { SharedArrayBuffer?: unknown }).SharedArrayBuffer;
            }
        }
    });

    it('allocates and attaches typed views over the same SharedArrayBuffers', () => {
        const writerViews = allocSharedSnapshotViews(2, 3);
        const readerViews = attachSharedSnapshotViews({
            control: writerViews.controlSAB,
            outputGrid: writerViews.outputGridSAB,
            neuronGrids: writerViews.neuronGridsSAB,
            gridSize: 2,
            neuronCount: 3,
        });

        expect(writerViews.control).toHaveLength(8);
        expect(writerViews.outputGrid).toHaveLength(4);
        expect(writerViews.neuronGrids).toHaveLength(12);
        expect(readerViews.gridSize).toBe(2);
        expect(readerViews.neuronCount).toBe(3);

        writerViews.outputGrid.set([1, 2, 3, 4]);
        writerViews.neuronGrids.set([5, 6, 7, 8], 4);
        Atomics.store(writerViews.control, CTL_FLAGS, FLAG_OUTPUT_GRID);

        expect(Array.from(readerViews.outputGrid)).toEqual([1, 2, 3, 4]);
        expect(Array.from(readerViews.neuronGrids.slice(4, 8))).toEqual([5, 6, 7, 8]);
        expect(Atomics.load(readerViews.control, CTL_FLAGS)).toBe(FLAG_OUTPUT_GRID);
    });

    it('publishes and reads only the payloads marked by flags', () => {
        const views = allocSharedSnapshotViews(2, 1);
        const outputDst = new Float32Array(4);
        const neuronDst = new Float32Array([9, 9, 9, 9]);

        const firstSeq = publishSharedSnapshot(
            views,
            new Float32Array([0.1, 0.2, 0.3, 0.4]),
            null,
            FLAG_OUTPUT_GRID,
        );

        expect(firstSeq).toBe(1);
        expect(Atomics.load(views.control, CTL_SEQ_START)).toBe(1);
        expect(Atomics.load(views.control, CTL_SEQ_END)).toBe(1);

        expect(readSharedSnapshot(views, outputDst, neuronDst)).toEqual({
            seq: 1,
            flags: FLAG_OUTPUT_GRID,
        });
        expect(outputDst).toEqual(new Float32Array([0.1, 0.2, 0.3, 0.4]));
        expect(neuronDst).toEqual(new Float32Array([9, 9, 9, 9]));

        const secondSeq = publishSharedSnapshot(
            views,
            null,
            new Float32Array([0.5, 0.6, 0.7, 0.8]),
            FLAG_NEURON_GRIDS,
        );

        expect(secondSeq).toBe(2);
        expect(readSharedSnapshot(views, outputDst, neuronDst)).toEqual({
            seq: 2,
            flags: FLAG_NEURON_GRIDS,
        });
        expect(outputDst).toEqual(new Float32Array([0.1, 0.2, 0.3, 0.4]));
        expect(neuronDst).toEqual(new Float32Array([0.5, 0.6, 0.7, 0.8]));
    });

    it('retries torn reads before returning a consistent snapshot', () => {
        const views = allocSharedSnapshotViews(2, 1);
        publishSharedSnapshot(
            views,
            new Float32Array([1, 2, 3, 4]),
            new Float32Array([4, 3, 2, 1]),
            FLAG_OUTPUT_GRID | FLAG_NEURON_GRIDS,
        );

        let startLoads = 0;
        const load = atomicsLoadMock((array, index, actual) => {
            if (array === views.control && index === CTL_SEQ_START) {
                startLoads++;
                if (startLoads === 1) return 99;
            }
            return actual;
        });
        vi.spyOn(Atomics, 'load').mockImplementation(load);

        const outputDst = new Float32Array(4);
        const neuronDst = new Float32Array(4);

        expect(readSharedSnapshot(views, outputDst, neuronDst, 2)).toEqual({
            seq: 1,
            flags: FLAG_OUTPUT_GRID | FLAG_NEURON_GRIDS,
        });
        expect(startLoads).toBe(2);
        expect(outputDst).toEqual(new Float32Array([1, 2, 3, 4]));
        expect(neuronDst).toEqual(new Float32Array([4, 3, 2, 1]));
    });

    it('returns null when every read attempt is torn', () => {
        const views = allocSharedSnapshotViews(2, 1);
        publishSharedSnapshot(views, new Float32Array([1, 2, 3, 4]), null, FLAG_OUTPUT_GRID);

        const load = atomicsLoadMock((array, index, actual) => {
            if (array === views.control && index === CTL_SEQ_START) {
                return actual + 1;
            }
            return actual;
        });
        vi.spyOn(Atomics, 'load').mockImplementation(load);

        expect(readSharedSnapshot(views, new Float32Array(4), null, 2)).toBeNull();
    });
});
