import { afterEach, describe, expect, it, vi } from 'vitest';
import {
    CTL_FLAGS,
    CTL_SEQ_END,
    CTL_SEQ_START,
    FLAG_NEURON_GRIDS,
    FLAG_OUTPUT_GRID,
    allocSharedSnapshotViews,
    attachSharedSnapshotViews,
    publishSharedSnapshot,
    readSharedSnapshot,
} from './sharedSnapshot.ts';

describe('sharedSnapshot', () => {
    afterEach(() => {
        vi.restoreAllMocks();
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

        const realLoad = Atomics.load;
        let startLoads = 0;
        vi.spyOn(Atomics, 'load').mockImplementation((array, index) => {
            if (array === views.control && index === CTL_SEQ_START) {
                startLoads++;
                if (startLoads === 1) return 99;
            }
            return realLoad(array, index);
        });

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

        const realLoad = Atomics.load;
        vi.spyOn(Atomics, 'load').mockImplementation((array, index) => {
            if (array === views.control && index === CTL_SEQ_START) {
                return realLoad(array, index) + 1;
            }
            return realLoad(array, index);
        });

        expect(readSharedSnapshot(views, new Float32Array(4), null, 2)).toBeNull();
    });
});
