import { describe, expect, it } from 'vitest';
import {
    createMiniBatchScratch,
    fillMiniBatchScratch,
    getTrainingStepsForTick,
    normalizeTrainingSpeed,
} from './trainingLoop.ts';

describe('training loop speed bounds', () => {
    it('maps the selected speed to a fixed number of steps per tick', () => {
        expect(getTrainingStepsForTick(1)).toBe(1);
        expect(getTrainingStepsForTick(5)).toBe(5);
        expect(getTrainingStepsForTick(50)).toBe(50);
    });

    it('clamps malformed speed values before the worker loop uses them', () => {
        expect(normalizeTrainingSpeed(0)).toBe(1);
        expect(normalizeTrainingSpeed(Number.POSITIVE_INFINITY)).toBe(1);
        expect(normalizeTrainingSpeed(5000)).toBe(100);
    });
});

describe('mini-batch scratch buffers', () => {
    it('reuses the same batch arrays across fills', () => {
        const inputs = [[0], [1], [2], [3]];
        const targets = [[0], [1], [0], [1]];
        const indices = [3, 2, 1, 0];
        const scratch = createMiniBatchScratch(3);

        const first = fillMiniBatchScratch(scratch, inputs, targets, indices, 0, 3);
        const firstInputs = first.inputs;
        const firstTargets = first.targets;
        expect(first.inputs).toEqual([[3], [2], [1]]);
        expect(first.targets).toEqual([[1], [0], [1]]);

        const second = fillMiniBatchScratch(scratch, inputs, targets, indices, 3, 4);
        expect(second.inputs).toBe(firstInputs);
        expect(second.targets).toBe(firstTargets);
        expect(second.inputs).toEqual([[0]]);
        expect(second.targets).toEqual([[0]]);
    });
});
