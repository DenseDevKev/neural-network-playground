import { describe, expect, it } from 'vitest';
import { getTrainingStepsForTick, normalizeTrainingSpeed } from './trainingLoop.ts';

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
