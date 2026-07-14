import { describe, expect, it } from 'vitest';
import {
    APP_PERFORMANCE_MEASURE_PREFIX,
    SLOW_INTERACTION_THRESHOLD_MS,
    createAppMeasureName,
    shouldReportSlowInteraction,
} from './interactionMeasures.ts';

describe('interaction performance measures', () => {
    it('prefixes app-owned measure names', () => {
        expect(createAppMeasureName('tooltip-show:example')).toBe(
            'nn-playground:tooltip-show:example',
        );
        expect(APP_PERFORMANCE_MEASURE_PREFIX).toBe('nn-playground:');
    });

    it('reports app-owned measures above the slow interaction threshold', () => {
        expect(shouldReportSlowInteraction({
            name: createAppMeasureName('tooltip-show:example'),
            duration: 16.01,
        })).toBe(true);
    });

    it('does not report app-owned measures exactly at the threshold', () => {
        expect(SLOW_INTERACTION_THRESHOLD_MS).toBe(16);
        expect(shouldReportSlowInteraction({
            name: createAppMeasureName('panel-toggle:Data'),
            duration: SLOW_INTERACTION_THRESHOLD_MS,
        })).toBe(false);
    });

    it('ignores browser tooling Promise Resolved measures', () => {
        expect(shouldReportSlowInteraction({
            name: 'Promise Resolved',
            duration: 9250,
        })).toBe(false);
    });

    it('ignores browser tooling Cascading Update measures', () => {
        expect(shouldReportSlowInteraction({
            name: 'Cascading Update',
            duration: 17,
        })).toBe(false);
    });
});
