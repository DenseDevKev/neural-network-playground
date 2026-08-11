import { describe, expect, it } from 'vitest';
import { reconcileComparisonSelection } from './runComparisonSelection.ts';

const records = [
    { id: 'a' },
    { id: 'b' },
    { id: 'c' },
] as const;

describe('reconcileComparisonSelection', () => {
    it('keeps an empty untouched selection empty', () => {
        expect(reconcileComparisonSelection([], records, null)).toEqual([]);
    });

    it('appends existing records in stable user order until two are selected', () => {
        expect(reconcileComparisonSelection([], records, 'a')).toEqual(['a']);
        expect(reconcileComparisonSelection(['a'], records, 'b')).toEqual(['a', 'b']);
    });

    it('replaces the oldest choice deterministically when a third record is selected', () => {
        expect(reconcileComparisonSelection(['a', 'b'], records, 'c')).toEqual(['b', 'c']);
    });

    it('toggles a selected record off without reordering the remaining choice', () => {
        expect(reconcileComparisonSelection(['a', 'b'], records, 'a')).toEqual(['b']);
        expect(reconcileComparisonSelection(['a', 'b'], records, 'b')).toEqual(['a']);
    });

    it('prunes deleted and unknown IDs without filling their places', () => {
        expect(reconcileComparisonSelection(['a', 'missing'], records, null)).toEqual(['a']);
        expect(reconcileComparisonSelection(['missing'], records, 'also-missing')).toEqual([]);
    });

    it('deduplicates before retaining the newest two choices from malformed input', () => {
        expect(reconcileComparisonSelection(
            ['a', 'a', 'missing', 'b', 'c', 'b'],
            records,
            null,
        )).toEqual(['b', 'c']);
    });

    it('preserves user order when the records are reordered', () => {
        expect(reconcileComparisonSelection(
            ['c', 'a'],
            [records[1], records[0], records[2]],
            null,
        )).toEqual(['c', 'a']);
    });
});
