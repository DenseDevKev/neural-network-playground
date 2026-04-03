import { describe, it, expect } from 'vitest';
import { valueToColor, COLOR_BLUE, COLOR_DARK, COLOR_ORANGE } from '../colorScale';

describe('colorScale', () => {
    describe('valueToColor', () => {
        it('should map 0 to COLOR_BLUE', () => {
            expect(valueToColor(0)).toEqual(COLOR_BLUE);
        });

        it('should map 0.5 to COLOR_DARK', () => {
            expect(valueToColor(0.5)).toEqual(COLOR_DARK);
        });

        it('should map 1 to COLOR_ORANGE', () => {
            expect(valueToColor(1)).toEqual(COLOR_ORANGE);
        });

        it('should clamp values < 0 to 0', () => {
            expect(valueToColor(-0.5)).toEqual(COLOR_BLUE);
            expect(valueToColor(-100)).toEqual(COLOR_BLUE);
        });

        it('should clamp values > 1 to 1', () => {
            expect(valueToColor(1.5)).toEqual(COLOR_ORANGE);
            expect(valueToColor(100)).toEqual(COLOR_ORANGE);
        });

        it('should interpolate values between 0 and 0.5 (blue to dark)', () => {
            const color = valueToColor(0.25);
            // It should be exactly halfway between BLUE and DARK
            const expected = [
                Math.round(COLOR_BLUE[0] * 0.5 + COLOR_DARK[0] * 0.5),
                Math.round(COLOR_BLUE[1] * 0.5 + COLOR_DARK[1] * 0.5),
                Math.round(COLOR_BLUE[2] * 0.5 + COLOR_DARK[2] * 0.5),
            ];
            expect(color).toEqual(expected);
        });

        it('should interpolate values between 0.5 and 1 (dark to orange)', () => {
            const color = valueToColor(0.75);
            // It should be exactly halfway between DARK and ORANGE
            const expected = [
                Math.round(COLOR_DARK[0] * 0.5 + COLOR_ORANGE[0] * 0.5),
                Math.round(COLOR_DARK[1] * 0.5 + COLOR_ORANGE[1] * 0.5),
                Math.round(COLOR_DARK[2] * 0.5 + COLOR_ORANGE[2] * 0.5),
            ];
            expect(color).toEqual(expected);
        });

        describe('with discretize = true', () => {
            it('should map values < 0.5 to COLOR_BLUE', () => {
                expect(valueToColor(0, true)).toEqual(COLOR_BLUE);
                expect(valueToColor(0.25, true)).toEqual(COLOR_BLUE);
                expect(valueToColor(0.499, true)).toEqual(COLOR_BLUE);
            });

            it('should map values >= 0.5 to COLOR_ORANGE', () => {
                expect(valueToColor(0.5, true)).toEqual(COLOR_ORANGE);
                expect(valueToColor(0.75, true)).toEqual(COLOR_ORANGE);
                expect(valueToColor(1, true)).toEqual(COLOR_ORANGE);
            });

            it('should clamp and discretize values < 0 to COLOR_BLUE', () => {
                expect(valueToColor(-1, true)).toEqual(COLOR_BLUE);
            });

            it('should clamp and discretize values > 1 to COLOR_ORANGE', () => {
                expect(valueToColor(2, true)).toEqual(COLOR_ORANGE);
            });
        });
    });
});
