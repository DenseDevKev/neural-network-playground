import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render } from '@testing-library/react';
import { LossChart } from './LossChart.tsx';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';

const fillRect = vi.fn();

function createMockContext() {
    return {
        fillRect,
        fillText: vi.fn(),
        beginPath: vi.fn(),
        moveTo: vi.fn(),
        lineTo: vi.fn(),
        stroke: vi.fn(),
        fill: vi.fn(),
        closePath: vi.fn(),
        setLineDash: vi.fn(),
        createLinearGradient: vi.fn(() => ({ addColorStop: vi.fn() })),
        setTransform: vi.fn(),
        measureText: vi.fn(() => ({ width: 32 })),
        lineWidth: 1,
        lineCap: 'round' as const,
        lineJoin: 'round' as const,
        fillStyle: '',
        strokeStyle: '',
        font: '',
        textAlign: 'left' as const,
        shadowColor: '',
        shadowBlur: 0,
    };
}

describe('LossChart', () => {
    const originalGetContext = HTMLCanvasElement.prototype.getContext;

    beforeEach(() => {
        fillRect.mockClear();
        HTMLCanvasElement.prototype.getContext = vi.fn(
            () => createMockContext() as unknown as CanvasRenderingContext2D,
        ) as unknown as typeof HTMLCanvasElement.prototype.getContext;

        usePlaygroundStore.setState((state) => ({
            data: {
                ...state.data,
                problemType: 'classification',
            },
        }));

        useTrainingStore.getState().resetHistory();
        useTrainingStore.getState().addHistoryPoint({
            step: 0,
            trainLoss: 0.8,
            testLoss: 0.9,
            trainAccuracy: 0.5,
            testAccuracy: 0.45,
        });
        useTrainingStore.getState().addHistoryPoint({
            step: 1,
            trainLoss: 0.7,
            testLoss: 0.8,
            trainAccuracy: 0.55,
            testAccuracy: 0.5,
        });
    });

    it('fully redraws when new history points stream in', () => {
        render(<LossChart />);

        const initialPaints = fillRect.mock.calls.length;
        expect(initialPaints).toBeGreaterThan(0);

        act(() => {
            useTrainingStore.getState().addHistoryPoint({
                step: 2,
                trainLoss: 0.6,
                testLoss: 0.72,
                trainAccuracy: 0.6,
                testAccuracy: 0.54,
            });
        });

        expect(fillRect.mock.calls.length).toBeGreaterThan(initialPaints);
    });

    afterEach(() => {
        HTMLCanvasElement.prototype.getContext = originalGetContext;
    });
});
