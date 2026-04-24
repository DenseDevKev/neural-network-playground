import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { LossChart } from './LossChart.tsx';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';

const fillRect = vi.fn();
const DEFAULT_RECT = { width: 400, height: 140 };
let getBoundingClientRectSpy: ReturnType<typeof vi.spyOn>;

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
        getBoundingClientRectSpy = vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect');
        mockRenderedChartSize(DEFAULT_RECT.width, DEFAULT_RECT.height);
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

    it('uses a narrow rendered width for hover index, guide, and tooltip placement', () => {
        seedFiveHistoryPoints();
        mockRenderedChartSize(250, 140);

        render(<LossChart />);

        const canvas = screen.getByLabelText('Loss over training steps');
        fireEvent.mouseMove(canvas, { clientX: 187.5, clientY: 70 });

        const tooltip = getStepTooltip(3);
        const guide = getGuideLine(canvas);

        expect(guide.style.left).toBe('187.5px');
        expect(tooltip.style.right).toBe('70.5px');
        expect(tooltip.style.left).toBe('');
    });

    it('uses a wide rendered width for hover index, guide, and tooltip placement', () => {
        seedFiveHistoryPoints();
        mockRenderedChartSize(640, 140);

        render(<LossChart />);

        const canvas = screen.getByLabelText('Loss over training steps');
        fireEvent.mouseMove(canvas, { clientX: 192, clientY: 70 });

        const tooltip = getStepTooltip(1);
        const guide = getGuideLine(canvas);

        expect(guide.style.left).toBe('192px');
        expect(tooltip.style.left).toBe('200px');
        expect(tooltip.style.right).toBe('');
    });

    afterEach(() => {
        HTMLCanvasElement.prototype.getContext = originalGetContext;
        getBoundingClientRectSpy.mockRestore();
    });
});

function mockRenderedChartSize(width: number, height: number) {
    getBoundingClientRectSpy.mockImplementation(() => ({
        x: 0,
        y: 0,
        left: 0,
        top: 0,
        right: width,
        bottom: height,
        width,
        height,
        toJSON: () => ({}),
    } as DOMRect));
}

function seedFiveHistoryPoints() {
    useTrainingStore.getState().resetHistory();
    for (let i = 0; i < 5; i++) {
        useTrainingStore.getState().addHistoryPoint({
            step: i,
            trainLoss: 0.9 - i * 0.1,
            testLoss: 1 - i * 0.1,
            trainAccuracy: 0.45 + i * 0.05,
            testAccuracy: 0.4 + i * 0.05,
        });
    }
}

function getGuideLine(canvas: HTMLElement): HTMLElement {
    const wrapper = canvas.parentElement as HTMLElement;
    const guide = Array.from(wrapper.children).find((child) => {
        const element = child as HTMLElement;
        return element.style.width === '1px' && element.style.top === '20px';
    });

    expect(guide).toBeDefined();
    return guide as HTMLElement;
}

function getStepTooltip(step: number): HTMLElement {
    const label = screen.getByText((_, element) => element?.textContent === `Step ${step}`);
    return label.parentElement as HTMLElement;
}
