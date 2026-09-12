import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { LiveTrainingSignal, PairedEvaluation } from '@nn-playground/shared';
import { LossChart, deriveLossChartViewport } from './LossChart.tsx';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { createScientificTrustFixtures } from '../../test/scientificTrustFixtures.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import { getConceptById } from '../../concepts/conceptCatalog.ts';

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
        clearRect: vi.fn(),
        lineWidth: 1,
        lineCap: 'round' as const,
        lineJoin: 'round' as const,
        fillStyle: '',
        strokeStyle: '',
        font: '',
        textAlign: 'left' as const,
    };
}

function atStep<T extends LiveTrainingSignal | PairedEvaluation>(value: T, step: number): T {
    return {
        ...value,
        model: { ...value.model, revision: step, step, epoch: Math.floor(step / 100) },
        ...('basis' in value ? {
            basis: { ...value.basis, throughStep: step },
        } : {}),
    } as T;
}

describe('LossChart scientific evidence series', () => {
    const originalGetContext = HTMLCanvasElement.prototype.getContext;
    const originalResizeObserver = window.ResizeObserver;
    let live: LiveTrainingSignal;
    let pair: PairedEvaluation;

    beforeEach(async () => {
        const fixture = await createScientificTrustFixtures();
        live = atStep({ ...fixture.liveSignal, dataLoss: 0.6 }, 1240);
        pair = atStep({
            ...fixture.evaluation,
            evaluationId: 31,
            train: {
                ...fixture.evaluation.train,
                values: { ...fixture.evaluation.train.values, dataLoss: 0.7 },
            },
            test: {
                ...fixture.evaluation.test,
                values: { ...fixture.evaluation.test.values, dataLoss: 0.8 },
            },
            objective: { regularizationPenalty: 0.05, trainTotalObjective: 0.75 },
        }, 1230);

        window.ResizeObserver = vi.fn().mockImplementation(() => ({
            observe: vi.fn(),
            unobserve: vi.fn(),
            disconnect: vi.fn(),
        }));
        fillRect.mockClear();
        HTMLCanvasElement.prototype.getContext = vi.fn(
            () => createMockContext() as unknown as CanvasRenderingContext2D,
        ) as unknown as typeof HTMLCanvasElement.prototype.getContext;

        useTrainingStore.getState().resetEvidence();
        useTrainingStore.getState().applyEvidence({
            type: 'evidence',
            protocolVersion: 2,
            liveSignal: live,
            latestEvaluation: pair,
        });
        useLayoutStore.setState({ audienceMode: 'explore' });
    });

    it('derives finite readable bounds at phone, tablet, and desktop widths', () => {
        for (const width of [0, -10, Number.NaN, Infinity]) {
            expect(deriveLossChartViewport(width)).toEqual({ width: 1, height: 220 });
        }
        expect(deriveLossChartViewport(320.2)).toEqual({ width: 320, height: 220 });
        expect(deriveLossChartViewport(735)).toEqual({ width: 735, height: 331 });
        expect(deriveLossChartViewport(1437)).toEqual({ width: 1437, height: 360 });
    });

    it('coalesces resize paints, ignores subpixel churn, and cancels pending work on unmount', () => {
        let notify!: ResizeObserverCallback;
        const disconnect = vi.fn();
        window.ResizeObserver = vi.fn().mockImplementation((callback: ResizeObserverCallback) => {
            notify = callback;
            return { observe: vi.fn(), disconnect };
        });
        let flush!: FrameRequestCallback;
        const schedule = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((callback) => { flush = callback; return 42; });
        const cancel = vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});
        const emitWidth = (width: number) => notify([{ contentRect: { width } } as ResizeObserverEntry], {} as ResizeObserver);
        const view = render(<LossChart />);
        const canvas = screen.getByLabelText('Scientific loss evidence by actual model step');
        expect(canvas).toHaveStyle({ height: '220px' });
        const initialPaints = fillRect.mock.calls.length;
        act(() => { emitWidth(400.1); emitWidth(400.4); });
        expect(schedule).toHaveBeenCalledTimes(1);
        expect(fillRect).toHaveBeenCalledTimes(initialPaints);
        act(() => flush(0));
        expect(canvas).toHaveStyle({ height: '220px' });
        expect(fillRect).toHaveBeenCalledTimes(initialPaints + 1);
        act(() => { emitWidth(400.2); flush(0); });
        expect(fillRect).toHaveBeenCalledTimes(initialPaints + 1);
        act(() => emitWidth(200));
        view.unmount();
        expect(cancel).toHaveBeenCalledWith(42);
        expect(disconnect).toHaveBeenCalledTimes(1);
        schedule.mockRestore();
        cancel.mockRestore();
    });

    it('uses actual model steps and keeps live/evaluation labels distinct', () => {
        render(<LossChart />);

        expect(screen.getByText('Batch trend (EMA)')).toBeInTheDocument();
        expect(screen.getByText('Train data loss (full split)')).toBeInTheDocument();
        expect(screen.getByText('Test data loss (full split)')).toBeInTheDocument();
        expect(screen.getByText('Training objective')).toBeInTheDocument();
        expect(screen.getByText(/Batch trend through step 1,240/i)).toBeInTheDocument();
        expect(screen.getByText(/Full evaluation 31 at step 1,230/i)).toBeInTheDocument();
    });

    it('opens the catalog training-objective definition beside the unchanged legend label', async () => {
        const user = userEvent.setup();
        render(<LossChart />);

        expect(screen.getByText('Training objective')).toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Learn about Training objective' }));

        expect(screen.getByText(getConceptById('training-objective')?.plainDefinition ?? ''))
            .toBeInTheDocument();
        expect(screen.queryByText(getConceptById('training-objective')?.examples?.[0] ?? ''))
            .not.toBeInTheDocument();
    });

    it('does not move paired best-test or gap diagnostics when only live evidence advances', () => {
        render(<LossChart />);
        expect(screen.getByText('Best test 0.8000')).toBeInTheDocument();
        expect(screen.getByText('Gap +0.1000')).toBeInTheDocument();

        act(() => {
            useTrainingStore.getState().applyEvidence({
                type: 'evidence',
                protocolVersion: 2,
                liveSignal: atStep({ ...live, dataLoss: 0.2 }, 1250),
            });
        });

        expect(screen.getByText('Best test 0.8000')).toBeInTheDocument();
        expect(screen.getByText('Gap +0.1000')).toBeInTheDocument();
        expect(screen.getByText(/Batch trend through step 1,250/i)).toBeInTheDocument();
        expect(screen.getByText(/Full evaluation 31 at step 1,230/i)).toBeInTheDocument();
    });

    it('redraws when scientific metric versions advance', () => {
        render(<LossChart />);
        const initialPaints = fillRect.mock.calls.length;

        act(() => {
            useTrainingStore.getState().applyEvidence({
                type: 'evidence',
                protocolVersion: 2,
                liveSignal: atStep({ ...live, dataLoss: 0.5 }, 1250),
            });
        });

        expect(fillRect.mock.calls.length).toBeGreaterThan(initialPaints);
    });

    afterEach(() => {
        HTMLCanvasElement.prototype.getContext = originalGetContext;
        window.ResizeObserver = originalResizeObserver;
    });
});
