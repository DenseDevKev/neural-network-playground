import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import type { LiveTrainingSignal, PairedEvaluation } from '@nn-playground/shared';
import { LossChart } from './LossChart.tsx';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { createScientificTrustFixtures } from '../../test/scientificTrustFixtures.ts';

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
