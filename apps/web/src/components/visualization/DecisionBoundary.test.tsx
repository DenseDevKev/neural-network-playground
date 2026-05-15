import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import { DecisionBoundary } from './DecisionBoundary.tsx';
import { classifyPointFromGrid } from './DecisionBoundary.tsx';
import {
    resetFrameBuffer,
    updateFrameBuffer,
    getFrameVersion,
    getFrameVersions,
} from '../../worker/frameBuffer.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';

const drawImage = vi.fn();

function createMockContext() {
    return {
        createImageData: (width: number, height: number) => ({
            width,
            height,
            data: new Uint8ClampedArray(width * height * 4),
        }),
        putImageData: vi.fn(),
        drawImage,
        fillRect: vi.fn(),
        beginPath: vi.fn(),
        moveTo: vi.fn(),
        lineTo: vi.fn(),
        arc: vi.fn(),
        fill: vi.fn(),
        stroke: vi.fn(),
        save: vi.fn(),
        restore: vi.fn(),
        clearRect: vi.fn(),
        setTransform: vi.fn(),
        globalCompositeOperation: 'source-over' as GlobalCompositeOperation,
        imageSmoothingEnabled: false,
        imageSmoothingQuality: 'low' as const,
        fillStyle: '',
        strokeStyle: '',
        lineWidth: 1,
        shadowColor: '',
        shadowBlur: 0,
        shadowOffsetX: 0,
        shadowOffsetY: 0,
    };
}

describe('DecisionBoundary', () => {
    const originalGetContext = HTMLCanvasElement.prototype.getContext;
    const originalRequestAnimationFrame = window.requestAnimationFrame;
    const originalCancelAnimationFrame = window.cancelAnimationFrame;
    const originalResizeObserver = window.ResizeObserver;

    beforeEach(() => {
        drawImage.mockClear();
        resetFrameBuffer();
        useTrainingStore.setState({
            snapshot: null,
            frameVersion: 0,
            trainPoints: [],
            testPoints: [],
            multiclassBoundaryVersion: 0,
        });
        usePlaygroundStore.setState((state) => ({
            data: {
                ...state.data,
                problemType: 'classification',
            },
            network: {
                ...state.network,
                outputSize: 1,
                outputActivation: 'sigmoid',
            },
        }));

        HTMLCanvasElement.prototype.getContext = vi.fn(
            () => createMockContext() as unknown as CanvasRenderingContext2D,
        ) as unknown as typeof HTMLCanvasElement.prototype.getContext;
        window.requestAnimationFrame = vi.fn(() => 1);
        window.cancelAnimationFrame = vi.fn();

        class ResizeObserverMock {
            observe() {}
            disconnect() {}
        }

        Object.defineProperty(window, 'ResizeObserver', {
            configurable: true,
            value: ResizeObserverMock,
        });
    });

    afterEach(() => {
        HTMLCanvasElement.prototype.getContext = originalGetContext;
        window.requestAnimationFrame = originalRequestAnimationFrame;
        window.cancelAnimationFrame = originalCancelAnimationFrame;

        Object.defineProperty(window, 'ResizeObserver', {
            configurable: true,
            value: originalResizeObserver,
        });
    });

    it('repaints immediately when a streamed frame arrives', () => {
        updateFrameBuffer({
            outputGrid: new Float32Array([0, 0.25, 0.75, 1]),
            gridSize: 2,
        });
        useTrainingStore.setState({ frameVersion: getFrameVersion() });

        render(
            <DecisionBoundary
                trainPoints={[{ x: -0.5, y: 0.5, label: 0 }]}
                testPoints={[]}
                showTestData={false}
                discretize={false}
            />,
        );

        expect(drawImage).toHaveBeenCalledTimes(1);

        act(() => {
            updateFrameBuffer({
                outputGrid: new Float32Array([1, 0.75, 0.25, 0]),
                gridSize: 2,
            });
            useTrainingStore.setState({ frameVersion: getFrameVersion() });
        });

        expect(drawImage).toHaveBeenCalledTimes(2);
        expect(window.requestAnimationFrame).not.toHaveBeenCalled();
    });

    it('keeps rendering scalar snapshot grids when no streamed frame is cached', () => {
        useTrainingStore.setState({
            snapshot: {
                step: 1,
                epoch: 0,
                weights: [[[0.1, -0.2]]],
                biases: [[0.05]],
                trainLoss: 0.3,
                testLoss: 0.4,
                trainMetrics: { loss: 0.3, accuracy: 0.8 },
                testMetrics: { loss: 0.4, accuracy: 0.75 },
                outputGrid: new Float32Array([0, 0.25, 0.75, 1]),
                gridSize: 2,
                historyPoint: { step: 1, trainLoss: 0.3, testLoss: 0.4 },
            },
        });

        render(
            <DecisionBoundary
                trainPoints={[{ x: -0.5, y: 0.5, label: 0 }]}
                testPoints={[]}
                showTestData={false}
                discretize={false}
            />,
        );

        expect(drawImage).toHaveBeenCalledTimes(1);
        expect(screen.getByText('Negative')).toBeInTheDocument();
        expect(screen.getByText('Positive')).toBeInTheDocument();
    });

    it('renders uncertainty, misclassification, and split overlay badges when requested', () => {
        const { rerender, container } = render(
            <DecisionBoundary
                trainPoints={[{ x: 0, y: 0, label: 0 }]}
                testPoints={[]}
                showTestData={false}
                discretize={false}
                overlayMode="uncertainty"
            />,
        );

        expect(container.querySelector('[data-overlay-mode="uncertainty"]')).not.toBeNull();

        rerender(
            <DecisionBoundary
                trainPoints={[{ x: 0, y: 0, label: 0 }]}
                testPoints={[]}
                showTestData={false}
                discretize={false}
                overlayMode="misclassification"
            />,
        );

        expect(container.querySelector('[data-overlay-mode="misclassification"]')).not.toBeNull();

        rerender(
            <DecisionBoundary
                trainPoints={[{ x: 0, y: 0, label: 0 }]}
                testPoints={[{ x: 0.5, y: 0.5, label: 1 }]}
                showTestData={false}
                discretize={false}
                overlayMode="split"
            />,
        );

        expect(container.querySelector('[data-overlay-mode="split"]')).not.toBeNull();
    });

    it('describes the selected overlay mode for assistive technology', () => {
        const { rerender } = render(
            <DecisionBoundary
                trainPoints={[{ x: 0, y: 0, label: 0 }]}
                testPoints={[]}
                showTestData={false}
                discretize={false}
                overlayMode="uncertainty"
            />,
        );

        const canvas = screen.getByRole('img', {
            name: /decision boundary visualization/i,
        });
        const descriptionId = canvas.getAttribute('aria-describedby');

        expect(descriptionId).toBeTruthy();
        expect(document.getElementById(descriptionId ?? '')).toHaveTextContent(/least sure/i);

        rerender(
            <DecisionBoundary
                trainPoints={[{ x: 0, y: 0, label: 0 }]}
                testPoints={[{ x: 0.5, y: 0.5, label: 1 }]}
                showTestData={true}
                discretize={false}
                overlayMode="misclassification"
            />,
        );

        expect(document.getElementById(descriptionId ?? '')).toHaveTextContent(/visible test points/i);
    });

    it('does not render binary legend copy for multiclass classification labels', () => {
        render(
            <DecisionBoundary
                trainPoints={[
                    { x: -0.5, y: 0.5, label: 0 },
                    { x: 0, y: 0, label: 1 },
                    { x: 0.5, y: -0.5, label: 2 },
                ]}
                testPoints={[]}
                showTestData={false}
                discretize={false}
            />,
        );

        expect(screen.getByText('Binary decision boundary unavailable')).toBeInTheDocument();
        expect(screen.getByText(/supports two-class outputs/i)).toBeInTheDocument();
        expect(screen.queryByText('Negative')).not.toBeInTheDocument();
        expect(screen.queryByText('Positive')).not.toBeInTheDocument();
    });

    it('guards binary boundary copy when hidden test data contains multiclass labels', () => {
        render(
            <DecisionBoundary
                trainPoints={[
                    { x: -0.5, y: 0.5, label: 0 },
                    { x: 0, y: 0, label: 1 },
                ]}
                testPoints={[{ x: 0.5, y: -0.5, label: 2 }]}
                showTestData={false}
                discretize={false}
            />,
        );

        expect(screen.getByText('Binary decision boundary unavailable')).toBeInTheDocument();
        expect(screen.queryByText('Negative')).not.toBeInTheDocument();
        expect(screen.queryByText('Positive')).not.toBeInTheDocument();
    });

    it('guards binary boundary copy for multiclass output configs even with binary labels', () => {
        usePlaygroundStore.setState((state) => ({
            network: {
                ...state.network,
                outputSize: 3,
                outputActivation: 'softmax' as const,
            },
        }));

        render(
            <DecisionBoundary
                trainPoints={[
                    { x: -0.5, y: 0.5, label: 0 },
                    { x: 0, y: 0, label: 1 },
                ]}
                testPoints={[]}
                showTestData={false}
                discretize={false}
            />,
        );

        expect(screen.getByText('Binary decision boundary unavailable')).toBeInTheDocument();
        expect(screen.queryByText('Negative')).not.toBeInTheDocument();
        expect(screen.queryByText('Positive')).not.toBeInTheDocument();
    });

    it('renders bounded multiclass boundary data with a text confidence summary', () => {
        usePlaygroundStore.setState((state) => ({
            network: {
                ...state.network,
                outputSize: 3,
                outputActivation: 'softmax' as const,
            },
        }));
        updateFrameBuffer({
            gridSize: 2,
            multiclassClassGrid: new Uint8Array([0, 1, 2, 2]),
            multiclassConfidenceGrid: new Float32Array([0.9, 0.62, 0.74, 0.58]),
            multiclassBoundaryLayout: {
                gridSize: 2,
                classCount: 3,
                classLabels: [0, 1, 2],
            },
        });
        useTrainingStore.setState(getFrameVersions());

        render(
            <DecisionBoundary
                trainPoints={[
                    { x: -0.5, y: 0.5, label: 0 },
                    { x: 0, y: 0, label: 1 },
                    { x: 0.5, y: -0.5, label: 2 },
                ]}
                testPoints={[]}
                showTestData={false}
                discretize={false}
            />,
        );

        const canvas = screen.getByRole('img', {
            name: /multiclass decision boundary/i,
        });
        const descriptionId = canvas.getAttribute('aria-describedby');

        expect(screen.queryByText('Binary decision boundary unavailable')).not.toBeInTheDocument();
        expect(screen.getByText('Class 0')).toBeInTheDocument();
        expect(screen.getByText('Class 1')).toBeInTheDocument();
        expect(screen.getByText('Class 2')).toBeInTheDocument();
        expect(screen.getByText(/Dominant class: Class 2/i)).toBeInTheDocument();
        expect(screen.getByText(/Average confidence: 71%/i)).toBeInTheDocument();
        expect(document.getElementById(descriptionId ?? '')).toHaveTextContent(/25% of cells are below 60% confidence/i);
        expect(drawImage).toHaveBeenCalledTimes(1);
    });

    it('repaints when the multiclass boundary frame version changes', () => {
        usePlaygroundStore.setState((state) => ({
            network: {
                ...state.network,
                outputSize: 3,
                outputActivation: 'softmax' as any,
            },
        }));
        updateFrameBuffer({
            gridSize: 2,
            multiclassClassGrid: new Uint8Array([0, 1, 2, 2]),
            multiclassConfidenceGrid: new Float32Array([0.9, 0.62, 0.74, 0.58]),
            multiclassBoundaryLayout: {
                gridSize: 2,
                classCount: 3,
                classLabels: [0, 1, 2],
            },
        });
        useTrainingStore.setState(getFrameVersions());

        render(
            <DecisionBoundary
                trainPoints={[
                    { x: -0.5, y: 0.5, label: 0 },
                    { x: 0, y: 0, label: 1 },
                    { x: 0.5, y: -0.5, label: 2 },
                ]}
                testPoints={[]}
                showTestData={false}
                discretize={false}
            />,
        );
        expect(drawImage).toHaveBeenCalledTimes(1);

        act(() => {
            updateFrameBuffer({
                gridSize: 2,
                multiclassClassGrid: new Uint8Array([2, 2, 1, 0]),
                multiclassConfidenceGrid: new Float32Array([0.82, 0.76, 0.69, 0.61]),
                multiclassBoundaryLayout: {
                    gridSize: 2,
                    classCount: 3,
                    classLabels: [0, 1, 2],
                },
            });
            useTrainingStore.setState(getFrameVersions());
        });

        expect(drawImage).toHaveBeenCalledTimes(2);
    });

    it('classifies a point from the nearest decision grid cell', () => {
        const grid = new Float32Array([
            0.1, 0.8,
            0.2, 0.9,
        ]);

        expect(classifyPointFromGrid({ x: -1, y: 1, label: 0 }, grid, 2)).toBe(0);
        expect(classifyPointFromGrid({ x: 1, y: -1, label: 0 }, grid, 2)).toBe(1);
    });
});
