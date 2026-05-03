import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render } from '@testing-library/react';
import { DecisionBoundary } from './DecisionBoundary.tsx';
import { classifyPointFromGrid } from './DecisionBoundary.tsx';
import { resetFrameBuffer, updateFrameBuffer, getFrameVersion } from '../../worker/frameBuffer.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';

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
        });

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

    it('classifies a point from the nearest decision grid cell', () => {
        const grid = new Float32Array([
            0.1, 0.8,
            0.2, 0.9,
        ]);

        expect(classifyPointFromGrid({ x: -1, y: 1, label: 0 }, grid, 2)).toBe(0);
        expect(classifyPointFromGrid({ x: 1, y: -1, label: 0 }, grid, 2)).toBe(1);
    });
});
