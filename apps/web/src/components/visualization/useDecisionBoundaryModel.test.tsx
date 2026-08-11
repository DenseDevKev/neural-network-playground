import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { PREPARED_PRESETS } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import {
    getFrameVersions,
    resetFrameBuffer,
    updateFrameBuffer,
} from '../../worker/frameBuffer.ts';
import * as frameBufferModule from '../../worker/frameBuffer.ts';
import { useDecisionBoundaryModel } from './useDecisionBoundaryModel.ts';

const BINARY_PREPARED = PREPARED_PRESETS.find((entry) => entry.id === 'single-neuron')!.prepared;
const MULTICLASS_PREPARED = PREPARED_PRESETS.find(
    (entry) => entry.id === 'three-class-clusters',
)!.prepared;

const PROPS = {
    trainPoints: [{ x: -0.5, y: 0.5, label: 0 }],
    testPoints: [],
    showTestData: false,
    discretize: false,
};

describe('useDecisionBoundaryModel', () => {
    beforeEach(() => {
        resetFrameBuffer();
        useTrainingStore.setState({
            outputGridVersion: 0,
            multiclassBoundaryVersion: 0,
            layerStatsVersion: 0,
        });
        usePlaygroundStore.setState({
            access: { status: 'ready', prepared: BINARY_PREPARED },
        });
    });

    it('refreshes exactly when the output grid version changes', () => {
        updateFrameBuffer({ outputGrid: new Float32Array([0, 0.25, 0.75, 1]), gridSize: 2 });
        useTrainingStore.setState(getFrameVersions());
        const { result } = renderHook(() => useDecisionBoundaryModel(PROPS));
        const initial = result.current;

        act(() => {
            updateFrameBuffer({ outputGrid: new Float32Array([1, 0.75, 0.25, 0]), gridSize: 2 });
            useTrainingStore.setState(getFrameVersions());
        });

        expect(result.current).not.toBe(initial);
        expect(result.current.kind).toBe('scalar');
        if (result.current.kind !== 'scalar') return;
        expect(result.current.grid).toEqual(new Float32Array([1, 0.75, 0.25, 0]));
    });

    it('refreshes exactly when the multiclass boundary version changes', () => {
        usePlaygroundStore.setState({
            access: { status: 'ready', prepared: MULTICLASS_PREPARED },
        });
        updateFrameBuffer({
            gridSize: 2,
            multiclassClassGrid: new Uint8Array([0, 1, 2, 2]),
            multiclassConfidenceGrid: new Float32Array([0.9, 0.62, 0.74, 0.58]),
            multiclassBoundaryLayout: { gridSize: 2, classCount: 3, classLabels: [0, 1, 2] },
        });
        useTrainingStore.setState(getFrameVersions());
        const { result } = renderHook(() => useDecisionBoundaryModel(PROPS));
        const initial = result.current;

        act(() => {
            updateFrameBuffer({
                gridSize: 2,
                multiclassClassGrid: new Uint8Array([2, 2, 1, 0]),
                multiclassConfidenceGrid: new Float32Array([0.82, 0.76, 0.69, 0.61]),
                multiclassBoundaryLayout: { gridSize: 2, classCount: 3, classLabels: [0, 1, 2] },
            });
            useTrainingStore.setState(getFrameVersions());
        });

        expect(result.current).not.toBe(initial);
        expect(result.current.kind).toBe('multiclass');
    });

    it('does not refresh for unrelated layer-stat changes', () => {
        const { result } = renderHook(() => useDecisionBoundaryModel(PROPS));
        const initial = result.current;

        act(() => {
            useTrainingStore.setState({ layerStatsVersion: 1 });
        });

        expect(result.current).toBe(initial);
    });

    it('reads one frame snapshot for each derived model', () => {
        const getFrameBuffer = vi.spyOn(frameBufferModule, 'getFrameBuffer');

        const { result } = renderHook(() => useDecisionBoundaryModel(PROPS));

        expect(result.current.kind).toBe('scalar');
        expect(getFrameBuffer).toHaveBeenCalledTimes(1);
    });
});
