import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { PREPARED_PRESETS } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import * as frames from '../../worker/frameBuffer.ts';
import { updateCompiledForTest } from '../../test/playgroundStoreTestUtils.ts';
import { useNetworkSelectionController } from './useNetworkSelectionController.ts';

const prepared = PREPARED_PRESETS.find((p) => p.id === 'xor-hidden')!.prepared;
let snapshot: frames.FrameBuffer;

describe('useNetworkSelectionController', () => {
    beforeEach(() => {
        usePlaygroundStore.setState({ access: { status: 'ready', prepared } });
        useTrainingStore.setState({ evidenceGenerationId: 7, paramsVersion: 0, neuronGridsVersion: 0, frameVersion: 0 });
        const layers = [prepared.compiled.network.inputSize, ...prepared.compiled.network.hiddenLayers, prepared.compiled.network.outputSize];
        const count = layers.slice(1).reduce((sum, n) => sum + n, 0);
        const model = { generationId: 7, revision: 10, step: 10, epoch: 1 };
        snapshot = { ...frames.getFrameBuffer(), weightLayout: { layerSizes: layers },
            weights: new Float32Array(layers.slice(1).reduce((sum, n, i) => sum + n * layers[i], 0)).fill(1), biases: new Float32Array(count).fill(2),
            parameterProvenance: { model, recipeFingerprint: prepared.identities.recipeFingerprint },
            neuronGrids: new Float32Array(count * 4).fill(3), neuronGridLayout: { count, gridSize: 2 },
            neuronGridsProvenance: { model: { ...model, step: 8, revision: 8 }, dataset: { datasetKey: 'd', generatorVersion: 1, trainCount: 50, testCount: 50 }, objectiveKey: 'o', basis: { kind: 'prediction-grid', pointCount: 4, domain: [-1, 1, -1, 1] } } };
        vi.spyOn(frames, 'getFrameBuffer').mockImplementation(() => snapshot);
    });
    afterEach(() => vi.restoreAllMocks());
    it('preserves selection through frame/status/profile changes and reads only relevant versions', () => {
        const { result } = renderHook(() => useNetworkSelectionController());
        const before = usePlaygroundStore.getState().access;
        act(() => result.current.commands.selectNode({ layerIdx: 1, nodeIdx: 0 }));
        const model = result.current.model;
        act(() => { useTrainingStore.setState({ status: 'paused', frameVersion: 3 }); useLayoutStore.setState({ audienceMode: 'lab',  }); });
        expect(result.current.model).toBe(model);
        expect(usePlaygroundStore.getState().access).toBe(before);
        act(() => { snapshot = { ...snapshot, biases: new Float32Array(snapshot.biases!.length).fill(4) }; useTrainingStore.setState({ paramsVersion: 1 }); });
        expect(result.current.model).toMatchObject({ bias: 4, parameterStep: 10, activationStep: 8 });
        expect(result.current.selectedNode).toEqual({ layerIdx: 1, nodeIdx: 0 });
    });
    it('clears on a new generation and never blends previous-generation arrays', () => {
        const { result } = renderHook(() => useNetworkSelectionController());
        act(() => result.current.commands.selectNode({ layerIdx: 1, nodeIdx: 0 }));
        act(() => useTrainingStore.setState({ evidenceGenerationId: 8 }));
        expect(result.current.selectedNode).toBeNull();
        act(() => result.current.commands.selectNode({ layerIdx: 1, nodeIdx: 0 }));
        expect(result.current.model).toMatchObject({ kind: 'selected', bias: null, grid: null, incoming: [], outgoing: [] });
    });
    it('clears on architecture changes and never resurrects the old selection', () => {
        const { result } = renderHook(() => useNetworkSelectionController());
        act(() => result.current.commands.selectNode({ layerIdx: 1, nodeIdx: 0 }));
        act(() => updateCompiledForTest((c) => ({ ...c, network: { ...c.network, hiddenLayers: [3] } })));
        expect(result.current.selectedNode).toBeNull();
        act(() => usePlaygroundStore.setState({ access: { status: 'ready', prepared } }));
        expect(result.current.selectedNode).toBeNull();
    });
    it('preserves selection on a same-generation revision and exposes distinct artifact steps', () => {
        const { result } = renderHook(() => useNetworkSelectionController());
        act(() => result.current.commands.selectNode({ layerIdx: 1, nodeIdx: 0 }));
        act(() => { snapshot = { ...snapshot, parameterProvenance: { ...snapshot.parameterProvenance!, model: { generationId: 7, revision: 11, step: 1, epoch: 0 } } }; useTrainingStore.setState({ paramsVersion: 1 }); });
        expect(result.current.selectedNode).toEqual({ layerIdx: 1, nodeIdx: 0 });
        expect(result.current.model).toMatchObject({ parameterStep: 1 });
    });
    it('rejects invalid selection and clears explicitly without a recipe transaction', () => {
        const { result } = renderHook(() => useNetworkSelectionController());
        act(() => result.current.commands.selectNode({ layerIdx: 10, nodeIdx: 0 }));
        expect(result.current.model).toEqual({ kind: 'empty' });
        act(() => result.current.commands.selectNode({ layerIdx: 1, nodeIdx: 0 }));
        act(() => result.current.commands.clearSelection());
        expect(result.current.model).toEqual({ kind: 'empty' });
    });
});
