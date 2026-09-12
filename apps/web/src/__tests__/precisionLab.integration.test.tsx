import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { PREPARED_PRESETS, encodeExperimentUrl } from '@nn-playground/shared';
import App from '../App.tsx';
import { useLayoutStore } from '../store/useLayoutStore.ts';
import { useTrainingStore } from '../store/useTrainingStore.ts';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import { useExperimentMemoryStore } from '../store/experimentMemoryStore.ts';
import * as frames from '../worker/frameBuffer.ts';
import type { NetworkSelectionController } from '../components/visualization/useNetworkSelectionController.ts';

const mountedTraining = vi.hoisted(() => vi.fn());
vi.mock('../hooks/useTraining.ts', async () => {
    const { useEffect } = await import('react');
    return { useTraining: () => {
        useEffect(() => { mountedTraining(); }, []);
        return { play: vi.fn(), pause: vi.fn(), step: vi.fn(), reset: vi.fn(), restoreCheckpoint: vi.fn() };
    } };
});
// Only the paint boundary is substituted. App, controllers, stores, shell and deck are real.
vi.mock('../components/visualization/NetworkGraph.tsx', () => ({ NetworkGraph: ({ selectionController }: { selectionController?: NetworkSelectionController }) =>
    <button onClick={() => selectionController?.commands.selectNode({ layerIdx: 1, nodeIdx: 0 })}>Select graph neuron</button> }));
vi.mock('../components/visualization/DecisionBoundaryCanvas.tsx', () => ({ DecisionBoundaryCanvas: () => <canvas data-decision-boundary-canvas aria-label="Live boundary paint" /> }));
const prepared = PREPARED_PRESETS.find((preset) => preset.id === 'xor-hidden')!.prepared;

describe('Signal Atelier production composition', () => {
    beforeEach(() => {
        localStorage.clear(); mountedTraining.mockClear();
        window.history.replaceState(null, '', encodeExperimentUrl(prepared.document));
        usePlaygroundStore.setState({ access: { status: 'ready', prepared }, preparation: { status: 'ready', requestId: 0, issues: [] } });
        useLayoutStore.setState({ destination:'playground',workspaceTab:'network',setupTab:'dataset',resultsTab:'boundary',inspectTab:'trace',activeLessonId:null,   audienceMode: 'explore',     codeExportTab: 'numpy', lessonCueDismissed: true });
        useTrainingStore.getState().resetEvidence();
        useTrainingStore.setState({ status: 'paused', evidenceGenerationId: 7, paramsVersion: 1, neuronGridsVersion: 1, workerError: null, pendingConfigSource: null, trainedRecipe: prepared.document.recipe, trainedRecipeFingerprint: prepared.identities.recipeFingerprint });
        useExperimentMemoryStore.setState({ hydrationStatus: 'ready', records: Object.freeze([]), rejectedRecords: Object.freeze([]), incompatibleEnvelope: null, pendingSave: null });
        const layers = [prepared.compiled.network.inputSize, ...prepared.compiled.network.hiddenLayers, prepared.compiled.network.outputSize];
        const count = layers.slice(1).reduce((sum, n) => sum + n, 0);
        vi.spyOn(frames, 'getFrameBuffer').mockReturnValue({ ...frames.getFrameBuffer(),
            weightLayout: { layerSizes: layers },
            weights: new Float32Array(layers.slice(1).reduce((sum, n, i) => sum + n * layers[i], 0)).fill(1),
            biases: new Float32Array(count).fill(2),
            parameterProvenance: { model: { generationId: 7, revision: 10, step: 10, epoch: 1 }, recipeFingerprint: prepared.identities.recipeFingerprint },
        });
    });
    afterEach(() => vi.restoreAllMocks());

    it('mounts one live boundary only in views that need it while keeping the model owner alive', async () => {
        const user = userEvent.setup();
        const { container } = render(<App />);
        expect(container.querySelectorAll('[data-decision-boundary-canvas]')).toHaveLength(1);
        await user.click(screen.getByRole('tab', {name:'Setup'}));
        expect(container.querySelectorAll('[data-decision-boundary-canvas]')).toHaveLength(0);
        expect(usePlaygroundStore.getState().demand.needDecisionBoundary).toBe(false);
        await user.click(screen.getByRole('tab', {name:'Results'}));
        expect(container.querySelectorAll('[data-decision-boundary-canvas]')).toHaveLength(1);
        expect(usePlaygroundStore.getState().demand.needDecisionBoundary).toBe(true);
        await user.click(screen.getByRole('tab', {name:'Learning progress'}));
        await waitFor(() => expect(container.querySelectorAll('[data-decision-boundary-canvas]')).toHaveLength(0));
        expect(usePlaygroundStore.getState().demand.needDecisionBoundary).toBe(false);
        expect(mountedTraining).toHaveBeenCalledTimes(1);
    });

    it('keeps graph selection and scientific state across navigation and guidance changes', async () => {
        const user = userEvent.setup(); render(<App />);
        const url = window.location.href;
        const access = usePlaygroundStore.getState().access;
        const runtime = useTrainingStore.getState();
        await user.click(screen.getByRole('button', {name:'Select graph neuron'}));
        expect(screen.getByRole('region', {name:'Selected neuron details'})).toHaveTextContent('Weights at step 10');
        for (const mode of ['beginner','explore','lab'] as const) {
            act(() => { useLayoutStore.getState().setAudienceMode(mode); useLayoutStore.getState().navigate('playground','setup'); });
            expect(screen.queryByRole('region', {name:'Selected neuron details'})).not.toBeInTheDocument();
            act(() => useLayoutStore.getState().navigate('playground','network'));
            expect(screen.getByRole('region', {name:'Selected neuron details'})).toHaveTextContent('2.000');
            expect(useTrainingStore.getState()).toBe(runtime); expect(usePlaygroundStore.getState().access).toBe(access);
            expect(window.location.href).toBe(url); expect(useLayoutStore.getState().codeExportTab).toBe('numpy');
        }
        await user.click(screen.getByRole('button',{name:'Clear selection'}));
        expect(screen.queryByRole('region',{name:'Selected neuron details'})).not.toBeInTheDocument();
        expect(mountedTraining).toHaveBeenCalledTimes(1);
    });

    it('opens dataset setup and expanded prediction without changing the recipe', async () => {
        const user = userEvent.setup(); render(<App />);
        const access = usePlaygroundStore.getState().access;
        await user.click(screen.getByRole('button', {name:'Edit dataset'}));
        expect(screen.getByRole('heading', {name:'Choose a pattern'})).toBeInTheDocument();
        await user.click(screen.getByRole('tab', {name:'Network'}));
        await user.click(screen.getByRole('button', {name:'Expand prediction results'}));
        expect(screen.getByRole('tab',{name:'Prediction'})).toHaveAttribute('aria-selected','true');
        expect(screen.getByRole('checkbox', {name:'Show test data'})).toBeInTheDocument();
        expect(usePlaygroundStore.getState().access).toBe(access);
    });
});
