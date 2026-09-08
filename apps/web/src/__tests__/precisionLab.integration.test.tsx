import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, within } from '@testing-library/react';
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

describe('Precision Lab production composition', () => {
    beforeEach(() => {
        localStorage.clear(); mountedTraining.mockClear();
        window.history.replaceState(null, '', encodeExperimentUrl(prepared.document));
        usePlaygroundStore.setState({ access: { status: 'ready', prepared }, preparation: { status: 'ready', requestId: 0, issues: [] } });
        useLayoutStore.setState({ view: 'build', phase: 'build', audienceMode: 'explore', advancedToolsOpen: false, buildContextOpen: false, activeRecipeSection: 'data', activeEvidenceView: 'boundary', codeExportTab: 'numpy', lessonCueDismissed: true });
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

    it('mounts a single live boundary in Build and every Run evidence view without replacing it', async () => {
        const { container } = render(<App />);
        expect(container.querySelector('[data-precision-workspace]')).not.toBeNull();
        const canvas = container.querySelector('[data-decision-boundary-canvas]');
        expect(canvas).not.toBeNull();
        for (const view of ['build', 'run'] as const) {
            for (const evidence of ['boundary', 'loss', 'confusion', 'inspection', 'code'] as const) {
                act(() => useLayoutStore.setState({ view, advancedToolsOpen: true, activeEvidenceView: evidence }));
                expect(container.querySelectorAll('[data-decision-boundary-canvas]')).toHaveLength(1);
                expect(container.querySelector('[data-decision-boundary-canvas]')).toBe(canvas);
                expect(screen.getByRole('tabpanel').querySelector('[data-decision-boundary-canvas]')).toBeNull();
                expect(usePlaygroundStore.getState().demand.needDecisionBoundary).toBe(true);
            }
        }
        expect(mountedTraining).toHaveBeenCalledTimes(1);
    });

    it('shares the graph selection with the deck and preserves it across pure layout changes', async () => {
        const user = userEvent.setup();
        render(<App />);
        const url = window.location.href;
        const access = usePlaygroundStore.getState().access;
        const runtime = useTrainingStore.getState();
        await user.click(screen.getByRole('button', { name: 'Select graph neuron' }));
        expect(screen.getByRole('region', { name: 'Selected neuron details' })).toHaveTextContent('Weights at step 10');
        for (const audienceMode of ['beginner', 'explore', 'lab'] as const) {
            act(() => useLayoutStore.getState().setAudienceMode(audienceMode));
            act(() => useLayoutStore.getState().setView('run'));
            act(() => useLayoutStore.getState().setAdvancedToolsOpen(true));
            expect(screen.getByRole('region', { name: 'Selected neuron details' })).toHaveTextContent('2.000');
            expect(useTrainingStore.getState()).toBe(runtime);
            expect(usePlaygroundStore.getState().access).toBe(access);
            expect(window.location.href).toBe(url);
            expect(useLayoutStore.getState().codeExportTab).toBe('numpy');
        }
        await user.click(screen.getByRole('button', { name: 'Clear selection' }));
        expect(screen.queryByRole('region', { name: 'Selected neuron details' })).not.toBeInTheDocument();
        expect(mountedTraining).toHaveBeenCalledTimes(1);
    });

    it('opens real Build controls and focuses Boundary details without mutating the recipe', async () => {
        const user = userEvent.setup();
        const { container } = render(<App />);
        const access = usePlaygroundStore.getState().access;
        const before = container.querySelector('[data-decision-boundary-canvas]');
        await user.click(within(screen.getByRole('navigation', { name: 'Build tools' })).getByRole('button', { name: 'Data' }));
        expect(await screen.findByRole('region', { name: 'Data context' })).toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Close Data context' }));
        await user.click(screen.getByRole('button', { name: 'Boundary details' }));
        expect(screen.getByRole('tab', { name: 'Boundary' })).toHaveAttribute('aria-selected', 'true');
        await vi.waitFor(() => expect(screen.getByRole('tab', { name: 'Boundary' })).toHaveFocus());
        expect(screen.getByRole('checkbox', { name: 'Show test data' })).toBeInTheDocument();
        expect(container.querySelector('[data-decision-boundary-canvas]')).toBe(before);
        expect(usePlaygroundStore.getState().access).toBe(access);
    });
});
