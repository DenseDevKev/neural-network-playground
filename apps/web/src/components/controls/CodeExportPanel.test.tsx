// ── CodeExportPanel Tests ──
// Verifies tab switching, code generation, and clipboard copy.

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, fireEvent } from '@testing-library/react';
import { CodeExportPanel } from './CodeExportPanel';
import { useTrainingStore } from '../../store/useTrainingStore';
import { usePlaygroundStore } from '../../store/usePlaygroundStore';
import { useLayoutStore } from '../../store/useLayoutStore';
import { resetFrameBuffer, updateFrameBuffer } from '../../worker/frameBuffer';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
    PREPARED_PRESETS,
    prepareExperimentDocument,
    type ExperimentDocumentV2,
    type PreparedExperimentDocumentV2,
    type ModelRevision,
} from '@nn-playground/shared';

const fakeSnapshot = {
    step: 5,
    epoch: 0,
    trainLoss: 0.5,
    testLoss: 0.6,
    trainMetrics: { loss: 0.5, accuracy: 0.5 },
    testMetrics: { loss: 0.6, accuracy: 0.4 },
    weights: [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6]]],
    biases: [[0.1, 0.2], [0.3]],
    outputGrid: [],
    gridSize: 40,
};

async function prepareWithHiddenLayers(
    hiddenLayers: number[],
): Promise<PreparedExperimentDocumentV2> {
    const document = structuredClone(DEFAULT_EXPERIMENT_DOCUMENT) as ExperimentDocumentV2;
    document.recipe.model.hiddenLayers = hiddenLayers;
    const result = await prepareExperimentDocument(document);
    if (!result.ok) throw new Error(result.issues.map((issue) => issue.message).join('; '));
    return result.value;
}

function installPrepared(prepared: PreparedExperimentDocumentV2) {
    usePlaygroundStore.setState({
        access: { status: 'ready', prepared },
        prepared,
    });
}

function installParameters(
    prepared: PreparedExperimentDocumentV2,
    model: ModelRevision = { generationId: 1, revision: 5, step: 5, epoch: 0 },
    recipeFingerprint = prepared.identities.recipeFingerprint,
) {
    updateFrameBuffer({
        weights: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        biases: new Float32Array([0.1, 0.2, 0.3]),
        weightLayout: { layerSizes: [2, 2, 1] },
        parameterProvenance: { model, recipeFingerprint },
    });
}

describe('CodeExportPanel', () => {
    beforeEach(async () => {
        resetFrameBuffer();

        const prepared = await prepareWithHiddenLayers([2]);
        installPrepared(prepared);

        useTrainingStore.getState().resetEvidence();
        useTrainingStore.setState({
            status: 'idle',
            snapshot: fakeSnapshot as any,
            frameVersion: 0,
            trainPoints: [],
            testPoints: [],
            stepsPerFrame: 5,
            dataConfigLoading: false,
            networkConfigLoading: false,
            pendingConfigSource: null,
            configError: null,
            configErrorSource: null,
            configSyncNonce: 0,
            workerError: null,
            trainedRecipe: prepared.document.recipe,
            trainedRecipeFingerprint: prepared.identities.recipeFingerprint,
            latestLiveSignal: {
                model: { generationId: 1, revision: 5, step: 5, epoch: 0 },
                dataset: {
                    generatorVersion: 1,
                    datasetKey: prepared.identities.datasetKey,
                    trainCount: 150,
                    testCount: 150,
                },
                objectiveKey: prepared.identities.objectiveKey,
                basis: {
                    kind: 'mini-batch-ema',
                    alpha: 0.1,
                    latestBatchSize: 10,
                    throughStep: 5,
                },
                dataLoss: 0.5,
            },
            latestEvaluation: null,
        });

        useLayoutStore.setState({
            layout: 'dock',
            phase: 'build',
            activeTabLeft: 'data',
            activeTabRight: 'code',
            codeExportTab: 'pseudocode',
        } as any);

        // Seed the frame buffer with weights
        installParameters(prepared);
        updateFrameBuffer({
            outputGrid: null,
            gridSize: 40,
            neuronGrids: null,
            neuronGridLayout: null,
            layerStats: null,
            confusionMatrix: null,
        });

        // Mock clipboard
        Object.defineProperty(navigator, 'clipboard', {
            value: { writeText: vi.fn().mockResolvedValue(undefined) },
            configurable: true,
        });
    });

    it('renders the Pseudocode tab by default', () => {
        render(<CodeExportPanel />);
        const pseudocodeBtn = screen.getByRole('button', { name: 'Pseudocode' });
        expect(pseudocodeBtn).toHaveClass('active');
    });

    it('shows code in the Pseudocode tab', () => {
        render(<CodeExportPanel />);
        const code = document.querySelector('.code-export__code');
        expect(code).toBeTruthy();
        expect(code!.textContent).toBeTruthy();
    });

    it('preserves an existing multiclass output size in generated code', () => {
        const multiclass = PREPARED_PRESETS.find((entry) => entry.id === 'three-class-clusters')!.prepared;
        installPrepared(multiclass);
        useTrainingStore.setState({
            snapshot: null,
            trainedRecipe: null,
            trainedRecipeFingerprint: null,
            latestLiveSignal: null,
        });
        resetFrameBuffer();

        render(<CodeExportPanel />);

        const code = document.querySelector('.code-export__code')?.textContent ?? '';
        expect(code).toContain('Architecture: 2 → 6 → 6 → 3');
        expect(code).toContain('activation = Softmax');
        expect(code).toContain('loss = Categorical Cross-Entropy');
        expect(code).not.toContain('Architecture: 2 → 2 → 1');
    });

    it('never pairs learned parameters with a different current recipe fingerprint', async () => {
        render(<CodeExportPanel />);

        expect(document.querySelector('.code-export__code')?.textContent).toContain(
            '# Trained weights (step 5)',
        );

        const changed = await prepareWithHiddenLayers([3]);
        act(() => installPrepared(changed));

        expect(document.querySelector('.code-export__code')?.textContent).toContain(
            '# Architecture: 2 → 3 → 1',
        );
        expect(document.querySelector('.code-export__code')?.textContent).not.toContain(
            '# Trained weights',
        );
    });

    it('labels parameters with their own revision when newer evidence arrives first', () => {
        useTrainingStore.setState({
            latestLiveSignal: {
                ...useTrainingStore.getState().latestLiveSignal!,
                model: { generationId: 1, revision: 6, step: 6, epoch: 0 },
            },
        });

        render(<CodeExportPanel />);

        const code = document.querySelector('.code-export__code')?.textContent ?? '';
        expect(code).toContain('# Trained weights (step 5)');
        expect(code).not.toContain('# Trained weights (step 6)');
    });

    it.each([
        ['generation', { generationId: 2, revision: 5, step: 5, epoch: 0 }, null],
        ['fingerprint', { generationId: 1, revision: 5, step: 5, epoch: 0 }, `r2.1.${'Z'.repeat(43)}`],
    ] as const)('suppresses parameters on a %s mismatch', (_kind, model, fingerprint) => {
        const prepared = usePlaygroundStore.getState().access;
        if (prepared.status !== 'ready') throw new Error('expected prepared experiment');
        installParameters(prepared.prepared, model, fingerprint ?? prepared.prepared.identities.recipeFingerprint);

        render(<CodeExportPanel />);

        expect(document.querySelector('.code-export__code')?.textContent)
            .not.toContain('# Trained weights');
    });

    it('switches to NumPy tab and shows numpy code', async () => {
        render(<CodeExportPanel />);

        const codeBeforeSwitch = document.querySelector('.code-export__code')?.textContent;

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: 'NumPy' }));
        });

        expect(screen.getByRole('button', { name: 'NumPy' })).toHaveClass('active');
        const codeAfterSwitch = document.querySelector('.code-export__code')?.textContent;
        expect(codeAfterSwitch).toBeTruthy();
        // NumPy code will differ from pseudocode
        expect(codeAfterSwitch).not.toBe(codeBeforeSwitch);
    });

    it('switches to TF.js tab and shows tfjs code', async () => {
        render(<CodeExportPanel />);

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: 'TF.js' }));
        });

        expect(screen.getByRole('button', { name: 'TF.js' })).toHaveClass('active');
        const code = document.querySelector('.code-export__code')?.textContent;
        expect(code).toBeTruthy();
    });

    it('each tab generates distinct code', async () => {
        render(<CodeExportPanel />);

        const pseudocode = document.querySelector('.code-export__code')?.textContent ?? '';

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: 'NumPy' }));
        });
        const numpyCode = document.querySelector('.code-export__code')?.textContent ?? '';

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: 'TF.js' }));
        });
        const tfjsCode = document.querySelector('.code-export__code')?.textContent ?? '';

        // All three should be non-empty and distinct
        expect(pseudocode.length).toBeGreaterThan(0);
        expect(numpyCode.length).toBeGreaterThan(0);
        expect(tfjsCode.length).toBeGreaterThan(0);
        expect(new Set([pseudocode, numpyCode, tfjsCode]).size).toBe(3);
    });

    it('calls clipboard.writeText when copy button is clicked', async () => {
        render(<CodeExportPanel />);

        const copyBtn = screen.getByRole('button', { name: /copy code/i });
        await act(async () => {
            fireEvent.click(copyBtn);
        });

        expect(navigator.clipboard.writeText).toHaveBeenCalledTimes(1);
        const callArg = (navigator.clipboard.writeText as ReturnType<typeof vi.fn>).mock.calls[0][0];
        expect(typeof callArg).toBe('string');
        expect(callArg.length).toBeGreaterThan(0);
    });

    it('shows announced failure feedback when copying code fails', async () => {
        (navigator.clipboard.writeText as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error('Denied'));
        render(<CodeExportPanel />);

        const copyBtn = screen.getByRole('button', { name: /copy code/i });
        await act(async () => {
            fireEvent.click(copyBtn);
        });

        expect(screen.getByRole('alert')).toHaveTextContent('Could not copy code');
    });

    it('shows announced failure feedback when the Clipboard API is unavailable', async () => {
        Object.defineProperty(navigator, 'clipboard', {
            value: undefined,
            configurable: true,
        });
        render(<CodeExportPanel />);

        const copyBtn = screen.getByRole('button', { name: /copy code/i });
        await act(async () => {
            fireEvent.click(copyBtn);
        });

        expect(screen.getByRole('alert')).toHaveTextContent('Could not copy code');
    });

    it('passes the current tab code to clipboard.writeText', async () => {
        render(<CodeExportPanel />);

        // Switch to NumPy tab
        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: 'NumPy' }));
        });

        const copyBtn = screen.getByRole('button', { name: /copy code/i });
        await act(async () => {
            fireEvent.click(copyBtn);
        });

        const callArg = (navigator.clipboard.writeText as ReturnType<typeof vi.fn>).mock.calls[0][0];
        const renderedCode = document.querySelector('.code-export__code')?.textContent ?? '';
        expect(callArg).toBe(renderedCode);
    });

    it('preserves the selected export tab after the panel remounts', async () => {
        const { unmount } = render(<CodeExportPanel />);

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: 'NumPy' }));
        });

        unmount();
        render(<CodeExportPanel />);

        expect(screen.getByRole('button', { name: 'NumPy' })).toHaveClass('active');
    });
});
