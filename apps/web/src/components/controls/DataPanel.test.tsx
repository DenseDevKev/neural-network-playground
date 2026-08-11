import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
    type ExperimentDocumentV2,
} from '@nn-playground/shared';
import { DataPanel } from './DataPanel';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import {
    currentPreparedForTest,
    installLegacyDataProjectionForTest,
} from '../../test/playgroundStoreTestUtils.ts';

async function restoreDocument(document: ExperimentDocumentV2 = DEFAULT_EXPERIMENT_DOCUMENT) {
    const restored = await usePlaygroundStore.getState().replaceDocument(document);
    expect(restored.ok).toBe(true);
}

function recipe() {
    return currentPreparedForTest()!.document.recipe;
}

function resetTrainingTransaction() {
    useTrainingStore.getState().resetEvidence();
    useTrainingStore.setState({
        status: 'idle',
        trainPoints: [{ x: 0, y: 0, label: 0 }, { x: 1, y: 1, label: 1 }],
        testPoints: [{ x: -1, y: -1, label: 0 }],
        stepsPerFrame: 5,
        dataConfigLoading: false,
        networkConfigLoading: false,
        featuresConfigLoading: false,
        trainingConfigLoading: false,
        presetConfigLoading: false,
        pendingConfigSource: null,
        configError: null,
        configErrorSource: null,
        configSyncNonce: 0,
    });
}

describe('DataPanel V2 recipe controls', () => {
    beforeEach(async () => {
        await restoreDocument();
        resetTrainingTransaction();
    });

    it('renders the canonical recipe instead of a contradictory legacy data projection', () => {
        const removeLegacyProjection = installLegacyDataProjectionForTest({
            dataset: 'xor',
            problemType: 'classification',
            noise: 49,
            trainTestRatio: 0.9,
            numSamples: 1_000,
        });
        try {
            render(<DataPanel onReset={vi.fn()} />);

            expect(screen.getByLabelText('Dataset settings: 300 samples, 0 noise, 50% train'))
                .toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'Circle' }))
                .toHaveAttribute('aria-pressed', 'true');
            expect(screen.getByRole('button', { name: 'XOR' }))
                .toHaveAttribute('aria-pressed', 'false');
        } finally {
            removeLegacyProjection();
        }
    });

    it('presents problem kind as a derived read-only value while keeping every dataset reachable', () => {
        render(<DataPanel onReset={vi.fn()} />);

        expect(screen.getByLabelText('Problem type: Binary classification'))
            .toHaveTextContent('Binary classification');
        expect(screen.queryByRole('button', { name: 'Classification' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Regression' })).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Three-Class' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Plane' })).toBeInTheDocument();
    });

    it('associates exact slider values with stable native labels and outputs', async () => {
        render(<DataPanel onReset={vi.fn()} />);

        const trainSlider = screen.getByRole('slider', { name: 'Train ratio' });
        const noiseSlider = screen.getByRole('slider', { name: 'Noise' });
        const trainId = trainSlider.id;
        const noiseId = noiseSlider.id;
        const trainOutputId = trainSlider.getAttribute('aria-describedby') ?? '';
        const noiseOutputId = noiseSlider.getAttribute('aria-describedby') ?? '';

        expect(trainId).not.toBe('');
        expect(noiseId).not.toBe('');
        expect(trainOutputId).not.toBe('');
        expect(noiseOutputId).not.toBe('');
        expect(document.querySelector(`label[for="${trainId}"]`))
            .toHaveTextContent('Train ratio');
        expect(document.querySelector(`label[for="${noiseId}"]`))
            .toHaveTextContent('Noise');

        const trainOutput = document.getElementById(trainOutputId);
        const noiseOutput = document.getElementById(noiseOutputId);
        expect(trainOutput?.tagName).toBe('OUTPUT');
        expect(noiseOutput?.tagName).toBe('OUTPUT');
        expect(trainOutput).toHaveAttribute('for', trainId);
        expect(noiseOutput).toHaveAttribute('for', noiseId);
        expect(trainOutput).toHaveTextContent('50%');
        expect(noiseOutput).toHaveTextContent('0%');
        expect((trainSlider as HTMLInputElement).value).toBe('50');
        expect((noiseSlider as HTMLInputElement).value).toBe('0');
        expect(trainSlider).toHaveAttribute(
            'aria-valuetext',
            '50 percent training, 50 percent test',
        );
        expect(noiseSlider).toHaveAttribute('aria-valuetext', '0 percent noise');

        fireEvent.change(trainSlider, { target: { value: '70' } });
        await waitFor(() => expect(recipe().data.trainFraction).toBe(0.7));
        expect(screen.getByRole('slider', { name: 'Train ratio' })).toHaveAttribute(
            'aria-valuetext',
            '70 percent training, 30 percent test',
        );
        expect(document.getElementById(trainOutputId)).toHaveTextContent('70%');
        expect(screen.getByRole('slider', { name: 'Train ratio' }).id).toBe(trainId);
        expect(screen.getByRole('slider', { name: 'Noise' }).id).toBe(noiseId);

        fireEvent.change(noiseSlider, { target: { value: '15' } });
        await waitFor(() => expect(recipe().data.noise).toBe(15));
        expect(screen.getByRole('slider', { name: 'Noise' }))
            .toHaveAttribute('aria-valuetext', '15 percent noise');
        expect(document.getElementById(noiseOutputId)).toHaveTextContent('15%');
        expect(screen.getByRole('slider', { name: 'Train ratio' }).id).toBe(trainId);
        expect(screen.getByRole('slider', { name: 'Noise' }).id).toBe(noiseId);
    });

    it('keeps slider and output ownership unique across two panels', () => {
        render(
            <>
                <DataPanel onReset={vi.fn()} />
                <DataPanel onReset={vi.fn()} />
            </>,
        );

        const sliders = [
            ...screen.getAllByRole('slider', { name: 'Train ratio' }),
            ...screen.getAllByRole('slider', { name: 'Noise' }),
        ];
        const inputIds = sliders.map((slider) => slider.id);
        const outputIds = sliders.map(
            (slider) => slider.getAttribute('aria-describedby') ?? '',
        );

        expect(inputIds).not.toContain('');
        expect(outputIds).not.toContain('');
        expect(new Set([...inputIds, ...outputIds]).size).toBe(8);

        for (const slider of sliders) {
            const outputId = slider.getAttribute('aria-describedby') ?? '';
            expect(document.querySelector(`label[for="${slider.id}"]`))
                .toHaveTextContent(slider.getAttribute('aria-valuetext')?.includes('noise')
                    ? 'Noise'
                    : 'Train ratio');
            expect(document.getElementById(outputId)).toHaveAttribute('for', slider.id);
        }
    });

    it('moves binary to multiclass to regression to binary with exact derived contracts', async () => {
        const user = userEvent.setup();
        render(<DataPanel onReset={vi.fn()} />);

        await user.click(screen.getByRole('button', { name: 'Three-Class' }));
        await waitFor(() => expect(recipe().task).toEqual({
            kind: 'multiclass-classification',
            dataset: 'three-class-clusters',
        }));
        expect(recipe().objective.dataLoss).toEqual({
            kind: 'categorical-cross-entropy-with-logits',
        });
        expect(currentPreparedForTest()?.compiled.task).toMatchObject({
            outputSize: 3,
            outputActivation: 'softmax',
        });

        await user.click(screen.getByRole('button', { name: 'Plane' }));
        await waitFor(() => expect(recipe().task).toEqual({
            kind: 'regression',
            dataset: 'reg-plane',
        }));
        expect(recipe().objective.dataLoss).toEqual({ kind: 'mean-squared-error' });
        expect(currentPreparedForTest()?.compiled.task).toMatchObject({
            outputSize: 1,
            outputActivation: 'linear',
        });

        await user.click(screen.getByRole('button', { name: 'XOR' }));
        await waitFor(() => expect(recipe().task).toEqual({
            kind: 'binary-classification',
            dataset: 'xor',
        }));
        expect(recipe().objective.dataLoss).toEqual({
            kind: 'binary-cross-entropy-with-logits',
        });
        expect(currentPreparedForTest()?.compiled.task).toMatchObject({
            outputSize: 1,
            outputActivation: 'sigmoid',
        });
    });

    it('publishes exact sample, split, noise, and seed edits with a new fingerprint', async () => {
        const user = userEvent.setup();
        const beforeFingerprint = currentPreparedForTest()!.identities.recipeFingerprint;
        render(<DataPanel onReset={vi.fn()} />);

        await user.click(screen.getByRole('button', { name: '600 samples' }));
        fireEvent.change(screen.getByRole('slider', { name: 'Train ratio' }), {
            target: { value: '70' },
        });
        fireEvent.change(screen.getByRole('slider', { name: 'Noise' }), {
            target: { value: '18' },
        });
        await waitFor(() => expect(recipe().data).toMatchObject({
            sampleCount: 600,
            trainFraction: 0.7,
            noise: 18,
        }));
        act(() => useTrainingStore.getState().finishConfigChange());
        await user.click(screen.getByRole('button', { name: 'Reshuffle split' }));

        await waitFor(() => expect(recipe().data).toEqual({
            sampleCount: 600,
            trainFraction: 0.7,
            noise: 18,
            seed: 43,
        }));
        expect(currentPreparedForTest()!.identities.recipeFingerprint)
            .not.toBe(beforeFingerprint);
        expect(useTrainingStore.getState().pendingConfigSource).toBe('data');
    });

    it('preserves rapid orthogonal edits and makes the last same-field edit win', async () => {
        render(<DataPanel onReset={vi.fn()} />);
        const split = screen.getByRole('slider', { name: 'Train ratio' });
        const noise = screen.getByRole('slider', { name: 'Noise' });

        fireEvent.change(split, { target: { value: '60' } });
        fireEvent.change(noise, { target: { value: '7' } });
        fireEvent.change(noise, { target: { value: '23' } });

        await waitFor(() => {
            expect(recipe().data.trainFraction).toBe(0.6);
            expect(recipe().data.noise).toBe(23);
        });
    });

    it('rejects a sample-count edit that would make batch exceed the training population', async () => {
        const document: ExperimentDocumentV2 = {
            ...DEFAULT_EXPERIMENT_DOCUMENT,
            recipe: {
                ...DEFAULT_EXPERIMENT_DOCUMENT.recipe,
                training: {
                    ...DEFAULT_EXPERIMENT_DOCUMENT.recipe.training,
                    batchSize: 100,
                },
            },
        };
        await restoreDocument(document);
        const before = currentPreparedForTest();
        const user = userEvent.setup();
        render(<DataPanel onReset={vi.fn()} />);

        await user.click(screen.getByRole('button', { name: '100 samples' }));

        await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(
            'recipe.data.sampleCount: batch size 100 exceeds training population 50',
        ));
        expect(currentPreparedForTest()).toBe(before);
        expect(recipe().data.sampleCount).toBe(300);
    });

    it('rejects uint32 reshuffle overflow without changing the prepared reference', async () => {
        const document: ExperimentDocumentV2 = {
            ...DEFAULT_EXPERIMENT_DOCUMENT,
            recipe: {
                ...DEFAULT_EXPERIMENT_DOCUMENT.recipe,
                data: {
                    ...DEFAULT_EXPERIMENT_DOCUMENT.recipe.data,
                    seed: 4_294_967_295,
                },
            },
        };
        await restoreDocument(document);
        const before = currentPreparedForTest();
        const user = userEvent.setup();
        render(<DataPanel onReset={vi.fn()} />);

        await user.click(screen.getByRole('button', { name: 'Reshuffle split' }));

        await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(
            'recipe.data.seed: cannot reshuffle seed 4294967295 beyond the uint32 maximum',
        ));
        expect(currentPreparedForTest()).toBe(before);
    });

    it('keeps loading, retry, split-count, tooltip, and explicit reset behavior accessible', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();
        useTrainingStore.setState({
            dataConfigLoading: true,
            pendingConfigSource: 'data',
            configError: 'Failed to generate data',
            configErrorSource: 'data',
        });
        render(<DataPanel onReset={onReset} />);

        const loadingStatus = screen.getByText('Generating data...').closest('[role="status"]');
        expect(loadingStatus).toHaveTextContent('Generating data...');
        expect(screen.getByRole('alert')).toHaveTextContent('Failed to generate data');
        expect(screen.getByLabelText('Train/test split: 2 train, 1 test')).toBeInTheDocument();
        expect(screen.getByText(
            'Cause: XOR alternates labels by quadrant. Effect: a straight boundary fails, so hidden layers have something meaningful to learn.',
        )).toBeInTheDocument();
        expect(screen.getByRole('button', { name: '600 samples' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Reshuffle split' })).toBeDisabled();

        await user.click(screen.getByRole('button', { name: 'Retry' }));
        expect(useTrainingStore.getState().configSyncNonce).toBe(1);

        await user.click(screen.getByRole('button', { name: 'Reset model & data' }));
        expect(onReset).toHaveBeenCalledTimes(1);
    });
});
