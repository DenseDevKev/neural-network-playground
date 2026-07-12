import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DEFAULT_EXPERIMENT_DOCUMENT } from '@nn-playground/shared';
import {
    setBatchSize,
    setLearningRate,
    switchDataset,
} from '../../store/recipeEdits.ts';
import {
    usePlaygroundStore,
    type PlaygroundStore,
} from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { currentPreparedForTest } from '../../test/playgroundStoreTestUtils.ts';
import { HyperparamPanel } from './HyperparamPanel.tsx';

const originalEditRecipe: PlaygroundStore['editRecipe'] = usePlaygroundStore.getState().editRecipe;

function resetTrainingTransaction() {
    useTrainingStore.setState({
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

async function editNumber(name: string, value: string) {
    const input = screen.getByRole('spinbutton', { name });
    fireEvent.change(input, { target: { value } });
    fireEvent.blur(input);
    await waitFor(() => {
        expect(input).toHaveValue(Number(value));
        expect(usePlaygroundStore.getState().preparation.status).not.toBe('preparing');
    });
}

describe('HyperparamPanel canonical V2 controls', () => {
    beforeEach(async () => {
        usePlaygroundStore.setState({ editRecipe: originalEditRecipe });
        const restored = await usePlaygroundStore.getState()
            .replaceDocument(DEFAULT_EXPERIMENT_DOCUMENT);
        expect(restored.ok).toBe(true);
        resetTrainingTransaction();
    });

    afterEach(() => {
        act(() => usePlaygroundStore.setState({ editRecipe: originalEditRecipe }));
        vi.restoreAllMocks();
    });

    it('reads canonical training values and renders classification objective/output as derived', () => {
        usePlaygroundStore.setState({
            training: {
                ...usePlaygroundStore.getState().training,
                learningRate: 10,
                batchSize: 64,
            },
        });
        render(<HyperparamPanel />);

        expect(screen.getByRole('combobox', { name: 'Learning rate' })).toHaveValue('0.03');
        expect(screen.getByRole('combobox', { name: 'Batch size' })).toHaveValue('10');
        expect(screen.getByText('Binary cross-entropy with logits')).toBeInTheDocument();
        expect(screen.getByText('sigmoid')).toBeInTheDocument();
        expect(screen.queryByRole('combobox', { name: 'Loss' })).not.toBeInTheDocument();
        expect(screen.queryByRole('combobox', { name: /output activation/i })).not.toBeInTheDocument();
        expect(screen.queryByRole('combobox', { name: 'Weight initialization' }))
            .not.toBeInTheDocument();
    });

    it('switches step, cosine, and constant schedules with exact explicit defaults', async () => {
        const user = userEvent.setup();
        render(<HyperparamPanel />);
        const schedule = screen.getByRole('combobox', { name: 'LR schedule' });

        await user.selectOptions(schedule, 'step');
        await waitFor(() => expect(currentPreparedForTest()?.document.recipe.training.schedule)
            .toEqual({ kind: 'step', interval: 100, gamma: 0.5 }));
        await editNumber('Step schedule interval', '250');
        await editNumber('Step schedule gamma', '0.75');
        expect(currentPreparedForTest()?.document.recipe.training.schedule)
            .toEqual({ kind: 'step', interval: 250, gamma: 0.75 });

        await user.selectOptions(schedule, 'cosine');
        await waitFor(() => expect(currentPreparedForTest()?.document.recipe.training.schedule)
            .toEqual({ kind: 'cosine', totalSteps: 1000, minimumRate: 0 }));
        expect(screen.queryByRole('spinbutton', { name: 'Step schedule interval' }))
            .not.toBeInTheDocument();
        await editNumber('Cosine total steps', '2000');
        await editNumber('Cosine minimum learning rate', '0.001');
        expect(currentPreparedForTest()?.document.recipe.training.schedule)
            .toEqual({ kind: 'cosine', totalSteps: 2000, minimumRate: 0.001 });

        await user.selectOptions(schedule, 'constant');
        await waitFor(() => expect(currentPreparedForTest()?.document.recipe.training.schedule)
            .toEqual({ kind: 'constant' }));
    });

    it('switches all optimizer variants without retaining inactive fields, including Adam epsilon', async () => {
        const user = userEvent.setup();
        render(<HyperparamPanel />);
        const optimizer = screen.getByRole('combobox', { name: 'Optimizer' });

        await user.selectOptions(optimizer, 'sgd-momentum');
        await waitFor(() => expect(currentPreparedForTest()?.document.recipe.training.optimizer)
            .toEqual({ kind: 'sgd-momentum', momentum: 0.9 }));
        await editNumber('Momentum', '0.8');
        expect(currentPreparedForTest()?.document.recipe.training.optimizer)
            .toEqual({ kind: 'sgd-momentum', momentum: 0.8 });

        await user.selectOptions(optimizer, 'adam');
        await waitFor(() => expect(currentPreparedForTest()?.document.recipe.training.optimizer)
            .toEqual({ kind: 'adam', beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }));
        await editNumber('Adam beta 1', '0.85');
        await editNumber('Adam beta 2', '0.99');
        await editNumber('Adam epsilon', '0.0000001');
        expect(currentPreparedForTest()?.document.recipe.training.optimizer)
            .toEqual({ kind: 'adam', beta1: 0.85, beta2: 0.99, epsilon: 1e-7 });

        await user.selectOptions(optimizer, 'sgd');
        await waitFor(() => expect(currentPreparedForTest()?.document.recipe.training.optimizer)
            .toEqual({ kind: 'sgd' }));
    });

    it('round-trips clipping and penalty variants with their exact scopes and defaults', async () => {
        const user = userEvent.setup();
        render(<HyperparamPanel />);

        await user.selectOptions(
            screen.getByRole('combobox', { name: 'Gradient clipping' }),
            'global-norm',
        );
        await waitFor(() => expect(currentPreparedForTest()?.document.recipe.training.gradientClipping)
            .toEqual({
                kind: 'global-norm',
                maximumNorm: 1,
                scope: 'total-objective-gradient',
            }));
        expect(screen.getByText('total objective gradient')).toBeInTheDocument();
        await editNumber('Maximum gradient norm', '2');

        await user.selectOptions(screen.getByRole('combobox', { name: 'Penalty' }), 'l1');
        await waitFor(() => expect(currentPreparedForTest()?.document.recipe.objective.penalty)
            .toEqual({ kind: 'l1', coefficient: 0.001, applyTo: 'weights' }));
        expect(screen.getByText('weights only')).toBeInTheDocument();
        await editNumber('Penalty coefficient', '0.03');
        await user.selectOptions(screen.getByRole('combobox', { name: 'Penalty' }), 'l2');
        await waitFor(() => expect(currentPreparedForTest()?.document.recipe.objective.penalty)
            .toEqual({ kind: 'l2', coefficient: 0.001, applyTo: 'weights' }));

        await user.selectOptions(
            screen.getByRole('combobox', { name: 'Gradient clipping' }),
            'none',
        );
        await user.selectOptions(screen.getByRole('combobox', { name: 'Penalty' }), 'none');
        await waitFor(() => {
            const recipe = currentPreparedForTest()?.document.recipe;
            expect(recipe?.training.gradientClipping).toEqual({ kind: 'none' });
            expect(recipe?.objective.penalty).toEqual({ kind: 'none' });
        });
    });

    it('moves binary to multiclass to regression and back with exact derived labels and losses', async () => {
        render(<HyperparamPanel />);
        expect(screen.getByText('Binary cross-entropy with logits')).toBeInTheDocument();

        await act(async () => {
            const result = await originalEditRecipe((recipe) => (
                switchDataset(recipe, 'three-class-clusters')
            ));
            expect(result.ok).toBe(true);
        });
        expect(screen.getByText('Categorical cross-entropy with logits')).toBeInTheDocument();
        expect(screen.getByText('softmax')).toBeInTheDocument();
        expect(screen.queryByRole('combobox', { name: 'Loss' })).not.toBeInTheDocument();

        await act(async () => {
            const result = await originalEditRecipe((recipe) => switchDataset(recipe, 'reg-plane'));
            expect(result.ok).toBe(true);
        });
        expect(screen.getByText('linear')).toBeInTheDocument();
        expect(screen.getByRole('combobox', { name: 'Loss' }))
            .toHaveValue('mean-squared-error');

        await act(async () => {
            const result = await originalEditRecipe((recipe) => switchDataset(recipe, 'circle'));
            expect(result.ok).toBe(true);
        });
        expect(screen.getByText('Binary cross-entropy with logits')).toBeInTheDocument();
        expect(screen.getByText('sigmoid')).toBeInTheDocument();
        expect(screen.queryByRole('combobox', { name: 'Loss' })).not.toBeInTheDocument();
    });

    it('offers true MSE and Huber only for regression and preserves exact Huber delta', async () => {
        await originalEditRecipe((recipe) => switchDataset(recipe, 'reg-gauss'));
        const user = userEvent.setup();
        render(<HyperparamPanel />);
        const loss = screen.getByRole('combobox', { name: 'Loss' });

        expect(within(loss).getByRole('option', { name: 'Mean squared error' }))
            .toBeInTheDocument();
        expect(within(loss).getByRole('option', { name: 'Huber' })).toBeInTheDocument();
        await user.selectOptions(loss, 'huber');
        await waitFor(() => expect(currentPreparedForTest()?.document.recipe.objective.dataLoss)
            .toEqual({ kind: 'huber', delta: 1 }));
        await editNumber('Huber delta', '2.25');
        expect(currentPreparedForTest()?.document.recipe.objective.dataLoss)
            .toEqual({ kind: 'huber', delta: 2.25 });

        await user.selectOptions(loss, 'mean-squared-error');
        await waitFor(() => expect(currentPreparedForTest()?.document.recipe.objective.dataLoss)
            .toEqual({ kind: 'mean-squared-error' }));
        expect(screen.queryByRole('spinbutton', { name: 'Huber delta' }))
            .not.toBeInTheDocument();
    });

    it('commits exact learning-rate and batch values and changes the fingerprint', async () => {
        const user = userEvent.setup();
        render(<HyperparamPanel />);
        const firstFingerprint = currentPreparedForTest()?.identities.recipeFingerprint;

        await user.selectOptions(screen.getByRole('combobox', { name: 'Learning rate' }), '0.1');
        await user.selectOptions(screen.getByRole('combobox', { name: 'Batch size' }), '32');

        await waitFor(() => {
            const prepared = currentPreparedForTest();
            expect(prepared?.document.recipe.training).toMatchObject({
                learningRate: 0.1,
                batchSize: 32,
            });
            expect(prepared?.identities.recipeFingerprint).not.toBe(firstFingerprint);
        });
        expect(useTrainingStore.getState().pendingConfigSource).toBe('training');
    });

    it('keeps valid imported values visible when they are outside the quick-pick lists', async () => {
        const result = await originalEditRecipe((recipe) => {
            const withRate = setLearningRate(recipe, 0.02);
            if (!withRate.ok) return withRate;
            return setBatchSize(withRate.recipe, 7);
        });
        expect(result.ok).toBe(true);

        render(<HyperparamPanel />);

        expect(screen.getByRole('combobox', { name: 'Learning rate' })).toHaveValue('0.02');
        expect(screen.getByRole('combobox', { name: 'Batch size' })).toHaveValue('7');
        expect(screen.getByRole('option', { name: '0.02 (current)' })).toBeInTheDocument();
        expect(screen.getByRole('option', { name: '7 (current)' })).toBeInTheDocument();
    });

    it('does not offer batch sizes above the derived training population', async () => {
        const result = await originalEditRecipe((recipe) => ({
            ok: true,
            recipe: {
                ...recipe,
                data: { ...recipe.data, sampleCount: 20, trainFraction: 0.5 },
            },
        }));
        expect(result.ok).toBe(true);
        render(<HyperparamPanel />);

        const batch = screen.getByRole('combobox', { name: 'Batch size' });
        expect(within(batch).getByRole('option', { name: '16' })).toBeDisabled();
        expect(within(batch).getByRole('option', { name: '64' })).toBeDisabled();
        expect(screen.getByText('Maximum for this split: 10')).toBeInTheDocument();
    });

    it('retains the exact prepared state and invalid draft when numeric validation fails', async () => {
        const user = userEvent.setup();
        render(<HyperparamPanel />);
        await user.selectOptions(screen.getByRole('combobox', { name: 'LR schedule' }), 'step');
        await waitFor(() => expect(currentPreparedForTest()?.document.recipe.training.schedule.kind)
            .toBe('step'));
        const beforeInvalid = currentPreparedForTest();

        await editNumber('Step schedule interval', '0');

        expect(await screen.findByRole('alert')).toHaveTextContent(
            'recipe.training.schedule.interval',
        );
        expect(currentPreparedForTest()).toBe(beforeInvalid);
        expect(screen.getByRole('spinbutton', { name: 'Step schedule interval' }))
            .toHaveValue(0);
    });

    it('does not rebuild the runtime when an exact numeric field is blurred unchanged', async () => {
        const user = userEvent.setup();
        render(<HyperparamPanel />);
        await user.selectOptions(screen.getByRole('combobox', { name: 'LR schedule' }), 'step');
        await waitFor(() => expect(currentPreparedForTest()?.document.recipe.training.schedule.kind)
            .toBe('step'));
        act(() => useTrainingStore.getState().finishConfigChange());
        const before = currentPreparedForTest();

        fireEvent.blur(screen.getByRole('spinbutton', { name: 'Step schedule interval' }));

        expect(currentPreparedForTest()).toBe(before);
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
    });

    it('preserves rapid orthogonal edits and last-write order without partial projections', async () => {
        render(<HyperparamPanel />);
        const learningRate = screen.getByRole('combobox', { name: 'Learning rate' });
        const optimizer = screen.getByRole('combobox', { name: 'Optimizer' });

        fireEvent.change(learningRate, { target: { value: '0.1' } });
        fireEvent.change(optimizer, { target: { value: 'adam' } });
        fireEvent.change(learningRate, { target: { value: '0.3' } });

        await waitFor(() => {
            const training = currentPreparedForTest()?.document.recipe.training;
            expect(training?.learningRate).toBe(0.3);
            expect(training?.optimizer).toEqual({
                kind: 'adam',
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            });
        });
    });

    it('keeps preparation failures persistent and retryable', async () => {
        const before = currentPreparedForTest();
        usePlaygroundStore.setState({
            editRecipe: vi.fn(async () => ({
                ok: false as const,
                issues: [{
                    code: 'invalid-field' as const,
                    path: 'recipe.training.learningRate',
                    message: 'learning rate failed external validation',
                }],
            })),
        });
        const user = userEvent.setup();
        render(<HyperparamPanel />);

        await user.selectOptions(screen.getByRole('combobox', { name: 'Learning rate' }), '0.1');

        expect(await screen.findByRole('alert')).toHaveTextContent(
            'recipe.training.learningRate: learning rate failed external validation',
        );
        expect(currentPreparedForTest()).toBe(before);
        await user.click(screen.getByRole('button', { name: 'Retry' }));
        expect(useTrainingStore.getState().pendingConfigSource).toBe('training');
        expect(useTrainingStore.getState().configSyncNonce).toBe(1);
    });

    it('preserves cause/effect explanations for learning rate and batch size', () => {
        render(<HyperparamPanel />);

        expect(screen.getByText(/Cause: larger learning rates take bigger weight updates/))
            .toBeInTheDocument();
        expect(screen.getByText(/Cause: larger batches average more samples per update/))
            .toBeInTheDocument();
    });
});
