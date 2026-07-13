import { beforeEach, describe, expect, it } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
    type ExperimentDocumentV2,
} from '@nn-playground/shared';
import { FeaturesPanel } from './FeaturesPanel';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { currentPreparedForTest } from '../../test/playgroundStoreTestUtils.ts';

async function restoreDocument(document: ExperimentDocumentV2 = DEFAULT_EXPERIMENT_DOCUMENT) {
    const restored = await usePlaygroundStore.getState().replaceDocument(document);
    expect(restored.ok).toBe(true);
}

function featureIds() {
    return currentPreparedForTest()!.document.recipe.inputs.featureIds;
}

function resetTrainingTransaction() {
    useTrainingStore.getState().resetEvidence();
    useTrainingStore.setState({
        status: 'idle',
        snapshot: null,
        trainPoints: [],
        testPoints: [],
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

describe('FeaturesPanel V2 recipe controls', () => {
    beforeEach(async () => {
        await restoreDocument();
        resetTrainingTransaction();
    });

    it('renders canonical feature IDs instead of a conflicting legacy projection', () => {
        usePlaygroundStore.setState((state) => ({
            features: {
                ...state.features,
                x: false,
                y: false,
                xSquared: true,
            },
        }));

        render(<FeaturesPanel />);

        expect(screen.getByRole('button', { name: 'X₁' })).toHaveAttribute('aria-pressed', 'true');
        expect(screen.getByRole('button', { name: 'X₂' })).toHaveAttribute('aria-pressed', 'true');
        expect(screen.getByRole('button', { name: 'X₁²' })).toHaveAttribute('aria-pressed', 'false');
    });

    it('publishes exact canonical feature order and a new recipe fingerprint', async () => {
        const user = userEvent.setup();
        const beforeFingerprint = currentPreparedForTest()!.identities.recipeFingerprint;
        render(<FeaturesPanel />);

        await user.click(screen.getByRole('button', { name: 'X₂²' }));
        await user.click(screen.getByRole('button', { name: 'X₁²' }));

        await waitFor(() => expect(featureIds()).toEqual(['x', 'y', 'xSquared', 'ySquared']));
        expect(currentPreparedForTest()!.identities.recipeFingerprint)
            .not.toBe(beforeFingerprint);
        expect(useTrainingStore.getState().pendingConfigSource).toBe('features');
    });

    it('preserves rapid orthogonal toggles and lets the final same-feature intent win', async () => {
        render(<FeaturesPanel />);
        const xSquared = screen.getByRole('button', { name: 'X₁²' });
        const ySquared = screen.getByRole('button', { name: 'X₂²' });

        fireEvent.click(xSquared);
        fireEvent.click(ySquared);
        fireEvent.click(xSquared);

        await waitFor(() => expect(featureIds()).toEqual(['x', 'y', 'ySquared']));
    });

    it('disables the final active feature without starting a rejected transaction', async () => {
        const document: ExperimentDocumentV2 = {
            ...DEFAULT_EXPERIMENT_DOCUMENT,
            recipe: {
                ...DEFAULT_EXPERIMENT_DOCUMENT.recipe,
                inputs: { featureIds: ['x'] },
            },
        };
        await restoreDocument(document);
        const before = currentPreparedForTest();
        const user = userEvent.setup();
        render(<FeaturesPanel />);

        const finalFeature = screen.getByRole('button', { name: 'X₁' });
        expect(finalFeature).toBeDisabled();
        await user.click(finalFeature);

        expect(currentPreparedForTest()).toBe(before);
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
        expect(screen.queryByRole('status')).not.toBeInTheDocument();
    });

    it('keeps loading, persistent error, retry, and tooltip behavior accessible', async () => {
        const user = userEvent.setup();
        useTrainingStore.setState({
            featuresConfigLoading: true,
            pendingConfigSource: 'features',
            configError: 'Failed to update features',
            configErrorSource: 'features',
        });
        render(<FeaturesPanel />);

        expect(screen.getByRole('status')).toHaveTextContent('Updating features...');
        expect(screen.getByRole('alert')).toHaveTextContent('Failed to update features');
        expect(screen.getByText(
            'Cause: x squared turns distance from the vertical center into a feature. Effect: circles and rings become easier to separate.',
        )).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Retry' }));

        expect(useTrainingStore.getState().pendingConfigSource).toBe('features');
        expect(useTrainingStore.getState().configSyncNonce).toBe(1);
    });
});
