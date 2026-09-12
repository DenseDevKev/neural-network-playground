import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { DEFAULT_EXPERIMENT_DOCUMENT } from '@nn-playground/shared';
import { DataPanel } from './DataPanel.tsx';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import { currentPreparedForTest } from '../../test/playgroundStoreTestUtils.ts';

const recipe = () => currentPreparedForTest()!.document.recipe;

describe('DataPanel production previews', () => {
    beforeEach(async () => {
        expect((await usePlaygroundStore.getState().replaceDocument(DEFAULT_EXPERIMENT_DOCUMENT)).ok).toBe(true);
        useTrainingStore.getState().resetEvidence();
        useTrainingStore.setState({ dataConfigLoading: false, networkConfigLoading: false, featuresConfigLoading: false, trainingConfigLoading: false, presetConfigLoading: false, pendingConfigSource: null, configError: null, configErrorSource: null });
        useLayoutStore.setState({ audienceMode: 'beginner' });
    });
    it('gives every named dataset a decorative production preview', () => {
        const { container } = render(<DataPanel onReset={vi.fn()} />);
        expect(container.querySelectorAll('canvas[aria-hidden="true"]')).toHaveLength(11);
        for (const name of ['Gaussian', 'Checker', 'Three-Class', 'Plane', 'Multi-Gauss']) {
            expect(screen.getByRole('button', { name }).querySelector('canvas')).not.toBeNull();
        }
    });
    it('preserves newer dataset edits while the previous worker synchronization is loading', async () => {
        useTrainingStore.setState({ dataConfigLoading: true });
        render(<DataPanel onReset={vi.fn()} />);
        const gaussian = screen.getByRole('button', { name: 'Gaussian' });
        expect(gaussian).toBeEnabled();
        fireEvent.click(gaussian);
        await waitFor(() => expect(recipe().task.dataset).toBe('gauss'));
    });
    it.each([['Gaussian', 'gauss'], ['Checker', 'checkerboard'], ['Three-Class', 'three-class-clusters'], ['Plane', 'reg-plane'], ['Multi-Gauss', 'reg-gauss']])('keeps the real recipe transaction for %s', async (name, dataset) => {
        const reset = vi.fn();
        render(<DataPanel onReset={reset} />);
        fireEvent.click(screen.getByRole('button', { name }));
        await waitFor(() => expect(recipe().task.dataset).toBe(dataset));
        expect(reset).not.toHaveBeenCalled();
    });
});
