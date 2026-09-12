import { act, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { DEFAULT_EXPERIMENT_DOCUMENT, PREPARED_PRESETS } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useDecisionBoundaryController } from './useDecisionBoundaryController.ts';
import { BoundaryEvidencePanel } from './BoundaryEvidencePanel.tsx';
import { DecisionBoundaryCanvas } from './DecisionBoundaryCanvas.tsx';
import { getConceptById } from '../../concepts/conceptCatalog.ts';

vi.mock('./DecisionBoundaryCanvas.tsx', () => ({ DecisionBoundaryCanvas: () => <canvas data-testid="live-boundary" /> }));

const current = () => {
    const access = usePlaygroundStore.getState().access;
    if (access.status !== 'ready') throw new Error('Expected a prepared experiment');
    return access.prepared.document;
};
describe('shared decision boundary controller', () => {
    beforeEach(async () => {
        await usePlaygroundStore.getState().replaceDocument(DEFAULT_EXPERIMENT_DOCUMENT);
        useTrainingStore.getState().resetEvidence();
    });
    it('shares the existing document-view command without editing the recipe or runtime', async () => {
        const before = current().recipe;
        const training = useTrainingStore.getState();
        const { result } = renderHook(() => useDecisionBoundaryController());
        const next = !result.current.showTestData;
        await act(async () => { await result.current.commands.setShowTestData(next); });
        expect(result.current.showTestData).toBe(next);
        expect(current().recipe).toEqual(before);
        expect(useTrainingStore.getState()).toBe(training);
    });
    it('keeps overlays local, distinct from recipe and document view', () => {
        const before = usePlaygroundStore.getState().access;
        const { result } = renderHook(() => useDecisionBoundaryController());
        act(() => result.current.commands.setOverlayMode('uncertainty'));
        expect(result.current.overlayMode).toBe('uncertainty');
        expect(usePlaygroundStore.getState().access).toBe(before);
    });
    it('renders details without creating another live canvas', async () => {
        const { result } = renderHook(() => useDecisionBoundaryController());
        const { container } = render(<BoundaryEvidencePanel controller={result.current} />);
        expect(container.querySelectorAll('canvas')).toHaveLength(0);
        const previous = current().view.showTestData;
        fireEvent.click(screen.getByRole('checkbox', { name: 'Show test data' }));
        await waitFor(() => expect(current().view.showTestData).toBe(!previous));
    });
    it('publishes both decision view toggles through the canonical edit transaction', async () => {
        const editView = vi.spyOn(usePlaygroundStore.getState(), 'editView');
        const { result } = renderHook(() => useDecisionBoundaryController());
        render(<BoundaryEvidencePanel controller={result.current} />);

        fireEvent.click(screen.getByRole('checkbox', { name: 'Show test data' }));
        fireEvent.click(screen.getByRole('checkbox', { name: 'Discretize output' }));

        await waitFor(() => {
            expect(editView).toHaveBeenCalledTimes(2);
            expect(current().view).toEqual({
                showTestData: true,
                discretizeOutput: true,
            });
        });
        editView.mockRestore();
    });
    it('updates the detailed overlay explanation when its local mode changes', () => {
        function BoundaryHarness() {
            return <BoundaryEvidencePanel controller={useDecisionBoundaryController()} />;
        }
        render(<BoundaryHarness />);

        expect(screen.getByText(/output mode shows/i)).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Uncertain' }));
        expect(screen.getAllByText(/least sure/i).length).toBeGreaterThan(0);
        fireEvent.click(screen.getByRole('button', { name: 'Errors' }));
        expect(screen.getAllByText(/training points whose predicted class does not match/i).length)
            .toBeGreaterThan(0);
    });
    it('offers decision-boundary guidance for classification', () => {
        function BoundaryHarness() {
            return <BoundaryEvidencePanel controller={useDecisionBoundaryController()} />;
        }
        render(<BoundaryHarness />);

        fireEvent.click(screen.getByRole('button', { name: 'Learn about Decision boundary' }));
        expect(screen.getByText(getConceptById('decision-boundary')?.plainDefinition ?? ''))
            .toBeInTheDocument();
    });
    it('does not label a regression output field as a classification decision boundary', () => {
        const regression = PREPARED_PRESETS.find((entry) => entry.id === 'regression-plane')?.prepared;
        if (!regression) throw new Error('missing regression-plane preset');
        usePlaygroundStore.setState({ access: { status: 'ready', prepared: regression } });
        function BoundaryHarness() {
            return <BoundaryEvidencePanel controller={useDecisionBoundaryController()} />;
        }

        render(<BoundaryHarness />);

        expect(screen.getByLabelText('Decision overlay controls')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Learn about Decision boundary' }))
            .not.toBeInTheDocument();
    });
    it('keeps exactly one live canvas mounted while expanding evidence', () => {
        const { result } = renderHook(() => useDecisionBoundaryController());
        const { container, rerender } = render(<DecisionBoundaryCanvas model={result.current.model} />);
        const canvas = screen.getByTestId('live-boundary');
        rerender(<><DecisionBoundaryCanvas model={result.current.model} /><BoundaryEvidencePanel controller={result.current} /></>);
        expect(container.querySelectorAll('canvas')).toHaveLength(1);
        expect(screen.getByTestId('live-boundary')).toBe(canvas);
    });
});
