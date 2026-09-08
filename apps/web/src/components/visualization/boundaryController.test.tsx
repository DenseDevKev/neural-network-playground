import { act, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { DEFAULT_EXPERIMENT_DOCUMENT } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useDecisionBoundaryController } from './useDecisionBoundaryController.ts';
import { BoundaryEvidencePanel } from './BoundaryEvidencePanel.tsx';
import { PinnedBoundaryRail } from './PinnedBoundaryRail.tsx';

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
    it('keeps exactly one live canvas mounted while expanding evidence', () => {
        const expand = vi.fn();
        const { result } = renderHook(() => useDecisionBoundaryController());
        const { container, rerender } = render(<PinnedBoundaryRail controller={result.current} onExpand={expand} />);
        const canvas = screen.getByTestId('live-boundary');
        fireEvent.click(screen.getByRole('button', { name: 'Boundary details' }));
        expect(expand).toHaveBeenCalledOnce();
        rerender(<><PinnedBoundaryRail controller={result.current} onExpand={expand} /><BoundaryEvidencePanel controller={result.current} /></>);
        expect(container.querySelectorAll('canvas')).toHaveLength(1);
        expect(screen.getByTestId('live-boundary')).toBe(canvas);
    });
});
