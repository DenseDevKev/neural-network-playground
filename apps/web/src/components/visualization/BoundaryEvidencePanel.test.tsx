import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { expect, it } from 'vitest';
import { PREPARED_PRESETS } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { BoundaryEvidencePanel } from './BoundaryEvidencePanel.tsx';
import { useDecisionBoundaryController } from './useDecisionBoundaryController.ts';
function Harness() { return <BoundaryEvidencePanel controller={useDecisionBoundaryController()} />; }
it('describes three-class confidence and flat winning regions when discretized', async () => {
    const prepared = PREPARED_PRESETS.find((entry) => entry.id === 'three-class-clusters')!.prepared;
    await usePlaygroundStore.getState().replaceDocument(prepared.document);
    await act(async () => { await usePlaygroundStore.getState().editView((view) => ({ ...view, discretizeOutput: false })); });
    render(<Harness />);
    expect(screen.getByText(/color strength indicating winning-class confidence/)).toBeInTheDocument();
    expect(screen.queryByText(/negative blue|positive orange/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByLabelText('Discretize output'));
    await waitFor(() => expect(screen.getByText(/flat winning-class regions for Class 0, Class 1, and Class 2/)).toBeInTheDocument());
});
