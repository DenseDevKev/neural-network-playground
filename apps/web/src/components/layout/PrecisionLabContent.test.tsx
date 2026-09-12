import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ConfigurationContent, LossContent } from './PrecisionLabContent.tsx';
import { useTrainingStore } from '../../store/useTrainingStore.ts';

vi.mock('../controls/ConfigPanel.tsx', () => ({
    ConfigPanel: ({ onReset }: { onReset: () => void }) => (
        <button type="button" onClick={onReset}>Lazy configuration controls</button>
    ),
}));

vi.mock('../visualization/LossChart.tsx', () => ({
    LossChart: () => <div>Loss chart</div>,
}));

describe('Precision Lab content', () => {
    beforeEach(() => {
        useTrainingStore.getState().resetEvidence();
        useTrainingStore.setState({ pauseReason: null });
    });

    it('loads Configuration through its lazy boundary and forwards reset', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();

        render(<ConfigurationContent onReset={onReset} />);

        expect(screen.getByText('Loading configuration…')).toBeInTheDocument();
        await user.click(await screen.findByRole('button', { name: 'Lazy configuration controls' }));
        expect(onReset).toHaveBeenCalledTimes(1);
    });

    it('renders the training explanation surface with the loss evidence', async () => {
        useTrainingStore.setState({ pauseReason: 'diverged' });

        render(<LossContent />);

        expect(screen.getByText('Loss chart')).toBeInTheDocument();
        expect(await screen.findByText('Why did this happen?')).toBeInTheDocument();
        expect(screen.getByText('Training diverged')).toBeInTheDocument();
    });
});
