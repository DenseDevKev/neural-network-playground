import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TrainingProgressBar } from './TrainingProgressBar';

describe('TrainingProgressBar', () => {
    it('renders nothing when training is not running', () => {
        render(<TrainingProgressBar isTraining={false} currentEpoch={5} targetEpoch={20} />);

        expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    });

    it('renders percentage progress when a target epoch is provided', () => {
        render(<TrainingProgressBar isTraining={true} currentEpoch={5} targetEpoch={20} />);

        const progressBar = screen.getByRole('progressbar', { name: 'Training progress' });
        const fill = document.querySelector('.training-progress__fill');

        expect(progressBar).toHaveAttribute('aria-valuenow', '25');
        expect(fill).toHaveStyle({ width: '25%' });
    });

    it('renders an indeterminate state when no target epoch is available', () => {
        render(<TrainingProgressBar isTraining={true} currentEpoch={5} />);

        const progressBar = screen.getByRole('progressbar', { name: 'Training progress' });

        expect(progressBar).toHaveClass('training-progress--indeterminate');
        expect(progressBar).not.toHaveAttribute('aria-valuenow');
    });
});
