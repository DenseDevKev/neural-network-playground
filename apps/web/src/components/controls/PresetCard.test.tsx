import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { PresetCard } from './PresetCard';
import type { Preset } from '@nn-playground/shared';

const preset: Preset = {
    id: 'xor-hidden',
    title: 'XOR Needs Hidden Layers',
    description: 'XOR is not linearly separable.',
    learningGoal: 'See that hidden layers enable non-linear decision boundaries.',
    difficulty: 'beginner',
    config: {},
};

describe('PresetCard', () => {
    it('renders preset content and difficulty badge', () => {
        render(<PresetCard preset={preset} isSelected={false} onSelect={vi.fn()} />);

        expect(screen.getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers' })).toBeInTheDocument();
        expect(screen.getByText('XOR Needs Hidden Layers')).toBeInTheDocument();
        expect(screen.getByText('XOR is not linearly separable.')).toBeInTheDocument();
        expect(screen.getByText('See that hidden layers enable non-linear decision boundaries.')).toBeInTheDocument();
        expect(screen.getByText('Beginner')).toBeInTheDocument();
    });

    it('applies selected state styling and aria-pressed', () => {
        render(<PresetCard preset={preset} isSelected={true} onSelect={vi.fn()} />);

        const button = screen.getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers' });
        expect(button).toHaveClass('preset-card--selected');
        expect(button).toHaveAttribute('aria-pressed', 'true');
    });

    it('calls onSelect when clicked', async () => {
        const user = userEvent.setup();
        const onSelect = vi.fn();

        render(<PresetCard preset={preset} isSelected={false} onSelect={onSelect} />);

        await user.click(screen.getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers' }));

        expect(onSelect).toHaveBeenCalledWith(preset);
    });

    it('supports keyboard interaction', async () => {
        const user = userEvent.setup();
        const onSelect = vi.fn();

        render(<PresetCard preset={preset} isSelected={false} onSelect={onSelect} />);

        await user.tab();
        expect(screen.getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers' })).toHaveFocus();

        await user.keyboard('{Enter}');
        expect(onSelect).toHaveBeenCalledWith(preset);
    });
});
