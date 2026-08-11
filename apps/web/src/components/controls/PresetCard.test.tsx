import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { resolveRecipe } from '@nn-playground/shared';
import { PresetCard } from './PresetCard';
import { STATE_EFFECTS } from '../../copy/stateEffects.ts';

const preset = resolveRecipe({ id: 'xor-hidden', revision: 1 });
if (!preset) throw new Error('Missing test recipe xor-hidden@1');

describe('PresetCard', () => {
    it('renders only declared catalog metadata and no thumbnail assumption', () => {
        render(<PresetCard preset={preset} isSelected={false} onSelect={vi.fn()} />);

        expect(screen.getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers' })).toBeInTheDocument();
        expect(screen.getByText(preset.title)).toBeInTheDocument();
        expect(screen.getByText(preset.description)).toBeInTheDocument();
        expect(screen.getByText(preset.learningGoal!)).toBeInTheDocument();
        expect(screen.getByText('Beginner')).toBeInTheDocument();
        expect(screen.queryByRole('img')).not.toBeInTheDocument();
    });

    it('exposes exact selected state with native button semantics', () => {
        render(<PresetCard preset={preset} isSelected onSelect={vi.fn()} />);

        const button = screen.getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers' });
        expect(button).toHaveAttribute('type', 'button');
        expect(button).toHaveClass('preset-card--selected');
        expect(button).toHaveAttribute('aria-pressed', 'true');
    });

    it('returns the actual resolved catalog entry when clicked', async () => {
        const user = userEvent.setup();
        const onSelect = vi.fn();

        render(<PresetCard preset={preset} isSelected={false} onSelect={onSelect} />);

        await user.click(screen.getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers' }));

        expect(onSelect).toHaveBeenCalledWith(preset);
    });

    it('supports keyboard activation', async () => {
        const user = userEvent.setup();
        const onSelect = vi.fn();

        render(<PresetCard preset={preset} isSelected={false} onSelect={onSelect} />);

        await user.tab();
        expect(screen.getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers' })).toHaveFocus();

        await user.keyboard('{Enter}');
        expect(onSelect).toHaveBeenCalledWith(preset);
    });

    it('uses native disabled semantics and cannot be activated', async () => {
        const user = userEvent.setup();
        const onSelect = vi.fn();

        render(<PresetCard preset={preset} isSelected={false} disabled onSelect={onSelect} />);

        const button = screen.getByRole('button', { name: 'Apply preset: XOR Needs Hidden Layers' });
        expect(button).toBeDisabled();
        await user.click(button);
        expect(onSelect).not.toHaveBeenCalled();
    });

    it('directly describes each preset action from unique persistent non-live copy', () => {
        render(
            <>
                <PresetCard preset={preset} isSelected={false} onSelect={vi.fn()} />
                <PresetCard preset={preset} isSelected={false} onSelect={vi.fn()} />
            </>,
        );

        const buttons = screen.getAllByRole('button', {
            name: 'Apply preset: XOR Needs Hidden Layers',
        });
        const descriptionIds = buttons.map((button) => {
            expect(button).toHaveAccessibleDescription(STATE_EFFECTS['preset-apply']);
            const ids = (button.getAttribute('aria-describedby') ?? '')
                .split(/\s+/u)
                .filter(Boolean);
            expect(ids).toHaveLength(1);
            const description = document.getElementById(ids[0]);
            expect(description).toHaveTextContent(STATE_EFFECTS['preset-apply']);
            expect(description?.closest('[aria-live], [role="status"], [role="alert"]'))
                .toBeNull();
            return ids[0];
        });

        expect(new Set(descriptionIds).size).toBe(buttons.length);
        expect(
            [...document.querySelectorAll('.tooltip__content')]
                .filter((content) => content.textContent === STATE_EFFECTS['preset-apply']),
        ).toHaveLength(buttons.length);
    });
});
