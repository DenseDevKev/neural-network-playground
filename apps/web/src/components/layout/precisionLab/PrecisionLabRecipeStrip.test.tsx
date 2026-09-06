import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { PrecisionLabRecipeStrip } from './PrecisionLabRecipeStrip.tsx';

const model = Object.freeze({
    dataset: 'three-class-clusters',
    architecture: '2 -> 6 -> 6 -> 3',
    hiddenActivation: 'tanh',
    output: 'softmax',
    seed: '42',
    evaluationLabel: 'Evaluation 12 steps behind',
    tone: 'stale' as const,
});

describe('PrecisionLabRecipeStrip', () => {
    it('renders six textual recipe/evaluation facts in a labelled region', () => {
        render(<PrecisionLabRecipeStrip model={model} onEditRecipe={() => undefined} />);
        const region = screen.getByRole('region', { name: 'Recipe summary' });
        expect(region).toHaveAttribute('data-tone', 'stale');
        for (const label of ['Data', 'Architecture', 'Activation', 'Output', 'Seed', 'Evaluation']) {
            expect(screen.getByText(label)).toBeVisible();
        }
        for (const value of [
            'three-class-clusters',
            '2 -> 6 -> 6 -> 3',
            'tanh',
            'softmax',
            '42',
            'Evaluation 12 steps behind',
        ]) {
            expect(screen.getByText(value)).toBeVisible();
        }
    });

    it('calls the explicit recipe-edit command exactly once', async () => {
        const user = userEvent.setup();
        const onEditRecipe = vi.fn();
        render(<PrecisionLabRecipeStrip model={model} onEditRecipe={onEditRecipe} />);
        await user.click(screen.getByRole('button', { name: 'Edit recipe' }));
        expect(onEditRecipe).toHaveBeenCalledTimes(1);
    });
});
