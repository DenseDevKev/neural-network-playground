import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Sidebar } from './Sidebar';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { resolveRecipe } from '@nn-playground/shared';

const lazyPanelWait = { timeout: 10000 };

describe('Sidebar panels', () => {
    beforeEach(async () => {
        const recipe = resolveRecipe({ id: 'xor-hidden', revision: 1 });
        if (!recipe) throw new Error('Missing test recipe xor-hidden@1');
        const result = await usePlaygroundStore.getState().applyRecipe(recipe);
        expect(result.ok).toBe(true);
    });

    it('renders core panel headings (Presets, Data, Features)', async () => {
        render(<Sidebar onReset={vi.fn()} />);

        expect(screen.getByText('Presets')).toBeInTheDocument();
        expect(screen.getByText('Data')).toBeInTheDocument();
        expect(screen.getByText('Features')).toBeInTheDocument();
        await screen.findByText('Learning rate', undefined, lazyPanelWait);
    });

    it('shows the network panel with hidden-layer count badge', async () => {
        render(<Sidebar onReset={vi.fn()} />);
        expect(screen.getByText('Network (2)')).toBeInTheDocument();
        await screen.findByText('Learning rate', undefined, lazyPanelWait);
    });

    it('shows no count badge when there are no hidden layers', async () => {
        const recipe = resolveRecipe({ id: 'single-neuron', revision: 1 });
        if (!recipe) throw new Error('Missing test recipe single-neuron@1');
        const result = await usePlaygroundStore.getState().applyRecipe(recipe);
        expect(result.ok).toBe(true);
        render(<Sidebar onReset={vi.fn()} />);
        expect(screen.getByText('Network')).toBeInTheDocument();
        expect(screen.queryByText('Network (0)')).not.toBeInTheDocument();
        await screen.findByText('Learning rate', undefined, lazyPanelWait);
    });

    it('renders preset buttons without requiring any expansion toggle', async () => {
        render(<Sidebar onReset={vi.fn()} />);
        const presetButtons = screen.getAllByRole('button', { name: /Apply preset/i });
        expect(presetButtons.length).toBeGreaterThan(0);
        await screen.findByText('Learning rate', undefined, lazyPanelWait);
    });

    it('loads hyperparameters panel via Suspense', async () => {
        render(<Sidebar onReset={vi.fn()} />);
        expect(await screen.findByText('Learning rate', undefined, lazyPanelWait)).toBeInTheDocument();
    });

    it('renders the sidebar with accessible complementary role', async () => {
        render(<Sidebar onReset={vi.fn()} />);
        expect(screen.getByRole('complementary', { name: 'Configuration' })).toBeInTheDocument();
        await screen.findByText('Learning rate', undefined, lazyPanelWait);
    });
});
