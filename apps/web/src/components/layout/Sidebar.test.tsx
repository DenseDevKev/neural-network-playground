import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Sidebar } from './Sidebar';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';

describe('Sidebar panels', () => {
    beforeEach(() => {
        usePlaygroundStore.setState((state) => ({
            network: {
                ...state.network,
                hiddenLayers: [4, 4],
            },
        }));
    });

    it('renders core panel headings (Presets, Data, Features)', () => {
        render(<Sidebar onReset={vi.fn()} />);

        expect(screen.getByText('Presets')).toBeInTheDocument();
        expect(screen.getByText('Data')).toBeInTheDocument();
        expect(screen.getByText('Features')).toBeInTheDocument();
    });

    it('shows the network panel with hidden-layer count badge', () => {
        render(<Sidebar onReset={vi.fn()} />);
        expect(screen.getByText('Network (2)')).toBeInTheDocument();
    });

    it('shows no count badge when there are no hidden layers', () => {
        usePlaygroundStore.setState((state) => ({
            network: { ...state.network, hiddenLayers: [] },
        }));
        render(<Sidebar onReset={vi.fn()} />);
        expect(screen.getByText('Network')).toBeInTheDocument();
        expect(screen.queryByText('Network (0)')).not.toBeInTheDocument();
    });

    it('renders preset buttons without requiring any expansion toggle', () => {
        render(<Sidebar onReset={vi.fn()} />);
        const presetButtons = screen.getAllByRole('button', { name: /Apply preset/i });
        expect(presetButtons.length).toBeGreaterThan(0);
    });

    it('loads hyperparameters panel via Suspense', async () => {
        render(<Sidebar onReset={vi.fn()} />);
        expect(await screen.findByText('Learning rate')).toBeInTheDocument();
    });

    it('renders the sidebar with accessible complementary role', () => {
        render(<Sidebar onReset={vi.fn()} />);
        expect(screen.getByRole('complementary', { name: 'Configuration' })).toBeInTheDocument();
    });
});
