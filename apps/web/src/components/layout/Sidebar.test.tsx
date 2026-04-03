import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Sidebar } from './Sidebar';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';

function getPanelToggle(label: RegExp): HTMLButtonElement {
    const matches = screen.getAllByRole('button', { name: label });
    const panelToggle = matches.find((button) => button.classList.contains('panel__header'));
    if (!panelToggle) {
        throw new Error(`Panel toggle not found for label: ${label.toString()}`);
    }
    return panelToggle as HTMLButtonElement;
}

describe('Sidebar collapsible panel flow', () => {
    beforeEach(() => {
        const store = new Map<string, string>();
        Object.defineProperty(window, 'localStorage', {
            configurable: true,
            value: {
                getItem: (key: string) => store.get(key) ?? null,
                setItem: (key: string, value: string) => {
                    store.set(key, value);
                },
                removeItem: (key: string) => {
                    store.delete(key);
                },
                clear: () => {
                    store.clear();
                },
            },
        });

        usePlaygroundStore.setState((state) => ({
            network: {
                ...state.network,
                hiddenLayers: [4, 4],
            },
        }));
    });

    it('collapses the expanded sidebar panels when their headers are clicked', async () => {
        const user = userEvent.setup();

        render(<Sidebar onReset={vi.fn()} />);

        const presets = getPanelToggle(/Presets/);
        const data = getPanelToggle(/Data/);
        const features = getPanelToggle(/Features/);
        const network = getPanelToggle(/Network/);

        await user.click(presets);
        await user.click(data);
        await user.click(features);
        await user.click(network);

        expect(presets).toHaveAttribute('aria-expanded', 'false');
        expect(data).toHaveAttribute('aria-expanded', 'false');
        expect(features).toHaveAttribute('aria-expanded', 'false');
        expect(network).toHaveAttribute('aria-expanded', 'false');
    });

    it('expands a panel that is collapsed by default', async () => {
        const user = userEvent.setup();

        render(<Sidebar onReset={vi.fn()} />);

        const hyperparameters = getPanelToggle(/Hyperparameters/);

        expect(hyperparameters).toHaveAttribute('aria-expanded', 'false');

        await user.click(hyperparameters);

        expect(hyperparameters).toHaveAttribute('aria-expanded', 'true');
    });

    it('restores panel collapse state across remounts', async () => {
        const user = userEvent.setup();

        const { unmount } = render(<Sidebar onReset={vi.fn()} />);

        const data = getPanelToggle(/Data/);
        await user.click(data);

        expect(window.localStorage.getItem('panel-data')).toBe('false');

        unmount();
        render(<Sidebar onReset={vi.fn()} />);

        expect(getPanelToggle(/Data/)).toHaveAttribute('aria-expanded', 'false');
    });
});
