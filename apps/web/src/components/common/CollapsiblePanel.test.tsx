import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CollapsiblePanel } from './CollapsiblePanel';

describe('CollapsiblePanel', () => {
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
    });

    it('toggles expanded state when the header is clicked', async () => {
        const user = userEvent.setup();

        render(
            <CollapsiblePanel title="Data">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        const toggle = screen.getByRole('button', { name: /Data/ });
        const content = document.getElementById(toggle.getAttribute('aria-controls') ?? '');

        expect(toggle).toHaveAttribute('aria-expanded', 'true');

        await user.click(toggle);

        expect(toggle).toHaveAttribute('aria-expanded', 'false');
        expect(content).toHaveStyle({ maxHeight: '0px' });
        expect(screen.getByText('Data collapsed')).toBeInTheDocument();
    });

    it('persists the panel state in localStorage', async () => {
        const user = userEvent.setup();

        render(
            <CollapsiblePanel title="Hyperparameters">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        await user.click(screen.getByRole('button', { name: /Hyperparameters/ }));

        expect(window.localStorage.getItem('panel-hyperparameters')).toBe('false');
    });

    it('restores the saved state from localStorage on mount', () => {
        window.localStorage.setItem('panel-network', 'false');

        render(
            <CollapsiblePanel title="Network">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        expect(screen.getByRole('button', { name: /Network/ })).toHaveAttribute('aria-expanded', 'false');
    });

    it('clears invalid localStorage state on mount', () => {
        window.localStorage.setItem('panel-network', 'invalid');

        render(
            <CollapsiblePanel title="Network">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        expect(window.localStorage.getItem('panel-network')).toBeNull();
    });

    it('applies the expected max-height transition for the collapse animation', () => {
        render(
            <CollapsiblePanel title="Features">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        expect(document.querySelector('.panel__content')).toHaveStyle({
            transition: 'max-height 300ms cubic-bezier(0.16, 1, 0.3, 1)',
        });
    });

    it('renders badge and ARIA wiring', () => {
        render(
            <CollapsiblePanel title="Network" badge={3}>
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        const toggle = screen.getByRole('button', { name: /Network/ });
        const contentId = toggle.getAttribute('aria-controls');

        expect(toggle).toHaveAttribute('aria-expanded', 'true');
        expect(contentId).toBeTruthy();
        expect(document.getElementById(contentId!)).toBeInTheDocument();
        expect(screen.getByText('3')).toHaveClass('panel__badge');
    });

    it('disconnects ResizeObserver on unmount', () => {
        const disconnect = vi.fn();

        class ResizeObserverMock {
            observe() {}
            disconnect = disconnect;
        }

        Object.defineProperty(window, 'ResizeObserver', {
            configurable: true,
            value: ResizeObserverMock,
        });

        const { unmount } = render(
            <CollapsiblePanel title="Network">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        unmount();

        expect(disconnect).toHaveBeenCalled();
    });
});
