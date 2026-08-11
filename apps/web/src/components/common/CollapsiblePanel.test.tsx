import { StrictMode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CollapsiblePanel } from './CollapsiblePanel';

const sharedLocalStorageDescriptor = Object.getOwnPropertyDescriptor(window, 'localStorage');

if (!sharedLocalStorageDescriptor) {
    throw new Error('CollapsiblePanel tests require window.localStorage');
}

const storagePrototype = Object.getPrototypeOf(window.localStorage) as Storage;

describe('CollapsiblePanel', () => {
    beforeEach(() => {
        Object.defineProperty(window, 'localStorage', sharedLocalStorageDescriptor);
        window.localStorage.clear();
    });

    afterEach(() => {
        vi.unstubAllEnvs();
        vi.unstubAllGlobals();
        vi.restoreAllMocks();
        Object.defineProperty(window, 'localStorage', sharedLocalStorageDescriptor);
        window.localStorage.clear();
    });

    it.each(['', 'Network', 'network_config', '-network', 'network-', 'network--config'])(
        'rejects the invalid development storage ID %j before storage access',
        (storageId) => {
            const getItem = vi.spyOn(storagePrototype, 'getItem');
            vi.spyOn(console, 'error').mockImplementation(() => {});

            expect(() => render(
                <CollapsiblePanel storageId={storageId} title="Network">
                    <div>Panel content</div>
                </CollapsiblePanel>,
            )).toThrow(
                `Invalid CollapsiblePanel storageId "${storageId}". Expected lowercase kebab-case matching /^[a-z0-9]+(?:-[a-z0-9]+)*$/.`,
            );
            expect(getItem).not.toHaveBeenCalled();
        },
    );

    it.each([
        ['true', 'true'],
        ['false', 'false'],
    ])('uses an exact v2 %s value without reading or removing legacy state', (saved, expanded) => {
        window.localStorage.setItem('panel-v2-network', saved);
        window.localStorage.setItem('panel-old-title', saved === 'true' ? 'false' : 'true');
        const getItem = vi.spyOn(storagePrototype, 'getItem');
        const removeItem = vi.spyOn(storagePrototype, 'removeItem');

        render(
            <CollapsiblePanel storageId="network" title="Old Title">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        expect(screen.getByRole('button', { name: /Old Title/ })).toHaveAttribute(
            'aria-expanded',
            expanded,
        );
        expect(getItem).toHaveBeenCalledTimes(1);
        expect(getItem).toHaveBeenCalledWith('panel-v2-network');
        expect(removeItem).not.toHaveBeenCalled();
        expect(window.localStorage.getItem('panel-old-title')).toBe(
            saved === 'true' ? 'false' : 'true',
        );
    });

    it('falls back for invalid v2 state, removes it after commit, and never revives legacy state', () => {
        window.localStorage.setItem('panel-v2-network', 'invalid');
        window.localStorage.setItem('panel-old-title', 'true');
        const getItem = vi.spyOn(storagePrototype, 'getItem');
        const removeItem = vi.spyOn(storagePrototype, 'removeItem');

        render(
            <CollapsiblePanel storageId="network" title="Old Title" defaultExpanded={false}>
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        expect(screen.getByRole('button', { name: /Old Title/ })).toHaveAttribute(
            'aria-expanded',
            'false',
        );
        expect(getItem).toHaveBeenCalledTimes(1);
        expect(getItem).toHaveBeenCalledWith('panel-v2-network');
        expect(removeItem).toHaveBeenCalledTimes(1);
        expect(removeItem).toHaveBeenCalledWith('panel-v2-network');
        expect(window.localStorage.getItem('panel-old-title')).toBe('true');
    });

    it('migrates valid legacy state by writing v2 before removing the old key', () => {
        window.localStorage.setItem('panel-old-title', 'false');
        const originalSetItem = window.localStorage.setItem.bind(window.localStorage);
        const originalRemoveItem = window.localStorage.removeItem.bind(window.localStorage);
        const operations: string[] = [];
        vi.spyOn(storagePrototype, 'setItem').mockImplementation((key, value) => {
            operations.push(`set:${key}:${value}`);
            originalSetItem(key, value);
        });
        vi.spyOn(storagePrototype, 'removeItem').mockImplementation((key) => {
            operations.push(`remove:${key}`);
            originalRemoveItem(key);
        });

        render(
            <CollapsiblePanel storageId="network" title="Old Title">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        expect(screen.getByRole('button', { name: /Old Title/ })).toHaveAttribute(
            'aria-expanded',
            'false',
        );
        expect(operations).toEqual([
            'set:panel-v2-network:false',
            'remove:panel-old-title',
        ]);
        expect(window.localStorage.getItem('panel-v2-network')).toBe('false');
        expect(window.localStorage.getItem('panel-old-title')).toBeNull();
    });

    it('removes invalid legacy state without writing a v2 value', () => {
        window.localStorage.setItem('panel-old-title', 'invalid');
        const setItem = vi.spyOn(storagePrototype, 'setItem');
        const removeItem = vi.spyOn(storagePrototype, 'removeItem');

        render(
            <CollapsiblePanel storageId="network" title="Old Title" defaultExpanded={false}>
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        expect(screen.getByRole('button', { name: /Old Title/ })).toHaveAttribute(
            'aria-expanded',
            'false',
        );
        expect(setItem).not.toHaveBeenCalled();
        expect(removeItem).toHaveBeenCalledTimes(1);
        expect(removeItem).toHaveBeenCalledWith('panel-old-title');
        expect(window.localStorage.getItem('panel-v2-network')).toBeNull();
        expect(window.localStorage.getItem('panel-old-title')).toBeNull();
    });

    it('retains valid legacy state when the v2 migration write fails', () => {
        window.localStorage.setItem('panel-old-title', 'false');
        const setItem = vi.spyOn(storagePrototype, 'setItem').mockImplementation(() => {
            throw new Error('quota exceeded');
        });
        const removeItem = vi.spyOn(storagePrototype, 'removeItem');

        expect(() => render(
            <CollapsiblePanel storageId="network" title="Old Title">
                <div>Panel content</div>
            </CollapsiblePanel>,
        )).not.toThrow();

        expect(setItem).toHaveBeenCalledTimes(1);
        expect(setItem).toHaveBeenCalledWith('panel-v2-network', 'false');
        expect(removeItem).not.toHaveBeenCalled();
        expect(window.localStorage.getItem('panel-v2-network')).toBeNull();
        expect(window.localStorage.getItem('panel-old-title')).toBe('false');
    });

    it('retains both keys when legacy removal fails after a successful v2 write', () => {
        window.localStorage.setItem('panel-old-title', 'false');
        const originalRemoveItem = window.localStorage.removeItem.bind(window.localStorage);
        const removeItem = vi.spyOn(storagePrototype, 'removeItem').mockImplementation((key) => {
            if (key === 'panel-old-title') {
                throw new Error('storage is busy');
            }
            originalRemoveItem(key);
        });

        expect(() => render(
            <CollapsiblePanel storageId="network" title="Old Title">
                <div>Panel content</div>
            </CollapsiblePanel>,
        )).not.toThrow();

        expect(removeItem).toHaveBeenCalledTimes(1);
        expect(window.localStorage.getItem('panel-v2-network')).toBe('false');
        expect(window.localStorage.getItem('panel-old-title')).toBe('false');
    });

    it('stops initialization after a v2 read failure', () => {
        window.localStorage.setItem('panel-old-title', 'false');
        const getItem = vi.spyOn(storagePrototype, 'getItem').mockImplementation(() => {
            throw new Error('read blocked');
        });
        const setItem = vi.spyOn(storagePrototype, 'setItem');
        const removeItem = vi.spyOn(storagePrototype, 'removeItem');

        expect(() => render(
            <CollapsiblePanel storageId="network" title="Old Title" defaultExpanded>
                <div>Panel content</div>
            </CollapsiblePanel>,
        )).not.toThrow();

        expect(screen.getByRole('button', { name: /Old Title/ })).toHaveAttribute(
            'aria-expanded',
            'true',
        );
        expect(getItem).toHaveBeenCalledTimes(1);
        expect(getItem).toHaveBeenCalledWith('panel-v2-network');
        expect(setItem).not.toHaveBeenCalled();
        expect(removeItem).not.toHaveBeenCalled();
    });

    it('stops initialization when the legacy read throws after an absent v2 key', () => {
        const originalGetItem = window.localStorage.getItem.bind(window.localStorage);
        const getItem = vi.spyOn(storagePrototype, 'getItem').mockImplementation((key) => {
            if (key === 'panel-old-title') {
                throw new Error('legacy read blocked');
            }
            return originalGetItem(key);
        });
        const setItem = vi.spyOn(storagePrototype, 'setItem');
        const removeItem = vi.spyOn(storagePrototype, 'removeItem');

        expect(() => render(
            <CollapsiblePanel storageId="network" title="Old Title" defaultExpanded={false}>
                <div>Panel content</div>
            </CollapsiblePanel>,
        )).not.toThrow();

        expect(getItem.mock.calls.map(([key]) => key)).toEqual([
            'panel-v2-network',
            'panel-old-title',
        ]);
        expect(setItem).not.toHaveBeenCalled();
        expect(removeItem).not.toHaveBeenCalled();
    });

    it('falls back and keeps toggling in memory when the localStorage property getter throws', async () => {
        const user = userEvent.setup();
        Object.defineProperty(window, 'localStorage', {
            configurable: true,
            get() {
                throw new Error('storage unavailable');
            },
        });

        try {
            expect(() => render(
                <CollapsiblePanel storageId="network" title="Network" defaultExpanded={false}>
                    <div>Panel content</div>
                </CollapsiblePanel>,
            )).not.toThrow();
            expect(screen.getByRole('button', { name: /Network/ })).toHaveAttribute(
                'aria-expanded',
                'false',
            );
            await user.click(screen.getByRole('button', { name: /Network/ }));
            expect(screen.getByRole('button', { name: /Network/ })).toHaveAttribute(
                'aria-expanded',
                'true',
            );
        } finally {
            Object.defineProperty(window, 'localStorage', sharedLocalStorageDescriptor);
        }
    });

    it('updates React state when a toggle persistence write throws', async () => {
        const user = userEvent.setup();
        vi.spyOn(storagePrototype, 'setItem').mockImplementation(() => {
            throw new Error('quota exceeded');
        });

        render(
            <CollapsiblePanel storageId="data" title="Data">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        const toggle = screen.getByRole('button', { name: /Data/ });
        await user.click(toggle);

        expect(toggle).toHaveAttribute('aria-expanded', 'false');
        expect(screen.getByText('Data collapsed')).toBeInTheDocument();
    });

    it('retains state and the initial v2 key when only the title changes', async () => {
        const user = userEvent.setup();
        window.localStorage.setItem('panel-old-title', 'false');
        const getItem = vi.spyOn(storagePrototype, 'getItem');
        const { rerender } = render(
            <CollapsiblePanel storageId="network" title="Old Title">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );
        expect(window.localStorage.getItem('panel-v2-network')).toBe('false');
        getItem.mockClear();

        rerender(
            <CollapsiblePanel storageId="network" title="New Title">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        const toggle = screen.getByRole('button', { name: /New Title/ });
        expect(toggle).toHaveAttribute('aria-expanded', 'false');
        expect(getItem).not.toHaveBeenCalled();
        await user.click(toggle);
        expect(window.localStorage.getItem('panel-v2-network')).toBe('true');
        expect(window.localStorage.getItem('panel-v2-new-title')).toBeNull();
        expect(window.localStorage.getItem('panel-new-title')).toBeNull();
    });

    it('rejects a development storageId change while mounted before new storage access', () => {
        const getItem = vi.spyOn(storagePrototype, 'getItem');
        vi.spyOn(console, 'error').mockImplementation(() => {});
        const { rerender } = render(
            <CollapsiblePanel storageId="network" title="Network">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );
        getItem.mockClear();

        expect(() => rerender(
            <CollapsiblePanel storageId="training" title="Network">
                <div>Panel content</div>
            </CollapsiblePanel>,
        )).toThrow(
            'CollapsiblePanel storageId cannot change while mounted. Remount with key={storageId}. Initial: "network"; received: "training".',
        );
        expect(getItem).not.toHaveBeenCalled();
    });

    it('uses an invalid production ID verbatim and keeps the initial mounted identity', async () => {
        vi.stubEnv('DEV', false);
        const user = userEvent.setup();
        window.localStorage.setItem('panel-v2-Network_ID', 'false');
        const { rerender } = render(
            <CollapsiblePanel storageId="Network_ID" title="Network">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        const toggle = screen.getByRole('button', { name: /Network/ });
        expect(toggle).toHaveAttribute('aria-expanded', 'false');

        rerender(
            <CollapsiblePanel storageId="Other_ID" title="Network renamed">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );
        await user.click(screen.getByRole('button', { name: /Network renamed/ }));

        expect(window.localStorage.getItem('panel-v2-Network_ID')).toBe('true');
        expect(window.localStorage.getItem('panel-v2-Other_ID')).toBeNull();
    });

    it('migrates only once after commit under StrictMode and never writes during render', () => {
        window.localStorage.setItem('panel-old-title', 'false');
        const originalSetItem = window.localStorage.setItem.bind(window.localStorage);
        const originalRemoveItem = window.localStorage.removeItem.bind(window.localStorage);
        const committedAtOperation: boolean[] = [];
        const setItem = vi.spyOn(storagePrototype, 'setItem').mockImplementation((key, value) => {
            committedAtOperation.push(document.querySelector('.panel__header') !== null);
            originalSetItem(key, value);
        });
        const removeItem = vi.spyOn(storagePrototype, 'removeItem').mockImplementation((key) => {
            committedAtOperation.push(document.querySelector('.panel__header') !== null);
            originalRemoveItem(key);
        });

        render(
            <StrictMode>
                <CollapsiblePanel storageId="network" title="Old Title">
                    <div>Panel content</div>
                </CollapsiblePanel>
            </StrictMode>,
        );

        expect(setItem).toHaveBeenCalledTimes(1);
        expect(removeItem).toHaveBeenCalledTimes(1);
        expect(committedAtOperation).toEqual([true, true]);
    });

    it('keeps same-title legacy collisions independent when IDs differ', async () => {
        const user = userEvent.setup();
        render(
            <>
                <CollapsiblePanel storageId="data-primary" title="Same Title">
                    <div>Primary content</div>
                </CollapsiblePanel>
                <CollapsiblePanel storageId="data-secondary" title={'Same   Title'}>
                    <div>Secondary content</div>
                </CollapsiblePanel>
            </>,
        );
        const toggles = screen.getAllByRole('button', { name: /Same Title/ });

        await user.click(toggles[0]!);
        expect(toggles[0]).toHaveAttribute('aria-expanded', 'false');
        expect(toggles[1]).toHaveAttribute('aria-expanded', 'true');
        expect(window.localStorage.getItem('panel-v2-data-primary')).toBe('false');
        expect(window.localStorage.getItem('panel-v2-data-secondary')).toBeNull();

        await user.click(toggles[1]!);
        expect(window.localStorage.getItem('panel-v2-data-primary')).toBe('false');
        expect(window.localStorage.getItem('panel-v2-data-secondary')).toBe('false');
    });

    it('toggles expanded state when the header is clicked', async () => {
        const user = userEvent.setup();

        render(
            <CollapsiblePanel storageId="data-toggle" title="Data">
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

    it('persists the panel state under its v2 storage ID', async () => {
        const user = userEvent.setup();

        render(
            <CollapsiblePanel storageId="hyperparameters" title="Hyperparameters">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        await user.click(screen.getByRole('button', { name: /Hyperparameters/ }));

        expect(window.localStorage.getItem('panel-v2-hyperparameters')).toBe('false');
        expect(window.localStorage.getItem('panel-hyperparameters')).toBeNull();
    });

    it('records panel toggles with an app-owned performance measure name', async () => {
        const user = userEvent.setup();
        const measure = vi.spyOn(performance, 'measure');

        render(
            <CollapsiblePanel storageId="data-performance" title="Data">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        await user.click(screen.getByRole('button', { name: /Data/ }));

        await waitFor(() => {
            expect(measure).toHaveBeenCalledWith(
                expect.stringMatching(/^nn-playground:panel-toggle:Data:/),
                expect.stringMatching(/^nn-playground:panel-toggle:Data:.*:start$/),
                expect.stringMatching(/^nn-playground:panel-toggle:Data:.*:end$/),
            );
        });
    });

    it('restores v2 saved state on mount', () => {
        window.localStorage.setItem('panel-v2-network-restore', 'false');

        render(
            <CollapsiblePanel storageId="network-restore" title="Network">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        expect(screen.getByRole('button', { name: /Network/ })).toHaveAttribute('aria-expanded', 'false');
    });

    it('does not mount lazy content until the panel is expanded', async () => {
        const user = userEvent.setup();
        let renderCount = 0;

        function LazyChild() {
            renderCount++;
            return <div>Lazy panel content</div>;
        }

        render(
            <CollapsiblePanel storageId="config-lazy" title="Config" defaultExpanded={false} lazyMount>
                <LazyChild />
            </CollapsiblePanel>,
        );

        expect(renderCount).toBe(0);
        expect(screen.queryByText('Lazy panel content')).not.toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: /Config/ }));

        expect(screen.getByText('Lazy panel content')).toBeInTheDocument();
        expect(renderCount).toBe(1);
    });

    it('keeps lazy content mounted after collapsing again', async () => {
        const user = userEvent.setup();

        render(
            <CollapsiblePanel storageId="code-export-lazy" title="Code Export" defaultExpanded={false} lazyMount>
                <div>Code tools</div>
            </CollapsiblePanel>,
        );

        const toggle = screen.getByRole('button', { name: /Code Export/ });

        await user.click(toggle);
        expect(screen.getByText('Code tools')).toBeInTheDocument();

        await user.click(toggle);

        expect(toggle).toHaveAttribute('aria-expanded', 'false');
        expect(screen.getByText('Code tools')).toBeInTheDocument();
    });

    it('mounts lazy content immediately when v2 state restores an expanded panel', () => {
        window.localStorage.setItem('panel-v2-inspection-lazy', 'true');

        render(
            <CollapsiblePanel storageId="inspection-lazy" title="Inspection" defaultExpanded={false} lazyMount>
                <div>Inspection tools</div>
            </CollapsiblePanel>,
        );

        expect(screen.getByRole('button', { name: /Inspection/ })).toHaveAttribute('aria-expanded', 'true');
        expect(screen.getByText('Inspection tools')).toBeInTheDocument();
    });

    it('applies the expected max-height transition for the collapse animation', () => {
        render(
            <CollapsiblePanel storageId="features-animation" title="Features">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        expect(document.querySelector('.panel__content')).toHaveStyle({
            transition: 'max-height 300ms cubic-bezier(0.16, 1, 0.3, 1)',
        });
    });

    it('renders badge and ARIA wiring', () => {
        render(
            <CollapsiblePanel storageId="network-accessibility" title="Network" badge={3}>
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

        vi.stubGlobal('ResizeObserver', ResizeObserverMock);

        const { unmount } = render(
            <CollapsiblePanel storageId="network-observer" title="Network">
                <div>Panel content</div>
            </CollapsiblePanel>,
        );

        unmount();

        expect(disconnect).toHaveBeenCalled();
    });
});
