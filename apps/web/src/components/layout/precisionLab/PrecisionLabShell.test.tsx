import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { PrecisionLabShell, type PrecisionLabShellProps } from './PrecisionLabShell.tsx';

function props(overrides: Partial<PrecisionLabShellProps> = {}): PrecisionLabShellProps {
    return {
        view: 'build',
        status: 'idle',
        activeRecipeSection: 'data',
        activeEvidenceView: 'boundary',
        audienceMode: 'beginner',
        advancedToolsOpen: false,
        buildContextOpen: false,
        openSurface: null,
        onSelectRecipeSection: vi.fn(),
        onCloseRecipeSection: vi.fn(),
        onSelectEvidence: vi.fn(),
        onCloseSurface: vi.fn(),
        recipeStripContent: <div data-testid="recipe-strip">recipe</div>,
        runSummaryContent: <div>run summary</div>,
        buildContent: {
            data: <div>data context</div>,
            network: <div>network context</div>,
            features: <div>features context</div>,
            hyperparams: <div>hyperparams context</div>,
            config: <div>config context</div>,
        },
        topologyContent: <div data-testid="topology-content">topology</div>,
        boundaryRailContent: <div data-testid="live-boundary" data-decision-boundary-canvas>boundary</div>,
        selectionContent: <div>selection</div>,
        evidenceContent: {
            boundary: <div>boundary details</div>,
            loss: <div>loss details</div>,
            confusion: <div>confusion details</div>,
            inspection: <div>inspection details</div>,
            code: <div>code details</div>,
        },
        transportContent: <div>transport</div>,
        presetContent: <div>presets drawer</div>,
        lessonContent: <div>lessons drawer</div>,
        historyContent: <div>history drawer</div>,
        ...overrides,
    };
}

describe('PrecisionLabShell', () => {
    it('keeps topology and the pinned boundary mounted exactly once while evidence changes', async () => {
        const user = userEvent.setup();
        const onSelectEvidence = vi.fn();
        const { rerender } = render(<PrecisionLabShell {...props({ onSelectEvidence })} />);

        expect(screen.getByLabelText('Neural network')).toBeVisible();
        expect(screen.getByLabelText('Pinned decision boundary')).toBeVisible();
        expect(screen.getAllByTestId('live-boundary')).toHaveLength(1);
        expect(screen.getByRole('tabpanel')).toHaveTextContent('boundary details');

        await user.click(screen.getByRole('tab', { name: 'Loss' }));
        expect(onSelectEvidence).toHaveBeenCalledWith('loss');
        rerender(<PrecisionLabShell {...props({ activeEvidenceView: 'loss', onSelectEvidence })} />);
        expect(screen.getAllByTestId('topology-content')).toHaveLength(1);
        expect(screen.getAllByTestId('live-boundary')).toHaveLength(1);
        expect(screen.getByRole('tabpanel')).toHaveTextContent('loss details');
    });

    it('mounts only profile-visible Build tools and exposes the full union with Advanced Tools', () => {
        const { rerender } = render(<PrecisionLabShell {...props()} />);
        const buildTools = screen.getByRole('navigation', { name: 'Build tools' });
        expect(buildTools).toHaveTextContent('Data');
        expect(buildTools).toHaveTextContent('Network');
        expect(buildTools).not.toHaveTextContent('Features');
        expect(buildTools).not.toHaveTextContent('Hyperparameters');
        expect(buildTools).not.toHaveTextContent('Configuration');

        rerender(<PrecisionLabShell {...props({ advancedToolsOpen: true })} />);
        expect(buildTools).toHaveTextContent('Features');
        expect(buildTools).toHaveTextContent('Hyperparameters');
        expect(buildTools).toHaveTextContent('Configuration');
        expect(screen.getByRole('tab', { name: 'Inspect' })).toBeVisible();
        expect(screen.getByRole('tab', { name: 'Code' })).toBeVisible();
    });

    it('uses one evidence tabpanel with roving focus for arrows, Home, and End', async () => {
        const user = userEvent.setup();
        const onSelectEvidence = vi.fn();
        render(<PrecisionLabShell {...props({
            audienceMode: 'lab',
            advancedToolsOpen: true,
            activeEvidenceView: 'loss',
            onSelectEvidence,
        })} />);

        const loss = screen.getByRole('tab', { name: 'Loss' });
        loss.focus();
        await user.keyboard('{ArrowRight}');
        expect(onSelectEvidence).toHaveBeenLastCalledWith('confusion');
        expect(screen.getByRole('tab', { name: 'Confusion' })).toHaveFocus();

        await user.keyboard('{End}');
        expect(onSelectEvidence).toHaveBeenLastCalledWith('code');
        expect(screen.getByRole('tab', { name: 'Code' })).toHaveFocus();

        await user.keyboard('{Home}');
        expect(onSelectEvidence).toHaveBeenLastCalledWith('boundary');
        expect(screen.getByRole('tab', { name: 'Boundary' })).toHaveFocus();
    });

    it('selects Build modules and renders the supplied context when disclosure is open', async () => {
        const user = userEvent.setup();
        const onSelectRecipeSection = vi.fn();
        const { rerender } = render(<PrecisionLabShell {...props({ onSelectRecipeSection })} />);
        await user.click(screen.getByRole('button', { name: 'Network' }));
        expect(onSelectRecipeSection).toHaveBeenCalledWith('network');

        rerender(<PrecisionLabShell {...props({
            activeRecipeSection: 'network',
            buildContextOpen: true,
            onSelectRecipeSection,
        })} />);
        expect(screen.getByRole('region', { name: 'Network context' })).toHaveTextContent('network context');
    });

    it.each(['close button', 'Escape'] as const)('closes Build context with %s and restores rail focus', async (command) => {
        const user = userEvent.setup();
        const onCloseRecipeSection = vi.fn();
        render(<PrecisionLabShell {...props({
            activeRecipeSection: 'network',
            buildContextOpen: true,
            onCloseRecipeSection,
        })} />);

        if (command === 'close button') {
            await user.click(screen.getByRole('button', { name: 'Close Network context' }));
        } else {
            screen.getByRole('region', { name: 'Network context' }).focus();
            await user.keyboard('{Escape}');
        }
        expect(onCloseRecipeSection).toHaveBeenCalledTimes(1);
        await waitFor(() => {
            expect(screen.getByRole('button', { name: 'Network' })).toHaveFocus();
        });
    });

    it.each([
        ['presets', 'Presets', 'presets drawer'],
        ['lessons', 'Lessons', 'lessons drawer'],
        ['history', 'History', 'history drawer'],
    ] as const)('preserves the %s drawer dialog and close command', async (surface, label, content) => {
        const user = userEvent.setup();
        const onCloseSurface = vi.fn();
        render(<PrecisionLabShell {...props({ openSurface: surface, onCloseSurface })} />);
        const dialog = screen.getByRole('dialog', { name: label });
        expect(dialog).toHaveTextContent(content);
        await user.click(screen.getByRole('button', { name: `Close ${label}` }));
        expect(onCloseSurface).toHaveBeenCalledTimes(1);
    });

    it('does not steal focus from the drawer trigger when a disclosed Build context remains open', () => {
        const trigger = document.createElement('button');
        trigger.textContent = 'History trigger';
        document.body.append(trigger);
        try {
            const { rerender } = render(<PrecisionLabShell {...props({ buildContextOpen: true, openSurface: 'history' })} />);
            trigger.focus();
            rerender(<PrecisionLabShell {...props({ buildContextOpen: true, openSurface: null })} />);
            expect(trigger).toHaveFocus();
        } finally {
            trigger.remove();
        }
    });

    it('keeps detailed Boundary evidence free of a second live decision-boundary canvas', () => {
        render(<PrecisionLabShell {...props()} />);
        const panel = screen.getByRole('tabpanel');
        expect(panel.querySelector('[data-decision-boundary-canvas]')).toBeNull();
        expect(screen.getAllByTestId('live-boundary')).toHaveLength(1);
    });
});
