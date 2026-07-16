import { useState, type ReactNode } from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import type { AudienceMode } from '../../productShell/audienceProfiles.ts';
import type { EvidenceViewId } from '../../productShell/shellTypes.ts';
import { ADVANCED_TOOLS_REGION_ID } from '../../productShell/shellTypes.ts';
import { BuildRunShell } from './BuildRunShell.tsx';

const BUILD_MARKERS = {
    data: 'Data controls',
    network: 'Network controls',
    features: 'Feature controls',
    hyperparams: 'Hyperparameter controls',
    config: 'Configuration controls',
} as const;

const EVIDENCE_MARKERS: Record<EvidenceViewId, ReactNode> = {
    boundary: <div>Boundary evidence</div>,
    loss: <div>Loss evidence</div>,
    confusion: <div>Confusion evidence</div>,
    inspection: <div>Inspection evidence</div>,
    code: <div>Code evidence</div>,
    history: <div>History evidence</div>,
};

function shellProps(overrides: Partial<React.ComponentProps<typeof BuildRunShell>> = {}) {
    return {
        view: 'build' as const,
        status: 'idle' as const,
        activeEvidenceView: 'boundary' as EvidenceViewId,
        audienceMode: 'explore' as AudienceMode,
        advancedToolsOpen: false,
        onSelectEvidence: vi.fn(),
        openSurface: null,
        onCloseSurface: vi.fn(),
        recipeContent: <div>Recipe content</div>,
        runContent: <div>Run content</div>,
        dataContent: <div>{BUILD_MARKERS.data}</div>,
        networkContent: <div>{BUILD_MARKERS.network}</div>,
        featuresContent: <div>{BUILD_MARKERS.features}</div>,
        hyperparamContent: <div>{BUILD_MARKERS.hyperparams}</div>,
        configurationContent: <div>{BUILD_MARKERS.config}</div>,
        topologyContent: <div>Topology content</div>,
        transportContent: <div>Transport content</div>,
        evidenceContent: EVIDENCE_MARKERS,
        presetContent: <div>Preset drawer content</div>,
        lessonContent: <div>Lesson drawer content</div>,
        historyContent: <div>History drawer content</div>,
        ...overrides,
    };
}

const MATRIX = [
    {
        mode: 'beginner' as const,
        open: false,
        build: ['data', 'network'] as const,
        evidence: ['boundary', 'loss'] as const,
    },
    {
        mode: 'beginner' as const,
        open: true,
        build: ['data', 'network', 'features', 'hyperparams', 'config'] as const,
        evidence: ['boundary', 'loss', 'confusion', 'inspection', 'code'] as const,
    },
    {
        mode: 'explore' as const,
        open: false,
        build: ['data', 'network', 'features', 'hyperparams'] as const,
        evidence: ['boundary', 'loss', 'confusion'] as const,
    },
    {
        mode: 'explore' as const,
        open: true,
        build: ['data', 'network', 'features', 'hyperparams', 'config'] as const,
        evidence: ['boundary', 'loss', 'confusion', 'inspection', 'code'] as const,
    },
    {
        mode: 'lab' as const,
        open: false,
        build: ['data', 'network', 'features', 'hyperparams'] as const,
        evidence: ['boundary', 'loss', 'confusion'] as const,
    },
    {
        mode: 'lab' as const,
        open: true,
        build: ['data', 'network', 'features', 'hyperparams', 'config'] as const,
        evidence: ['boundary', 'loss', 'confusion', 'inspection', 'code'] as const,
    },
] as const;

describe('BuildRunShell', () => {
    it.each(MATRIX)(
        'mounts the exact $mode profile tools when advanced open is $open',
        ({ mode, open, build, evidence }) => {
            const buildRender = render(
                <BuildRunShell {...shellProps({ audienceMode: mode, advancedToolsOpen: open })} />,
            );

            expect(screen.getByText('Recipe content')).toBeInTheDocument();
            expect(screen.getByText('Topology content')).toBeInTheDocument();
            expect(screen.getByText('Transport content')).toBeInTheDocument();
            for (const [id, marker] of Object.entries(BUILD_MARKERS)) {
                const query = build.includes(id as never) ? screen.getByText(marker) : screen.queryByText(marker);
                if (build.includes(id as never)) expect(query).toBeInTheDocument();
                else expect(query).not.toBeInTheDocument();
            }

            buildRender.unmount();
            render(
                <BuildRunShell
                    {...shellProps({
                        view: 'run',
                        audienceMode: mode,
                        advancedToolsOpen: open,
                    })}
                />,
            );

            expect(screen.getByText('Run content')).toBeInTheDocument();
            expect(screen.getByText('Recipe content')).toBeInTheDocument();
            expect(screen.getByText('Topology content')).toBeInTheDocument();
            expect(screen.getByText('Transport content')).toBeInTheDocument();
            expect(screen.getAllByRole('tab').map((tab) => tab.textContent)).toEqual(
                evidence.map((id) => ({
                    boundary: 'Boundary',
                    loss: 'Loss',
                    confusion: 'Confusion',
                    inspection: 'Inspect',
                    code: 'Code',
                })[id]),
            );
        },
    );

    it('uses one stable controlled workspace region and explains the open disclosure', () => {
        const { rerender } = render(<BuildRunShell {...shellProps()} />);

        const collapsedRegion = screen.getByRole('region', { name: 'Workspace tools' });
        expect(collapsedRegion).toHaveAttribute('id', ADVANCED_TOOLS_REGION_ID);
        expect(screen.queryByText(/advanced tools are visible/i)).not.toBeInTheDocument();

        rerender(<BuildRunShell {...shellProps({ advancedToolsOpen: true })} />);

        const expandedRegion = screen.getByRole('region', { name: 'Workspace tools' });
        expect(expandedRegion).toHaveAttribute('id', ADVANCED_TOOLS_REGION_ID);
        expect(screen.getByText(/advanced tools are visible/i)).toBeInTheDocument();
    });

    it('resolves the legacy History evidence alias to Boundary', () => {
        render(
            <BuildRunShell
                {...shellProps({ view: 'run', activeEvidenceView: 'history' })}
            />,
        );

        expect(screen.getByRole('tab', { name: 'Boundary' })).toHaveAttribute('aria-selected', 'true');
        expect(screen.getByText('Boundary evidence')).toBeInTheDocument();
        expect(screen.queryByText('History evidence')).not.toBeInTheDocument();
    });

    it('keeps every evidence tab associated with the one mounted tabpanel', async () => {
        const user = userEvent.setup();

        function Harness() {
            const [activeEvidenceView, setActiveEvidenceView] = useState<EvidenceViewId>('boundary');
            return (
                <BuildRunShell
                    {...shellProps({
                        view: 'run',
                        activeEvidenceView,
                        onSelectEvidence: setActiveEvidenceView,
                    })}
                />
            );
        }

        render(<Harness />);

        const tabs = screen.getAllByRole('tab');
        const controlledIds = tabs.map((tab) => tab.getAttribute('aria-controls'));
        expect(new Set(controlledIds).size).toBe(1);
        expect(controlledIds[0]).toBeTruthy();
        expect(document.getElementById(controlledIds[0]!)).toBe(
            screen.getByRole('tabpanel', { name: 'Boundary' }),
        );

        await user.click(screen.getByRole('tab', { name: 'Loss' }));

        expect(screen.getByRole('tabpanel', { name: 'Loss' })).toHaveAttribute(
            'id',
            controlledIds[0],
        );
        expect(screen.queryByText('Boundary evidence')).not.toBeInTheDocument();
    });

    it('uses roving tab focus with arrows, Home, End, and wrapping', async () => {
        const user = userEvent.setup();
        const onSelectEvidence = vi.fn();

        function Harness() {
            const [activeEvidenceView, setActiveEvidenceView] = useState<EvidenceViewId>('boundary');
            return (
                <BuildRunShell
                    {...shellProps({
                        view: 'run',
                        activeEvidenceView,
                        onSelectEvidence: (view) => {
                            onSelectEvidence(view);
                            setActiveEvidenceView(view);
                        },
                    })}
                />
            );
        }

        render(<Harness />);
        const boundary = screen.getByRole('tab', { name: 'Boundary' });
        const loss = screen.getByRole('tab', { name: 'Loss' });
        const confusion = screen.getByRole('tab', { name: 'Confusion' });

        expect(boundary).toHaveAttribute('tabindex', '0');
        expect(loss).toHaveAttribute('tabindex', '-1');
        expect(confusion).toHaveAttribute('tabindex', '-1');

        boundary.focus();
        await user.keyboard('{ArrowRight}');
        expect(loss).toHaveFocus();
        expect(loss).toHaveAttribute('aria-selected', 'true');

        await user.keyboard('{End}');
        expect(confusion).toHaveFocus();
        await user.keyboard('{ArrowRight}');
        expect(boundary).toHaveFocus();
        await user.keyboard('{ArrowLeft}');
        expect(confusion).toHaveFocus();
        await user.keyboard('{Home}');
        expect(boundary).toHaveFocus();
        await user.keyboard('{ArrowDown}');
        expect(loss).toHaveFocus();
        await user.keyboard('{ArrowUp}');
        expect(boundary).toHaveFocus();
        expect(onSelectEvidence).toHaveBeenCalled();
    });

    it('focuses the drawer close control when a drawer opens', async () => {
        render(<BuildRunShell {...shellProps({ openSurface: 'presets' })} />);

        const drawer = screen.getByRole('dialog', { name: 'Presets' });
        expect(drawer).toHaveAttribute('id', 'forge-surface-presets');
        await waitFor(() => {
            expect(screen.getByRole('button', { name: 'Close Presets' })).toHaveFocus();
        });
    });
});
