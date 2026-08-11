import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'jest-axe';
import { Header } from './Header';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import { installLegacyTrainingSnapshotForTest } from '../../test/playgroundStoreTestUtils.ts';
import type { TrainingHook } from '../../hooks/useTraining.ts';
import {
    ADVANCED_TOOLS_REGION_ID,
    ADVANCED_TOOLS_TRIGGER_ID,
    type DrawerSurfaceId,
} from '../../productShell/shellTypes.ts';

function createTrainingMock(): Pick<TrainingHook, 'play' | 'pause'> {
    return {
        play: vi.fn(),
        pause: vi.fn(),
    };
}

function renderHeader({
    training = createTrainingMock(),
    openSurface = null,
    onToggleSurface = vi.fn(),
    advancedToolsOpen = false,
    onToggleAdvancedTools = vi.fn(),
}: {
    training?: Pick<TrainingHook, 'play' | 'pause'>;
    openSurface?: DrawerSurfaceId | null;
    onToggleSurface?: (surface: DrawerSurfaceId) => void;
    advancedToolsOpen?: boolean;
    onToggleAdvancedTools?: () => void;
} = {}) {
    return render(
        <Header
            training={training}
            openSurface={openSurface}
            onToggleSurface={onToggleSurface}
            advancedToolsOpen={advancedToolsOpen}
            onToggleAdvancedTools={onToggleAdvancedTools}
        />,
    );
}

describe('Header', () => {
    beforeEach(() => {
        window.localStorage.clear();
        useTrainingStore.getState().resetEvidence();
        useTrainingStore.setState({
            status: 'idle',
            trainPoints: [],
            testPoints: [],
            stepsPerFrame: 5,
            dataConfigLoading: false,
            networkConfigLoading: false,
            featuresConfigLoading: false,
            trainingConfigLoading: false,
            presetConfigLoading: false,
            pendingConfigSource: null,
            configError: null,
            configErrorSource: null,
            configSyncNonce: 0,
            pauseReason: null,
        });
        useLayoutStore.setState({
            view: 'build',
            activeRecipeSection: 'data',
            activeEvidenceView: 'boundary',
            audienceMode: 'explore',
            advancedToolsOpen: false,
            layout: 'dock',
            phase: 'build',
            activeTabLeft: 'data',
            activeTabRight: 'boundary',
        });
    });

    it('labels a newer batch trend separately from the paired full evaluation', () => {
        useTrainingStore.setState({
            latestLiveSignal: {
                model: { generationId: 4, revision: 1240, step: 1240, epoch: 12 },
                dataset: { generatorVersion: 1, datasetKey: 'd', trainCount: 210, testCount: 90 },
                objectiveKey: 'o',
                basis: { kind: 'mini-batch-ema', alpha: 0.1, latestBatchSize: 10, throughStep: 1240 },
                dataLoss: 0.1234,
            },
            latestEvaluation: {
                evaluationId: 31,
                trigger: 'cadence',
                model: { generationId: 4, revision: 1230, step: 1230, epoch: 11 },
                dataset: { generatorVersion: 1, datasetKey: 'd', trainCount: 210, testCount: 90 },
                objectiveKey: 'o',
                train: { basis: { kind: 'full-split', split: 'train', sampleCount: 210, populationCount: 210 }, values: { dataLoss: 0.2345, accuracy: 0.527 } },
                test: { basis: { kind: 'full-split', split: 'test', sampleCount: 90, populationCount: 90 }, values: { dataLoss: 0.5678, accuracy: 0.493 } },
                objective: { regularizationPenalty: 0.01, trainTotalObjective: 0.2445 },
            },
        });
        const removeLegacySnapshot = installLegacyTrainingSnapshotForTest({
            epoch: 99,
            trainLoss: 9,
            testLoss: 8,
        });
        try {
            renderHeader();

            expect(screen.getByText('0012')).toBeInTheDocument();
            expect(screen.getByText('0.1234')).toBeInTheDocument();
            expect(screen.getByText(/Batch trend \(EMA\).*step 1,240/i)).toBeInTheDocument();
            expect(screen.getByText(/Train data loss \(full split\).*step 1,230/i)).toBeInTheDocument();
            expect(screen.getByText(/Test data loss \(full split\).*step 1,230/i)).toBeInTheDocument();
            expect(screen.getByText('0.2345')).toBeInTheDocument();
            expect(screen.getByText('0.5678')).toBeInTheDocument();
            expect(screen.getByText('49.3%')).toBeInTheDocument();
            expect(screen.queryByText('9.0000')).not.toBeInTheDocument();
        } finally {
            removeLegacySnapshot();
        }
    });

    it('uses the primary header play button to start and pause training', async () => {
        const user = userEvent.setup();
        const training = createTrainingMock();

        const { rerender } = renderHeader({ training });

        await user.click(screen.getByRole('button', { name: 'Start training' }));
        expect(training.play).toHaveBeenCalledTimes(1);

        act(() => {
            useTrainingStore.setState({ status: 'running' });
        });
        rerender(
            <Header
                training={training}
                openSurface={null}
                onToggleSurface={vi.fn()}
                advancedToolsOpen={false}
                onToggleAdvancedTools={vi.fn()}
            />,
        );

        await user.click(screen.getByRole('button', { name: 'Pause training' }));
        expect(training.pause).toHaveBeenCalledTimes(1);
    });

    it('disables the primary start control while config sync is pending', async () => {
        const user = userEvent.setup();
        const training = createTrainingMock();
        useTrainingStore.setState({ pendingConfigSource: 'preset', presetConfigLoading: true });

        renderHeader({ training });

        const button = screen.getByRole('button', { name: 'Start training' });
        expect(button).toBeDisabled();

        await user.click(button);

        expect(training.play).not.toHaveBeenCalled();
    });

    it('renders only Build and Run as global workspace views', () => {
        renderHeader();

        const switcher = screen.getByRole('group', { name: 'Workspace view' });
        expect(switcher).toBeInTheDocument();
        expect(switcher).toHaveAccessibleDescription(
            'Build changes the recipe. Run trains and inspects it. Switching views does not start or reset training.',
        );

        expect(screen.getByRole('button', { name: /build/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /run/i })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'dock' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'focus' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'grid' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'split' })).not.toBeInTheDocument();
    });

    it('explains Build and Run by pointer, keyboard, and touch-safe activation', async () => {
        const user = userEvent.setup();
        renderHeader();

        const trigger = screen.getByRole('button', { name: 'About workspace views' });
        const contentId = trigger.getAttribute('aria-controls');
        const descriptionId = screen.getByRole('group', { name: 'Workspace view' })
            .getAttribute('aria-describedby');

        expect(trigger).toHaveAttribute('aria-expanded', 'false');
        expect(descriptionId).toBe(contentId);
        expect(Array.from(document.querySelectorAll('[id]')).filter(
            (element) => element.id === contentId,
        )).toHaveLength(1);
        expect(document.getElementById(contentId ?? '')).toHaveClass('sr-only');

        fireEvent.mouseEnter(trigger);
        const hoverExplanation = screen.getByRole('region', { name: 'Build and Run views' });
        expect(document.getElementById(contentId ?? '')).toBe(hoverExplanation);
        expect(hoverExplanation).toHaveStyle({ position: 'fixed' });
        expect(screen.getByRole('banner')).not.toContainElement(hoverExplanation);
        expect(document.body).toContainElement(hoverExplanation);
        expect(contentId).toMatch(/^forge-workspace-view-description-/);
        fireEvent.mouseLeave(trigger);
        expect(screen.queryByRole('region', { name: 'Build and Run views' })).not.toBeInTheDocument();
        expect(document.getElementById(contentId ?? '')).toBeInTheDocument();

        fireEvent.focus(trigger);
        expect(screen.getByRole('region', { name: 'Build and Run views' })).toBeInTheDocument();
        await user.keyboard('{Enter}');
        expect(trigger).toHaveAttribute('aria-expanded', 'true');
        fireEvent.blur(trigger);
        expect(screen.queryByRole('region', { name: 'Build and Run views' })).not.toBeInTheDocument();

        fireEvent.pointerDown(trigger, { pointerType: 'touch' });
        fireEvent.pointerUp(trigger, { pointerType: 'touch' });
        fireEvent.click(trigger);
        expect(trigger).toHaveAttribute('aria-expanded', 'true');
        const visibleExplanation = screen.getByRole('region', { name: 'Build and Run views' });
        expect(visibleExplanation).toBeVisible();
        expect(visibleExplanation).not.toHaveClass('sr-only');
        expect(visibleExplanation).toHaveTextContent(
            'Build changes the recipe. Run trains and inspects it. Switching views does not start or reset training.',
        );

        trigger.focus();
        await user.keyboard('{Escape}');
        expect(trigger).toHaveAttribute('aria-expanded', 'false');
        expect(screen.queryByRole('region', { name: 'Build and Run views' })).not.toBeInTheDocument();
        expect(trigger).toHaveFocus();
    });

    it('owns a unique persistent workspace description for every Header instance', () => {
        render(
            <>
                <Header
                    training={createTrainingMock()}
                    openSurface={null}
                    onToggleSurface={vi.fn()}
                    advancedToolsOpen={false}
                    onToggleAdvancedTools={vi.fn()}
                />
                <Header
                    training={createTrainingMock()}
                    openSurface={null}
                    onToggleSurface={vi.fn()}
                    advancedToolsOpen={false}
                    onToggleAdvancedTools={vi.fn()}
                />
            </>,
        );

        const groups = screen.getAllByRole('group', { name: 'Workspace view' });
        const triggers = screen.getAllByRole('button', { name: 'About workspace views' });
        const descriptionIds = groups.map((group) => group.getAttribute('aria-describedby'));

        expect(new Set(descriptionIds).size).toBe(2);
        groups.forEach((_, index) => {
            const descriptionId = descriptionIds[index];
            expect(descriptionId).toMatch(/^forge-workspace-view-description-/);
            expect(triggers[index]).toHaveAttribute('aria-controls', descriptionId);
            expect(document.getElementById(descriptionId ?? '')).toHaveTextContent(
                'Build changes the recipe. Run trains and inspects it. Switching views does not start or reset training.',
            );
        });
    });

    it('updates only workspace view state when Build or Run is clicked', async () => {
        const user = userEvent.setup();
        const training = {
            ...createTrainingMock(),
            reset: vi.fn(),
        };
        useTrainingStore.setState({ status: 'running' });
        const trainingStateBefore = useTrainingStore.getState();
        renderHeader({ training });

        await user.click(screen.getByRole('button', { name: /run/i }));
        expect(useLayoutStore.getState().view).toBe('run');

        await user.click(screen.getByRole('button', { name: /build/i }));
        expect(useLayoutStore.getState().view).toBe('build');
        expect(useTrainingStore.getState()).toBe(trainingStateBefore);
        expect(training.play).not.toHaveBeenCalled();
        expect(training.pause).not.toHaveBeenCalled();
        expect(training.reset).not.toHaveBeenCalled();
    });

    it('has no automated accessibility violations with the workspace explanation open', async () => {
        const user = userEvent.setup();
        const { baseElement } = renderHeader();

        await user.click(screen.getByRole('button', { name: 'About workspace views' }));

        expect((await axe(baseElement)).violations).toHaveLength(0);
    });

    it('changes audience mode through a described native select and announces only explicit choices', async () => {
        const user = userEvent.setup();
        renderHeader();

        const mode = screen.getByRole('combobox', { name: 'Audience mode' });
        expect(mode).toHaveValue('explore');
        expect(mode).toHaveAccessibleDescription(
            'Adds feature, hyperparameter, and confusion tools for guided experimentation. Mode changes visible tools only.',
        );
        expect(screen.getByText(
            'Adds feature, hyperparameter, and confusion tools for guided experimentation.',
        )).toBeInTheDocument();
        expect(screen.getByText('Mode changes visible tools only.')).toBeInTheDocument();
        expect(screen.getByRole('option', { name: 'Beginner' })).toHaveValue('beginner');
        expect(screen.getByRole('option', { name: 'Explore' })).toHaveValue('explore');
        expect(screen.getByRole('option', { name: 'Lab' })).toHaveValue('lab');
        expect(screen.getByRole('status', { name: 'Audience mode change' })).toBeEmptyDOMElement();

        await user.selectOptions(mode, 'lab');

        expect(useLayoutStore.getState()).toMatchObject({
            audienceMode: 'lab',
            advancedToolsOpen: true,
        });
        expect(screen.getByRole('status', { name: 'Audience mode change' }))
            .toHaveTextContent('Mode: Lab. Mode changes visible tools only.');
        const stored = JSON.parse(window.localStorage.getItem('nn-playground-layout') ?? '{}');
        expect(stored.state).toMatchObject({ audienceMode: 'lab', advancedToolsOpen: true });

        await user.selectOptions(mode, 'beginner');
        expect(useLayoutStore.getState()).toMatchObject({
            audienceMode: 'beginner',
            advancedToolsOpen: false,
        });
        expect(screen.getByRole('status', { name: 'Audience mode change' }))
            .toHaveTextContent('Mode: Beginner. Mode changes visible tools only.');
        expect(screen.getByText(
            'Keeps the core data, network, boundary, and loss tools visible with more guidance.',
        )).toBeInTheDocument();
    });

    it('opens only the retained drawer surfaces through stable top-bar controls', async () => {
        const user = userEvent.setup();
        const onToggleSurface = vi.fn();
        renderHeader({ onToggleSurface, openSurface: 'history' });

        expect(screen.getByRole('button', { name: 'History' })).toHaveAttribute('aria-pressed', 'true');

        await user.click(screen.getByRole('button', { name: 'Presets' }));
        await user.click(screen.getByRole('button', { name: 'Lessons' }));

        expect(onToggleSurface).toHaveBeenNthCalledWith(1, 'presets');
        expect(onToggleSurface).toHaveBeenNthCalledWith(2, 'lessons');
        expect(screen.getByRole('button', { name: 'Presets' })).toHaveAttribute(
            'id',
            'forge-surface-trigger-presets',
        );
        expect(screen.getByRole('button', { name: 'History' })).toHaveAttribute(
            'aria-controls',
            'forge-surface-history',
        );
        expect(screen.queryByRole('button', { name: 'More' })).not.toBeInTheDocument();
    });

    it('exposes Advanced Tools as a described controlled disclosure', async () => {
        const user = userEvent.setup();
        const onToggleAdvancedTools = vi.fn();
        const { rerender } = renderHeader({ onToggleAdvancedTools });

        const trigger = screen.getByRole('button', { name: 'Advanced Tools' });
        expect(trigger).toHaveAttribute('id', ADVANCED_TOOLS_TRIGGER_ID);
        expect(trigger).toHaveAttribute('aria-expanded', 'false');
        expect(trigger).toHaveAttribute('aria-controls', ADVANCED_TOOLS_REGION_ID);
        expect(trigger).toHaveAccessibleDescription(/shows configuration and diagnostic tools/i);

        await user.click(trigger);
        expect(onToggleAdvancedTools).toHaveBeenCalledTimes(1);

        rerender(
            <Header
                training={createTrainingMock()}
                openSurface={null}
                onToggleSurface={vi.fn()}
                advancedToolsOpen
                onToggleAdvancedTools={onToggleAdvancedTools}
            />,
        );
        expect(screen.getByRole('button', { name: 'Advanced Tools' }))
            .toHaveAttribute('aria-expanded', 'true');
    });

    it('shows the NN·FORGE brand name', () => {
        renderHeader();
        expect(screen.getByText('NN·FORGE')).toBeInTheDocument();
    });
});
