import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
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

    it('keeps desktop metrics and exposes the paired classification evaluation in a native compact disclosure', async () => {
        const user = userEvent.setup();
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

            const metrics = screen.getByRole('group', { name: 'Training metrics' });
            expect(within(metrics).getByText('0012')).toBeInTheDocument();
            expect(within(metrics).getByText('0.1234')).toBeInTheDocument();
            expect(within(metrics).getByText(/Batch trend \(EMA\).*step 1,240/i)).toBeInTheDocument();
            expect(within(metrics).getByText(/Train data loss \(full split\).*step 1,230/i)).toBeInTheDocument();
            expect(within(metrics).getByText(/Test data loss \(full split\).*step 1,230/i)).toBeInTheDocument();
            expect(within(metrics).getByText('0.2345')).toBeInTheDocument();
            expect(within(metrics).getByText('0.5678')).toBeInTheDocument();
            expect(within(metrics).getByText('49.3%')).toBeInTheDocument();
            expect(screen.queryByText('9.0000')).not.toBeInTheDocument();

            expect(metrics).not.toHaveAttribute('aria-live');
            expect(metrics.closest('[aria-live], [role="status"], [role="alert"]')).toBeNull();

            const outcome = screen.getByRole('group', { name: 'Evaluation outcome' });
            const summary = within(outcome).getByText(
                'Step 1,230 · Test accuracy 49.3%',
                { selector: 'summary' },
            );
            const body = outcome.querySelector('.forge-compact-outcome__body');
            expect(outcome).not.toHaveAttribute('open');
            expect(body).not.toBeVisible();
            expect(outcome).not.toHaveAttribute('aria-live');
            expect(outcome.closest('[aria-live], [role="status"], [role="alert"]')).toBeNull();

            await user.click(summary);

            expect(outcome).toHaveAttribute('open');
            expect(body).toBeVisible();
            expect(outcome).toHaveTextContent('Full evaluation at step 1,230');
            expect(outcome).toHaveTextContent('Train data loss (full split) 0.2345');
            expect(outcome).toHaveTextContent('Test data loss (full split) 0.5678');
            expect(outcome).toHaveTextContent('Test accuracy 49.3%');

            act(() => {
                useTrainingStore.setState({
                    latestLiveSignal: {
                        ...useTrainingStore.getState().latestLiveSignal!,
                        model: { generationId: 4, revision: 1241, step: 1241, epoch: 13 },
                        basis: { kind: 'mini-batch-ema', alpha: 0.1, latestBatchSize: 10, throughStep: 1241 },
                        dataLoss: 0.1111,
                    },
                });
            });
            expect(screen.getByText('0013').closest('[aria-live], [role="status"], [role="alert"]')).toBeNull();
        } finally {
            removeLegacySnapshot();
        }
    });

    it('uses full-evaluation test loss as the compact regression outcome', async () => {
        const user = userEvent.setup();
        useTrainingStore.setState({
            latestLiveSignal: {
                model: { generationId: 7, revision: 85, step: 85, epoch: 2 },
                dataset: { generatorVersion: 1, datasetKey: 'regression', trainCount: 240, testCount: 60 },
                objectiveKey: 'mse',
                basis: { kind: 'mini-batch-ema', alpha: 0.1, latestBatchSize: 10, throughStep: 85 },
                dataLoss: 0.8765,
            },
            latestEvaluation: {
                evaluationId: 9,
                trigger: 'cadence',
                model: { generationId: 7, revision: 80, step: 80, epoch: 2 },
                dataset: { generatorVersion: 1, datasetKey: 'regression', trainCount: 240, testCount: 60 },
                objectiveKey: 'mse',
                train: { basis: { kind: 'full-split', split: 'train', sampleCount: 240, populationCount: 240 }, values: { dataLoss: 0.34567 } },
                test: { basis: { kind: 'full-split', split: 'test', sampleCount: 60, populationCount: 60 }, values: { dataLoss: 0.45678 } },
                objective: { regularizationPenalty: 0, trainTotalObjective: 0.34567 },
            },
        });
        renderHeader();

        const outcome = screen.getByRole('group', { name: 'Evaluation outcome' });
        const summary = within(outcome).getByText(
            'Step 80 · Test data loss 0.4568',
            { selector: 'summary' },
        );
        expect(outcome).not.toHaveTextContent('0.8765');
        expect(outcome).not.toHaveTextContent('Batch trend');

        await user.click(summary);

        expect(outcome).toHaveTextContent('Full evaluation at step 80');
        expect(outcome).toHaveTextContent('Train data loss (full split) 0.3457');
        expect(outcome).toHaveTextContent('Test data loss (full split) 0.4568');
        expect(outcome).not.toHaveTextContent('Test accuracy');
    });

    it('keeps a real zero-percent full-evaluation accuracy on the classification branch', () => {
        useTrainingStore.setState({
            latestEvaluation: {
                evaluationId: 10,
                trigger: 'cadence',
                model: { generationId: 8, revision: 25, step: 25, epoch: 1 },
                dataset: { generatorVersion: 1, datasetKey: 'classification', trainCount: 150, testCount: 150 },
                objectiveKey: 'bce',
                train: { basis: { kind: 'full-split', split: 'train', sampleCount: 150, populationCount: 150 }, values: { dataLoss: 1.11111, accuracy: 0.1 } },
                test: { basis: { kind: 'full-split', split: 'test', sampleCount: 150, populationCount: 150 }, values: { dataLoss: 2.22222, accuracy: 0 } },
                objective: { regularizationPenalty: 0, trainTotalObjective: 1.11111 },
            },
        });
        renderHeader();

        const outcome = screen.getByRole('group', { name: 'Evaluation outcome' });
        expect(within(outcome).getByText(
            'Step 25 · Test accuracy 0.0%',
            { selector: 'summary' },
        )).toBeInTheDocument();
        expect(outcome).not.toHaveTextContent('Test data loss 2.2222');
    });

    it('reports missing full evaluation without substituting the live batch EMA', () => {
        useTrainingStore.setState({
            latestLiveSignal: {
                model: { generationId: 3, revision: 18, step: 18, epoch: 1 },
                dataset: { generatorVersion: 1, datasetKey: 'd', trainCount: 210, testCount: 90 },
                objectiveKey: 'o',
                basis: { kind: 'mini-batch-ema', alpha: 0.1, latestBatchSize: 10, throughStep: 18 },
                dataLoss: 0.2468,
            },
            latestEvaluation: null,
        });
        renderHeader();

        const outcome = screen.getByRole('group', { name: 'Evaluation outcome' });
        expect(within(outcome).getByText('Not evaluated yet', { selector: 'summary' }))
            .toBeInTheDocument();
        expect(outcome).not.toHaveAttribute('open');
        expect(outcome).not.toHaveTextContent('0.2468');
        expect(outcome).not.toHaveTextContent('Batch trend');
        expect(outcome).not.toHaveAttribute('aria-live');
        expect(outcome.closest('[aria-live], [role="status"], [role="alert"]')).toBeNull();
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

    it('changes workspace profile through a described native select and announces only explicit choices', async () => {
        const user = userEvent.setup();
        renderHeader();

        expect(screen.getByText('Workspace')).toBeInTheDocument();
        const mode = screen.getByRole('combobox', { name: 'Workspace profile' });
        expect(mode).toHaveValue('explore');
        expect(mode).toHaveAccessibleDescription(
            'Adds feature, hyperparameter, and confusion tools for guided experimentation. Profiles change visible tools and guidance only.',
        );
        expect(screen.getByText(
            'Adds feature, hyperparameter, and confusion tools for guided experimentation.',
        )).toBeInTheDocument();
        expect(screen.getByText('Profiles change visible tools and guidance only.')).toBeInTheDocument();
        expect(screen.getByRole('option', { name: 'Beginner' })).toHaveValue('beginner');
        expect(screen.getByRole('option', { name: 'Explore' })).toHaveValue('explore');
        expect(screen.getByRole('option', { name: 'Lab' })).toHaveValue('lab');
        expect(screen.getByRole('status', { name: 'Workspace profile change' })).toBeEmptyDOMElement();
        const initialStored = JSON.parse(window.localStorage.getItem('nn-playground-layout') ?? '{}');
        expect(initialStored.version).toBe(0);
        expect(initialStored.state.audienceMode).toBe('explore');

        await user.selectOptions(mode, 'lab');

        expect(useLayoutStore.getState()).toMatchObject({
            audienceMode: 'lab',
            advancedToolsOpen: true,
        });
        expect(screen.getByRole('status', { name: 'Workspace profile change' }))
            .toHaveTextContent('Workspace profile: Lab. Profiles change visible tools and guidance only.');
        const stored = JSON.parse(window.localStorage.getItem('nn-playground-layout') ?? '{}');
        expect(stored.state.audienceMode).toBe('lab');
        expect(stored.state.advancedToolsOpen).toBe(true);

        await user.selectOptions(mode, 'beginner');
        expect(useLayoutStore.getState()).toMatchObject({
            audienceMode: 'beginner',
            advancedToolsOpen: false,
        });
        expect(screen.getByRole('status', { name: 'Workspace profile change' }))
            .toHaveTextContent('Workspace profile: Beginner. Profiles change visible tools and guidance only.');
        const beginnerStored = JSON.parse(window.localStorage.getItem('nn-playground-layout') ?? '{}');
        expect(beginnerStored.state.audienceMode).toBe('beginner');
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
