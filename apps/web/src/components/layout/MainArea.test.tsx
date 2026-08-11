import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';
import { BoundaryContent, ConfigurationContent, MainArea } from './MainArea.tsx';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import {
    PREPARED_PRESETS,
} from '@nn-playground/shared';
import type { TrainingHook } from '../../hooks/useTraining.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import { getConceptById } from '../../concepts/conceptCatalog.ts';

vi.mock('../controls/TrainingControls.tsx', () => ({
    TrainingControls: () => <div>Training controls</div>,
}));

vi.mock('../visualization/NetworkGraph.tsx', () => ({
    NetworkGraph: () => <div>Network graph</div>,
}));

vi.mock('../visualization/DecisionBoundary.tsx', async (importOriginal) => ({
    ...(await importOriginal<typeof import('../visualization/DecisionBoundary.tsx')>()),
    DecisionBoundary: () => <div>Decision boundary</div>,
}));

vi.mock('../visualization/LossChart.tsx', () => ({
    LossChart: () => <div>Loss chart</div>,
}));

vi.mock('../visualization/ConfusionMatrix.tsx', () => ({
    ConfusionMatrix: () => <div>Confusion matrix</div>,
}));

vi.mock('../controls/ConfigPanel.tsx', () => ({
    ConfigPanel: ({ onReset }: { onReset: () => void }) => (
        <button type="button" onClick={onReset}>Lazy configuration controls</button>
    ),
}));

vi.mock('../common/ErrorBoundary.tsx', () => ({
    ErrorBoundary: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

function createTrainingMock(): TrainingHook {
    return {
        play: vi.fn(),
        pause: vi.fn(),
        step: vi.fn(),
        reset: vi.fn(),
        restoreCheckpoint: vi.fn(),
    };
}

describe('MainArea right-panel content', () => {
    beforeEach(() => {
        const classification = PREPARED_PRESETS.find((entry) => entry.id === 'xor-hidden')?.prepared;
        if (!classification) throw new Error('missing xor-hidden preset');
        usePlaygroundStore.setState({
            access: { status: 'ready', prepared: classification },
        });

        useTrainingStore.getState().resetEvidence();
        useTrainingStore.setState({
            status: 'idle',
            frameVersion: 0,
            trainPoints: [],
            testPoints: [],
            stepsPerFrame: 5,
            dataConfigLoading: false,
            networkConfigLoading: false,
            pendingConfigSource: null,
            configError: null,
            configErrorSource: null,
            configSyncNonce: 0,
            workerError: null,
            pauseReason: null,
        });
        useLayoutStore.setState({ audienceMode: 'explore' });
    });

    it('renders the code export panel in the right panel without requiring a toggle', () => {
        render(<MainArea training={createTrainingMock()} />);

        // Code Export panel is always visible — no collapse needed.
        // CodeExportPanel owns detailed tab/code assertions in its component tests.
        expect(screen.getByText('Code Export')).toBeInTheDocument();
        expect(screen.getByText('Loading code export…')).toBeInTheDocument();
    });

    it('loads Configuration through the advanced lazy boundary and forwards reset', async () => {
        const user = userEvent.setup();
        const onReset = vi.fn();

        render(<ConfigurationContent onReset={onReset} />);

        expect(screen.getByText('Loading configuration…')).toBeInTheDocument();
        await user.click(await screen.findByRole('button', { name: 'Lazy configuration controls' }));
        expect(onReset).toHaveBeenCalledTimes(1);
    });

    it('renders all main visualization sections', () => {
        render(<MainArea training={createTrainingMock()} />);

        expect(screen.getByText('Decision boundary')).toBeInTheDocument();
        expect(screen.getByText('Loss chart')).toBeInTheDocument();
        expect(screen.getByText('Confusion matrix')).toBeInTheDocument();
        expect(screen.getByText('Training controls')).toBeInTheDocument();
    });

    it('updates the decision overlay explanation when controls change', async () => {
        const user = userEvent.setup();
        render(<MainArea training={createTrainingMock()} />);

        expect(screen.getByText(/output mode shows/i)).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Uncertain' }));
        expect(screen.getByText(/least sure/i)).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Errors' }));
        expect(screen.getByText(/training points whose predicted class does not match/i)).toBeInTheDocument();
    });

    it('publishes decision view toggles through the canonical editView transaction', async () => {
        const user = userEvent.setup();
        const editView = vi.spyOn(usePlaygroundStore.getState(), 'editView');
        render(<BoundaryContent />);

        await user.click(screen.getByRole('checkbox', { name: 'Show test data' }));
        await user.click(screen.getByRole('checkbox', { name: 'Discretize output' }));

        await waitFor(() => {
            expect(editView).toHaveBeenCalledTimes(2);
            const { access } = usePlaygroundStore.getState();
            expect(access.status === 'ready' ? access.prepared.document.view : null).toEqual({
                showTestData: true,
                discretizeOutput: true,
            });
        });
        editView.mockRestore();
    });

    it('places decision-boundary guidance above the existing controls', async () => {
        const user = userEvent.setup();
        render(<BoundaryContent />);

        expect(screen.getByLabelText('Decision overlay controls'))
            .toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Learn about Decision boundary' }));

        expect(screen.getByText(getConceptById('decision-boundary')?.plainDefinition ?? ''))
            .toBeInTheDocument();
    });

    it('does not describe a regression output field as a classification decision boundary', () => {
        const regression = PREPARED_PRESETS.find((entry) => entry.id === 'regression-plane')?.prepared;
        if (!regression) throw new Error('missing regression-plane preset');
        usePlaygroundStore.setState({
            access: { status: 'ready', prepared: regression },
        });

        render(<BoundaryContent />);

        expect(screen.getByLabelText('Decision overlay controls')).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Learn about Decision boundary' }))
            .not.toBeInTheDocument();
    });

    it('renders the training explanation surface with the loss chart', () => {
        useTrainingStore.setState({ pauseReason: 'diverged' });

        render(<MainArea training={createTrainingMock()} />);

        expect(screen.getByText('Why did this happen?')).toBeInTheDocument();
        expect(screen.getByText('Training diverged')).toBeInTheDocument();
    });
});
