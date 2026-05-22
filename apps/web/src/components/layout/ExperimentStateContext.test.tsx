import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    type AppConfig,
} from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import {
    DiagnosticCockpitStrip,
    EvidenceContextLine,
    EvidenceFrame,
    TopologyStateBadge,
} from './ExperimentStateContext.tsx';

function makeConfig(overrides: Partial<AppConfig> = {}): AppConfig {
    return {
        data: overrides.data ?? { ...DEFAULT_DATA },
        features: overrides.features ?? { ...DEFAULT_FEATURES },
        network: overrides.network ?? {
            ...DEFAULT_NETWORK,
            inputSize: 2,
            hiddenLayers: [...DEFAULT_NETWORK.hiddenLayers],
            seed: DEFAULT_DATA.seed,
        },
        training: overrides.training ?? { ...DEFAULT_TRAINING },
        ui: overrides.ui ?? { showTestData: false, discretizeOutput: false },
    };
}

describe('ExperimentStateContext', () => {
    beforeEach(() => {
        usePlaygroundStore.setState(makeConfig());
        useTrainingStore.setState({
            status: 'idle',
            snapshot: null,
            trainedRecipeConfig: null,
            trainedRecipeRecordedAt: null,
            trainedRecipeSource: null,
            pendingConfigSource: null,
            testMetricsStale: false,
            workerError: null,
            configError: null,
        });
        useLayoutStore.setState({
            view: 'build',
            activeRecipeSection: 'data',
            activeEvidenceView: 'boundary',
            layout: 'dock',
            phase: 'build',
            activeTabLeft: 'data',
            activeTabRight: 'boundary',
        });
    });

    it('labels topology as a draft blueprint before a snapshot exists', () => {
        render(<TopologyStateBadge />);

        expect(screen.getByLabelText('Topology state')).toHaveTextContent('Draft Blueprint');
    });

    it('labels topology as live while training runs', () => {
        useTrainingStore.setState({
            status: 'running',
            snapshot: { step: 40, epoch: 2 } as any,
        });

        render(<TopologyStateBadge />);

        expect(screen.getByLabelText('Topology state')).toHaveTextContent('Live Run');
    });

    it('labels topology and evidence as drifted when current recipe changes after training', () => {
        useTrainingStore.getState().markTrainedRecipe(makeConfig(), 'initialize');
        useTrainingStore.setState({ snapshot: { step: 24, epoch: 1 } as any });
        usePlaygroundStore.setState({
            ...makeConfig(),
            network: {
                ...DEFAULT_NETWORK,
                inputSize: 2,
                hiddenLayers: [8],
                seed: DEFAULT_DATA.seed,
            },
        });

        render(
            <>
                <TopologyStateBadge />
                <EvidenceContextLine view="Boundary" />
            </>,
        );

        expect(screen.getByLabelText('Topology state')).toHaveTextContent('Drifted Recipe');
        expect(screen.getByText('Boundary evidence belongs to trained snapshot step 24; current recipe has drift.')).toBeInTheDocument();
        expect(screen.getByText('Shows decision regions and sample outcomes.')).toBeInTheDocument();
    });

    it('explains empty evidence before training', () => {
        render(<EvidenceContextLine view="Loss" />);

        expect(screen.getByText('Train to see Loss evidence.')).toBeInTheDocument();
        expect(screen.getByText('Empty')).toBeInTheDocument();
        expect(screen.getByText('Shows learning curves over time.')).toBeInTheDocument();
    });

    it('marks evidence unavailable when the worker has failed', () => {
        useTrainingStore.setState({ workerError: 'Worker channel closed unexpectedly.' });

        render(<EvidenceContextLine view="Inspection" />);

        expect(screen.getByText('Inspection evidence is unavailable until the worker reconnects.')).toBeInTheDocument();
        expect(screen.getByText('Unavailable')).toBeInTheDocument();
        expect(screen.getByText('Shows layer activations, gradients, and probes.')).toBeInTheDocument();
    });

    it('wraps evidence views in a shared frame without forcing their internal layout', () => {
        render(
            <EvidenceFrame view="Boundary">
                <div>Boundary plot content</div>
            </EvidenceFrame>,
        );

        expect(screen.getByRole('region', { name: 'Boundary evidence view' })).toBeInTheDocument();
        expect(screen.getByText('Boundary')).toBeInTheDocument();
        expect(screen.getByText('Plot-based')).toBeInTheDocument();
        expect(screen.getByText('Shows decision regions and sample outcomes.')).toBeInTheDocument();
        expect(screen.getByText('Boundary plot content')).toBeInTheDocument();
    });

    it('summarizes the live diagnostic cockpit around the selected evidence view', () => {
        useLayoutStore.setState({
            view: 'run',
            activeEvidenceView: 'loss',
            phase: 'run',
            activeTabRight: 'loss',
        });
        useTrainingStore.setState({
            status: 'running',
            snapshot: {
                step: 128,
                epoch: 4,
                trainLoss: 0.22,
                testLoss: 0.31,
                trainMetrics: { loss: 0.22, accuracy: 0.9 },
                testMetrics: { loss: 0.31, accuracy: 0.8 },
            } as any,
        });

        render(<DiagnosticCockpitStrip />);

        expect(screen.getByRole('status', { name: 'Diagnostic cockpit state' })).toHaveTextContent(
            'Topology and Loss are reading live run step 128.',
        );
        expect(screen.queryByText('focus pair')).not.toBeInTheDocument();
        expect(screen.getByText('train 0.2200')).toBeInTheDocument();
        expect(screen.getByText('test 0.3100')).toBeInTheDocument();
    });

    it('labels mixed draft and snapshot state in the diagnostic cockpit', () => {
        useTrainingStore.getState().markTrainedRecipe(makeConfig(), 'initialize');
        useTrainingStore.setState({ snapshot: { step: 24, epoch: 1, trainLoss: 0.4, testLoss: 0.5 } as any });
        usePlaygroundStore.setState({
            ...makeConfig(),
            network: {
                ...DEFAULT_NETWORK,
                inputSize: 2,
                hiddenLayers: [8],
                seed: DEFAULT_DATA.seed,
            },
        });

        render(<DiagnosticCockpitStrip />);

        expect(screen.getByRole('status', { name: 'Diagnostic cockpit state' })).toHaveTextContent(
            'Topology shows the draft recipe while Boundary evidence belongs to trained snapshot step 24.',
        );
    });
});
