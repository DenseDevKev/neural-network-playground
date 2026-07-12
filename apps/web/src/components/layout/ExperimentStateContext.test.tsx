import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    type AppConfig,
    type LiveTrainingSignal,
    type PairedEvaluation,
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

function live(step: number): LiveTrainingSignal {
    return {
        model: { generationId: 3, revision: step, step, epoch: 4 },
        dataset: { generatorVersion: 1, datasetKey: 'd', trainCount: 210, testCount: 90 },
        objectiveKey: 'o',
        basis: { kind: 'mini-batch-ema', alpha: 0.1, latestBatchSize: 10, throughStep: step },
        dataLoss: 0.22,
    };
}

function evaluation(step: number): PairedEvaluation {
    return {
        evaluationId: 3,
        trigger: 'cadence',
        model: { generationId: 3, revision: step, step, epoch: 4 },
        dataset: { generatorVersion: 1, datasetKey: 'd', trainCount: 210, testCount: 90 },
        objectiveKey: 'o',
        train: {
            basis: { kind: 'full-split', split: 'train', sampleCount: 210, populationCount: 210 },
            values: { dataLoss: 0.22, accuracy: 0.9 },
        },
        test: {
            basis: { kind: 'full-split', split: 'test', sampleCount: 90, populationCount: 90 },
            values: { dataLoss: 0.31, accuracy: 0.8 },
        },
        objective: { regularizationPenalty: 0.01, trainTotalObjective: 0.23 },
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
            latestLiveSignal: null,
            latestEvaluation: null,
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
            latestLiveSignal: live(40),
        });

        render(<TopologyStateBadge />);

        expect(screen.getByLabelText('Topology state')).toHaveTextContent('Live Run');
    });

    it('labels topology and evidence as drifted when current recipe changes after training', () => {
        useTrainingStore.getState().markTrainedRecipe(makeConfig(), 'initialize');
        useTrainingStore.setState({ latestLiveSignal: live(24) });
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
        expect(screen.getByText('Boundary evidence belongs to trained model step 24; current recipe has drift.')).toBeInTheDocument();
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
            latestLiveSignal: live(128),
            latestEvaluation: evaluation(120),
        });

        render(<DiagnosticCockpitStrip />);

        expect(screen.getByRole('status', { name: 'Diagnostic cockpit state' })).toHaveTextContent(
            'Loss has batch trend through step 128 and full evaluation 3 at step 120 using all 210 train and 90 test samples.',
        );
        expect(screen.queryByText('focus pair')).not.toBeInTheDocument();
        expect(screen.getByText('Batch trend (EMA) 0.2200')).toBeInTheDocument();
        expect(screen.getByText('Train data loss (full split) 0.2200')).toBeInTheDocument();
        expect(screen.getByText('Test data loss (full split) 0.3100')).toBeInTheDocument();
    });

    it('states exact evaluation age instead of a global stale-metrics claim', () => {
        useTrainingStore.getState().markTrainedRecipe(makeConfig(), 'initialize');
        useLayoutStore.setState({
            view: 'run',
            activeEvidenceView: 'confusion',
            phase: 'run',
            activeTabRight: 'confusion',
        });
        useTrainingStore.setState({
            status: 'running',
            latestLiveSignal: live(128),
            latestEvaluation: evaluation(120),
        });

        render(
            <>
                <EvidenceContextLine view="Confusion" />
                <DiagnosticCockpitStrip />
            </>,
        );

        expect(screen.getAllByText('Evaluation age')).toHaveLength(2);
        expect(screen.getAllByText('Confusion uses full evaluation 3 at step 120 across all 90 test samples; the current model is at step 128.')).toHaveLength(2);
        expect(screen.getByRole('status', { name: 'Diagnostic cockpit state' })).toHaveTextContent(
            'Confusion uses full evaluation 3 at step 120 across all 90 test samples; the current model is at step 128.',
        );
    });

    it('labels mixed draft and snapshot state in the diagnostic cockpit', () => {
        useTrainingStore.getState().markTrainedRecipe(makeConfig(), 'initialize');
        useTrainingStore.setState({ latestLiveSignal: live(24) });
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
            'Topology shows the draft recipe while Boundary evidence belongs to trained model step 24.',
        );
    });
});
