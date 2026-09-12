// ── Data Panel ──
import { memo, useId, useMemo } from 'react';
import type { DatasetId } from '@nn-playground/engine';
import { deriveDatasetPreviewModel, type DatasetPreviewModel } from './datasetPreviewModel.ts';
import { DatasetPreviewCanvas } from './DatasetPreviewCanvas.tsx';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { commitRecipeEdit } from '../../store/commitRecipeEdit.ts';
import {
    reshuffleDataSeed,
    setNoise,
    setSampleCount,
    setTrainFraction,
    switchDataset,
} from '../../store/recipeEdits.ts';
import { LoadingState } from '../common/LoadingState.tsx';
import { ConceptHelp } from '../common/ConceptHelp.tsx';
import { Tooltip } from '../common/Tooltip.tsx';
import { DATASET_TOOLTIPS } from '../../data/datasetInsights.ts';
import { useAudienceGuidanceLevel } from '../../hooks/useAudienceGuidanceLevel.ts';
import { STATE_EFFECTS } from '../../copy/stateEffects.ts';

const CLASSIFICATION_DATASETS: { id: DatasetId; label: string }[] = [
    { id: 'circle', label: 'Circle' },
    { id: 'xor', label: 'XOR' },
    { id: 'gauss', label: 'Gaussian' },
    { id: 'spiral', label: 'Spiral' },
    { id: 'moons', label: 'Moons' },
    { id: 'checkerboard', label: 'Checker' },
    { id: 'rings', label: 'Rings' },
    { id: 'heart', label: 'Heart' },
    { id: 'three-class-clusters', label: 'Three-Class' },
];

const REGRESSION_DATASETS: { id: DatasetId; label: string }[] = [
    { id: 'reg-plane', label: 'Plane' },
    { id: 'reg-gauss', label: 'Multi-Gauss' },
];

const SAMPLE_COUNT_PRESETS = [100, 300, 600, 1000] as const;

interface DataPanelProps {
    onReset: () => void;
}

function problemLabel(kind: 'binary-classification' | 'multiclass-classification' | 'regression') {
    if (kind === 'binary-classification') return 'Binary classification';
    if (kind === 'multiclass-classification') return 'Multiclass classification';
    return 'Regression';
}

export const DataPanel = memo(function DataPanel({ onReset }: DataPanelProps) {
    const recipe = usePlaygroundStore((state) => state.access.status === 'ready'
        ? state.access.prepared.document.recipe
        : null);
    const isLoading = useTrainingStore((state) => state.dataConfigLoading);
    const trainCount = useTrainingStore((state) => state.trainPoints.length);
    const testCount = useTrainingStore((state) => state.testPoints.length);
    const configError = useTrainingStore((state) => state.configError);
    const configErrorSource = useTrainingStore((state) => state.configErrorSource);
    const guidanceLevel = useAudienceGuidanceLevel();
    const controlId = useId();
    const trainRatioId = `${controlId}-train-ratio`;
    const trainRatioOutputId = `${controlId}-train-ratio-output`;
    const noiseId = `${controlId}-noise`;
    const noiseOutputId = `${controlId}-noise-output`;
    const reshuffleEffectsId = `${controlId}-reshuffle-split-effects`;
    const trainingResetEffectsId = `${controlId}-training-reset-effects`;

    const dataSeed = recipe?.data.seed;
    const dataNoise = recipe?.data.noise;
    const previews = useMemo(() => {
        if (dataSeed === undefined || dataNoise === undefined) return new Map<DatasetId, DatasetPreviewModel>();
        return new Map([...CLASSIFICATION_DATASETS, ...REGRESSION_DATASETS].map(({ id }) => [
            id, deriveDatasetPreviewModel({ datasetId: id, seed: dataSeed, noise: dataNoise }),
        ]));
    }, [dataSeed, dataNoise]);

    const retryDataChange = () => useTrainingStore.getState().retryConfigSync();

    if (!recipe) {
        return (
            <div>
                <LoadingState isLoading={isLoading} inline message="Generating data..." />
                <div className="config-feedback config-feedback--error" role="alert">
                    No compatible version-2 experiment is active.
                </div>
            </div>
        );
    }

    const dataset = recipe.task.dataset;
    const { noise, trainFraction, sampleCount } = recipe.data;
    const trainPercent = Math.round(trainFraction * 100);
    const testPercent = 100 - trainPercent;
    const currentProblemLabel = problemLabel(recipe.task.kind);

    const chooseDataset = (nextDataset: DatasetId) => {
        if (nextDataset === dataset) return;
        void commitRecipeEdit('data', (current) => switchDataset(current, nextDataset));
    };

    return (
        <div aria-busy={isLoading}>
            <LoadingState isLoading={isLoading} inline announce={false} message="Generating data..." />
            {configError && configErrorSource === 'data' && (
                <div className="config-feedback config-feedback--error">
                    <span>{configError}</span>
                    <button type="button" className="btn btn--ghost btn--sm" onClick={retryDataChange}>
                        Retry
                    </button>
                </div>
            )}

            <div className="control-row" style={{ marginBottom: 12 }}>
                <span className="control-label">Problem</span>
                <span
                    className="control-value"
                    aria-label={`Problem type: ${currentProblemLabel}`}
                >
                    {currentProblemLabel}
                </span>
            </div>

            <div className="control-row" style={{ alignItems: 'flex-start' }}>
                <span className="control-label">Classification</span>
                <div
                    className="precision-dataset-grid"
                    aria-label="Classification datasets"
                    style={{ marginBottom: 8 }}
                >
                    {CLASSIFICATION_DATASETS.map((candidate) => (
                        <Tooltip key={candidate.id} content={DATASET_TOOLTIPS[candidate.id]}>
                            <button
                                type="button"
                                className={`precision-dataset-choice ${dataset === candidate.id ? 'is-active' : ''}`}
                                onClick={() => chooseDataset(candidate.id)}
                                aria-pressed={dataset === candidate.id}
                            >
                                <DatasetPreviewCanvas model={previews.get(candidate.id)!} />
                                <span>{candidate.label}</span>
                            </button>
                        </Tooltip>
                    ))}
                </div>
            </div>

            <div className="control-row" style={{ alignItems: 'flex-start', marginBottom: 12 }}>
                <span className="control-label">Regression</span>
                <div className="precision-dataset-grid" aria-label="Regression datasets">
                    {REGRESSION_DATASETS.map((candidate) => (
                        <Tooltip key={candidate.id} content={DATASET_TOOLTIPS[candidate.id]}>
                            <button
                                type="button"
                                className={`precision-dataset-choice ${dataset === candidate.id ? 'is-active' : ''}`}
                                onClick={() => chooseDataset(candidate.id)}
                                aria-pressed={dataset === candidate.id}
                            >
                                <DatasetPreviewCanvas model={previews.get(candidate.id)!} />
                                <span>{candidate.label}</span>
                            </button>
                        </Tooltip>
                    ))}
                </div>
            </div>

            <div
                className="control-row"
                aria-label={`Dataset settings: ${sampleCount} samples, ${noise} noise, ${trainPercent}% train`}
                style={{ marginBottom: 8 }}
            >
                <span className="control-label">Dataset lab</span>
                <span className="control-value">{sampleCount.toLocaleString()} samples</span>
                <span className="control-value">{noise} noise</span>
            </div>

            <div className="control-row">
                <span className="control-label">Samples</span>
                <div className="chip-group" aria-label="Sample count presets">
                    {SAMPLE_COUNT_PRESETS.map((count) => (
                        <button
                            key={count}
                            type="button"
                            className={`chip ${sampleCount === count ? 'active' : ''}`}
                            aria-pressed={sampleCount === count}
                            disabled={isLoading}
                            onClick={() => {
                                if (sampleCount === count) return;
                                void commitRecipeEdit(
                                    'data',
                                    (current) => setSampleCount(current, count),
                                );
                            }}
                        >
                            {count} samples
                        </button>
                    ))}
                </div>
            </div>

            <div className="control-row">
                <span className="control-label">
                    <label htmlFor={trainRatioId}>Train ratio</label>
                    <ConceptHelp
                        conceptId="train-test-split"
                        guidanceLevel={guidanceLevel}
                        className="concept-help--viewport-overlay"
                    />
                </span>
                <output
                    className="control-value"
                    id={trainRatioOutputId}
                    htmlFor={trainRatioId}
                >
                    {trainPercent}%
                </output>
            </div>
            <Tooltip content="Cause: a higher train ratio gives the model more examples to fit. Effect: the test set gets smaller, so generalization estimates become noisier." block>
                <input
                    id={trainRatioId}
                    type="range"
                    min="10"
                    max="90"
                    value={trainPercent}
                    onChange={(event) => {
                        const nextFraction = Number(event.target.value) / 100;
                        if (nextFraction === trainFraction) return;
                        void commitRecipeEdit(
                            'data',
                            (current) => setTrainFraction(current, nextFraction),
                        );
                    }}
                    aria-describedby={trainRatioOutputId}
                    aria-valuetext={`${trainPercent} percent training, ${testPercent} percent test`}
                />
            </Tooltip>

            <div
                className="control-row"
                style={{ marginTop: 8 }}
                aria-label={`Train/test split: ${trainCount} train, ${testCount} test`}
            >
                <span className="control-label">Split</span>
                <span className="control-value">Train {trainCount}</span>
                <span className="control-value">Test {testCount}</span>
            </div>

            <Tooltip content={STATE_EFFECTS['reshuffle-split']} block>
                <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    style={{ marginTop: 8, width: '100%' }}
                    disabled={isLoading}
                    aria-describedby={reshuffleEffectsId}
                    onClick={() => {
                        void commitRecipeEdit('data', reshuffleDataSeed);
                    }}
                >
                    Reshuffle split
                </button>
            </Tooltip>
            <span id={reshuffleEffectsId} className="sr-only">
                {STATE_EFFECTS['reshuffle-split']}
            </span>

            <div className="control-row" style={{ marginTop: 8 }}>
                <label className="control-label" htmlFor={noiseId}>Noise</label>
                <output
                    className="control-value"
                    id={noiseOutputId}
                    htmlFor={noiseId}
                >
                    {noise}%
                </output>
            </div>
            <Tooltip content="Cause: more noise blurs class edges. Effect: training loss may flatten and test accuracy becomes harder to improve." block>
                <input
                    id={noiseId}
                    type="range"
                    min="0"
                    max="50"
                    value={noise}
                    onChange={(event) => {
                        const nextNoise = Number(event.target.value);
                        if (nextNoise === noise) return;
                        void commitRecipeEdit(
                            'data',
                            (current) => setNoise(current, nextNoise),
                        );
                    }}
                    aria-describedby={noiseOutputId}
                    aria-valuetext={`${noise} percent noise`}
                />
            </Tooltip>

            <Tooltip content={STATE_EFFECTS['training-reset']} block>
                <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    style={{ marginTop: 10, width: '100%' }}
                    onClick={onReset}
                    aria-describedby={trainingResetEffectsId}
                >
                    <span aria-hidden="true">↻</span>{' '}
                    <span>Reset training</span>
                </button>
            </Tooltip>
            <span id={trainingResetEffectsId} className="sr-only">
                {STATE_EFFECTS['training-reset']}
            </span>
        </div>
    );
});
