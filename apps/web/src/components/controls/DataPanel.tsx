// ── Data Panel ──
import { memo } from 'react';
import type { DatasetId } from '@nn-playground/engine';
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
import { Tooltip } from '../common/Tooltip.tsx';
import { DATASET_TOOLTIPS } from '../../data/datasetInsights.ts';

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
    const currentProblemLabel = problemLabel(recipe.task.kind);

    const chooseDataset = (nextDataset: DatasetId) => {
        if (nextDataset === dataset) return;
        void commitRecipeEdit('data', (current) => switchDataset(current, nextDataset));
    };

    return (
        <div aria-busy={isLoading}>
            <LoadingState isLoading={isLoading} inline message="Generating data..." />
            {configError && configErrorSource === 'data' && (
                <div className="config-feedback config-feedback--error" role="alert">
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
                    className="chip-group"
                    aria-label="Classification datasets"
                    style={{ marginBottom: 8 }}
                >
                    {CLASSIFICATION_DATASETS.map((candidate) => (
                        <Tooltip key={candidate.id} content={DATASET_TOOLTIPS[candidate.id]}>
                            <button
                                type="button"
                                className={`chip ${dataset === candidate.id ? 'active' : ''}`}
                                onClick={() => chooseDataset(candidate.id)}
                                aria-pressed={dataset === candidate.id}
                            >
                                {candidate.label}
                            </button>
                        </Tooltip>
                    ))}
                </div>
            </div>

            <div className="control-row" style={{ alignItems: 'flex-start', marginBottom: 12 }}>
                <span className="control-label">Regression</span>
                <div className="chip-group" aria-label="Regression datasets">
                    {REGRESSION_DATASETS.map((candidate) => (
                        <Tooltip key={candidate.id} content={DATASET_TOOLTIPS[candidate.id]}>
                            <button
                                type="button"
                                className={`chip ${dataset === candidate.id ? 'active' : ''}`}
                                onClick={() => chooseDataset(candidate.id)}
                                aria-pressed={dataset === candidate.id}
                            >
                                {candidate.label}
                            </button>
                        </Tooltip>
                    ))}
                </div>
            </div>

            <div
                className="control-row"
                aria-label={`Dataset settings: ${sampleCount} samples, ${noise} noise, ${Math.round(trainFraction * 100)}% train`}
                aria-live="polite"
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
                <span className="control-label">Train ratio</span>
                <span className="control-value">{Math.round(trainFraction * 100)}%</span>
            </div>
            <Tooltip content="Cause: a higher train ratio gives the model more examples to fit. Effect: the test set gets smaller, so generalization estimates become noisier." block>
                <input
                    type="range"
                    min="10"
                    max="90"
                    value={Math.round(trainFraction * 100)}
                    onChange={(event) => {
                        const nextFraction = Number(event.target.value) / 100;
                        if (nextFraction === trainFraction) return;
                        void commitRecipeEdit(
                            'data',
                            (current) => setTrainFraction(current, nextFraction),
                        );
                    }}
                    aria-label="Train/test split percentage"
                />
            </Tooltip>

            <div
                className="control-row"
                style={{ marginTop: 8 }}
                aria-label={`Train/test split: ${trainCount} train, ${testCount} test`}
                aria-live="polite"
            >
                <span className="control-label">Split</span>
                <span className="control-value">Train {trainCount}</span>
                <span className="control-value">Test {testCount}</span>
            </div>

            <Tooltip content="Cause: reshuffle changes the existing data seed. Effect: the generated examples and train/test split are rebuilt deterministically without adding a new schema field." block>
                <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    style={{ marginTop: 8, width: '100%' }}
                    disabled={isLoading}
                    onClick={() => {
                        void commitRecipeEdit('data', reshuffleDataSeed);
                    }}
                >
                    Reshuffle split
                </button>
            </Tooltip>

            <div className="control-row" style={{ marginTop: 8 }}>
                <span className="control-label">Noise</span>
                <span className="control-value">{noise}</span>
            </div>
            <Tooltip content="Cause: more noise blurs class edges. Effect: training loss may flatten and test accuracy becomes harder to improve." block>
                <input
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
                    aria-label="Noise level"
                />
            </Tooltip>

            <Tooltip content="Cause: regenerating samples redraws the same dataset pattern with the current settings. Effect: you can check whether behavior is robust or seed-specific." block>
                <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    style={{ marginTop: 10, width: '100%' }}
                    onClick={onReset}
                    aria-label="Reset model & data"
                >
                    ↻ Reset model & data
                </button>
            </Tooltip>
        </div>
    );
});
