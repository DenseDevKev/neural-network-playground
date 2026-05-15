// ── Data Panel ──
import { memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import type { DatasetType } from '@nn-playground/engine';
import { LoadingState } from '../common/LoadingState.tsx';
import { Tooltip } from '../common/Tooltip.tsx';
import { DATASET_TOOLTIPS } from '../../data/datasetInsights.ts';

const CLASSIFICATION_DATASETS: { id: DatasetType; label: string }[] = [
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

const REGRESSION_DATASETS: { id: DatasetType; label: string }[] = [
    { id: 'reg-plane', label: 'Plane' },
    { id: 'reg-gauss', label: 'Multi-Gauss' },
];

const SAMPLE_COUNT_PRESETS = [100, 300, 600, 1000] as const;

interface DataPanelProps {
    onReset: () => void;
}

export const DataPanel = memo(function DataPanel({ onReset }: DataPanelProps) {
    const dataset = usePlaygroundStore((s) => s.data.dataset);
    const problemType = usePlaygroundStore((s) => s.data.problemType);
    const noise = usePlaygroundStore((s) => s.data.noise);
    const trainTestRatio = usePlaygroundStore((s) => s.data.trainTestRatio);
    const numSamples = usePlaygroundStore((s) => s.data.numSamples);
    const isLoading = useTrainingStore((s) => s.dataConfigLoading);
    const trainCount = useTrainingStore((s) => s.trainPoints.length);
    const testCount = useTrainingStore((s) => s.testPoints.length);
    const configError = useTrainingStore((s) => s.configError);
    const configErrorSource = useTrainingStore((s) => s.configErrorSource);
    const store = usePlaygroundStore;

    const datasets = problemType === 'regression' ? REGRESSION_DATASETS : CLASSIFICATION_DATASETS;

    const beginDataChange = () => useTrainingStore.getState().beginConfigChange('data');
    const retryDataChange = () => useTrainingStore.getState().retryConfigSync();

    return (
        <div>
            <LoadingState isLoading={isLoading} inline message="Generating data..." />
            {configError && configErrorSource === 'data' && (
                <div className="config-feedback config-feedback--error" role="alert">
                    <span>{configError}</span>
                    <button type="button" className="btn btn--ghost btn--sm" onClick={retryDataChange}>
                        Retry
                    </button>
                </div>
            )}

            {/* Problem type toggle */}
            <div className="control-row" style={{ marginBottom: 12 }}>
                <span className="control-label">Problem</span>
                <div className="chip-group">
                    <Tooltip content="Cause: classification uses class labels. Effect: the boundary view shows which region the model assigns to each class.">
                        <button
                            type="button"
                            className={`chip ${problemType === 'classification' ? 'active' : ''}`}
                            aria-pressed={problemType === 'classification'}
                            onClick={() => {
                                if (problemType === 'classification' && dataset === 'circle') return;
                                beginDataChange();
                                store.getState().setDataset('circle');
                            }}
                        >
                            Classification
                        </button>
                    </Tooltip>
                    <Tooltip content="Cause: regression predicts a continuous value. Effect: loss tracks distance from a surface instead of class mistakes.">
                        <button
                            type="button"
                            className={`chip ${problemType === 'regression' ? 'active' : ''}`}
                            aria-pressed={problemType === 'regression'}
                            onClick={() => {
                                if (problemType === 'regression' && dataset === 'reg-plane') return;
                                beginDataChange();
                                store.getState().setDataset('reg-plane');
                            }}
                        >
                            Regression
                        </button>
                    </Tooltip>
                </div>
            </div>

            {/* Dataset selector */}
            <div className="chip-group" style={{ marginBottom: 12 }}>
                {datasets.map((ds) => (
                    <Tooltip key={ds.id} content={DATASET_TOOLTIPS[ds.id]}>
                        <button
                            type="button"
                            className={`chip ${dataset === ds.id ? 'active' : ''}`}
                            onClick={() => {
                                if (dataset === ds.id) return;
                                beginDataChange();
                                store.getState().setDataset(ds.id);
                            }}
                            aria-pressed={dataset === ds.id}
                        >
                            {ds.label}
                        </button>
                    </Tooltip>
                ))}
            </div>

            <div
                className="control-row"
                aria-label={`Dataset settings: ${numSamples} samples, ${noise} noise, ${Math.round(trainTestRatio * 100)}% train`}
                aria-live="polite"
                style={{ marginBottom: 8 }}
            >
                <span className="control-label">Dataset lab</span>
                <span className="control-value">{numSamples.toLocaleString()} samples</span>
                <span className="control-value">{noise} noise</span>
            </div>

            <div className="control-row">
                <span className="control-label">Samples</span>
                <div className="chip-group" aria-label="Sample count presets">
                    {SAMPLE_COUNT_PRESETS.map((count) => (
                        <button
                            key={count}
                            type="button"
                            className={`chip ${numSamples === count ? 'active' : ''}`}
                            aria-pressed={numSamples === count}
                            disabled={isLoading}
                            onClick={() => {
                                if (numSamples === count) return;
                                beginDataChange();
                                store.getState().setNumSamples(count);
                            }}
                        >
                            {count} samples
                        </button>
                    ))}
                </div>
            </div>

            {/* Train/test ratio */}
            <div className="control-row">
                <span className="control-label">Train ratio</span>
                <span className="control-value">{Math.round(trainTestRatio * 100)}%</span>
            </div>
            <Tooltip content="Cause: a higher train ratio gives the model more examples to fit. Effect: the test set gets smaller, so generalization estimates become noisier." block>
                <input
                    type="range"
                    min="10"
                    max="90"
                    value={Math.round(trainTestRatio * 100)}
                    onChange={(e) => {
                        const nextRatio = Number(e.target.value) / 100;
                        if (nextRatio === trainTestRatio) return;
                        beginDataChange();
                        store.getState().setTrainTestRatio(nextRatio);
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
                        beginDataChange();
                        store.getState().reshuffleDataSeed();
                    }}
                >
                    Reshuffle split
                </button>
            </Tooltip>

            {/* Noise */}
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
                    onChange={(e) => {
                        const nextNoise = Number(e.target.value);
                        if (nextNoise === noise) return;
                        beginDataChange();
                        store.getState().setNoise(nextNoise);
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
                >
                    ↻ Reset model & data
                </button>
            </Tooltip>
        </div>
    );
});
