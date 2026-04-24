// ── Data Panel ──
import { memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import type { DatasetType } from '@nn-playground/engine';
import { LoadingState } from '../common/LoadingState.tsx';
import { Tooltip } from '../common/Tooltip.tsx';

const CLASSIFICATION_DATASETS: { id: DatasetType; label: string }[] = [
    { id: 'circle', label: 'Circle' },
    { id: 'xor', label: 'XOR' },
    { id: 'gauss', label: 'Gaussian' },
    { id: 'spiral', label: 'Spiral' },
    { id: 'moons', label: 'Moons' },
    { id: 'checkerboard', label: 'Checker' },
    { id: 'rings', label: 'Rings' },
    { id: 'heart', label: 'Heart' },
];

const REGRESSION_DATASETS: { id: DatasetType; label: string }[] = [
    { id: 'reg-plane', label: 'Plane' },
    { id: 'reg-gauss', label: 'Multi-Gauss' },
];

interface DataPanelProps {
    onReset: () => void;
}

export const DataPanel = memo(function DataPanel({ onReset }: DataPanelProps) {
    const dataset = usePlaygroundStore((s) => s.data.dataset);
    const problemType = usePlaygroundStore((s) => s.data.problemType);
    const noise = usePlaygroundStore((s) => s.data.noise);
    const trainTestRatio = usePlaygroundStore((s) => s.data.trainTestRatio);
    const isLoading = useTrainingStore((s) => s.dataConfigLoading);
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
                    <Tooltip content="Switch to classification datasets">
                        <button
                            type="button"
                            className={`chip ${problemType === 'classification' ? 'active' : ''}`}
                            onClick={() => {
                                beginDataChange();
                                store.getState().setDataset('circle');
                            }}
                        >
                            Classification
                        </button>
                    </Tooltip>
                    <Tooltip content="Switch to regression datasets">
                        <button
                            type="button"
                            className={`chip ${problemType === 'regression' ? 'active' : ''}`}
                            onClick={() => {
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
                    <Tooltip key={ds.id} content={`Use the ${ds.label} dataset`}>
                        <button
                            type="button"
                            className={`chip ${dataset === ds.id ? 'active' : ''}`}
                            onClick={() => {
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

            {/* Train/test ratio */}
            <div className="control-row">
                <span className="control-label">Train ratio</span>
                <span className="control-value">{Math.round(trainTestRatio * 100)}%</span>
            </div>
            <Tooltip content="Control the percentage of samples used for training" block>
                <input
                    type="range"
                    min="10"
                    max="90"
                    value={Math.round(trainTestRatio * 100)}
                    onChange={(e) => {
                        beginDataChange();
                        store.getState().setTrainTestRatio(Number(e.target.value) / 100);
                    }}
                    aria-label="Train/test split percentage"
                />
            </Tooltip>

            {/* Noise */}
            <div className="control-row" style={{ marginTop: 8 }}>
                <span className="control-label">Noise</span>
                <span className="control-value">{noise}</span>
            </div>
            <Tooltip content="Adjust how much randomness is added to the dataset" block>
                <input
                    type="range"
                    min="0"
                    max="50"
                    value={noise}
                    onChange={(e) => {
                        beginDataChange();
                        store.getState().setNoise(Number(e.target.value));
                    }}
                    aria-label="Noise level"
                />
            </Tooltip>

            <Tooltip content="Reset the model and regenerate the current dataset with the latest settings" block>
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
