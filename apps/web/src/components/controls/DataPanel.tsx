// ── Data Panel ──
import { memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import type { DatasetType } from '@nn-playground/engine';

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
    const store = usePlaygroundStore;

    const datasets = problemType === 'regression' ? REGRESSION_DATASETS : CLASSIFICATION_DATASETS;

    return (
        <div className="panel">
            <div className="panel__title">Data</div>

            {/* Problem type toggle */}
            <div className="control-row" style={{ marginBottom: 12 }}>
                <span className="control-label">Problem</span>
                <div className="chip-group">
                    <button
                        className={`chip ${problemType === 'classification' ? 'active' : ''}`}
                        onClick={() => store.getState().setDataset('circle')}
                    >
                        Classification
                    </button>
                    <button
                        className={`chip ${problemType === 'regression' ? 'active' : ''}`}
                        onClick={() => store.getState().setDataset('reg-plane')}
                    >
                        Regression
                    </button>
                </div>
            </div>

            {/* Dataset selector */}
            <div className="chip-group" style={{ marginBottom: 12 }}>
                {datasets.map((ds) => (
                    <button
                        key={ds.id}
                        className={`chip ${dataset === ds.id ? 'active' : ''}`}
                        onClick={() => store.getState().setDataset(ds.id)}
                        aria-pressed={dataset === ds.id}
                    >
                        {ds.label}
                    </button>
                ))}
            </div>

            {/* Train/test ratio */}
            <div className="control-row">
                <span className="control-label">Train ratio</span>
                <span className="control-value">{Math.round(trainTestRatio * 100)}%</span>
            </div>
            <input
                type="range"
                min="10"
                max="90"
                value={Math.round(trainTestRatio * 100)}
                onChange={(e) => store.getState().setTrainTestRatio(Number(e.target.value) / 100)}
                aria-label="Train/test split percentage"
            />

            {/* Noise */}
            <div className="control-row" style={{ marginTop: 8 }}>
                <span className="control-label">Noise</span>
                <span className="control-value">{noise}</span>
            </div>
            <input
                type="range"
                min="0"
                max="50"
                value={noise}
                onChange={(e) => store.getState().setNoise(Number(e.target.value))}
                aria-label="Noise level"
            />

            <button
                className="btn btn--ghost btn--sm"
                style={{ marginTop: 10, width: '100%' }}
                onClick={onReset}
            >
                ↻ Regenerate
            </button>
        </div>
    );
});
