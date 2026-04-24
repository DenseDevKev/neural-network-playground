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

const DATASET_TOOLTIPS: Record<DatasetType, string> = {
    circle: 'Cause: circle data wraps one class around another. Effect: hidden layers or squared features help make a curved boundary.',
    xor: 'Cause: XOR alternates labels by quadrant. Effect: a straight boundary fails, so hidden layers have something meaningful to learn.',
    gauss: 'Cause: Gaussian blobs are mostly separable clusters. Effect: simple models learn quickly unless noise overlaps the classes.',
    spiral: 'Cause: spiral arms twist around each other. Effect: deeper networks usually need more training steps to untangle the boundary.',
    moons: 'Cause: moon shapes curve past each other. Effect: extra neurons help bend the decision boundary between the arcs.',
    checkerboard: 'Cause: checkerboard labels alternate in many small regions. Effect: the model needs more local bends and may train slowly.',
    rings: 'Cause: rings stack circular bands. Effect: curved features or hidden layers make the class transitions easier to fit.',
    heart: 'Cause: the heart outline has tight curves and a notch. Effect: low-capacity networks underfit the shape.',
    'reg-plane': 'Cause: plane regression is almost linear. Effect: a simple network can fit it without hidden layers.',
    'reg-gauss': 'Cause: multi-Gauss regression has several smooth bumps. Effect: hidden layers help approximate the changing surface.',
};

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
                    <button className="btn btn--ghost btn--sm" onClick={retryDataChange}>
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
                            className={`chip ${problemType === 'classification' ? 'active' : ''}`}
                            onClick={() => {
                                beginDataChange();
                                store.getState().setDataset('circle');
                            }}
                        >
                            Classification
                        </button>
                    </Tooltip>
                    <Tooltip content="Cause: regression predicts a continuous value. Effect: loss tracks distance from a surface instead of class mistakes.">
                        <button
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
                    <Tooltip key={ds.id} content={DATASET_TOOLTIPS[ds.id]}>
                        <button
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
            <Tooltip content="Cause: a higher train ratio gives the model more examples to fit. Effect: the test set gets smaller, so generalization estimates become noisier." block>
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
            <Tooltip content="Cause: more noise blurs class edges. Effect: training loss may flatten and test accuracy becomes harder to improve." block>
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

            <Tooltip content="Cause: regenerating samples redraws the same dataset pattern with the current settings. Effect: you can check whether behavior is robust or seed-specific." block>
                <button
                    className="btn btn--ghost btn--sm"
                    style={{ marginTop: 10, width: '100%' }}
                    onClick={onReset}
                >
                    ↻ Regenerate
                </button>
            </Tooltip>
        </div>
    );
});
