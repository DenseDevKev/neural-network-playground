// ── Hyperparameter Panel ──
import { memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { LEARNING_RATES, REGULARIZATION_RATES, BATCH_SIZES } from '@nn-playground/shared';
import { LOSS_LABELS } from '@nn-playground/engine';
import type { LossType, OptimizerType, RegularizationType } from '@nn-playground/engine';
import { LoadingState } from '../common/LoadingState.tsx';
import { Tooltip } from '../common/Tooltip.tsx';

export const HyperparamPanel = memo(function HyperparamPanel() {
    // Granular selectors — only re-render when the specific field changes
    const learningRate = usePlaygroundStore((s) => s.training.learningRate);
    const lossType = usePlaygroundStore((s) => s.training.lossType);
    const optimizer = usePlaygroundStore((s) => s.training.optimizer);
    const batchSize = usePlaygroundStore((s) => s.training.batchSize);
    const regularization = usePlaygroundStore((s) => s.training.regularization);
    const regularizationRate = usePlaygroundStore((s) => s.training.regularizationRate);
    const isLoading = useTrainingStore((s) => s.trainingConfigLoading);
    const configError = useTrainingStore((s) => s.configError);
    const configErrorSource = useTrainingStore((s) => s.configErrorSource);

    const beginTrainingChange = () => useTrainingStore.getState().beginConfigChange('training');
    const retryTrainingChange = () => useTrainingStore.getState().retryConfigSync();

    return (
        <div>
            <LoadingState isLoading={isLoading} inline message="Updating training..." />
            {configError && configErrorSource === 'training' && (
                <div className="config-feedback config-feedback--error" role="alert">
                    <span>{configError}</span>
                    <button type="button" className="btn btn--ghost btn--sm" onClick={retryTrainingChange}>
                        Retry
                    </button>
                </div>
            )}

            {/* Learning rate */}
            <div className="control-row">
                <span className="control-label">Learning rate</span>
                <Tooltip content="Control how large each gradient update is">
                    <select
                        className="select"
                        aria-label="Learning rate"
                        value={learningRate}
                        onChange={(e) => {
                            beginTrainingChange();
                            usePlaygroundStore.getState().setLearningRate(Number(e.target.value));
                        }}
                    >
                        {LEARNING_RATES.map((lr) => (
                            <option key={lr} value={lr}>{lr}</option>
                        ))}
                    </select>
                </Tooltip>
            </div>

            {/* Loss */}
            <div className="control-row">
                <span className="control-label">Loss</span>
                <Tooltip content="Choose how training error is measured">
                    <select
                        className="select"
                        aria-label="Loss"
                        value={lossType}
                        onChange={(e) => {
                            beginTrainingChange();
                            usePlaygroundStore.getState().setLossType(e.target.value as LossType);
                        }}
                    >
                        {(Object.keys(LOSS_LABELS) as LossType[]).map((l) => (
                            <option key={l} value={l}>{LOSS_LABELS[l]}</option>
                        ))}
                    </select>
                </Tooltip>
            </div>

            {/* Optimizer */}
            <div className="control-row">
                <span className="control-label">Optimizer</span>
                <Tooltip content="Select the optimization algorithm used during training">
                    <select
                        className="select"
                        aria-label="Optimizer"
                        value={optimizer}
                        onChange={(e) => {
                            beginTrainingChange();
                            usePlaygroundStore.getState().setOptimizer(e.target.value as OptimizerType);
                        }}
                    >
                        <option value="sgd">SGD</option>
                        <option value="sgdMomentum">SGD + Momentum</option>
                        <option value="adam">Adam</option>
                    </select>
                </Tooltip>
            </div>

            {/* Batch size */}
            <div className="control-row">
                <span className="control-label">Batch size</span>
                <Tooltip content="Choose how many samples are processed per update">
                    <select
                        className="select"
                        aria-label="Batch size"
                        value={batchSize}
                        onChange={(e) => {
                            beginTrainingChange();
                            usePlaygroundStore.getState().setBatchSize(Number(e.target.value));
                        }}
                    >
                        {BATCH_SIZES.map((bs) => (
                            <option key={bs} value={bs}>{bs}</option>
                        ))}
                    </select>
                </Tooltip>
            </div>

            {/* Regularization */}
            <div className="control-row">
                <span className="control-label">Regularization</span>
                <Tooltip content="Apply a penalty to discourage overfitting">
                    <select
                        className="select"
                        aria-label="Regularization"
                        value={regularization}
                        onChange={(e) => {
                            beginTrainingChange();
                            usePlaygroundStore.getState().setRegularization(e.target.value as RegularizationType);
                        }}
                    >
                        <option value="none">None</option>
                        <option value="l1">L1</option>
                        <option value="l2">L2</option>
                    </select>
                </Tooltip>
            </div>

            {/* Regularization rate */}
            {regularization !== 'none' && (
                <div className="control-row">
                    <span className="control-label">Reg. rate</span>
                    <Tooltip content="Set the strength of the regularization penalty">
                        <select
                            className="select"
                            aria-label="Regularization rate"
                            value={regularizationRate}
                            onChange={(e) => {
                                beginTrainingChange();
                                usePlaygroundStore.getState().setRegularizationRate(Number(e.target.value));
                            }}
                        >
                            {REGULARIZATION_RATES.map((r) => (
                                <option key={r} value={r}>{r}</option>
                            ))}
                        </select>
                    </Tooltip>
                </div>
            )}
        </div>
    );
});
