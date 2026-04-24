// ── Hyperparameter Panel ──
import { memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { LEARNING_RATES, REGULARIZATION_RATES, BATCH_SIZES } from '@nn-playground/shared';
import { LOSS_LABELS } from '@nn-playground/engine';
import type { LossType, OptimizerType, RegularizationType } from '@nn-playground/engine';
import { Tooltip } from '../common/Tooltip.tsx';

export const HyperparamPanel = memo(function HyperparamPanel() {
    // Granular selectors — only re-render when the specific field changes
    const learningRate = usePlaygroundStore((s) => s.training.learningRate);
    const lossType = usePlaygroundStore((s) => s.training.lossType);
    const optimizer = usePlaygroundStore((s) => s.training.optimizer);
    const batchSize = usePlaygroundStore((s) => s.training.batchSize);
    const regularization = usePlaygroundStore((s) => s.training.regularization);
    const regularizationRate = usePlaygroundStore((s) => s.training.regularizationRate);

    return (
        <div>
            {/* Learning rate */}
            <div className="control-row">
                <span className="control-label">Learning rate</span>
                <Tooltip content="Cause: larger learning rates take bigger weight updates. Effect: training can move faster, but too large can overshoot and make loss jump.">
                    <select
                        className="select"
                        aria-label="Learning rate"
                        value={learningRate}
                        onChange={(e) => usePlaygroundStore.getState().setLearningRate(Number(e.target.value))}
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
                <Tooltip content="Cause: loss defines what the model is punished for. Effect: cross entropy sharpens classification confidence, while MSE fits numeric distance.">
                    <select
                        className="select"
                        aria-label="Loss"
                        value={lossType}
                        onChange={(e) => usePlaygroundStore.getState().setLossType(e.target.value as LossType)}
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
                <Tooltip content="Cause: optimizers choose how gradients become weight updates. Effect: momentum and Adam can smooth or adapt steps compared with plain SGD.">
                    <select
                        className="select"
                        aria-label="Optimizer"
                        value={optimizer}
                        onChange={(e) => usePlaygroundStore.getState().setOptimizer(e.target.value as OptimizerType)}
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
                <Tooltip content="Cause: larger batches average more samples per update. Effect: the path is steadier, but each visible update reacts less often.">
                    <select
                        className="select"
                        aria-label="Batch size"
                        value={batchSize}
                        onChange={(e) => usePlaygroundStore.getState().setBatchSize(Number(e.target.value))}
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
                <Tooltip content="Cause: regularization penalizes large or unnecessary weights. Effect: the boundary often smooths out and generalizes better on noisy data.">
                    <select
                        className="select"
                        aria-label="Regularization"
                        value={regularization}
                        onChange={(e) => usePlaygroundStore.getState().setRegularization(e.target.value as RegularizationType)}
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
                    <Tooltip content="Cause: increasing the penalty pushes weights harder toward simpler solutions. Effect: too much can underfit and leave the boundary too flat.">
                        <select
                            className="select"
                            aria-label="Regularization rate"
                            value={regularizationRate}
                            onChange={(e) => usePlaygroundStore.getState().setRegularizationRate(Number(e.target.value))}
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
