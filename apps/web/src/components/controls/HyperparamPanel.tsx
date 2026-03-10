// ── Hyperparameter Panel ──
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { LEARNING_RATES, REGULARIZATION_RATES, BATCH_SIZES } from '@nn-playground/shared';
import { LOSS_LABELS } from '@nn-playground/engine';
import type { LossType, OptimizerType, RegularizationType } from '@nn-playground/engine';

export function HyperparamPanel() {
    const training = usePlaygroundStore((s) => s.training);
    const store = usePlaygroundStore;

    return (
        <div className="panel">
            <div className="panel__title">Hyperparameters</div>

            {/* Learning rate */}
            <div className="control-row">
                <span className="control-label">Learning rate</span>
                <select
                    className="select"
                    value={training.learningRate}
                    onChange={(e) => store.getState().setLearningRate(Number(e.target.value))}
                >
                    {LEARNING_RATES.map((lr) => (
                        <option key={lr} value={lr}>{lr}</option>
                    ))}
                </select>
            </div>

            {/* Loss */}
            <div className="control-row">
                <span className="control-label">Loss</span>
                <select
                    className="select"
                    value={training.lossType}
                    onChange={(e) => store.getState().setLossType(e.target.value as LossType)}
                >
                    {(Object.keys(LOSS_LABELS) as LossType[]).map((l) => (
                        <option key={l} value={l}>{LOSS_LABELS[l]}</option>
                    ))}
                </select>
            </div>

            {/* Optimizer */}
            <div className="control-row">
                <span className="control-label">Optimizer</span>
                <select
                    className="select"
                    value={training.optimizer}
                    onChange={(e) => store.getState().setOptimizer(e.target.value as OptimizerType)}
                >
                    <option value="sgd">SGD</option>
                    <option value="sgdMomentum">SGD + Momentum</option>
                    <option value="adam">Adam</option>
                </select>
            </div>

            {/* Batch size */}
            <div className="control-row">
                <span className="control-label">Batch size</span>
                <select
                    className="select"
                    value={training.batchSize}
                    onChange={(e) => store.getState().setBatchSize(Number(e.target.value))}
                >
                    {BATCH_SIZES.map((bs) => (
                        <option key={bs} value={bs}>{bs}</option>
                    ))}
                </select>
            </div>

            {/* Regularization */}
            <div className="control-row">
                <span className="control-label">Regularization</span>
                <select
                    className="select"
                    value={training.regularization}
                    onChange={(e) => store.getState().setRegularization(e.target.value as RegularizationType)}
                >
                    <option value="none">None</option>
                    <option value="l1">L1</option>
                    <option value="l2">L2</option>
                </select>
            </div>

            {/* Regularization rate */}
            {training.regularization !== 'none' && (
                <div className="control-row">
                    <span className="control-label">Reg. rate</span>
                    <select
                        className="select"
                        value={training.regularizationRate}
                        onChange={(e) => store.getState().setRegularizationRate(Number(e.target.value))}
                    >
                        {REGULARIZATION_RATES.map((r) => (
                            <option key={r} value={r}>{r}</option>
                        ))}
                    </select>
                </div>
            )}
        </div>
    );
}
