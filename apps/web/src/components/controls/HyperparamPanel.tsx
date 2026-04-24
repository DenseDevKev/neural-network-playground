// ── Hyperparameter Panel ──
import { memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import {
    ADAM_BETA1_VALUES,
    ADAM_BETA2_VALUES,
    BATCH_SIZES,
    GRADIENT_CLIP_VALUES,
    HUBER_DELTA_VALUES,
    LEARNING_RATES,
    LR_SCHEDULE_GAMMA_VALUES,
    MOMENTUM_VALUES,
    REGULARIZATION_RATES,
} from '@nn-playground/shared';
import { ACTIVATION_LABELS, isLossCompatible, LOSS_LABELS } from '@nn-playground/engine';
import type {
    ActivationType,
    LossType,
    LRScheduleType,
    OptimizerType,
    RegularizationType,
    WeightInitType,
} from '@nn-playground/engine';
import { Tooltip } from '../common/Tooltip.tsx';

const OUTPUT_ACTIVATIONS: ActivationType[] = ['sigmoid', 'linear', 'tanh', 'relu', 'leakyRelu', 'elu', 'swish', 'softplus'];
const WEIGHT_INITS: Array<{ value: WeightInitType; label: string }> = [
    { value: 'xavier', label: 'Xavier' },
    { value: 'he', label: 'He' },
    { value: 'uniform', label: 'Uniform' },
    { value: 'zeros', label: 'Zeros' },
];

export const HyperparamPanel = memo(function HyperparamPanel() {
    // Granular selectors — only re-render when the specific field changes
    const learningRate = usePlaygroundStore((s) => s.training.learningRate);
    const lossType = usePlaygroundStore((s) => s.training.lossType);
    const optimizer = usePlaygroundStore((s) => s.training.optimizer);
    const batchSize = usePlaygroundStore((s) => s.training.batchSize);
    const regularization = usePlaygroundStore((s) => s.training.regularization);
    const regularizationRate = usePlaygroundStore((s) => s.training.regularizationRate);
    const momentum = usePlaygroundStore((s) => s.training.momentum);
    const gradientClip = usePlaygroundStore((s) => s.training.gradientClip);
    const adamBeta1 = usePlaygroundStore((s) => s.training.adamBeta1 ?? 0.9);
    const adamBeta2 = usePlaygroundStore((s) => s.training.adamBeta2 ?? 0.999);
    const huberDelta = usePlaygroundStore((s) => s.training.huberDelta ?? 1);
    const lrSchedule = usePlaygroundStore((s) => s.training.lrSchedule);
    const weightInit = usePlaygroundStore((s) => s.network.weightInit);
    const outputActivation = usePlaygroundStore((s) => s.network.outputActivation);

    const scheduleType = lrSchedule?.type ?? 'constant';
    const compatibleOutputActivations = OUTPUT_ACTIVATIONS.filter((act) => isLossCompatible(lossType, act));

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

            {/* LR schedule */}
            <div className="control-row">
                <span className="control-label">LR schedule</span>
                <Tooltip content="Shape the learning rate as training progresses">
                    <select
                        className="select"
                        aria-label="LR schedule"
                        value={scheduleType}
                        onChange={(e) => {
                            const type = e.target.value as LRScheduleType;
                            if (type === 'constant') {
                                usePlaygroundStore.getState().setLRSchedule(undefined);
                            } else if (type === 'step') {
                                usePlaygroundStore.getState().setLRSchedule({ type, stepSize: 100, gamma: 0.5 });
                            } else {
                                usePlaygroundStore.getState().setLRSchedule({ type, totalSteps: 1000, minLr: 0 });
                            }
                        }}
                    >
                        <option value="constant">Constant</option>
                        <option value="step">Step decay</option>
                        <option value="cosine">Cosine</option>
                    </select>
                </Tooltip>
            </div>

            {lrSchedule?.type === 'step' && (
                <>
                    <div className="control-row">
                        <span className="control-label">Step interval</span>
                        <Tooltip content="Apply the decay after this many updates">
                            <input
                                className="select"
                                type="number"
                                min="1"
                                step="1"
                                aria-label="Step schedule interval"
                                value={lrSchedule.stepSize}
                                onChange={(e) => {
                                    const raw = e.target.value;
                                    const stepSize = raw === '' ? 0 : Math.max(1, Math.trunc(Number(raw) || 1));
                                    usePlaygroundStore.getState().setLRSchedule({ ...lrSchedule, stepSize });
                                }}
                            />
                        </Tooltip>
                    </div>
                    <div className="control-row">
                        <span className="control-label">Step gamma</span>
                        <Tooltip content="Multiply the learning rate by this value at each interval">
                            <select
                                className="select"
                                aria-label="Step schedule gamma"
                                value={lrSchedule.gamma}
                                onChange={(e) => usePlaygroundStore.getState().setLRSchedule({
                                    ...lrSchedule,
                                    gamma: Number(e.target.value),
                                })}
                            >
                                {LR_SCHEDULE_GAMMA_VALUES.map((gamma) => (
                                    <option key={gamma} value={gamma}>{gamma}</option>
                                ))}
                            </select>
                        </Tooltip>
                    </div>
                </>
            )}

            {lrSchedule?.type === 'cosine' && (
                <>
                    <div className="control-row">
                        <span className="control-label">Cosine steps</span>
                        <Tooltip content="Anneal to the minimum learning rate over this many updates">
                            <input
                                className="select"
                                type="number"
                                min="1"
                                step="1"
                                aria-label="Cosine total steps"
                                value={lrSchedule.totalSteps}
                                onChange={(e) => {
                                    const raw = e.target.value;
                                    const totalSteps = raw === '' ? 0 : Math.max(1, Math.trunc(Number(raw) || 1));
                                    usePlaygroundStore.getState().setLRSchedule({ ...lrSchedule, totalSteps });
                                }}
                            />
                        </Tooltip>
                    </div>
                    <div className="control-row">
                        <span className="control-label">Min LR</span>
                        <Tooltip content="Lowest learning rate reached by the cosine schedule">
                            <select
                                className="select"
                                aria-label="Cosine minimum learning rate"
                                value={lrSchedule.minLr}
                                onChange={(e) => usePlaygroundStore.getState().setLRSchedule({
                                    ...lrSchedule,
                                    minLr: Number(e.target.value),
                                })}
                            >
                                {LEARNING_RATES.map((lr) => (
                                    <option key={lr} value={lr}>{lr}</option>
                                ))}
                                <option value={0}>0</option>
                            </select>
                        </Tooltip>
                    </div>
                </>
            )}

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

            {lossType === 'huber' && (
                <div className="control-row">
                    <span className="control-label">Huber delta</span>
                    <Tooltip content="Set the Huber loss transition point">
                        <select
                            className="select"
                            aria-label="Huber delta"
                            value={huberDelta}
                            onChange={(e) => usePlaygroundStore.getState().setHuberDelta(Number(e.target.value))}
                        >
                            {HUBER_DELTA_VALUES.map((delta) => (
                                <option key={delta} value={delta}>{delta}</option>
                            ))}
                        </select>
                    </Tooltip>
                </div>
            )}

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

            {/* Momentum */}
            <div className="control-row">
                <span className="control-label">Momentum</span>
                <Tooltip content="Set the momentum coefficient used by SGD + Momentum">
                    <select
                        className="select"
                        aria-label="Momentum"
                        value={momentum}
                        onChange={(e) => usePlaygroundStore.getState().setMomentum(Number(e.target.value))}
                    >
                        {MOMENTUM_VALUES.map((m) => (
                            <option key={m} value={m}>{m}</option>
                        ))}
                    </select>
                </Tooltip>
            </div>

            {optimizer === 'adam' && (
                <>
                    <div className="control-row">
                        <span className="control-label">Adam β1</span>
                        <Tooltip content="Set Adam's first-moment decay">
                            <select
                                className="select"
                                aria-label="Adam beta 1"
                                value={adamBeta1}
                                onChange={(e) => usePlaygroundStore.getState().setAdamBetas(Number(e.target.value), adamBeta2)}
                            >
                                {ADAM_BETA1_VALUES.map((beta) => (
                                    <option key={beta} value={beta}>{beta}</option>
                                ))}
                            </select>
                        </Tooltip>
                    </div>
                    <div className="control-row">
                        <span className="control-label">Adam β2</span>
                        <Tooltip content="Set Adam's second-moment decay">
                            <select
                                className="select"
                                aria-label="Adam beta 2"
                                value={adamBeta2}
                                onChange={(e) => usePlaygroundStore.getState().setAdamBetas(adamBeta1, Number(e.target.value))}
                            >
                                {ADAM_BETA2_VALUES.map((beta) => (
                                    <option key={beta} value={beta}>{beta}</option>
                                ))}
                            </select>
                        </Tooltip>
                    </div>
                </>
            )}

            {/* Gradient clipping */}
            <div className="control-row">
                <span className="control-label">Gradient clipping</span>
                <Tooltip content="Limit the global gradient norm before each update">
                    <select
                        className="select"
                        aria-label="Gradient clipping"
                        value={gradientClip ?? 'none'}
                        onChange={(e) => {
                            const value = e.target.value;
                            usePlaygroundStore.getState().setGradientClip(value === 'none' ? null : Number(value));
                        }}
                    >
                        <option value="none">Off</option>
                        {GRADIENT_CLIP_VALUES.map((clip) => (
                            <option key={clip} value={clip}>{clip}</option>
                        ))}
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

            {/* Weight initialization */}
            <div className="control-row">
                <span className="control-label">Weight init</span>
                <Tooltip content="Choose how network weights are initialized after reset">
                    <select
                        className="select"
                        aria-label="Weight initialization"
                        value={weightInit}
                        onChange={(e) => usePlaygroundStore.getState().setWeightInit(e.target.value as WeightInitType)}
                    >
                        {WEIGHT_INITS.map((init) => (
                            <option key={init.value} value={init.value}>{init.label}</option>
                        ))}
                    </select>
                </Tooltip>
            </div>

            {/* Output activation */}
            <div className="control-row">
                <span className="control-label">Output act.</span>
                <Tooltip content="Choose the activation function for the output layer">
                    <select
                        className="select"
                        aria-label="Output activation"
                        value={outputActivation}
                        onChange={(e) => usePlaygroundStore.getState().setOutputActivation(e.target.value as ActivationType)}
                    >
                        {compatibleOutputActivations.map((act) => (
                            <option key={act} value={act}>{ACTIVATION_LABELS[act]}</option>
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
