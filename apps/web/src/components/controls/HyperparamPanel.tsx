// ── Canonical V2 Training / Objective Panel ──
import { memo, useEffect, useState } from 'react';
import { BATCH_SIZES, LEARNING_RATES } from '@nn-playground/shared';
import { commitRecipeEdit } from '../../store/commitRecipeEdit.ts';
import {
    setBatchSize,
    setGradientClipping,
    setLearningRate,
    setOptimizer,
    setPenalty,
    setRegressionDataLoss,
    setSchedule,
} from '../../store/recipeEdits.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { LoadingState } from '../common/LoadingState.tsx';
import { ConceptHelp } from '../common/ConceptHelp.tsx';
import { Tooltip } from '../common/Tooltip.tsx';
import { useAudienceGuidanceLevel } from '../../hooks/useAudienceGuidanceLevel.ts';

type NumericCommit = (value: number) => Promise<boolean>;

function ExactNumberInput({
    ariaLabel,
    max,
    min,
    step = 'any',
    value,
    onCommit,
}: {
    ariaLabel: string;
    max?: number;
    min?: number;
    step?: number | 'any';
    value: number;
    onCommit: NumericCommit;
}) {
    const [draft, setDraft] = useState(String(value));

    useEffect(() => {
        setDraft(String(value));
    }, [value]);

    const commit = (raw: string) => {
        if (raw.trim() === '') {
            setDraft(String(value));
            return;
        }
        const nextValue = Number(raw);
        if (nextValue === value) return;
        // A rejected transaction leaves the stored value untouched; snap the
        // draft back so the input cannot keep displaying a rejected number.
        void onCommit(nextValue).then((committed) => {
            if (!committed) setDraft(String(value));
        });
    };

    return (
        <input
            className="input"
            type="number"
            aria-label={ariaLabel}
            min={min}
            max={max}
            step={step}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onBlur={(event) => commit(event.currentTarget.value)}
            onKeyDown={(event) => {
                if (event.key === 'Enter') event.currentTarget.blur();
            }}
        />
    );
}

function formatRate(value: number): string {
    return Number.isFinite(value) ? String(value) : 'n/a';
}

function objectiveLabel(kind: string): string {
    switch (kind) {
        case 'binary-cross-entropy-with-logits':
            return 'Binary cross-entropy with logits';
        case 'categorical-cross-entropy-with-logits':
            return 'Categorical cross-entropy with logits';
        case 'mean-squared-error':
            return 'Mean squared error';
        case 'huber':
            return 'Huber';
        default:
            return 'Unavailable objective';
    }
}

const OPTIMIZER_EXPLANATIONS = {
    sgd: 'Plain SGD uses the current gradient directly. It is simple and easy to inspect step by step.',
    'sgd-momentum': 'Momentum remembers recent gradients, so updates can keep moving through shallow valleys.',
    adam: 'Adam adapts each weight update from moving averages, often making noisy gradients easier to train.',
} as const;

export const HyperparamPanel = memo(function HyperparamPanel() {
    const prepared = usePlaygroundStore((state) => state.access.status === 'ready'
        ? state.access.prepared
        : null);
    const isLoading = useTrainingStore((state) => state.trainingConfigLoading);
    const configError = useTrainingStore((state) => state.configError);
    const configErrorSource = useTrainingStore((state) => state.configErrorSource);
    const guidanceLevel = useAudienceGuidanceLevel();

    if (!prepared) {
        return (
            <div>
                <LoadingState isLoading={isLoading} inline message="Updating training..." />
                <div className="config-feedback config-feedback--error" role="alert">
                    No compatible version-2 experiment is active.
                </div>
            </div>
        );
    }

    const recipe = prepared.document.recipe;
    const { training, objective } = recipe;
    const { schedule, optimizer, gradientClipping } = training;
    const trainCount = Math.floor(recipe.data.sampleCount * recipe.data.trainFraction);
    const isRegression = recipe.task.kind === 'regression';
    const regressionLoss = objective.dataLoss.kind === 'huber'
        ? objective.dataLoss
        : { kind: 'mean-squared-error' as const };
    const learningRateIsQuickPick = LEARNING_RATES.some(
        (rate) => rate === training.learningRate,
    );
    const learningRateOptions = learningRateIsQuickPick
        ? LEARNING_RATES
        : [...LEARNING_RATES, training.learningRate].sort((left, right) => left - right);
    const batchSizeIsQuickPick = BATCH_SIZES.some(
        (batchSize) => batchSize === training.batchSize,
    );
    const batchSizeOptions = batchSizeIsQuickPick
        ? BATCH_SIZES
        : [...BATCH_SIZES, training.batchSize].sort((left, right) => left - right);
    const retryTrainingChange = () => useTrainingStore.getState().retryConfigSync();
    const commit = (edit: Parameters<typeof commitRecipeEdit>[1]) => (
        commitRecipeEdit('training', edit)
    );

    let scheduleSummary = `Uses ${formatRate(training.learningRate)} every update.`;
    if (schedule.kind === 'step') {
        scheduleSummary = `Starts at ${formatRate(training.learningRate)}; multiplies by ${formatRate(schedule.gamma)} every ${schedule.interval} updates.`;
    } else if (schedule.kind === 'cosine') {
        scheduleSummary = `Anneals from ${formatRate(training.learningRate)} to ${formatRate(schedule.minimumRate)} over ${schedule.totalSteps} updates.`;
    }

    return (
        <div aria-busy={isLoading}>
            <LoadingState isLoading={isLoading} inline announce={false} message="Updating training..." />
            {configError && configErrorSource === 'training' && (
                <div className="config-feedback config-feedback--error">
                    <span>{configError}</span>
                    <button
                        type="button"
                        className="btn btn--ghost btn--sm"
                        onClick={retryTrainingChange}
                    >
                        Retry
                    </button>
                </div>
            )}

            <div className="forge-section">
                <div className="forge-section__label">Objective &amp; schedule</div>

                <div className="control-row">
                    <span className="control-label">
                        Learning rate
                        <ConceptHelp
                            conceptId="learning-rate"
                            guidanceLevel={guidanceLevel}
                            className="concept-help--viewport-overlay"
                        />
                    </span>
                    <Tooltip content="Cause: larger learning rates take bigger weight updates. Effect: training can move faster, but too large can overshoot and make loss jump.">
                        <select
                            className="select"
                            aria-label="Learning rate"
                            value={training.learningRate}
                            onChange={(event) => void commit((current) => (
                                setLearningRate(current, Number(event.target.value))
                            ))}
                        >
                            {learningRateOptions.map((rate) => (
                                <option key={rate} value={rate}>
                                    {rate}{!learningRateIsQuickPick && rate === training.learningRate
                                        ? ' (current)'
                                        : ''}
                                </option>
                            ))}
                        </select>
                    </Tooltip>
                </div>

                <div className="control-row">
                    <span className="control-label">LR schedule</span>
                    <Tooltip content="Shape the learning rate as training progresses">
                        <select
                            className="select"
                            aria-label="LR schedule"
                            value={schedule.kind}
                            onChange={(event) => {
                                const kind = event.target.value;
                                if (kind === 'constant') {
                                    void commit((current) => setSchedule(current, { kind: 'constant' }));
                                } else if (kind === 'step') {
                                    void commit((current) => setSchedule(current, {
                                        kind: 'step',
                                        interval: 100,
                                        gamma: 0.5,
                                    }));
                                } else {
                                    void commit((current) => setSchedule(current, {
                                        kind: 'cosine',
                                        totalSteps: 1000,
                                        minimumRate: 0,
                                    }));
                                }
                            }}
                        >
                            <option value="constant">Constant</option>
                            <option value="step">Step decay</option>
                            <option value="cosine">Cosine</option>
                        </select>
                    </Tooltip>
                </div>

                {schedule.kind === 'step' && (
                    <>
                        <div className="control-row">
                            <span className="control-label">Step interval</span>
                            <Tooltip content="Apply the decay after this many updates">
                                <ExactNumberInput
                                    ariaLabel="Step schedule interval"
                                    min={1}
                                    step={1}
                                    value={schedule.interval}
                                    onCommit={(interval) => commit((current) => {
                                        const currentSchedule = current.training.schedule;
                                        if (currentSchedule.kind !== 'step') {
                                            return { ok: true, recipe: current };
                                        }
                                        return setSchedule(current, { ...currentSchedule, interval });
                                    })}
                                />
                            </Tooltip>
                        </div>
                        <div className="control-row">
                            <span className="control-label">Step gamma</span>
                            <Tooltip content="Multiply the learning rate by this value at each interval">
                                <ExactNumberInput
                                    ariaLabel="Step schedule gamma"
                                    min={0}
                                    max={1}
                                    value={schedule.gamma}
                                    onCommit={(gamma) => commit((current) => {
                                        const currentSchedule = current.training.schedule;
                                        if (currentSchedule.kind !== 'step') {
                                            return { ok: true, recipe: current };
                                        }
                                        return setSchedule(current, { ...currentSchedule, gamma });
                                    })}
                                />
                            </Tooltip>
                        </div>
                    </>
                )}

                {schedule.kind === 'cosine' && (
                    <>
                        <div className="control-row">
                            <span className="control-label">Cosine steps</span>
                            <Tooltip content="Anneal to the minimum learning rate over this many updates">
                                <ExactNumberInput
                                    ariaLabel="Cosine total steps"
                                    min={1}
                                    step={1}
                                    value={schedule.totalSteps}
                                    onCommit={(totalSteps) => commit((current) => {
                                        const currentSchedule = current.training.schedule;
                                        if (currentSchedule.kind !== 'cosine') {
                                            return { ok: true, recipe: current };
                                        }
                                        return setSchedule(current, { ...currentSchedule, totalSteps });
                                    })}
                                />
                            </Tooltip>
                        </div>
                        <div className="control-row">
                            <span className="control-label">Minimum LR</span>
                            <Tooltip content="Lowest learning rate reached by the cosine schedule">
                                <ExactNumberInput
                                    ariaLabel="Cosine minimum learning rate"
                                    min={0}
                                    max={training.learningRate}
                                    value={schedule.minimumRate}
                                    onCommit={(minimumRate) => commit((current) => {
                                        const currentSchedule = current.training.schedule;
                                        if (currentSchedule.kind !== 'cosine') {
                                            return { ok: true, recipe: current };
                                        }
                                        return setSchedule(current, { ...currentSchedule, minimumRate });
                                    })}
                                />
                            </Tooltip>
                        </div>
                    </>
                )}

                <p className="control-note">{scheduleSummary}</p>

                <div className="control-row">
                    <span className="control-label">Output activation</span>
                    <span className="control-value">{prepared.compiled.task.outputActivation}</span>
                </div>

                <div className="control-row">
                    <span className="control-label">Data loss</span>
                    {isRegression ? (
                        <Tooltip content="Cause: loss defines how numeric prediction error is penalized. Effect: Huber reduces the influence of large outliers compared with MSE.">
                            <select
                                className="select"
                                aria-label="Loss"
                                value={regressionLoss.kind}
                                onChange={(event) => {
                                    if (event.target.value === 'huber') {
                                        void commit((current) => setRegressionDataLoss(
                                            current,
                                            { kind: 'huber', delta: 1 },
                                        ));
                                    } else {
                                        void commit((current) => setRegressionDataLoss(
                                            current,
                                            { kind: 'mean-squared-error' },
                                        ));
                                    }
                                }}
                            >
                                <option value="mean-squared-error">Mean squared error</option>
                                <option value="huber">Huber</option>
                            </select>
                        </Tooltip>
                    ) : (
                        <span className="control-value">
                            {objectiveLabel(objective.dataLoss.kind)}
                        </span>
                    )}
                </div>

                {isRegression && regressionLoss.kind === 'huber' && (
                    <div className="control-row">
                        <span className="control-label">Huber delta</span>
                        <Tooltip content="Set the Huber loss transition point">
                            <ExactNumberInput
                                ariaLabel="Huber delta"
                                min={0}
                                value={regressionLoss.delta}
                                onCommit={(delta) => commit((current) => (
                                    setRegressionDataLoss(current, { kind: 'huber', delta })
                                ))}
                            />
                        </Tooltip>
                    </div>
                )}
            </div>

            <div className="forge-section">
                <div className="forge-section__label">Optimizer</div>

                <div className="control-row">
                    <span className="control-label">Optimizer</span>
                    <Tooltip content="Cause: optimizers choose how gradients become weight updates. Effect: momentum and Adam can smooth or adapt steps compared with plain SGD.">
                        <select
                            className="select"
                            aria-label="Optimizer"
                            value={optimizer.kind}
                            onChange={(event) => {
                                const kind = event.target.value;
                                if (kind === 'sgd') {
                                    void commit((current) => setOptimizer(current, { kind: 'sgd' }));
                                } else if (kind === 'sgd-momentum') {
                                    void commit((current) => setOptimizer(current, {
                                        kind: 'sgd-momentum',
                                        momentum: 0.9,
                                    }));
                                } else {
                                    void commit((current) => setOptimizer(current, {
                                        kind: 'adam',
                                        beta1: 0.9,
                                        beta2: 0.999,
                                        epsilon: 1e-8,
                                    }));
                                }
                            }}
                        >
                            <option value="sgd">SGD</option>
                            <option value="sgd-momentum">SGD + Momentum</option>
                            <option value="adam">Adam</option>
                        </select>
                    </Tooltip>
                </div>

                <p className="control-note">
                    {OPTIMIZER_EXPLANATIONS[optimizer.kind]}
                </p>

                {optimizer.kind === 'sgd-momentum' && (
                    <div className="control-row">
                        <span className="control-label">Momentum</span>
                        <Tooltip content="Set the momentum coefficient used by SGD + Momentum">
                            <ExactNumberInput
                                ariaLabel="Momentum"
                                min={0}
                                max={1}
                                value={optimizer.momentum}
                                onCommit={(momentum) => commit((current) => {
                                    const currentOptimizer = current.training.optimizer;
                                    if (currentOptimizer.kind !== 'sgd-momentum') {
                                        return { ok: true, recipe: current };
                                    }
                                    return setOptimizer(current, { ...currentOptimizer, momentum });
                                })}
                            />
                        </Tooltip>
                    </div>
                )}

                {optimizer.kind === 'adam' && (
                    <>
                        <div className="control-row">
                            <span className="control-label">Adam β1</span>
                            <ExactNumberInput
                                ariaLabel="Adam beta 1"
                                min={0}
                                max={1}
                                value={optimizer.beta1}
                                onCommit={(beta1) => commit((current) => {
                                    const currentOptimizer = current.training.optimizer;
                                    if (currentOptimizer.kind !== 'adam') {
                                        return { ok: true, recipe: current };
                                    }
                                    return setOptimizer(current, { ...currentOptimizer, beta1 });
                                })}
                            />
                        </div>
                        <div className="control-row">
                            <span className="control-label">Adam β2</span>
                            <ExactNumberInput
                                ariaLabel="Adam beta 2"
                                min={0}
                                max={1}
                                value={optimizer.beta2}
                                onCommit={(beta2) => commit((current) => {
                                    const currentOptimizer = current.training.optimizer;
                                    if (currentOptimizer.kind !== 'adam') {
                                        return { ok: true, recipe: current };
                                    }
                                    return setOptimizer(current, { ...currentOptimizer, beta2 });
                                })}
                            />
                        </div>
                        <div className="control-row">
                            <span className="control-label">Adam epsilon</span>
                            <ExactNumberInput
                                ariaLabel="Adam epsilon"
                                min={0}
                                value={optimizer.epsilon}
                                onCommit={(epsilon) => commit((current) => {
                                    const currentOptimizer = current.training.optimizer;
                                    if (currentOptimizer.kind !== 'adam') {
                                        return { ok: true, recipe: current };
                                    }
                                    return setOptimizer(current, { ...currentOptimizer, epsilon });
                                })}
                            />
                        </div>
                    </>
                )}

                <div className="control-row">
                    <span className="control-label">Gradient clipping</span>
                    <Tooltip content="Limit the global norm of the exact total objective gradient before each update">
                        <select
                            className="select"
                            aria-label="Gradient clipping"
                            value={gradientClipping.kind}
                            onChange={(event) => {
                                if (event.target.value === 'none') {
                                    void commit((current) => setGradientClipping(
                                        current,
                                        { kind: 'none' },
                                    ));
                                } else {
                                    void commit((current) => setGradientClipping(current, {
                                        kind: 'global-norm',
                                        maximumNorm: 1,
                                        scope: 'total-objective-gradient',
                                    }));
                                }
                            }}
                        >
                            <option value="none">Off</option>
                            <option value="global-norm">Global norm</option>
                        </select>
                    </Tooltip>
                </div>

                {gradientClipping.kind === 'global-norm' && (
                    <>
                        <div className="control-row">
                            <span className="control-label">Maximum norm</span>
                            <ExactNumberInput
                                ariaLabel="Maximum gradient norm"
                                min={0}
                                value={gradientClipping.maximumNorm}
                                onCommit={(maximumNorm) => commit((current) => {
                                    const clipping = current.training.gradientClipping;
                                    if (clipping.kind !== 'global-norm') {
                                        return { ok: true, recipe: current };
                                    }
                                    return setGradientClipping(current, {
                                        ...clipping,
                                        maximumNorm,
                                    });
                                })}
                            />
                        </div>
                        <div className="control-row">
                            <span className="control-label">Clip scope</span>
                            <span className="control-value">total objective gradient</span>
                        </div>
                    </>
                )}

                <div className="control-row">
                    <span className="control-label">Batch size</span>
                    <Tooltip content="Cause: larger batches average more samples per update. Effect: the path is steadier, but each visible update reacts less often.">
                        <select
                            className="select"
                            aria-label="Batch size"
                            value={training.batchSize}
                            onChange={(event) => void commit((current) => (
                                setBatchSize(current, Number(event.target.value))
                            ))}
                        >
                            {batchSizeOptions.map((batchSize) => (
                                <option
                                    key={batchSize}
                                    value={batchSize}
                                    disabled={batchSize > trainCount}
                                >
                                    {batchSize}{!batchSizeIsQuickPick && batchSize === training.batchSize
                                        ? ' (current)'
                                        : ''}
                                </option>
                            ))}
                        </select>
                    </Tooltip>
                </div>
                <p className="control-note">Maximum for this split: {trainCount}</p>
            </div>

            <div className="forge-section">
                <div className="forge-section__label">Regularization</div>

                <div className="control-row">
                    <span className="control-label">Penalty</span>
                    <Tooltip content="Cause: regularization penalizes large or unnecessary weights. Effect: the boundary often smooths out and generalizes better on noisy data.">
                        <select
                            className="select"
                            aria-label="Penalty"
                            value={objective.penalty.kind}
                            onChange={(event) => {
                                const kind = event.target.value;
                                if (kind === 'none') {
                                    void commit((current) => setPenalty(current, { kind: 'none' }));
                                } else {
                                    void commit((current) => setPenalty(current, {
                                        kind: kind as 'l1' | 'l2',
                                        coefficient: 0.001,
                                        applyTo: 'weights',
                                    }));
                                }
                            }}
                        >
                            <option value="none">None</option>
                            <option value="l1">L1</option>
                            <option value="l2">L2</option>
                        </select>
                    </Tooltip>
                </div>

                {objective.penalty.kind !== 'none' && (
                    <>
                        <div className="control-row">
                            <span className="control-label">Coefficient</span>
                            <Tooltip content="Cause: increasing the penalty pushes weights harder toward simpler solutions. Effect: too much can underfit and leave the boundary too flat.">
                                <ExactNumberInput
                                    ariaLabel="Penalty coefficient"
                                    min={0}
                                    value={objective.penalty.coefficient}
                                    onCommit={(coefficient) => commit((current) => {
                                        const penalty = current.objective.penalty;
                                        if (penalty.kind === 'none') {
                                            return { ok: true, recipe: current };
                                        }
                                        return setPenalty(current, { ...penalty, coefficient });
                                    })}
                                />
                            </Tooltip>
                        </div>
                        <div className="control-row">
                            <span className="control-label">Penalty target</span>
                            <span className="control-value">weights only</span>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
});
