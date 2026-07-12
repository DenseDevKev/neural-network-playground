import {
    DEFAULT_EVALUATION_POLICY,
    parseLiveTrainingSignal,
    parsePairedEvaluation,
} from '@nn-playground/shared';
import type {
    DatasetRevision,
    EvaluationPolicy,
    EvaluationTrigger,
    EvaluationValues,
    LiveTrainingSignal,
    ModelRevision,
    PairedEvaluation,
} from '@nn-playground/shared';

export interface BatchMeasurement {
    readonly model: ModelRevision;
    readonly batchSize: number;
    readonly dataLoss: number;
}

export interface EvaluationRuntimeOptions {
    readonly generationId: number;
    readonly dataset: DatasetRevision;
    readonly objectiveKey: string;
    readonly emaAlpha: number;
    readonly policy?: EvaluationPolicy;
    readonly getCurrentModel: () => ModelRevision;
    readonly evaluateTrain: (model: ModelRevision) => EvaluationValues;
    readonly evaluateTest: (model: ModelRevision) => EvaluationValues;
    readonly evaluateRegularizationPenalty: (model: ModelRevision) => number;
}

export interface RestoreEvaluationPreparationOptions {
    readonly model: ModelRevision;
    readonly getCurrentModel: () => ModelRevision;
    readonly evaluateTrain: (model: ModelRevision) => EvaluationValues;
    readonly evaluateTest: (model: ModelRevision) => EvaluationValues;
    readonly evaluateRegularizationPenalty: (model: ModelRevision) => number;
}

export interface PreparedRestoreEvaluation {
    readonly evaluation: PairedEvaluation;
    commit(): PairedEvaluation;
}

export type ForcedEvaluationTrigger = Exclude<EvaluationTrigger, 'cadence'>;

/** A non-finite scientific value is terminal and must never be published. */
export class TerminalDivergenceError extends Error {
    readonly path: string;
    readonly value: number;

    constructor(path: string, value: number) {
        super(`terminal divergence: ${path} is non-finite`);
        this.name = 'TerminalDivergenceError';
        this.path = path;
        this.value = value;
    }
}

function assertFiniteScientificValue(value: number, path: string): void {
    if (!Number.isFinite(value)) throw new TerminalDivergenceError(path, value);
}

function assertFiniteEvaluationValues(values: EvaluationValues, path: string): void {
    assertFiniteScientificValue(values.dataLoss, `${path}.dataLoss`);
    if (values.accuracy !== undefined) {
        assertFiniteScientificValue(values.accuracy, `${path}.accuracy`);
    }
}

function deepFreeze<T>(value: T): Readonly<T> {
    if (typeof value !== 'object' || value === null || Object.isFrozen(value)) {
        return value;
    }
    for (const key of Reflect.ownKeys(value)) {
        deepFreeze((value as Record<PropertyKey, unknown>)[key]);
    }
    return Object.freeze(value);
}

function validatePolicy(policy: EvaluationPolicy): Readonly<EvaluationPolicy> {
    if (!Number.isSafeInteger(policy.everySteps) || policy.everySteps < 1) {
        throw new TypeError('policy.everySteps must be a positive safe integer');
    }
    if (policy.forceOnPause !== true
        || policy.forceOnManualStep !== true
        || policy.forceOnCheckpoint !== true
        || policy.forceOnSave !== true) {
        throw new TypeError('evaluation policy force flags must all be true');
    }
    return Object.freeze({
        everySteps: policy.everySteps,
        forceOnPause: true,
        forceOnManualStep: true,
        forceOnCheckpoint: true,
        forceOnSave: true,
    });
}

function sameModel(left: ModelRevision, right: ModelRevision): boolean {
    return left.generationId === right.generationId
        && left.revision === right.revision
        && left.step === right.step
        && left.epoch === right.epoch;
}

/**
 * Owns live-loss EMA and full-evaluation publication for one worker generation.
 * Evaluation callbacks are fixed for the generation so the atomic path exposes
 * no training/mutation hook between train and test evaluation.
 */
export class EvaluationRuntime {
    readonly generationId: number;
    readonly dataset: Readonly<DatasetRevision>;
    readonly objectiveKey: string;
    readonly emaAlpha: number;
    readonly policy: Readonly<EvaluationPolicy>;

    private readonly evaluateTrain: (model: ModelRevision) => EvaluationValues;
    private readonly evaluateTest: (model: ModelRevision) => EvaluationValues;
    private readonly evaluateRegularizationPenalty: (model: ModelRevision) => number;
    private readonly getCurrentModel: () => ModelRevision;
    private nextEvaluationId = 1;
    private pendingCadenceStep: number | null = null;
    private currentModel: ModelRevision;
    private _latestLiveSignal: LiveTrainingSignal | undefined;
    private _latestEvaluation: PairedEvaluation | undefined;

    constructor(options: EvaluationRuntimeOptions) {
        if (!Number.isSafeInteger(options.generationId) || options.generationId < 1) {
            throw new TypeError('generationId must be a positive safe integer');
        }
        if (!Number.isFinite(options.emaAlpha)
            || options.emaAlpha <= 0
            || options.emaAlpha > 1) {
            throw new TypeError('emaAlpha must be finite, greater than 0, and at most 1');
        }

        this.generationId = options.generationId;
        this.emaAlpha = options.emaAlpha;
        this.policy = validatePolicy(options.policy ?? DEFAULT_EVALUATION_POLICY);
        this.evaluateTrain = options.evaluateTrain;
        this.evaluateTest = options.evaluateTest;
        this.evaluateRegularizationPenalty = options.evaluateRegularizationPenalty;
        this.getCurrentModel = options.getCurrentModel;

        const seed = parseLiveTrainingSignal({
            model: {
                generationId: options.generationId,
                revision: 0,
                step: 0,
                epoch: 0,
            },
            dataset: options.dataset,
            objectiveKey: options.objectiveKey,
            basis: {
                kind: 'mini-batch-ema',
                alpha: options.emaAlpha,
                latestBatchSize: 1,
                throughStep: 0,
            },
            dataLoss: 0,
        });
        this.dataset = deepFreeze(seed.dataset) as Readonly<DatasetRevision>;
        this.objectiveKey = seed.objectiveKey;
        this.currentModel = this.readCurrentModel();
    }

    get latestLiveSignal(): LiveTrainingSignal | undefined {
        return this._latestLiveSignal;
    }

    get latestEvaluation(): PairedEvaluation | undefined {
        return this._latestEvaluation;
    }

    recordBatch(measurement: BatchMeasurement): LiveTrainingSignal {
        if (this.pendingCadenceStep !== null) {
            throw new Error(
                `pending cadence evaluation at step ${this.pendingCadenceStep} must be consumed`,
            );
        }
        assertFiniteScientificValue(measurement.dataLoss, '$.objective.dataLoss');
        const previous = this._latestLiveSignal;
        const ema = previous === undefined
            ? measurement.dataLoss
            : this.emaAlpha * measurement.dataLoss
                + (1 - this.emaAlpha) * previous.dataLoss;
        const parsed = parseLiveTrainingSignal({
            model: measurement.model,
            dataset: this.dataset,
            objectiveKey: this.objectiveKey,
            basis: {
                kind: 'mini-batch-ema',
                alpha: this.emaAlpha,
                latestBatchSize: measurement.batchSize,
                throughStep: measurement.model.step,
            },
            dataLoss: ema,
        });
        this.assertGeneration(parsed.model);
        const current = this.readCurrentModel();
        if (!sameModel(parsed.model, current)) {
            throw new RangeError('batch model must equal getCurrentModel() exactly');
        }
        if (parsed.model.step <= this.currentModel.step) {
            throw new RangeError('model.step must increase for every recorded batch');
        }
        if (parsed.model.revision <= this.currentModel.revision) {
            throw new RangeError('model.revision must increase for every recorded batch');
        }

        const published = deepFreeze(parsed) as LiveTrainingSignal;
        this._latestLiveSignal = published;
        this.currentModel = published.model;
        if (published.model.step > 0
            && published.model.step % this.policy.everySteps === 0) {
            this.pendingCadenceStep = published.model.step;
        }
        return published;
    }

    takeCadenceEvaluation(): PairedEvaluation | undefined {
        if (this.pendingCadenceStep === null) return undefined;
        const model = this.readCurrentModel();
        const latestModel = this._latestLiveSignal?.model;
        if (latestModel === undefined
            || model.step !== this.pendingCadenceStep
            || !sameModel(model, latestModel)) {
            throw new Error('cadence evaluation model must equal the due live-signal revision');
        }
        return this.publishEvaluation('cadence', model);
    }

    forceEvaluation(
        trigger: ForcedEvaluationTrigger,
    ): PairedEvaluation {
        if (trigger === ('cadence' as ForcedEvaluationTrigger)) {
            throw new TypeError('cadence evaluations must use takeCadenceEvaluation()');
        }
        const model = this.readCurrentModel();
        if (trigger === 'restore') {
            if (model.revision <= this.currentModel.revision) {
                throw new RangeError('restore model.revision must exceed the last observed model');
            }
        } else if (!sameModel(model, this.currentModel)) {
            throw new RangeError('forced evaluation model must equal the last observed model');
        }
        const evaluation = this.publishEvaluation(trigger, model);
        if (trigger === 'restore') {
            // A restored model starts a new live-trend segment. Reusing the
            // pre-restore EMA would attach old-generation loss to new weights.
            this._latestLiveSignal = undefined;
        }
        return evaluation;
    }

    /**
     * Evaluate a detached restore candidate without changing live runtime
     * evidence. The returned commit performs no scientific recomputation.
     */
    prepareRestoreEvaluation(
        options: RestoreEvaluationPreparationOptions,
    ): PreparedRestoreEvaluation {
        const model = this.validateAndFreezeModel(options.model);
        if (model.revision <= this.currentModel.revision) {
            throw new RangeError('restore model.revision must exceed the last observed model');
        }
        const expectedEvaluationId = this.nextEvaluationId;
        const expectedCurrentModel = this.currentModel;
        const evaluation = this.evaluateCandidate('restore', model, options);
        let committed = false;
        return Object.freeze({
            evaluation,
            commit: (): PairedEvaluation => {
                if (committed) return evaluation;
                if (this.nextEvaluationId !== expectedEvaluationId
                    || !sameModel(this.currentModel, expectedCurrentModel)) {
                    throw new Error('prepared restore evaluation is stale');
                }
                this.commitEvaluation(evaluation);
                this._latestLiveSignal = undefined;
                committed = true;
                return evaluation;
            },
        });
    }

    private publishEvaluation(
        trigger: EvaluationTrigger,
        model: ModelRevision,
    ): PairedEvaluation {
        const published = this.evaluateCandidate(trigger, model, {
            getCurrentModel: this.getCurrentModel,
            evaluateTrain: this.evaluateTrain,
            evaluateTest: this.evaluateTest,
            evaluateRegularizationPenalty: this.evaluateRegularizationPenalty,
        });
        this.commitEvaluation(published);
        return published;
    }

    private evaluateCandidate(
        trigger: EvaluationTrigger,
        model: ModelRevision,
        computation: Pick<
            RestoreEvaluationPreparationOptions,
            | 'getCurrentModel'
            | 'evaluateTrain'
            | 'evaluateTest'
            | 'evaluateRegularizationPenalty'
        >,
    ): PairedEvaluation {
        const frozenModel = this.validateAndFreezeModel(model);
        const trainValues = computation.evaluateTrain(frozenModel);
        assertFiniteEvaluationValues(trainValues, '$.evaluation.train.values');
        this.assertModelUnchangedDuringEvaluation(frozenModel, computation.getCurrentModel);
        const testValues = computation.evaluateTest(frozenModel);
        assertFiniteEvaluationValues(testValues, '$.evaluation.test.values');
        this.assertModelUnchangedDuringEvaluation(frozenModel, computation.getCurrentModel);
        const regularizationPenalty = computation.evaluateRegularizationPenalty(frozenModel);
        assertFiniteScientificValue(
            regularizationPenalty,
            '$.evaluation.objective.regularizationPenalty',
        );
        this.assertModelUnchangedDuringEvaluation(frozenModel, computation.getCurrentModel);
        const trainTotalObjective = trainValues.dataLoss + regularizationPenalty;
        assertFiniteScientificValue(
            trainTotalObjective,
            '$.evaluation.objective.trainTotalObjective',
        );
        const candidate = parsePairedEvaluation({
            evaluationId: this.nextEvaluationId,
            trigger,
            model: frozenModel,
            dataset: this.dataset,
            objectiveKey: this.objectiveKey,
            train: {
                basis: {
                    kind: 'full-split',
                    split: 'train',
                    sampleCount: this.dataset.trainCount,
                    populationCount: this.dataset.trainCount,
                },
                values: trainValues,
            },
            test: {
                basis: {
                    kind: 'full-split',
                    split: 'test',
                    sampleCount: this.dataset.testCount,
                    populationCount: this.dataset.testCount,
                },
                values: testValues,
            },
            objective: {
                regularizationPenalty,
                trainTotalObjective,
            },
        });
        return deepFreeze(candidate) as PairedEvaluation;
    }

    private commitEvaluation(published: PairedEvaluation): void {
        this.nextEvaluationId++;
        this._latestEvaluation = published;
        this.currentModel = published.model;
        // Any successful pair is current by construction, so it supersedes an
        // older cadence obligation (including after a validated restore).
        this.pendingCadenceStep = null;
    }

    private validateAndFreezeModel(model: ModelRevision): ModelRevision {
        const validated = parseLiveTrainingSignal({
            model,
            dataset: this.dataset,
            objectiveKey: this.objectiveKey,
            basis: {
                kind: 'mini-batch-ema',
                alpha: this.emaAlpha,
                latestBatchSize: 1,
                throughStep: model.step,
            },
            dataLoss: 0,
        }).model;
        this.assertGeneration(validated);
        return deepFreeze(validated) as ModelRevision;
    }

    private assertGeneration(model: ModelRevision): void {
        if (model.generationId !== this.generationId) {
            throw new RangeError(
                `model.generationId must equal runtime generationId ${this.generationId}`,
            );
        }
    }

    private readCurrentModel(): ModelRevision {
        return this.validateAndFreezeModel(this.getCurrentModel());
    }

    private assertModelUnchangedDuringEvaluation(
        expected: ModelRevision,
        getCurrentModel: () => ModelRevision,
    ): void {
        const current = this.validateAndFreezeModel(getCurrentModel());
        if (!sameModel(current, expected)) {
            throw new Error('current model changed during evaluation');
        }
    }
}
