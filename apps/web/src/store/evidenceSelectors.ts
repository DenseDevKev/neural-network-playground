import type {
    EvaluationTrigger,
    LiveTrainingSignal,
    ModelRevision,
    PairedEvaluation,
} from '@nn-playground/shared';

export interface ScientificEvidenceInput {
    readonly latestLiveSignal: LiveTrainingSignal | null;
    readonly latestEvaluation: PairedEvaluation | null;
}

export interface ScientificEvidence {
    readonly currentModel: ModelRevision | null;
    readonly batchTrend: {
        readonly step: number;
        readonly epoch: number;
        readonly dataLoss: number;
        readonly latestBatchSize: number;
    } | null;
    readonly fullEvaluation: {
        readonly evaluationId: number;
        readonly step: number;
        readonly epoch: number;
        readonly trigger: EvaluationTrigger;
        readonly trainDataLoss: number;
        readonly testDataLoss: number;
        readonly trainingObjective: number;
        readonly regularizationPenalty: number;
        readonly trainAccuracy: number | undefined;
        readonly testAccuracy: number | undefined;
        readonly trainSampleCount: number;
        readonly testSampleCount: number;
    } | null;
    readonly evaluationAgeSteps: number | null;
    readonly generalizationGap: number | null;
}

function newestModel(
    live: LiveTrainingSignal | null,
    evaluation: PairedEvaluation | null,
): ModelRevision | null {
    if (!live) return evaluation?.model ?? null;
    if (!evaluation || evaluation.model.generationId !== live.model.generationId) {
        return live.model;
    }
    return evaluation.model.revision > live.model.revision
        ? evaluation.model
        : live.model;
}

/**
 * The only scalar evidence projection used by the UI. It intentionally has
 * no snapshot or loose-history input, so live and paired claims cannot be
 * manufactured from unrelated revisions.
 */
export function selectScientificEvidence(
    input: ScientificEvidenceInput,
): ScientificEvidence {
    const live = input.latestLiveSignal;
    const evaluation = input.latestEvaluation;
    const currentModel = newestModel(live, evaluation);
    const sameGeneration = currentModel !== null
        && evaluation !== null
        && currentModel.generationId === evaluation.model.generationId;
    const age = sameGeneration
        ? currentModel.step - evaluation.model.step
        : -1;

    return Object.freeze({
        currentModel,
        batchTrend: live === null ? null : Object.freeze({
            step: live.model.step,
            epoch: live.model.epoch,
            dataLoss: live.dataLoss,
            latestBatchSize: live.basis.latestBatchSize,
        }),
        fullEvaluation: evaluation === null ? null : Object.freeze({
            evaluationId: evaluation.evaluationId,
            step: evaluation.model.step,
            epoch: evaluation.model.epoch,
            trigger: evaluation.trigger,
            trainDataLoss: evaluation.train.values.dataLoss,
            testDataLoss: evaluation.test.values.dataLoss,
            trainingObjective: evaluation.objective.trainTotalObjective,
            regularizationPenalty: evaluation.objective.regularizationPenalty,
            trainAccuracy: evaluation.train.values.accuracy,
            testAccuracy: evaluation.test.values.accuracy,
            trainSampleCount: evaluation.train.basis.sampleCount,
            testSampleCount: evaluation.test.basis.sampleCount,
        }),
        evaluationAgeSteps: age >= 0 ? age : null,
        generalizationGap: evaluation === null
            ? null
            : evaluation.test.values.dataLoss - evaluation.train.values.dataLoss,
    });
}
