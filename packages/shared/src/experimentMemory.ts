import { getDatasetContract } from '@nn-playground/engine';
import {
    canonicalizeJson,
    createStableJsonSnapshot,
    StableJsonSnapshotError,
} from './canonicalJson.js';
import {
    fingerprintDataset,
    fingerprintObjective,
    fingerprintRecipe,
    validateExperimentDocument,
} from './experimentSchema.js';
import {
    parseLiveTrainingSignal,
    parsePairedEvaluation,
} from './metricProvenance.js';
import type {
    DatasetRevision,
    EvaluationPoint,
    ModelRevision,
    PairedEvaluation,
    TrainingTrendPoint,
} from './metricProvenance.js';
import type {
    ExperimentSchemaIssueCode,
    StandardExperimentRecipeV2,
    ValidatedStandardExperimentRecipeV2,
} from './types.js';

export const EXPERIMENT_MEMORY_SCHEMA_VERSION = 2 as const;
export const EXPERIMENT_MEMORY_RECORD_KIND = 'nn-playground-run' as const;
export const EXPERIMENT_MEMORY_ENVELOPE_KIND = 'nn-playground-experiment-memory' as const;
export const EXPERIMENT_MEMORY_MAX_RECORDS = 20;
export const EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS = 120;
export const EXPERIMENT_MEMORY_MAX_TRENDS = 512;
export const EXPERIMENT_MEMORY_MAX_EVALUATIONS = 256;
export const EXPERIMENT_MEMORY_MAX_RECORD_BYTES = 512 * 1_024;
export const EXPERIMENT_MEMORY_MAX_ENVELOPE_BYTES = 4 * 1_024 * 1_024;

export interface ExperimentRunRecordV2 {
    readonly kind: typeof EXPERIMENT_MEMORY_RECORD_KIND;
    readonly schemaVersion: typeof EXPERIMENT_MEMORY_SCHEMA_VERSION;
    readonly id: string;
    readonly createdAt: string;
    readonly updatedAt: string;
    readonly title?: string;
    readonly recipe: StandardExperimentRecipeV2;
    readonly recipeFingerprint: string;
    readonly snapshot: {
        readonly model: ModelRevision;
        readonly evaluation: PairedEvaluation;
        readonly trendHistory: readonly TrainingTrendPoint[];
        readonly evaluationHistory: readonly EvaluationPoint[];
    };
}

export interface ExperimentMemoryEnvelopeV2 {
    readonly kind: typeof EXPERIMENT_MEMORY_ENVELOPE_KIND;
    readonly schemaVersion: typeof EXPERIMENT_MEMORY_SCHEMA_VERSION;
    readonly records: readonly ExperimentRunRecordV2[];
}

export interface ExperimentMemoryIssue {
    readonly code: ExperimentSchemaIssueCode;
    readonly path: string;
    readonly message: string;
}

export type ExperimentMemoryResult<T> =
    | { readonly ok: true; readonly value: T }
    | { readonly ok: false; readonly issues: readonly ExperimentMemoryIssue[] };

export interface RejectedExperimentRunRecordV2 {
    readonly sourceIndex: number;
    readonly rawJson: string;
    readonly issues: readonly ExperimentMemoryIssue[];
}

export interface ExperimentMemoryReadResultV2 {
    readonly records: readonly ExperimentRunRecordV2[];
    readonly rejectedRecords: readonly RejectedExperimentRunRecordV2[];
    readonly envelopeIssues: readonly ExperimentMemoryIssue[];
}

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/u;
const textEncoder = new TextEncoder();

function issue(
    code: ExperimentSchemaIssueCode,
    path: string,
    message: string,
): ExperimentMemoryIssue {
    return Object.freeze({ code, path, message });
}

function invalid(path: string, message: string): ExperimentMemoryIssue {
    return issue('invalid-field', path, message);
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
    const prototype = Object.getPrototypeOf(value);
    return prototype === Object.prototype || prototype === null;
}

function hasExactKeys(
    value: Record<string, unknown>,
    required: readonly string[],
    optional: readonly string[] = [],
): boolean {
    const allowed = new Set([...required, ...optional]);
    const keys = Object.keys(value);
    return required.every((key) => Object.prototype.hasOwnProperty.call(value, key))
        && keys.every((key) => allowed.has(key));
}

function serializedJson(value: unknown): string | null {
    try {
        const serialized = JSON.stringify(value);
        return typeof serialized === 'string' ? serialized : null;
    } catch {
        return null;
    }
}

function utf8Bytes(value: string): number {
    return textEncoder.encode(value).byteLength;
}

function sameModel(left: ModelRevision, right: ModelRevision): boolean {
    return left.generationId === right.generationId
        && left.revision === right.revision
        && left.step === right.step
        && left.epoch === right.epoch;
}

function sameDataset(left: DatasetRevision, right: DatasetRevision): boolean {
    return left.generatorVersion === right.generatorVersion
        && left.datasetKey === right.datasetKey
        && left.trainCount === right.trainCount
        && left.testCount === right.testCount;
}

function sameEvaluation(left: PairedEvaluation, right: PairedEvaluation): boolean {
    return canonicalizeJson(left) === canonicalizeJson(right);
}

function validateEvaluationSplitCounts(
    evaluation: PairedEvaluation,
    path: string,
    expectedTrainCount: number,
    expectedTestCount: number,
    issues: ExperimentMemoryIssue[],
): void {
    const checks: ReadonlyArray<readonly [number, number, string]> = [
        [evaluation.dataset.trainCount, expectedTrainCount, `${path}.dataset.trainCount`],
        [evaluation.dataset.testCount, expectedTestCount, `${path}.dataset.testCount`],
        [evaluation.train.basis.sampleCount, expectedTrainCount, `${path}.train.basis.sampleCount`],
        [evaluation.train.basis.populationCount, expectedTrainCount, `${path}.train.basis.populationCount`],
        [evaluation.test.basis.sampleCount, expectedTestCount, `${path}.test.basis.sampleCount`],
        [evaluation.test.basis.populationCount, expectedTestCount, `${path}.test.basis.populationCount`],
    ];
    for (const [actual, expected, issuePath] of checks) {
        if (actual !== expected) {
            issues.push(invalid(
                issuePath,
                `Saved split count ${actual} must equal the recipe-derived count ${expected}.`,
            ));
        }
    }
}

function parseRecordShape(snapshot: unknown): {
    record: Record<string, unknown> | null;
    issues: ExperimentMemoryIssue[];
} {
    if (!isPlainRecord(snapshot)) {
        return { record: null, issues: [invalid('$', 'Run record must be a plain object.')] };
    }
    const required = [
        'kind',
        'schemaVersion',
        'id',
        'createdAt',
        'updatedAt',
        'recipe',
        'recipeFingerprint',
        'snapshot',
    ];
    if (!hasExactKeys(snapshot, required, ['title'])) {
        return {
            record: null,
            issues: [invalid('$', 'Run record contains missing or unknown fields.')],
        };
    }
    return { record: snapshot, issues: [] };
}

function parseIsoTimestamp(value: unknown, path: string): ExperimentMemoryIssue | null {
    if (typeof value !== 'string' || value.length === 0) {
        return invalid(path, 'Timestamp must be a non-empty ISO-8601 string.');
    }
    const timestamp = Date.parse(value);
    if (!Number.isFinite(timestamp) || new Date(timestamp).toISOString() !== value) {
        return invalid(path, 'Timestamp must be a canonical ISO-8601 UTC string.');
    }
    return null;
}

function parseHistory(
    value: unknown,
    path: 'snapshot.trendHistory' | 'snapshot.evaluationHistory',
    maximum: number,
    parser: (entry: unknown) => TrainingTrendPoint | EvaluationPoint,
): {
    values: Array<TrainingTrendPoint | EvaluationPoint>;
    issues: ExperimentMemoryIssue[];
} {
    if (!Array.isArray(value)) {
        return { values: [], issues: [invalid(path, 'History must be a dense array.')] };
    }
    const issues: ExperimentMemoryIssue[] = [];
    if (value.length > maximum) {
        issues.push(issue(
            'resource-limit',
            path,
            `History contains ${value.length} points; the maximum is ${maximum}.`,
        ));
    }
    const values: Array<TrainingTrendPoint | EvaluationPoint> = [];
    for (let index = 0; index < Math.min(value.length, maximum + 1); index++) {
        try {
            values.push(parser(value[index]));
        } catch (error) {
            issues.push(invalid(
                `${path}[${index}]`,
                error instanceof Error ? error.message : 'History point is invalid.',
            ));
        }
    }
    return { values, issues };
}

/** Deterministic endpoint-preserving compaction for worker-owned histories. */
export function compactEvenly<T>(values: readonly T[], limit: number): readonly T[] {
    if (!Number.isSafeInteger(limit) || limit < 2) {
        throw new RangeError('History compaction limit must be a safe integer of at least 2.');
    }
    if (values.length <= limit) return Object.freeze([...values]);
    return Object.freeze(Array.from(
        { length: limit },
        (_, index) => values[Math.round(index * (values.length - 1) / (limit - 1))],
    ));
}

/**
 * Validate one worker-authored run record. The defensive snapshot and every
 * synchronous structural check happen before identity hashing yields control.
 */
export async function validateExperimentRunRecordV2(
    value: unknown,
): Promise<ExperimentMemoryResult<ExperimentRunRecordV2>> {
    let snapshot: unknown;
    try {
        snapshot = createStableJsonSnapshot(value);
    } catch (error) {
        const path = error instanceof StableJsonSnapshotError ? error.path : '$';
        return { ok: false, issues: [invalid(path, error instanceof Error ? error.message : 'Record snapshot failed.')] };
    }

    const rawJson = serializedJson(snapshot);
    if (rawJson === null) {
        return { ok: false, issues: [invalid('$', 'Run record is not valid JSON data.')] };
    }
    if (utf8Bytes(rawJson) > EXPERIMENT_MEMORY_MAX_RECORD_BYTES) {
        return {
            ok: false,
            issues: [issue(
                'resource-limit',
                '$',
                `Run record exceeds ${EXPERIMENT_MEMORY_MAX_RECORD_BYTES} UTF-8 bytes.`,
            )],
        };
    }

    const shape = parseRecordShape(snapshot);
    if (!shape.record) return { ok: false, issues: shape.issues };
    const record = shape.record;
    const issues: ExperimentMemoryIssue[] = [];

    if (record['kind'] !== EXPERIMENT_MEMORY_RECORD_KIND) {
        issues.push(invalid('kind', `kind must be ${EXPERIMENT_MEMORY_RECORD_KIND}.`));
    }
    if (record['schemaVersion'] !== EXPERIMENT_MEMORY_SCHEMA_VERSION) {
        issues.push(issue(
            'unsupported-version',
            'schemaVersion',
            'Only experiment run schemaVersion 2 is supported.',
        ));
    }
    if (typeof record['id'] !== 'string' || !UUID_PATTERN.test(record['id'])) {
        issues.push(invalid('id', 'Run record id must be a canonical lowercase 36-character UUID.'));
    }
    const createdIssue = parseIsoTimestamp(record['createdAt'], 'createdAt');
    if (createdIssue) issues.push(createdIssue);
    const updatedIssue = parseIsoTimestamp(record['updatedAt'], 'updatedAt');
    if (updatedIssue) issues.push(updatedIssue);
    if (typeof record['createdAt'] === 'string'
        && typeof record['updatedAt'] === 'string'
        && Date.parse(record['updatedAt']) < Date.parse(record['createdAt'])) {
        issues.push(invalid('updatedAt', 'updatedAt must not precede createdAt.'));
    }
    if (Object.prototype.hasOwnProperty.call(record, 'title')) {
        if (typeof record['title'] !== 'string'
            || record['title'].length === 0
            || Array.from(record['title']).length > EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS) {
            issues.push(invalid(
                'title',
                `Title must contain 1 to ${EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS} Unicode code points.`,
            ));
        }
    }
    if (typeof record['recipeFingerprint'] !== 'string') {
        issues.push(invalid('recipeFingerprint', 'Recipe fingerprint must be a string.'));
    }

    const documentResult = validateExperimentDocument({
        kind: 'nn-playground-experiment',
        schemaVersion: 2,
        recipe: record['recipe'],
        view: { showTestData: false, discretizeOutput: false },
    });
    if (!documentResult.ok) {
        for (const documentIssue of documentResult.issues) {
            issues.push(issue(
                documentIssue.code,
                documentIssue.path === '$' ? 'recipe' : `recipe.${documentIssue.path.replace(/^recipe\.?/u, '')}`,
                documentIssue.message,
            ));
        }
    }

    let model: ModelRevision | null = null;
    let evaluation: PairedEvaluation | null = null;
    let trendHistory: readonly TrainingTrendPoint[] = [];
    let evaluationHistory: readonly EvaluationPoint[] = [];
    if (!isPlainRecord(record['snapshot']) || !hasExactKeys(
        record['snapshot'],
        ['model', 'evaluation', 'trendHistory', 'evaluationHistory'],
    )) {
        issues.push(invalid('snapshot', 'Snapshot must contain exactly model, evaluation, trendHistory, and evaluationHistory.'));
    } else {
        const savedSnapshot = record['snapshot'];
        try {
            evaluation = parsePairedEvaluation(savedSnapshot['evaluation']);
            model = evaluation.model;
            if (!isPlainRecord(savedSnapshot['model']) || !hasExactKeys(
                savedSnapshot['model'],
                ['generationId', 'revision', 'step', 'epoch'],
            )) {
                issues.push(invalid('snapshot.model', 'Saved model revision is invalid.'));
            } else {
                const candidate = savedSnapshot['model'] as unknown as ModelRevision;
                if (!sameModel(candidate, evaluation.model)) {
                    issues.push(invalid('snapshot.model', 'Saved model must equal the paired evaluation model.'));
                }
            }
            if (evaluation.trigger !== 'save') {
                issues.push(invalid('snapshot.evaluation.trigger', 'Saved run evaluation trigger must be save.'));
            }
        } catch (error) {
            issues.push(invalid(
                'snapshot.evaluation',
                error instanceof Error ? error.message : 'Saved evaluation is invalid.',
            ));
        }

        const trends = parseHistory(
            savedSnapshot['trendHistory'],
            'snapshot.trendHistory',
            EXPERIMENT_MEMORY_MAX_TRENDS,
            parseLiveTrainingSignal,
        );
        issues.push(...trends.issues);
        trendHistory = trends.values as TrainingTrendPoint[];

        const evaluations = parseHistory(
            savedSnapshot['evaluationHistory'],
            'snapshot.evaluationHistory',
            EXPERIMENT_MEMORY_MAX_EVALUATIONS,
            parsePairedEvaluation,
        );
        issues.push(...evaluations.issues);
        evaluationHistory = evaluations.values as EvaluationPoint[];
    }

    if (evaluation) {
        const evaluationIds = new Set<number>();
        let previousEvaluationId = 0;
        for (let index = 0; index < evaluationHistory.length; index++) {
            const point = evaluationHistory[index];
            if (evaluationIds.has(point.evaluationId)) {
                issues.push(invalid(
                    `snapshot.evaluationHistory[${index}].evaluationId`,
                    `Duplicate evaluationId ${point.evaluationId} is not allowed.`,
                ));
            }
            if (point.evaluationId <= previousEvaluationId) {
                issues.push(invalid(
                    `snapshot.evaluationHistory[${index}].evaluationId`,
                    'Evaluation IDs must be strictly increasing.',
                ));
            }
            evaluationIds.add(point.evaluationId);
            previousEvaluationId = point.evaluationId;
            if (point.model.generationId !== evaluation.model.generationId) {
                issues.push(invalid(
                    `snapshot.evaluationHistory[${index}].model.generationId`,
                    'Evaluation history must use the saved model generation.',
                ));
            }
            if (!sameDataset(point.dataset, evaluation.dataset)) {
                issues.push(invalid(
                    `snapshot.evaluationHistory[${index}].datasetKey`,
                    'Evaluation history dataset identity must equal the saved evaluation dataset.',
                ));
            }
            if (point.objectiveKey !== evaluation.objectiveKey) {
                issues.push(invalid(
                    `snapshot.evaluationHistory[${index}].objectiveKey`,
                    'Evaluation history objective identity must equal the saved evaluation objective.',
                ));
            }
            if (point.model.revision > evaluation.model.revision) {
                issues.push(invalid(
                    `snapshot.evaluationHistory[${index}].model.revision`,
                    'Evaluation history cannot be newer than the saved evaluation.',
                ));
            }
        }
        const finalEvaluation = evaluationHistory.at(-1);
        if (!finalEvaluation || !sameEvaluation(finalEvaluation, evaluation)) {
            issues.push(invalid(
                'snapshot.evaluationHistory',
                'Evaluation history must end with the exact saved evaluation pair.',
            ));
        }

        for (let index = 0; index < trendHistory.length; index++) {
            const point = trendHistory[index];
            if (point.model.generationId !== evaluation.model.generationId) {
                issues.push(invalid(
                    `snapshot.trendHistory[${index}].model.generationId`,
                    'Trend history must use the saved model generation.',
                ));
            }
            if (!sameDataset(point.dataset, evaluation.dataset)) {
                issues.push(invalid(
                    `snapshot.trendHistory[${index}].datasetKey`,
                    'Trend history dataset identity must equal the saved evaluation dataset.',
                ));
            }
            if (point.objectiveKey !== evaluation.objectiveKey) {
                issues.push(invalid(
                    `snapshot.trendHistory[${index}].objectiveKey`,
                    'Trend history objective identity must equal the saved evaluation objective.',
                ));
            }
            if (point.model.revision > evaluation.model.revision) {
                issues.push(invalid(
                    `snapshot.trendHistory[${index}].model.revision`,
                    'Trend history cannot be newer than the saved evaluation.',
                ));
            }
        }
    }

    if (documentResult.ok && evaluation) {
        const validatedRecipe = documentResult.value.recipe;
        const expectedTrainCount = Math.floor(
            validatedRecipe.data.sampleCount * validatedRecipe.data.trainFraction,
        );
        const expectedTestCount = validatedRecipe.data.sampleCount - expectedTrainCount;
        validateEvaluationSplitCounts(
            evaluation,
            'snapshot.evaluation',
            expectedTrainCount,
            expectedTestCount,
            issues,
        );
        for (let index = 0; index < evaluationHistory.length; index++) {
            validateEvaluationSplitCounts(
                evaluationHistory[index],
                `snapshot.evaluationHistory[${index}]`,
                expectedTrainCount,
                expectedTestCount,
                issues,
            );
        }
    }

    if (issues.length > 0 || !documentResult.ok || !evaluation || !model) {
        return { ok: false, issues: Object.freeze(issues) };
    }

    const recipe = documentResult.value.recipe;
    const [recipeFingerprint, datasetKey, objectiveKey] = await Promise.all([
        fingerprintRecipe(recipe),
        fingerprintDataset(recipe),
        fingerprintObjective(recipe),
    ]);
    if (record['recipeFingerprint'] !== recipeFingerprint) {
        issues.push(invalid('recipeFingerprint', 'Recipe fingerprint does not match the saved recipe.'));
    }
    if (evaluation.dataset.datasetKey !== datasetKey) {
        issues.push(invalid('snapshot.evaluation.dataset.datasetKey', 'Saved dataset key does not match the recipe.'));
    }
    if (evaluation.dataset.generatorVersion !== getDatasetContract(recipe.task.dataset).generatorVersion) {
        issues.push(invalid('snapshot.evaluation.dataset.generatorVersion', 'Saved dataset generator version does not match the recipe.'));
    }
    if (evaluation.objectiveKey !== objectiveKey) {
        issues.push(invalid('snapshot.evaluation.objectiveKey', 'Saved objective key does not match the recipe.'));
    }
    if (issues.length > 0) return { ok: false, issues: Object.freeze(issues) };

    const validated: ExperimentRunRecordV2 = {
        kind: EXPERIMENT_MEMORY_RECORD_KIND,
        schemaVersion: EXPERIMENT_MEMORY_SCHEMA_VERSION,
        id: record['id'] as string,
        createdAt: record['createdAt'] as string,
        updatedAt: record['updatedAt'] as string,
        ...(typeof record['title'] === 'string' ? { title: record['title'] } : {}),
        recipe: recipe as ValidatedStandardExperimentRecipeV2,
        recipeFingerprint,
        snapshot: {
            model: evaluation.model,
            evaluation,
            trendHistory: Object.freeze([...trendHistory]),
            evaluationHistory: Object.freeze([...evaluationHistory]),
        },
    };
    return {
        ok: true,
        value: createStableJsonSnapshot(validated),
    };
}

function envelopeFailure(rawJson: string, issueValue: ExperimentMemoryIssue): ExperimentMemoryReadResultV2 {
    return Object.freeze({
        records: Object.freeze([]),
        rejectedRecords: Object.freeze([Object.freeze({
            sourceIndex: -1,
            rawJson,
            issues: Object.freeze([issueValue]),
        })]),
        envelopeIssues: Object.freeze([issueValue]),
    });
}

/** Read valid V2 siblings while isolating every rejected raw entry. */
export async function parseExperimentMemoryEnvelopeV2(
    rawJson: string,
): Promise<ExperimentMemoryReadResultV2> {
    if (typeof rawJson !== 'string') {
        return envelopeFailure('', invalid('$', 'Experiment memory must be supplied as JSON text.'));
    }
    if (utf8Bytes(rawJson) > EXPERIMENT_MEMORY_MAX_ENVELOPE_BYTES) {
        return envelopeFailure(rawJson, issue(
            'resource-limit',
            '$',
            `Experiment memory exceeds ${EXPERIMENT_MEMORY_MAX_ENVELOPE_BYTES} UTF-8 bytes.`,
        ));
    }

    let envelope: unknown;
    try {
        envelope = JSON.parse(rawJson) as unknown;
    } catch {
        return envelopeFailure(rawJson, invalid('$', 'Experiment memory is not valid JSON.'));
    }
    if (!isPlainRecord(envelope)
        || !hasExactKeys(envelope, ['kind', 'schemaVersion', 'records'])
        || envelope['kind'] !== EXPERIMENT_MEMORY_ENVELOPE_KIND
        || envelope['schemaVersion'] !== EXPERIMENT_MEMORY_SCHEMA_VERSION
        || !Array.isArray(envelope['records'])) {
        return envelopeFailure(rawJson, issue(
            envelope && isPlainRecord(envelope) && envelope['schemaVersion'] !== 2
                ? 'unsupported-version'
                : 'invalid-field',
            '$',
            'Experiment memory must be an exact version-2 envelope.',
        ));
    }

    const records: ExperimentRunRecordV2[] = [];
    const rejectedRecords: RejectedExperimentRunRecordV2[] = [];
    const ids = new Set<string>();
    for (let index = 0; index < envelope['records'].length; index++) {
        const candidate = envelope['records'][index];
        const candidateJson = serializedJson(candidate) ?? 'null';
        const result = await validateExperimentRunRecordV2(candidate);
        if (!result.ok) {
            rejectedRecords.push(Object.freeze({
                sourceIndex: index,
                rawJson: candidateJson,
                issues: result.issues,
            }));
            continue;
        }
        if (ids.has(result.value.id)) {
            rejectedRecords.push(Object.freeze({
                sourceIndex: index,
                rawJson: candidateJson,
                issues: Object.freeze([invalid(
                    `records[${index}].id`,
                    `Duplicate run id ${result.value.id} is not allowed.`,
                )]),
            }));
            continue;
        }
        if (records.length >= EXPERIMENT_MEMORY_MAX_RECORDS) {
            rejectedRecords.push(Object.freeze({
                sourceIndex: index,
                rawJson: candidateJson,
                issues: Object.freeze([issue(
                    'resource-limit',
                    `records[${index}]`,
                    `Experiment memory supports at most ${EXPERIMENT_MEMORY_MAX_RECORDS} valid records.`,
                )]),
            }));
            continue;
        }
        ids.add(result.value.id);
        records.push(result.value);
    }

    return Object.freeze({
        records: Object.freeze(records),
        rejectedRecords: Object.freeze(rejectedRecords),
        envelopeIssues: Object.freeze([]),
    });
}

/** Validate every candidate before returning bytes suitable for one storage write. */
export async function serializeExperimentMemoryEnvelopeV2(
    records: readonly ExperimentRunRecordV2[],
    preservedRejectedRawJson: readonly string[] = [],
): Promise<ExperimentMemoryResult<string>> {
    let recordSnapshots: readonly ExperimentRunRecordV2[];
    let rawSnapshots: readonly string[];
    try {
        recordSnapshots = createStableJsonSnapshot([...records]);
        rawSnapshots = createStableJsonSnapshot([...preservedRejectedRawJson]);
    } catch (error) {
        return { ok: false, issues: [invalid('$', error instanceof Error ? error.message : 'Storage snapshot failed.')] };
    }

    const totalCount = recordSnapshots.length + rawSnapshots.length;
    if (totalCount > EXPERIMENT_MEMORY_MAX_RECORDS) {
        return {
            ok: false,
            issues: [issue(
                'resource-limit',
                'records',
                `Saving ${totalCount} records would exceed the ${EXPERIMENT_MEMORY_MAX_RECORDS}-record limit.`,
            )],
        };
    }

    const validated: ExperimentRunRecordV2[] = [];
    const issues: ExperimentMemoryIssue[] = [];
    const ids = new Set<string>();
    for (let index = 0; index < recordSnapshots.length; index++) {
        const result = await validateExperimentRunRecordV2(recordSnapshots[index]);
        if (!result.ok) {
            issues.push(...result.issues.map((entry) => issue(
                entry.code,
                `records[${index}].${entry.path}`,
                entry.message,
            )));
            continue;
        }
        if (ids.has(result.value.id)) {
            issues.push(invalid(`records[${index}].id`, `Duplicate run id ${result.value.id} is not allowed.`));
            continue;
        }
        ids.add(result.value.id);
        validated.push(result.value);
    }
    if (issues.length > 0) return { ok: false, issues: Object.freeze(issues) };

    const rejectedValues: unknown[] = [];
    for (let index = 0; index < rawSnapshots.length; index++) {
        try {
            rejectedValues.push(JSON.parse(rawSnapshots[index]) as unknown);
        } catch {
            return {
                ok: false,
                issues: [invalid(
                    `preservedRejectedRawJson[${index}]`,
                    'Rejected record bytes must remain valid JSON values.',
                )],
            };
        }
    }

    const envelope = {
        kind: EXPERIMENT_MEMORY_ENVELOPE_KIND,
        schemaVersion: EXPERIMENT_MEMORY_SCHEMA_VERSION,
        records: [...validated, ...rejectedValues],
    };
    const json = JSON.stringify(envelope);
    if (utf8Bytes(json) > EXPERIMENT_MEMORY_MAX_ENVELOPE_BYTES) {
        return {
            ok: false,
            issues: [issue(
                'resource-limit',
                '$',
                `Experiment memory exceeds ${EXPERIMENT_MEMORY_MAX_ENVELOPE_BYTES} UTF-8 bytes.`,
            )],
        };
    }
    return { ok: true, value: json };
}
