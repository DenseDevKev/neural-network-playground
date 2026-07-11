import {
    DATASET_IDS,
    compileExperimentRecipe,
    getDatasetContract,
    type CompiledExperimentConfig,
    type DatasetId,
    type ScalarActivationType,
    type TaskKind,
    type WeightInitType,
} from '@nn-playground/engine';
import type {
    ExperimentSchemaIssue,
    ExperimentSchemaIssueCode,
    SchemaResult,
    ValidatedExperimentDocumentV2,
    ValidatedStandardExperimentRecipeV2,
} from './types.js';

export const EXPERIMENT_SCHEMA_VERSION = 2 as const;

const MAX_SCHEMA_ISSUES = 100;
const MAX_UINT32 = 4_294_967_295;
const MAX_PARAMETER_VALUE = 1_000_000;
const MAX_SCHEDULE_STEPS = 1_000_000_000;
const MAX_TRAINABLE_PARAMETERS = 2_000;

const FEATURE_IDS = [
    'x',
    'y',
    'xSquared',
    'ySquared',
    'xy',
    'sinX',
    'sinY',
    'cosX',
    'cosY',
] as const;
const FEATURE_INDEX = new Map<string, number>(
    FEATURE_IDS.map((featureId, index) => [featureId, index]),
);
const DATASET_ID_SET = new Set<string>(DATASET_IDS);
const HIDDEN_ACTIVATIONS = new Set<ScalarActivationType>([
    'relu',
    'tanh',
    'sigmoid',
    'linear',
    'leakyRelu',
    'elu',
    'swish',
    'softplus',
]);
const INITIALIZATIONS = new Set<WeightInitType>(['xavier', 'he', 'uniform', 'zeros']);
const TASK_KINDS = new Set<TaskKind>([
    'binary-classification',
    'multiclass-classification',
    'regression',
]);
const DATA_LOSS_KINDS = new Set([
    'binary-cross-entropy-with-logits',
    'categorical-cross-entropy-with-logits',
    'mean-squared-error',
    'huber',
]);

type UnknownRecord = Record<string, unknown>;

class IssueCollector {
    readonly issues: ExperimentSchemaIssue[] = [];

    add(code: ExperimentSchemaIssueCode, path: string, message: string): void {
        if (this.issues.length >= MAX_SCHEMA_ISSUES) return;
        this.issues.push({ code, path, message });
    }
}

function isRecord(value: unknown): value is UnknownRecord {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function hasOwn(record: UnknownRecord, key: string): boolean {
    return Object.prototype.hasOwnProperty.call(record, key);
}

function displayPath(path: string): string {
    return path || '$';
}

function childPath(path: string, key: string): string {
    return path ? `${path}.${key}` : key;
}

function requireRecord(
    value: unknown,
    path: string,
    collector: IssueCollector,
): UnknownRecord | undefined {
    if (isRecord(value)) return value;
    collector.add(
        'invalid-field',
        displayPath(path),
        `${displayPath(path)} must be a record`,
    );
    return undefined;
}

function checkExactKeys(
    record: UnknownRecord,
    path: string,
    expectedKeys: readonly string[],
    collector: IssueCollector,
): void {
    const expected = new Set(expectedKeys);
    for (const key of Object.keys(record)) {
        if (!expected.has(key)) {
            const issuePath = childPath(path, key);
            collector.add('unknown-field', issuePath, `${issuePath} is not allowed`);
        }
    }
    for (const key of expectedKeys) {
        if (!hasOwn(record, key)) {
            const issuePath = childPath(path, key);
            collector.add('missing-field', issuePath, `${issuePath} is required`);
        }
    }
}

function exactRecord(
    value: unknown,
    path: string,
    expectedKeys: readonly string[],
    collector: IssueCollector,
): UnknownRecord | undefined {
    const record = requireRecord(value, path, collector);
    if (record) checkExactKeys(record, path, expectedKeys, collector);
    return record;
}

interface NumberRules {
    minimum: number;
    maximum: number;
    integer?: boolean;
    minimumExclusive?: boolean;
    maximumExclusive?: boolean;
}

function validateNumberValue(
    value: unknown,
    path: string,
    rules: NumberRules,
    collector: IssueCollector,
): number | undefined {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
        collector.add('invalid-field', path, `${path} must be a finite number`);
        return undefined;
    }
    if (rules.integer && !Number.isInteger(value)) {
        collector.add('invalid-field', path, `${path} must be an integer`);
        return undefined;
    }
    const belowMinimum = rules.minimumExclusive
        ? value <= rules.minimum
        : value < rules.minimum;
    const aboveMaximum = rules.maximumExclusive
        ? value >= rules.maximum
        : value > rules.maximum;
    if (belowMinimum || aboveMaximum) {
        const left = rules.minimumExclusive ? '(' : '[';
        const right = rules.maximumExclusive ? ')' : ']';
        collector.add(
            'out-of-range',
            path,
            `${path} must be in ${left}${rules.minimum}, ${rules.maximum}${right}`,
        );
        return undefined;
    }
    return value;
}

function validateNumberField(
    record: UnknownRecord,
    key: string,
    path: string,
    rules: NumberRules,
    collector: IssueCollector,
): number | undefined {
    if (!hasOwn(record, key)) return undefined;
    return validateNumberValue(record[key], childPath(path, key), rules, collector);
}

function validateLiteralField(
    record: UnknownRecord,
    key: string,
    path: string,
    expected: unknown,
    collector: IssueCollector,
): boolean {
    if (!hasOwn(record, key)) return false;
    if (record[key] === expected) return true;
    const issuePath = childPath(path, key);
    collector.add(
        'invalid-field',
        issuePath,
        `${issuePath} must be ${JSON.stringify(expected)}`,
    );
    return false;
}

function validateData(
    value: unknown,
    collector: IssueCollector,
): { sampleCount?: number; trainFraction?: number; trainCount?: number } {
    const path = 'recipe.data';
    const data = exactRecord(
        value,
        path,
        ['sampleCount', 'trainFraction', 'noise', 'seed'],
        collector,
    );
    if (!data) return {};

    const sampleCount = validateNumberField(
        data,
        'sampleCount',
        path,
        { minimum: 2, maximum: 1_000, integer: true },
        collector,
    );
    const trainFraction = validateNumberField(
        data,
        'trainFraction',
        path,
        { minimum: 0.1, maximum: 0.9 },
        collector,
    );
    validateNumberField(
        data,
        'noise',
        path,
        { minimum: 0, maximum: 100 },
        collector,
    );
    validateNumberField(
        data,
        'seed',
        path,
        { minimum: 0, maximum: MAX_UINT32, integer: true },
        collector,
    );

    let trainCount: number | undefined;
    if (sampleCount !== undefined && trainFraction !== undefined) {
        trainCount = Math.floor(sampleCount * trainFraction);
        const testCount = sampleCount - trainCount;
        if (trainCount < 1 || testCount < 1) {
            collector.add(
                'resource-limit',
                `${path}.trainFraction`,
                'recipe.data split must produce positive train and test populations',
            );
            trainCount = undefined;
        }
    }
    return { sampleCount, trainFraction, trainCount };
}

function validateInputs(value: unknown, collector: IssueCollector): number | undefined {
    const path = 'recipe.inputs';
    const inputs = exactRecord(value, path, ['featureIds'], collector);
    if (!inputs || !hasOwn(inputs, 'featureIds')) return undefined;
    const featureIds = inputs.featureIds;
    if (!Array.isArray(featureIds)) {
        collector.add(
            'invalid-field',
            `${path}.featureIds`,
            'recipe.inputs.featureIds must be an array',
        );
        return undefined;
    }
    if (featureIds.length < 1 || featureIds.length > FEATURE_IDS.length) {
        collector.add(
            'resource-limit',
            `${path}.featureIds`,
            `recipe.inputs.featureIds must contain 1 through ${FEATURE_IDS.length} features`,
        );
    }

    const seen = new Set<string>();
    let previousIndex = -1;
    for (let index = 0; index < featureIds.length; index++) {
        const featureId = featureIds[index];
        const issuePath = `${path}.featureIds[${index}]`;
        if (typeof featureId !== 'string' || !FEATURE_INDEX.has(featureId)) {
            collector.add('invalid-field', issuePath, `${issuePath} is not a registered feature`);
            continue;
        }
        if (seen.has(featureId)) {
            collector.add('duplicate-feature', issuePath, `${issuePath} duplicates ${featureId}`);
            continue;
        }
        seen.add(featureId);
        const registryIndex = FEATURE_INDEX.get(featureId)!;
        if (registryIndex <= previousIndex) {
            collector.add(
                'invalid-field',
                issuePath,
                `${issuePath} is not in canonical feature registry order`,
            );
        }
        previousIndex = registryIndex;
    }
    return featureIds.length;
}

function validateHiddenLayers(
    value: unknown,
    collector: IssueCollector,
): number[] | undefined {
    const path = 'recipe.model.hiddenLayers';
    if (!Array.isArray(value)) {
        collector.add('invalid-field', path, `${path} must be an array`);
        return undefined;
    }
    if (value.length > 6) {
        collector.add('resource-limit', path, `${path} cannot contain more than 6 layers`);
    }
    const widths: number[] = [];
    let widthsAreIntegers = true;
    for (let index = 0; index < value.length; index++) {
        const width = value[index];
        const issuePath = `${path}[${index}]`;
        const validated = validateNumberValue(
            width,
            issuePath,
            { minimum: 1, maximum: 16, integer: true },
            collector,
        );
        if (validated !== undefined) {
            widths.push(validated);
        } else if (typeof width === 'number' && Number.isInteger(width) && width > 0) {
            widths.push(width);
        } else {
            widthsAreIntegers = false;
        }
    }
    return widthsAreIntegers ? widths : undefined;
}

function validateModel(
    value: unknown,
    collector: IssueCollector,
): number[] | undefined {
    const path = 'recipe.model';
    const model = exactRecord(
        value,
        path,
        ['hiddenLayers', 'hiddenActivation', 'initialization', 'seed'],
        collector,
    );
    if (!model) return undefined;

    const hiddenLayers = hasOwn(model, 'hiddenLayers')
        ? validateHiddenLayers(model.hiddenLayers, collector)
        : undefined;
    if (hasOwn(model, 'hiddenActivation')
        && (typeof model.hiddenActivation !== 'string'
            || !HIDDEN_ACTIVATIONS.has(model.hiddenActivation as ScalarActivationType))) {
        collector.add(
            'invalid-field',
            `${path}.hiddenActivation`,
            'recipe.model.hiddenActivation is not a scalar activation',
        );
    }
    if (hasOwn(model, 'initialization')
        && (typeof model.initialization !== 'string'
            || !INITIALIZATIONS.has(model.initialization as WeightInitType))) {
        collector.add(
            'invalid-field',
            `${path}.initialization`,
            'recipe.model.initialization is not supported',
        );
    }
    validateNumberField(
        model,
        'seed',
        path,
        { minimum: 0, maximum: MAX_UINT32, integer: true },
        collector,
    );
    return hiddenLayers;
}

function validateSchedule(
    value: unknown,
    learningRate: number | undefined,
    collector: IssueCollector,
): void {
    const path = 'recipe.training.schedule';
    const schedule = requireRecord(value, path, collector);
    if (!schedule) return;
    const kind = schedule.kind;
    if (kind === 'constant') {
        checkExactKeys(schedule, path, ['kind'], collector);
        return;
    }
    if (kind === 'step') {
        checkExactKeys(schedule, path, ['kind', 'interval', 'gamma'], collector);
        validateNumberField(
            schedule,
            'interval',
            path,
            { minimum: 1, maximum: MAX_SCHEDULE_STEPS, integer: true },
            collector,
        );
        validateNumberField(
            schedule,
            'gamma',
            path,
            { minimum: 0, maximum: 1, minimumExclusive: true },
            collector,
        );
        return;
    }
    if (kind === 'cosine') {
        checkExactKeys(schedule, path, ['kind', 'totalSteps', 'minimumRate'], collector);
        validateNumberField(
            schedule,
            'totalSteps',
            path,
            { minimum: 1, maximum: MAX_SCHEDULE_STEPS, integer: true },
            collector,
        );
        if (hasOwn(schedule, 'minimumRate')) {
            validateNumberValue(
                schedule.minimumRate,
                `${path}.minimumRate`,
                { minimum: 0, maximum: learningRate ?? Number.MAX_VALUE },
                collector,
            );
        }
        return;
    }
    checkExactKeys(schedule, path, ['kind'], collector);
    if (hasOwn(schedule, 'kind')) {
        collector.add('invalid-field', `${path}.kind`, `${path}.kind is not supported`);
    }
}

function validateOptimizer(value: unknown, collector: IssueCollector): void {
    const path = 'recipe.training.optimizer';
    const optimizer = requireRecord(value, path, collector);
    if (!optimizer) return;
    const kind = optimizer.kind;
    if (kind === 'sgd') {
        checkExactKeys(optimizer, path, ['kind'], collector);
        return;
    }
    if (kind === 'sgd-momentum') {
        checkExactKeys(optimizer, path, ['kind', 'momentum'], collector);
        validateNumberField(
            optimizer,
            'momentum',
            path,
            { minimum: 0, maximum: 1, maximumExclusive: true },
            collector,
        );
        return;
    }
    if (kind === 'adam') {
        checkExactKeys(optimizer, path, ['kind', 'beta1', 'beta2', 'epsilon'], collector);
        for (const key of ['beta1', 'beta2'] as const) {
            validateNumberField(
                optimizer,
                key,
                path,
                { minimum: 0, maximum: 1, maximumExclusive: true },
                collector,
            );
        }
        validateNumberField(
            optimizer,
            'epsilon',
            path,
            { minimum: 0, maximum: 1, minimumExclusive: true },
            collector,
        );
        return;
    }
    checkExactKeys(optimizer, path, ['kind'], collector);
    if (hasOwn(optimizer, 'kind')) {
        collector.add('invalid-field', `${path}.kind`, `${path}.kind is not supported`);
    }
}

function validateGradientClipping(value: unknown, collector: IssueCollector): void {
    const path = 'recipe.training.gradientClipping';
    const clipping = requireRecord(value, path, collector);
    if (!clipping) return;
    const kind = clipping.kind;
    if (kind === 'none') {
        checkExactKeys(clipping, path, ['kind'], collector);
        return;
    }
    if (kind === 'global-norm') {
        checkExactKeys(clipping, path, ['kind', 'maximumNorm', 'scope'], collector);
        validateNumberField(
            clipping,
            'maximumNorm',
            path,
            { minimum: 0, maximum: MAX_PARAMETER_VALUE, minimumExclusive: true },
            collector,
        );
        validateLiteralField(
            clipping,
            'scope',
            path,
            'total-objective-gradient',
            collector,
        );
        return;
    }
    checkExactKeys(clipping, path, ['kind'], collector);
    if (hasOwn(clipping, 'kind')) {
        collector.add('invalid-field', `${path}.kind`, `${path}.kind is not supported`);
    }
}

function validateTraining(
    value: unknown,
    trainCount: number | undefined,
    collector: IssueCollector,
): void {
    const path = 'recipe.training';
    const training = exactRecord(
        value,
        path,
        ['batchSize', 'learningRate', 'schedule', 'optimizer', 'gradientClipping'],
        collector,
    );
    if (!training) return;

    const batchSize = validateNumberField(
        training,
        'batchSize',
        path,
        { minimum: 1, maximum: 512, integer: true },
        collector,
    );
    const learningRate = validateNumberField(
        training,
        'learningRate',
        path,
        { minimum: 0, maximum: 10, minimumExclusive: true },
        collector,
    );
    if (batchSize !== undefined && trainCount !== undefined && batchSize > trainCount) {
        collector.add(
            'resource-limit',
            `${path}.batchSize`,
            'recipe.training.batchSize cannot exceed the derived training population',
        );
    }
    if (hasOwn(training, 'schedule')) {
        validateSchedule(training.schedule, learningRate, collector);
    }
    if (hasOwn(training, 'optimizer')) validateOptimizer(training.optimizer, collector);
    if (hasOwn(training, 'gradientClipping')) {
        validateGradientClipping(training.gradientClipping, collector);
    }
}

interface ValidatedTaskShape {
    kind?: TaskKind;
    outputSize?: 1 | 3;
}

function validateTask(value: unknown, collector: IssueCollector): ValidatedTaskShape {
    const path = 'recipe.task';
    const task = exactRecord(value, path, ['kind', 'dataset'], collector);
    if (!task) return {};

    let kind: TaskKind | undefined;
    if (hasOwn(task, 'kind')) {
        if (typeof task.kind === 'string' && TASK_KINDS.has(task.kind as TaskKind)) {
            kind = task.kind as TaskKind;
        } else {
            collector.add('invalid-field', `${path}.kind`, `${path}.kind is not supported`);
        }
    }

    if (hasOwn(task, 'dataset')) {
        if (typeof task.dataset !== 'string' || !DATASET_ID_SET.has(task.dataset)) {
            collector.add(
                'invalid-field',
                `${path}.dataset`,
                `${path}.dataset is not registered`,
            );
        } else if (kind !== undefined
            && getDatasetContract(task.dataset as DatasetId).taskKind !== kind) {
            collector.add(
                'incompatible-task',
                `${path}.dataset`,
                `${task.dataset} is incompatible with task ${kind}`,
            );
        }
    }

    return {
        kind,
        outputSize: kind === 'multiclass-classification' ? 3 : kind === undefined ? undefined : 1,
    };
}

function validateDataLoss(
    value: unknown,
    taskKind: TaskKind | undefined,
    collector: IssueCollector,
): void {
    const path = 'recipe.objective.dataLoss';
    const loss = requireRecord(value, path, collector);
    if (!loss) return;
    const kind = loss.kind;
    checkExactKeys(loss, path, kind === 'huber' ? ['kind', 'delta'] : ['kind'], collector);
    if (kind === 'huber') {
        validateNumberField(
            loss,
            'delta',
            path,
            { minimum: 0, maximum: MAX_PARAMETER_VALUE, minimumExclusive: true },
            collector,
        );
    }

    if (typeof kind !== 'string' || !DATA_LOSS_KINDS.has(kind)) {
        if (hasOwn(loss, 'kind')) {
            collector.add('invalid-field', `${path}.kind`, `${path}.kind is not supported`);
        }
        return;
    }
    const compatible = taskKind === 'binary-classification'
        ? kind === 'binary-cross-entropy-with-logits'
        : taskKind === 'multiclass-classification'
            ? kind === 'categorical-cross-entropy-with-logits'
            : taskKind === 'regression'
                ? kind === 'mean-squared-error' || kind === 'huber'
                : true;
    if (!compatible) {
        collector.add(
            'incompatible-task',
            `${path}.kind`,
            `${kind} is incompatible with task ${taskKind}`,
        );
    }
}

function validatePenalty(value: unknown, collector: IssueCollector): void {
    const path = 'recipe.objective.penalty';
    const penalty = requireRecord(value, path, collector);
    if (!penalty) return;
    const kind = penalty.kind;
    if (kind === 'none') {
        checkExactKeys(penalty, path, ['kind'], collector);
        return;
    }
    if (kind === 'l1' || kind === 'l2') {
        checkExactKeys(penalty, path, ['kind', 'coefficient', 'applyTo'], collector);
        validateNumberField(
            penalty,
            'coefficient',
            path,
            { minimum: 0, maximum: 1, minimumExclusive: true },
            collector,
        );
        validateLiteralField(penalty, 'applyTo', path, 'weights', collector);
        return;
    }
    checkExactKeys(penalty, path, ['kind'], collector);
    if (hasOwn(penalty, 'kind')) {
        collector.add('invalid-field', `${path}.kind`, `${path}.kind is not supported`);
    }
}

function validateObjective(
    value: unknown,
    taskKind: TaskKind | undefined,
    collector: IssueCollector,
): void {
    const path = 'recipe.objective';
    const objective = exactRecord(
        value,
        path,
        ['dataLoss', 'penalty', 'reduction'],
        collector,
    );
    if (!objective) return;
    if (hasOwn(objective, 'dataLoss')) {
        validateDataLoss(objective.dataLoss, taskKind, collector);
    }
    if (hasOwn(objective, 'penalty')) validatePenalty(objective.penalty, collector);
    validateLiteralField(objective, 'reduction', path, 'mean-per-sample', collector);
}

function validateParameterCount(
    featureCount: number | undefined,
    hiddenLayers: readonly number[] | undefined,
    outputSize: 1 | 3 | undefined,
    collector: IssueCollector,
): void {
    if (featureCount === undefined || hiddenLayers === undefined || outputSize === undefined) {
        return;
    }
    const sizes = [featureCount, ...hiddenLayers, outputSize];
    let parameterCount = 0;
    for (let index = 1; index < sizes.length; index++) {
        parameterCount += (sizes[index - 1] + 1) * sizes[index];
    }
    if (parameterCount > MAX_TRAINABLE_PARAMETERS) {
        collector.add(
            'resource-limit',
            'recipe.model.hiddenLayers',
            `architecture has ${parameterCount} trainable parameters; maximum is ${MAX_TRAINABLE_PARAMETERS}`,
        );
    }
}

function validateRecipe(value: unknown, collector: IssueCollector): void {
    const path = 'recipe';
    const recipe = exactRecord(
        value,
        path,
        ['data', 'inputs', 'model', 'training', 'task', 'objective'],
        collector,
    );
    if (!recipe) return;

    const data = hasOwn(recipe, 'data') ? validateData(recipe.data, collector) : {};
    const featureCount = hasOwn(recipe, 'inputs')
        ? validateInputs(recipe.inputs, collector)
        : undefined;
    const hiddenLayers = hasOwn(recipe, 'model')
        ? validateModel(recipe.model, collector)
        : undefined;
    const task = hasOwn(recipe, 'task') ? validateTask(recipe.task, collector) : {};
    if (hasOwn(recipe, 'training')) {
        validateTraining(recipe.training, data.trainCount, collector);
    }
    if (hasOwn(recipe, 'objective')) {
        validateObjective(recipe.objective, task.kind, collector);
    }
    validateParameterCount(featureCount, hiddenLayers, task.outputSize, collector);
}

function validateView(value: unknown, collector: IssueCollector): void {
    const path = 'view';
    const view = exactRecord(value, path, ['showTestData', 'discretizeOutput'], collector);
    if (!view) return;
    for (const key of ['showTestData', 'discretizeOutput'] as const) {
        if (hasOwn(view, key) && typeof view[key] !== 'boolean') {
            collector.add(
                'invalid-field',
                `${path}.${key}`,
                `${path}.${key} must be a boolean`,
            );
        }
    }
}

function looksLikeLegacyAppState(value: unknown): boolean {
    if (!isRecord(value) || hasOwn(value, 'schemaVersion')) return false;
    return ['network', 'training', 'data', 'features', 'ui'].every((key) => hasOwn(value, key));
}

export function validateExperimentDocument(
    value: unknown,
): SchemaResult<ValidatedExperimentDocumentV2> {
    const collector = new IssueCollector();
    if (looksLikeLegacyAppState(value)) {
        collector.add(
            'legacy-state',
            '$',
            'legacy application state is not a version-2 experiment document',
        );
        return { ok: false, issues: collector.issues };
    }

    const document = exactRecord(
        value,
        '',
        ['kind', 'schemaVersion', 'recipe', 'view'],
        collector,
    );
    if (document) {
        validateLiteralField(document, 'kind', '', 'nn-playground-experiment', collector);
        if (hasOwn(document, 'schemaVersion')) {
            if (typeof document.schemaVersion !== 'number'
                || !Number.isFinite(document.schemaVersion)
                || !Number.isInteger(document.schemaVersion)) {
                collector.add(
                    'invalid-field',
                    'schemaVersion',
                    'schemaVersion must be the integer 2',
                );
            } else if (document.schemaVersion !== EXPERIMENT_SCHEMA_VERSION) {
                collector.add(
                    'unsupported-version',
                    'schemaVersion',
                    `schemaVersion ${document.schemaVersion} is unsupported`,
                );
            }
        }
        if (hasOwn(document, 'recipe')) validateRecipe(document.recipe, collector);
        if (hasOwn(document, 'view')) validateView(document.view, collector);
    }

    if (collector.issues.length > 0) return { ok: false, issues: collector.issues };
    return {
        ok: true,
        value: value as ValidatedExperimentDocumentV2,
    };
}

/** Package-internal compiler seam; public callers enter through preparation in Task 5. */
export function compileValidatedExperiment(
    recipe: ValidatedStandardExperimentRecipeV2,
): CompiledExperimentConfig {
    return compileExperimentRecipe(recipe);
}
