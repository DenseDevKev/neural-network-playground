import {
    canonicalizeJson,
    parseLiveTrainingSignal,
    parsePairedEvaluation,
} from '@nn-playground/shared';
import type {
    EvaluationPoint,
    EvaluationTrigger,
    EvaluationValues,
    LiveTrainingSignal,
    PairedEvaluation,
    TrainingTrendPoint,
} from '@nn-playground/shared';

export const DEFAULT_TREND_HISTORY_CAPACITY = 4_096;
export const DEFAULT_EVALUATION_HISTORY_CAPACITY = 1_024;

export interface MetricHistoryBufferOptions {
    readonly trendCapacity?: number;
    readonly evaluationCapacity?: number;
}

export interface MetricHistoryVersions {
    readonly trendVersion: number;
    readonly evaluationVersion: number;
}

export interface MetricHistorySnapshot extends MetricHistoryVersions {
    readonly trendHistory: readonly TrainingTrendPoint[];
    readonly evaluationHistory: readonly EvaluationPoint[];
}

export interface PreparedMetricHistoryReplacement {
    readonly versions: MetricHistoryVersions;
    /** Idempotent, validation-free publication of the prepared replacement. */
    commit(): void;
}

export interface PreparedMetricHistoryAppend {
    readonly versions: MetricHistoryVersions;
    readonly evaluationAppended: boolean;
    commit(): void;
}

function capacity(value: number, path: string): number {
    if (!Number.isSafeInteger(value) || value < 1 || value > 1_000_000) {
        throw new RangeError(`${path} must be a safe integer from 1 to 1000000`);
    }
    return value;
}

function nextVersion(value: number, path: string): number {
    if (value >= Number.MAX_SAFE_INTEGER) {
        throw new RangeError(`${path} exhausted its safe integer range`);
    }
    return value + 1;
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

class PackedTrendStorage {
    private readonly generationId: Float64Array;
    private readonly revision: Float64Array;
    private readonly step: Float64Array;
    private readonly epoch: Float64Array;
    private readonly generatorVersion: Float64Array;
    private readonly trainCount: Float64Array;
    private readonly testCount: Float64Array;
    private readonly alpha: Float64Array;
    private readonly latestBatchSize: Float64Array;
    private readonly throughStep: Float64Array;
    private readonly dataLoss: Float64Array;
    private readonly datasetKey: string[];
    private readonly objectiveKey: string[];
    private readonly materialized: Array<TrainingTrendPoint | undefined>;
    private start = 0;
    private count = 0;

    constructor(private readonly capacity: number) {
        this.generationId = new Float64Array(capacity);
        this.revision = new Float64Array(capacity);
        this.step = new Float64Array(capacity);
        this.epoch = new Float64Array(capacity);
        this.generatorVersion = new Float64Array(capacity);
        this.trainCount = new Float64Array(capacity);
        this.testCount = new Float64Array(capacity);
        this.alpha = new Float64Array(capacity);
        this.latestBatchSize = new Float64Array(capacity);
        this.throughStep = new Float64Array(capacity);
        this.dataLoss = new Float64Array(capacity);
        this.datasetKey = Array.from({ length: capacity }, () => '');
        this.objectiveKey = Array.from({ length: capacity }, () => '');
        this.materialized = Array.from({ length: capacity });
    }

    append(point: LiveTrainingSignal): void {
        const index = this.nextWriteIndex();
        this.generationId[index] = point.model.generationId;
        this.revision[index] = point.model.revision;
        this.step[index] = point.model.step;
        this.epoch[index] = point.model.epoch;
        this.generatorVersion[index] = point.dataset.generatorVersion;
        this.datasetKey[index] = point.dataset.datasetKey;
        this.trainCount[index] = point.dataset.trainCount;
        this.testCount[index] = point.dataset.testCount;
        this.objectiveKey[index] = point.objectiveKey;
        this.alpha[index] = point.basis.alpha;
        this.latestBatchSize[index] = point.basis.latestBatchSize;
        this.throughStep[index] = point.basis.throughStep;
        this.dataLoss[index] = point.dataLoss;
        this.materialized[index] = deepFreeze(point) as TrainingTrendPoint;
    }

    read(): readonly TrainingTrendPoint[] {
        const points: TrainingTrendPoint[] = [];
        for (let offset = 0; offset < this.count; offset++) {
            const index = (this.start + offset) % this.capacity;
            const point = this.materialized[index];
            if (point !== undefined) points.push(point);
        }
        return Object.freeze(points);
    }

    reset(): void {
        this.start = 0;
        this.count = 0;
        this.datasetKey.fill('');
        this.objectiveKey.fill('');
        this.materialized.fill(undefined);
    }

    private nextWriteIndex(): number {
        if (this.count < this.capacity) {
            const index = (this.start + this.count) % this.capacity;
            this.count++;
            return index;
        }
        const index = this.start;
        this.start = (this.start + 1) % this.capacity;
        return index;
    }
}

class PackedEvaluationStorage {
    private readonly evaluationId: Float64Array;
    private readonly generationId: Float64Array;
    private readonly revision: Float64Array;
    private readonly step: Float64Array;
    private readonly epoch: Float64Array;
    private readonly generatorVersion: Float64Array;
    private readonly trainCount: Float64Array;
    private readonly testCount: Float64Array;
    private readonly trainDataLoss: Float64Array;
    private readonly testDataLoss: Float64Array;
    private readonly trainAccuracy: Float64Array;
    private readonly testAccuracy: Float64Array;
    private readonly hasTrainAccuracy: Uint8Array;
    private readonly hasTestAccuracy: Uint8Array;
    private readonly regularizationPenalty: Float64Array;
    private readonly trainTotalObjective: Float64Array;
    private readonly datasetKey: string[];
    private readonly objectiveKey: string[];
    private readonly trigger: EvaluationTrigger[];
    private readonly trainConfusionJson: Array<string | null>;
    private readonly testConfusionJson: Array<string | null>;
    private readonly materialized: Array<EvaluationPoint | undefined>;
    private readonly retainedEvaluationFingerprints = new Map<number, string>();
    private highestEvaluationId = 0;
    private start = 0;
    private count = 0;

    constructor(private readonly capacity: number) {
        this.evaluationId = new Float64Array(capacity);
        this.generationId = new Float64Array(capacity);
        this.revision = new Float64Array(capacity);
        this.step = new Float64Array(capacity);
        this.epoch = new Float64Array(capacity);
        this.generatorVersion = new Float64Array(capacity);
        this.trainCount = new Float64Array(capacity);
        this.testCount = new Float64Array(capacity);
        this.trainDataLoss = new Float64Array(capacity);
        this.testDataLoss = new Float64Array(capacity);
        this.trainAccuracy = new Float64Array(capacity);
        this.testAccuracy = new Float64Array(capacity);
        this.hasTrainAccuracy = new Uint8Array(capacity);
        this.hasTestAccuracy = new Uint8Array(capacity);
        this.regularizationPenalty = new Float64Array(capacity);
        this.trainTotalObjective = new Float64Array(capacity);
        this.datasetKey = Array.from({ length: capacity }, () => '');
        this.objectiveKey = Array.from({ length: capacity }, () => '');
        this.trigger = Array.from({ length: capacity }, () => 'initial' as const);
        this.trainConfusionJson = Array.from({ length: capacity }, () => null);
        this.testConfusionJson = Array.from({ length: capacity }, () => null);
        this.materialized = Array.from({ length: capacity });
    }

    preflight(point: PairedEvaluation): boolean {
        const existing = this.retainedEvaluationFingerprints.get(point.evaluationId);
        if (point.evaluationId <= this.highestEvaluationId) {
            if (existing === undefined) return false;
            const fingerprint = canonicalizeJson(point);
            if (existing === fingerprint) return false;
            throw new TypeError(`conflicting evaluationId ${point.evaluationId}`);
        }
        return true;
    }

    append(point: PairedEvaluation): boolean {
        if (!this.preflight(point)) return false;
        const fingerprint = canonicalizeJson(point);
        const evicted = this.count === this.capacity
            ? this.materialized[this.start]
            : undefined;
        const index = this.nextWriteIndex();
        this.evaluationId[index] = point.evaluationId;
        this.trigger[index] = point.trigger;
        this.generationId[index] = point.model.generationId;
        this.revision[index] = point.model.revision;
        this.step[index] = point.model.step;
        this.epoch[index] = point.model.epoch;
        this.generatorVersion[index] = point.dataset.generatorVersion;
        this.datasetKey[index] = point.dataset.datasetKey;
        this.trainCount[index] = point.dataset.trainCount;
        this.testCount[index] = point.dataset.testCount;
        this.objectiveKey[index] = point.objectiveKey;
        this.trainDataLoss[index] = point.train.values.dataLoss;
        this.testDataLoss[index] = point.test.values.dataLoss;
        this.storeAccuracy(index, point.train.values, true);
        this.storeAccuracy(index, point.test.values, false);
        this.trainConfusionJson[index] = this.confusionJson(point.train.values);
        this.testConfusionJson[index] = this.confusionJson(point.test.values);
        this.regularizationPenalty[index] = point.objective.regularizationPenalty;
        this.trainTotalObjective[index] = point.objective.trainTotalObjective;
        this.materialized[index] = deepFreeze(point) as EvaluationPoint;
        if (evicted !== undefined) {
            this.retainedEvaluationFingerprints.delete(evicted.evaluationId);
        }
        this.retainedEvaluationFingerprints.set(point.evaluationId, fingerprint);
        this.highestEvaluationId = point.evaluationId;
        return true;
    }

    read(): readonly EvaluationPoint[] {
        const points: EvaluationPoint[] = [];
        for (let offset = 0; offset < this.count; offset++) {
            const index = (this.start + offset) % this.capacity;
            const point = this.materialized[index];
            if (point !== undefined) points.push(point);
        }
        return Object.freeze(points);
    }

    reset(): void {
        this.start = 0;
        this.count = 0;
        this.retainedEvaluationFingerprints.clear();
        this.highestEvaluationId = 0;
        this.datasetKey.fill('');
        this.objectiveKey.fill('');
        this.trainConfusionJson.fill(null);
        this.testConfusionJson.fill(null);
        this.materialized.fill(undefined);
    }

    private nextWriteIndex(): number {
        if (this.count < this.capacity) {
            const index = (this.start + this.count) % this.capacity;
            this.count++;
            return index;
        }
        const index = this.start;
        this.start = (this.start + 1) % this.capacity;
        return index;
    }

    private storeAccuracy(index: number, values: EvaluationValues, train: boolean): void {
        const target = train ? this.trainAccuracy : this.testAccuracy;
        const flags = train ? this.hasTrainAccuracy : this.hasTestAccuracy;
        if (values.accuracy === undefined) {
            target[index] = 0;
            flags[index] = 0;
        } else {
            target[index] = values.accuracy;
            flags[index] = 1;
        }
    }

    private confusionJson(values: EvaluationValues): string | null {
        return values.confusionMatrix === undefined
            ? null
            : JSON.stringify(values.confusionMatrix);
    }

    getRetainedFingerprintCountForTests(): number {
        return this.retainedEvaluationFingerprints.size;
    }

}

/** Bounded, packed client-side mirrors of the two scientific metric series. */
export class MetricHistoryBuffer {
    private trends: PackedTrendStorage;
    private evaluations: PackedEvaluationStorage;
    private readonly trendCapacity: number;
    private readonly evaluationCapacity: number;
    private trendVersion = 0;
    private evaluationVersion = 0;
    private cachedSnapshot: MetricHistorySnapshot | undefined;

    constructor(options: MetricHistoryBufferOptions = {}) {
        this.trendCapacity = capacity(
            options.trendCapacity ?? DEFAULT_TREND_HISTORY_CAPACITY,
            'trendCapacity',
        );
        this.evaluationCapacity = capacity(
            options.evaluationCapacity ?? DEFAULT_EVALUATION_HISTORY_CAPACITY,
            'evaluationCapacity',
        );
        this.trends = new PackedTrendStorage(this.trendCapacity);
        this.evaluations = new PackedEvaluationStorage(this.evaluationCapacity);
    }

    get versions(): MetricHistoryVersions {
        return Object.freeze({
            trendVersion: this.trendVersion,
            evaluationVersion: this.evaluationVersion,
        });
    }

    /** Narrow test seam proving replay metadata is bounded by packed slots. */
    getRetainedEvaluationFingerprintCountForTests(): number {
        return this.evaluations.getRetainedFingerprintCountForTests();
    }

    appendTrend(point: LiveTrainingSignal): number {
        const parsed = parseLiveTrainingSignal(point);
        const version = nextVersion(this.trendVersion, 'trendVersion');
        this.trends.append(parsed);
        this.trendVersion = version;
        this.cachedSnapshot = undefined;
        return version;
    }

    appendEvaluation(point: PairedEvaluation): { readonly appended: boolean; readonly version: number } {
        const parsed = parsePairedEvaluation(point);
        if (!this.evaluations.preflight(parsed)) {
            return Object.freeze({ appended: false, version: this.evaluationVersion });
        }
        const version = nextVersion(this.evaluationVersion, 'evaluationVersion');
        this.evaluations.append(parsed);
        this.evaluationVersion = version;
        this.cachedSnapshot = undefined;
        return Object.freeze({ appended: true, version: this.evaluationVersion });
    }

    read(): MetricHistorySnapshot {
        if (this.cachedSnapshot !== undefined) return this.cachedSnapshot;
        this.cachedSnapshot = Object.freeze({
            trendHistory: this.trends.read(),
            evaluationHistory: this.evaluations.read(),
            trendVersion: this.trendVersion,
            evaluationVersion: this.evaluationVersion,
        });
        return this.cachedSnapshot;
    }

    reset(): MetricHistoryVersions {
        const trendVersion = nextVersion(this.trendVersion, 'trendVersion');
        const evaluationVersion = nextVersion(this.evaluationVersion, 'evaluationVersion');
        this.trends.reset();
        this.evaluations.reset();
        this.trendVersion = trendVersion;
        this.evaluationVersion = evaluationVersion;
        this.cachedSnapshot = undefined;
        return this.versions;
    }

    /**
     * Build an empty-generation replacement without touching the accepted
     * histories. All parsing, canonicalization, allocation, and version
     * overflow checks finish before the returned commit can publish.
     */
    prepareReplacement(
        liveSignal?: LiveTrainingSignal,
        evaluation?: PairedEvaluation,
    ): PreparedMetricHistoryReplacement {
        const baseTrendVersion = this.trendVersion;
        const baseEvaluationVersion = this.evaluationVersion;
        const replacement = new MetricHistoryBuffer({
            trendCapacity: this.trendCapacity,
            evaluationCapacity: this.evaluationCapacity,
        });
        replacement.trendVersion = nextVersion(this.trendVersion, 'trendVersion');
        replacement.evaluationVersion = nextVersion(
            this.evaluationVersion,
            'evaluationVersion',
        );
        if (evaluation !== undefined) replacement.appendEvaluation(evaluation);
        if (liveSignal !== undefined) replacement.appendTrend(liveSignal);

        const versions = replacement.versions;
        let committed = false;
        return Object.freeze({
            versions,
            commit: (): void => {
                if (committed) return;
                if (this.trendVersion !== baseTrendVersion
                    || this.evaluationVersion !== baseEvaluationVersion) {
                    throw new Error('prepared metric history replacement is stale');
                }
                this.trends = replacement.trends;
                this.evaluations = replacement.evaluations;
                this.trendVersion = replacement.trendVersion;
                this.evaluationVersion = replacement.evaluationVersion;
                this.cachedSnapshot = replacement.cachedSnapshot;
                committed = true;
            },
        });
    }

    prepareAppend(
        liveSignal?: LiveTrainingSignal,
        evaluation?: PairedEvaluation,
    ): PreparedMetricHistoryAppend {
        const baseTrendVersion = this.trendVersion;
        const baseEvaluationVersion = this.evaluationVersion;
        const parsedLive = liveSignal === undefined ? undefined : parseLiveTrainingSignal(liveSignal);
        const parsedEvaluation = evaluation === undefined
            ? undefined
            : parsePairedEvaluation(evaluation);
        const evaluationAppended = parsedEvaluation !== undefined
            && this.evaluations.preflight(parsedEvaluation);
        const trendVersion = parsedLive === undefined
            ? this.trendVersion
            : nextVersion(this.trendVersion, 'trendVersion');
        const evaluationVersion = !evaluationAppended
            ? this.evaluationVersion
            : nextVersion(this.evaluationVersion, 'evaluationVersion');
        let committed = false;
        return Object.freeze({
            versions: Object.freeze({ trendVersion, evaluationVersion }),
            evaluationAppended,
            commit: (): void => {
                if (committed) return;
                if (this.trendVersion !== baseTrendVersion
                    || this.evaluationVersion !== baseEvaluationVersion) {
                    throw new Error('prepared metric history append is stale');
                }
                if (evaluationAppended && parsedEvaluation !== undefined) {
                    this.evaluations.append(parsedEvaluation);
                    this.evaluationVersion = evaluationVersion;
                }
                if (parsedLive !== undefined) {
                    this.trends.append(parsedLive);
                    this.trendVersion = trendVersion;
                }
                this.cachedSnapshot = undefined;
                committed = true;
            },
        });
    }
}

/** Default app buffer; tests and isolated consumers may instantiate their own. */
export const metricHistoryBuffer = new MetricHistoryBuffer();
