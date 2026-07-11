import {
    canonicalizeJson,
    parseLiveTrainingSignal,
    parsePairedEvaluation,
} from '@nn-playground/shared';
import type {
    DatasetRevision,
    EvaluationPoint,
    LiveTrainingSignal,
    PairedEvaluation,
    TrainingTrendPoint,
} from '@nn-playground/shared';

export const DEFAULT_RUNTIME_TREND_HISTORY_CAPACITY = 4_096;
export const DEFAULT_RUNTIME_EVALUATION_HISTORY_CAPACITY = 1_024;

export interface RuntimeMetricHistoryIdentity {
    readonly generationId: number;
    readonly dataset: DatasetRevision;
    readonly objectiveKey: string;
    readonly trendCapacity?: number;
    readonly evaluationCapacity?: number;
}

export interface RuntimeMetricHistorySnapshot {
    readonly trendHistory: readonly TrainingTrendPoint[];
    readonly evaluationHistory: readonly EvaluationPoint[];
}

function capacity(value: number, path: string): number {
    if (!Number.isSafeInteger(value) || value < 2 || value > 1_000_000) {
        throw new RangeError(`${path} must be a safe integer from 2 to 1000000`);
    }
    return value;
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

function sameDataset(left: DatasetRevision, right: DatasetRevision): boolean {
    return left.generatorVersion === right.generatorVersion
        && left.datasetKey === right.datasetKey
        && left.trainCount === right.trainCount
        && left.testCount === right.testCount;
}

class FirstPreservingBuffer<T> {
    private readonly recent: Array<T | undefined>;
    private first: T | undefined;
    private recentStart = 0;
    private recentCount = 0;

    constructor(private readonly limit: number) {
        this.recent = Array.from({ length: limit - 1 });
    }

    append(value: T): T | undefined {
        if (this.first === undefined) {
            this.first = value;
            return undefined;
        }
        const recentLimit = this.limit - 1;
        if (this.recentCount < recentLimit) {
            const index = (this.recentStart + this.recentCount) % recentLimit;
            this.recent[index] = value;
            this.recentCount++;
            return undefined;
        }
        const evicted = this.recent[this.recentStart];
        this.recent[this.recentStart] = value;
        this.recentStart = (this.recentStart + 1) % recentLimit;
        return evicted;
    }

    read(): readonly T[] {
        const values: T[] = [];
        if (this.first !== undefined) values.push(this.first);
        for (let offset = 0; offset < this.recentCount; offset++) {
            const value = this.recent[(this.recentStart + offset) % (this.limit - 1)];
            if (value !== undefined) values.push(value);
        }
        return Object.freeze(values);
    }

    reset(): void {
        this.first = undefined;
        this.recent.fill(undefined);
        this.recentStart = 0;
        this.recentCount = 0;
    }
}

/** Worker-authoritative bounded histories for one immutable experiment identity. */
export class RuntimeMetricHistory {
    readonly generationId: number;
    readonly dataset: Readonly<DatasetRevision>;
    readonly objectiveKey: string;

    private readonly trends: FirstPreservingBuffer<TrainingTrendPoint>;
    private readonly evaluations: FirstPreservingBuffer<EvaluationPoint>;
    private readonly retainedEvaluationFingerprints = new Map<number, string>();
    private highestEvaluationId = 0;
    private cachedSnapshot: RuntimeMetricHistorySnapshot | undefined;

    constructor(identity: RuntimeMetricHistoryIdentity) {
        const seed = parseLiveTrainingSignal({
            model: {
                generationId: identity.generationId,
                revision: 0,
                step: 0,
                epoch: 0,
            },
            dataset: identity.dataset,
            objectiveKey: identity.objectiveKey,
            basis: {
                kind: 'mini-batch-ema',
                alpha: 1,
                latestBatchSize: 1,
                throughStep: 0,
            },
            dataLoss: 0,
        });
        this.generationId = seed.model.generationId;
        this.dataset = deepFreeze(seed.dataset) as Readonly<DatasetRevision>;
        this.objectiveKey = seed.objectiveKey;
        this.trends = new FirstPreservingBuffer(capacity(
            identity.trendCapacity ?? DEFAULT_RUNTIME_TREND_HISTORY_CAPACITY,
            'trendCapacity',
        ));
        this.evaluations = new FirstPreservingBuffer(capacity(
            identity.evaluationCapacity ?? DEFAULT_RUNTIME_EVALUATION_HISTORY_CAPACITY,
            'evaluationCapacity',
        ));
    }

    appendLiveSignal(signal: LiveTrainingSignal): void {
        const parsed = parseLiveTrainingSignal(signal);
        this.assertIdentity(parsed);
        this.trends.append(deepFreeze(parsed) as TrainingTrendPoint);
        this.cachedSnapshot = undefined;
    }

    appendEvaluation(evaluation: PairedEvaluation): boolean {
        const parsed = parsePairedEvaluation(evaluation);
        this.assertIdentity(parsed);
        const existing = this.retainedEvaluationFingerprints.get(parsed.evaluationId);
        if (parsed.evaluationId <= this.highestEvaluationId) {
            if (existing === undefined) return false;
            const fingerprint = canonicalizeJson(parsed);
            if (existing === fingerprint) return false;
            throw new TypeError(`conflicting evaluationId ${parsed.evaluationId}`);
        }

        const fingerprint = canonicalizeJson(parsed);
        const point = deepFreeze(parsed) as EvaluationPoint;
        const evicted = this.evaluations.append(point);
        if (evicted !== undefined) {
            this.retainedEvaluationFingerprints.delete(evicted.evaluationId);
        }
        this.retainedEvaluationFingerprints.set(point.evaluationId, fingerprint);
        this.highestEvaluationId = point.evaluationId;
        this.cachedSnapshot = undefined;
        return true;
    }

    read(): RuntimeMetricHistorySnapshot {
        if (this.cachedSnapshot !== undefined) return this.cachedSnapshot;
        this.cachedSnapshot = Object.freeze({
            trendHistory: this.trends.read(),
            evaluationHistory: this.evaluations.read(),
        });
        return this.cachedSnapshot;
    }

    reset(): void {
        this.trends.reset();
        this.evaluations.reset();
        this.retainedEvaluationFingerprints.clear();
        this.highestEvaluationId = 0;
        this.cachedSnapshot = undefined;
    }

    /** Narrow test seam proving replay metadata is bounded by retained slots. */
    getRetainedEvaluationFingerprintCountForTests(): number {
        return this.retainedEvaluationFingerprints.size;
    }

    private assertIdentity(
        evidence: Pick<LiveTrainingSignal, 'model' | 'dataset' | 'objectiveKey'>,
    ): void {
        if (evidence.model.generationId !== this.generationId) {
            throw new RangeError(
                `model.generationId must equal history generationId ${this.generationId}`,
            );
        }
        if (!sameDataset(evidence.dataset, this.dataset)) {
            throw new RangeError('dataset must equal the history dataset revision');
        }
        if (evidence.objectiveKey !== this.objectiveKey) {
            throw new RangeError('objectiveKey must equal the history objective identity');
        }
    }
}
