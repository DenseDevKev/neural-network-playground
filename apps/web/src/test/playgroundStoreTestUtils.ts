import type { PreparedExperimentDocumentV2 } from '@nn-playground/shared';
import type { CompiledExperimentConfig, ConfusionMatrixData } from '@nn-playground/engine';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../store/useTrainingStore.ts';

export interface LegacyDataProjectionForTest {
    readonly dataset: string;
    readonly problemType: string;
    readonly noise: number;
    readonly trainTestRatio: number;
    readonly numSamples: number;
}

export interface LegacyFeaturesProjectionForTest {
    readonly x: boolean;
    readonly y: boolean;
    readonly xSquared: boolean;
}

export interface LegacyTrainingProjectionForTest {
    readonly learningRate: number;
    readonly batchSize: number;
}

export interface LegacyTrainingSnapshotForTest {
    readonly epoch?: number;
    readonly trainLoss?: number;
    readonly testLoss?: number;
    readonly testMetrics?: {
        readonly confusionMatrix?: ConfusionMatrixData;
    };
}

function installRuntimeOnlyLegacyFieldForTest(
    getState: () => object,
    key: string,
    value: unknown,
): () => void {
    const installedTarget = getState();
    const priorDescriptor = Reflect.getOwnPropertyDescriptor(installedTarget, key);
    if (!Reflect.set(installedTarget, key, value)) {
        throw new Error(`Could not install runtime-only legacy field ${key}`);
    }

    return () => {
        const currentTarget = getState();
        if (currentTarget !== installedTarget) {
            Reflect.deleteProperty(currentTarget, key);
        }
        if (priorDescriptor) {
            Reflect.defineProperty(installedTarget, key, priorDescriptor);
        } else {
            Reflect.deleteProperty(installedTarget, key);
        }
    };
}

export function installLegacyDataProjectionForTest(
    value: LegacyDataProjectionForTest,
): () => void {
    return installRuntimeOnlyLegacyFieldForTest(usePlaygroundStore.getState, 'data', value);
}

export function installLegacyFeaturesProjectionForTest(
    value: LegacyFeaturesProjectionForTest,
): () => void {
    return installRuntimeOnlyLegacyFieldForTest(usePlaygroundStore.getState, 'features', value);
}

export function installLegacyTrainingProjectionForTest(
    value: LegacyTrainingProjectionForTest,
): () => void {
    return installRuntimeOnlyLegacyFieldForTest(usePlaygroundStore.getState, 'training', value);
}

export function installLegacyTrainingSnapshotForTest(
    value: LegacyTrainingSnapshotForTest,
): () => void {
    return installRuntimeOnlyLegacyFieldForTest(useTrainingStore.getState, 'snapshot', value);
}

export function currentPreparedForTest(): PreparedExperimentDocumentV2 | null {
    const { access } = usePlaygroundStore.getState();
    return access.status === 'ready' ? access.prepared : null;
}

export function installPreparedForTest(prepared: PreparedExperimentDocumentV2): void {
    usePlaygroundStore.setState({ access: { status: 'ready', prepared } });
}

export function updateCompiledForTest(
    update: (compiled: CompiledExperimentConfig) => CompiledExperimentConfig,
): void {
    const prepared = currentPreparedForTest();
    if (prepared === null) throw new Error('Test requires a ready V2 experiment');
    usePlaygroundStore.setState({
        access: {
            status: 'ready',
            prepared: { ...prepared, compiled: update(prepared.compiled) },
        },
    });
}
