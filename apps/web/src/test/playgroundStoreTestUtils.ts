import type { PreparedExperimentDocumentV2 } from '@nn-playground/shared';
import type { CompiledExperimentConfig } from '@nn-playground/engine';
import { usePlaygroundStore } from '../store/usePlaygroundStore.ts';

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
