import { generateDatasetV2, getDatasetContract, type DataPoint, type DatasetContract, type DatasetId } from '@nn-playground/engine';

export interface DatasetPreviewModel {
    readonly datasetId: DatasetId;
    readonly taskKind: DatasetContract['taskKind'];
    readonly points: readonly DataPoint[];
    readonly inputDomain: DatasetContract['inputDomain'];
    readonly valueDomain: readonly [number, number];
}

/** Preview the generator, not the current train/test evidence or a synthetic illustration. */
export function deriveDatasetPreviewModel({ datasetId, seed, noise, sampleCount = 72 }: {
    readonly datasetId: DatasetId;
    readonly seed: number;
    readonly noise: number;
    readonly sampleCount?: number;
}): DatasetPreviewModel {
    const split = generateDatasetV2({ dataset: datasetId, sampleCount, noise, seed, trainFraction: 0.5 });
    const contract = getDatasetContract(datasetId);
    const target = contract.targetDomain;
    const valueDomain: readonly [number, number] = target.kind === 'continuous'
        ? target.boundsForNoise(noise)
        : [0, target.kind === 'classes' ? target.classCount - 1 : 1];
    return Object.freeze({
        datasetId,
        taskKind: contract.taskKind,
        points: Object.freeze([...split.train, ...split.test]),
        inputDomain: contract.inputDomain,
        valueDomain: Object.freeze(valueDomain),
    });
}
