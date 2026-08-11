import type {
    ArtifactBasis,
    ArtifactProvenance,
    DatasetRevision,
} from '@nn-playground/shared';
import { beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { createScientificTrustFixtures } from '../test/scientificTrustFixtures.ts';
import {
    getFrameBuffer,
    resetFrameBuffer,
    type ParameterProvenance,
    updateFrameBuffer,
} from './frameBuffer.ts';

let dataset: DatasetRevision;
let objectiveKey: string;
let parameterProvenance: ParameterProvenance;

beforeAll(async () => {
    const fixtures = await createScientificTrustFixtures();
    dataset = fixtures.evaluation.dataset;
    objectiveKey = fixtures.evaluation.objectiveKey;
    parameterProvenance = {
        model: { generationId: 1, revision: 7, step: 70, epoch: 3 },
        recipeFingerprint: fixtures.prepared.identities.recipeFingerprint,
    };
});

function provenance(
    basis: ArtifactBasis,
    revision = 1,
): ArtifactProvenance {
    return {
        model: {
            generationId: 1,
            revision,
            step: revision,
            epoch: 0,
        },
        dataset,
        objectiveKey,
        basis,
    };
}

const strict = { requireArtifactProvenance: true } as const;
describe('frame buffer artifact provenance', () => {
    beforeEach(() => {
        resetFrameBuffer();
    });

    it('stores parameter bytes and their exact model provenance atomically', () => {
        const weights = new Float32Array([0.1, 0.2]);
        const biases = new Float32Array([0.3]);

        updateFrameBuffer({
            weights,
            biases,
            weightLayout: { layerSizes: [2, 1] },
            parameterProvenance,
        }, strict);

        expect(getFrameBuffer()).toMatchObject({
            weights,
            biases,
            parameterProvenance,
        });
    });

    it('rejects parameter bytes without exact model provenance atomically', () => {
        const before = getFrameBuffer();

        expect(() => updateFrameBuffer({
            weights: new Float32Array([0.1, 0.2]),
            biases: new Float32Array([0.3]),
            weightLayout: { layerSizes: [2, 1] },
        }, strict)).toThrow('parameter provenance');
        expect(getFrameBuffer()).toBe(before);
    });

    it('rejects strict payload/provenance mismatches without changing any frame state', () => {
        const before = getFrameBuffer();
        const decisionBoundaryProvenance = provenance({
            kind: 'prediction-grid',
            pointCount: 4,
            domain: [-1, 1, -1, 1],
        });

        expect(() => updateFrameBuffer({
            outputGrid: new Float32Array([0.1, 0.2, 0.3, 0.4]),
            gridSize: 2,
            multiclassClassGrid: null,
            multiclassConfidenceGrid: null,
            multiclassBoundaryLayout: null,
        }, strict)).toThrow('decision boundary payload and provenance');
        expect(getFrameBuffer()).toBe(before);

        expect(() => updateFrameBuffer({
            decisionBoundaryProvenance,
        }, strict)).toThrow('decision boundary payload and provenance');
        expect(getFrameBuffer()).toBe(before);
    });

    it('applies every strict artifact with its own provenance in one atomic frame', () => {
        const decisionBoundaryProvenance = provenance({
            kind: 'prediction-grid',
            pointCount: 4,
            domain: [-1, 1, -1, 1],
        }, 2);
        const layerStatsProvenance = provenance({
            kind: 'bounded-sample',
            split: 'train',
            sampleCount: Math.min(128, dataset.trainCount),
            populationCount: dataset.trainCount,
        }, 3);
        const outputGrid = new Float32Array([0.1, 0.2, 0.3, 0.4]);
        const layerStats = [{
            meanActivation: 0.25,
            activationStd: 0.1,
            meanAbsWeight: 0.2,
            meanAbsGradient: 0.05,
        }];

        updateFrameBuffer({
            outputGrid,
            gridSize: 2,
            multiclassClassGrid: null,
            multiclassConfidenceGrid: null,
            multiclassBoundaryLayout: null,
            decisionBoundaryProvenance,
            layerStats,
            layerStatsProvenance,
            layerStatsGradientRevision: 2,
        }, strict);

        const frame = getFrameBuffer();
        expect(frame.outputGrid).toBe(outputGrid);
        expect(frame.decisionBoundaryProvenance).toBe(decisionBoundaryProvenance);
        expect(frame.layerStats).toBe(layerStats);
        expect(frame.layerStatsProvenance).toBe(layerStatsProvenance);
        expect(frame.layerStatsGradientRevision).toBe(2);
        expect(frame.decisionBoundaryProvenance).not.toBe(frame.layerStatsProvenance);
    });

    it('retains both artifact bytes and provenance when a strict reuse frame omits them', () => {
        const decisionBoundaryProvenance = provenance({
            kind: 'prediction-grid',
            pointCount: 4,
            domain: [-1, 1, -1, 1],
        });
        const outputGrid = new Float32Array([0.1, 0.2, 0.3, 0.4]);
        updateFrameBuffer({
            outputGrid,
            gridSize: 2,
            multiclassClassGrid: null,
            multiclassConfidenceGrid: null,
            multiclassBoundaryLayout: null,
            decisionBoundaryProvenance,
        }, strict);
        const before = getFrameBuffer();

        updateFrameBuffer({}, strict);

        expect(getFrameBuffer().version).toBe(before.version);
        expect(getFrameBuffer().outputGrid).toBe(outputGrid);
        expect(getFrameBuffer().decisionBoundaryProvenance).toBe(decisionBoundaryProvenance);
    });

    it('clears artifact bytes and provenance together', () => {
        const decisionBoundaryProvenance = provenance({
            kind: 'prediction-grid',
            pointCount: 4,
            domain: [-1, 1, -1, 1],
        });
        updateFrameBuffer({
            outputGrid: new Float32Array([0.1, 0.2, 0.3, 0.4]),
            gridSize: 2,
            multiclassClassGrid: null,
            multiclassConfidenceGrid: null,
            multiclassBoundaryLayout: null,
            decisionBoundaryProvenance,
        }, strict);

        updateFrameBuffer({
            outputGrid: null,
            gridSize: 0,
            multiclassClassGrid: null,
            multiclassConfidenceGrid: null,
            multiclassBoundaryLayout: null,
            decisionBoundaryProvenance: null,
        }, strict);

        expect(getFrameBuffer().outputGrid).toBeNull();
        expect(getFrameBuffer().decisionBoundaryProvenance).toBeNull();
    });

    it('rejects a partial clear that would detach surviving alternate bytes from provenance', () => {
        const decisionBoundaryProvenance = provenance({
            kind: 'prediction-grid',
            pointCount: 4,
            domain: [-1, 1, -1, 1],
        });
        updateFrameBuffer({
            outputGrid: null,
            gridSize: 2,
            multiclassClassGrid: new Uint8Array([0, 1, 2, 1]),
            multiclassConfidenceGrid: new Float32Array([0.7, 0.6, 0.9, 0.5]),
            multiclassBoundaryLayout: {
                gridSize: 2,
                classCount: 3,
                classLabels: [0, 1, 2],
            },
            decisionBoundaryProvenance,
        }, strict);
        const before = getFrameBuffer();

        expect(() => updateFrameBuffer({
            outputGrid: null,
            gridSize: 0,
            decisionBoundaryProvenance: null,
        }, strict)).toThrow('complete domain');
        expect(getFrameBuffer()).toBe(before);
    });

    it('rejects partial or inconsistent strict scalar and multiclass boundary transactions', () => {
        const decisionBoundaryProvenance = provenance({
            kind: 'prediction-grid',
            pointCount: 4,
            domain: [-1, 1, -1, 1],
        });
        const before = getFrameBuffer();

        expect(() => updateFrameBuffer({
            gridSize: 2,
            decisionBoundaryProvenance,
        }, strict)).toThrow('complete domain');
        expect(getFrameBuffer()).toBe(before);

        expect(() => updateFrameBuffer({
            outputGrid: null,
            gridSize: 2,
            multiclassClassGrid: new Uint8Array([0, 1, 2, 1]),
            multiclassConfidenceGrid: null,
            multiclassBoundaryLayout: {
                gridSize: 2,
                classCount: 3,
                classLabels: [0, 1, 2],
            },
            decisionBoundaryProvenance,
        }, strict)).toThrow('incomplete or inconsistent');
        expect(getFrameBuffer()).toBe(before);
    });

    it('advances the public confusion version for either matrix kind', () => {
        const baseline = getFrameBuffer().confusionMatrixVersion;
        const confusionMatrixProvenance = provenance({
            kind: 'full-split',
            split: 'test',
            sampleCount: dataset.testCount,
            populationCount: dataset.testCount,
        });

        updateFrameBuffer({
            confusionMatrix: null,
            multiclassConfusionMatrix: {
                classCount: 3,
                classLabels: [0, 1, 2],
                counts: [1, 0, 0, 0, 1, 0, 0, 0, 1],
            },
            confusionMatrixProvenance,
            confusionMatrixEvaluationId: 1,
        }, strict);
        expect(getFrameBuffer().confusionMatrixVersion).toBe(baseline + 1);

        updateFrameBuffer({
            confusionMatrix: { tp: 1, tn: 1, fp: 0, fn: 0 },
            multiclassConfusionMatrix: null,
            confusionMatrixProvenance,
            confusionMatrixEvaluationId: 2,
        }, strict);
        expect(getFrameBuffer().confusionMatrixVersion).toBe(baseline + 2);
        expect(getFrameBuffer().confusionMatrixEvaluationId).toBe(2);
    });

    it('rejects missing, surplus, or invalid confusion evaluation IDs atomically', () => {
        const before = getFrameBuffer();
        const confusionMatrixProvenance = provenance({
            kind: 'full-split',
            split: 'test',
            sampleCount: dataset.testCount,
            populationCount: dataset.testCount,
        });
        const base = {
            confusionMatrix: { tp: 1, tn: 1, fp: 0, fn: 0 },
            multiclassConfusionMatrix: null,
            confusionMatrixProvenance,
        };

        expect(() => updateFrameBuffer(base, strict)).toThrow('complete domain');
        expect(() => updateFrameBuffer({
            ...base,
            confusionMatrixEvaluationId: 0,
        }, strict)).toThrow('evaluation ID');
        expect(() => updateFrameBuffer({
            confusionMatrix: null,
            multiclassConfusionMatrix: null,
            confusionMatrixProvenance: null,
            confusionMatrixEvaluationId: 1,
        }, strict)).toThrow('evaluation ID');
        expect(getFrameBuffer()).toBe(before);
    });
});
