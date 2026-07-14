import { describe, expect, it } from 'vitest';
import {
    createInspectionPanelDisplayModel,
    normalizeInspectionSampleIndex,
    type InspectionPanelModelInput,
} from './inspectionPanelModel.ts';

function baseInput(
    overrides: Partial<InspectionPanelModelInput> = {},
): InspectionPanelModelInput {
    return {
        hiddenLayerCount: 1,
        layerStats: null,
        activationBasis: null,
        histogram: null,
        selectedHistogramLayer: 0,
        traceSource: 'train',
        sampleIndex: 0,
        trainPointCount: 0,
        testPointCount: 0,
        hasCurrentModel: false,
        traceLoading: false,
        traceError: null,
        traceResult: null,
        backpropLoading: false,
        backpropError: null,
        backpropResult: null,
        landscapeLoading: false,
        landscapeError: null,
        landscapeResult: null,
        ...overrides,
    };
}

describe('createInspectionPanelDisplayModel', () => {
    it('normalizes raw sample input without conflating it with the effective index', () => {
        expect(normalizeInspectionSampleIndex(4.9)).toBe(4);
        expect(normalizeInspectionSampleIndex(-3)).toBe(0);
        expect(normalizeInspectionSampleIndex(Number.NaN)).toBe(0);
    });

    it('prepares layer labels, relative maxima, values, and provenance text', () => {
        const model = createInspectionPanelDisplayModel(baseInput({
            layerStats: [
                {
                    meanActivation: 0.25,
                    activationStd: 0.15,
                    meanAbsWeight: 0.2,
                    meanAbsGradient: 0.01,
                },
                {
                    meanActivation: 0.75,
                    activationStd: 0.05,
                    meanAbsWeight: 0.4,
                    meanAbsGradient: 0.02,
                },
            ],
            activationBasis: {
                sampleCount: 128,
                populationCount: 210,
                modelRevision: 12,
                gradientRevision: 11,
            },
        }));

        expect(model.activationBasis).toEqual({
            label: 'Activation statistics across 128 of 210 training examples',
            suffix: '; gradient summary comes from model revision 11.',
        });
        expect(model.layers).toEqual([
            {
                key: 0,
                name: 'Hidden 1',
                gradientWidth: '50%',
                weightWidth: '50%',
                gradientValue: '0.0100',
                weightValue: '0.2000',
                meanActivation: '0.2500',
                activationStd: '0.1500',
            },
            {
                key: 1,
                name: 'Output',
                gradientWidth: '100%',
                weightWidth: '100%',
                gradientValue: '0.0200',
                weightValue: '0.4000',
                meanActivation: '0.7500',
                activationStd: '0.0500',
            },
        ]);
    });

    it('resolves histogram selection without rewriting it and prepares accessible bins', () => {
        const model = createInspectionPanelDisplayModel(baseInput({
            selectedHistogramLayer: 99,
            histogram: {
                bins: [0, 2, 4, 0, 3, 0, 0, 6],
                layers: [
                    {
                        layerIndex: 0,
                        binCount: 4,
                        minActivation: -1,
                        maxActivation: 1,
                        nearZeroCount: 2,
                        saturatedCount: 1,
                        totalCount: 10,
                    },
                    {
                        layerIndex: 1,
                        binCount: 4,
                        minActivation: 0.00001,
                        maxActivation: 2000,
                        nearZeroCount: 6,
                        saturatedCount: 1,
                        totalCount: 10,
                    },
                ],
            },
        }));

        expect(model.histogramSelection).toBe(99);
        expect(model.histogram).toMatchObject({
            selectedLayerIndex: 1,
            title: 'Output activations',
            nearZeroText: '60.0% near zero',
            saturatedText: '10.0% near activation limits',
            rangeText: 'range 1.0e-5 to 2.0e+3',
            summary: 'Output activations: 60.0% near zero, 10.0% near activation limits. Many activations are near zero or inactive.',
        });
        expect(model.histogram?.bins).toEqual([
            { key: 0, height: '50%' },
            { key: 1, height: '6%' },
            { key: 2, height: '6%' },
            { key: 3, height: '100%' },
        ]);
    });

    it('describes an available histogram layer with zero sampled activations', () => {
        const model = createInspectionPanelDisplayModel(baseInput({
            histogram: {
                bins: [0, 0],
                layers: [{
                    layerIndex: 0,
                    binCount: 2,
                    minActivation: 0,
                    maxActivation: 0,
                    nearZeroCount: 0,
                    saturatedCount: 0,
                    totalCount: 0,
                }],
            },
        }));

        expect(model.histogram).toMatchObject({
            nearZeroText: '0.0% near zero',
            saturatedText: '0.0% near activation limits',
            summary: 'Hidden 1 activations: 0.0% near zero, 0.0% near activation limits. No activation samples yet.',
            bins: [{ key: 0, height: '6%' }, { key: 1, height: '6%' }],
        });
    });

    it('preserves the raw sample index while deriving an effective trace index', () => {
        const train = createInspectionPanelDisplayModel(baseInput({
            hasCurrentModel: true,
            trainPointCount: 2,
            sampleIndex: 9,
        }));
        const test = createInspectionPanelDisplayModel(baseInput({
            hasCurrentModel: true,
            traceSource: 'test',
            testPointCount: 0,
            sampleIndex: 9,
        }));

        expect(train.trace).toMatchObject({
            source: 'train',
            sampleIndex: 9,
            effectiveSampleIndex: 1,
            maxSampleIndex: 1,
            canRequest: true,
            buttonDisabled: false,
        });
        expect(test.trace).toMatchObject({
            source: 'test',
            sampleIndex: 9,
            effectiveSampleIndex: 0,
            maxSampleIndex: 0,
            canRequest: false,
            buttonDisabled: true,
            emptyMessage: 'No test samples are available yet.',
        });
        expect(test.backprop.buttonDisabled).toBe(false);
        expect(test.landscape.buttonDisabled).toBe(false);
    });

    it('formats trace, backprop, and landscape summaries as display-safe primitives', () => {
        const model = createInspectionPanelDisplayModel(baseInput({
            hiddenLayerCount: 0,
            traceResult: {
                source: 'train',
                sampleIndex: 4,
                modelStep: 12,
                modelRevision: 12,
                output: [0.82, 0.18],
                sampleDataLoss: 0.19,
                regularizationPenalty: 0.03,
                layers: [{ layerIndex: 0, activations: [0.82] }],
            },
            backpropResult: {
                modelStep: 12,
                modelEpoch: 1,
                summary: 'Backprop preview found 1 healthy layer update.',
                batchSize: 5,
                learningRate: 0.03,
                dataLoss: 0.12,
                regularizationPenalty: 0.01,
                totalObjective: 0.13,
                totalGradientNorm: 0.006,
                clippedGradientNorm: 0.006,
                clipScale: 1,
                layers: [{
                    layerIndex: 0,
                    status: 'healthy',
                    note: 'The previewed update is in a moderate range.',
                    meanAbsUpdate: 0.0009,
                    meanAbsGradient: 0.003,
                    meanAbsErrorSignal: 0.012,
                }],
            },
            landscapeResult: {
                modelStep: 12,
                modelEpoch: 1,
                summary: 'Training-objective surface found best objective 0.4000.',
                gridSize: 2,
                sampleCount: 12,
                parameterPositionCount: 4,
                objectives: [0.62, 0.58, 0.44, 0.4],
                centerObjective: 0.49,
                minObjective: 0.4,
                maxObjective: 0.62,
                axisALabel: 'W1[0,0]',
                axisBLabel: 'W1[1,0]',
                bestOffsetA: 0.1,
                bestOffsetB: -0.1,
            },
        }));

        expect(model.trace.result).toMatchObject({
            provenance: 'Trace from training sample 4 · model step 12 · revision 12',
            output: '0.8200, 0.1800',
            sampleDataLoss: '0.1900',
            regularizationPenalty: '0.0300',
            layers: [{ key: 0, label: 'Layer 1', activations: '0.820' }],
        });
        expect(model.backprop.result).toMatchObject({
            provenance: 'Preview from step 12 / epoch 1',
            objective: ['batch 5', 'data loss 0.120', 'penalty 0.010', 'training objective 0.130', 'lr 0.030'],
            gradient: ['complete objective gradient 0.006', 'not clipped'],
        });
        expect(model.landscape.result).toMatchObject({
            provenance: 'Probe from step 12 / epoch 1',
            title: 'Training objective on a parameter grid',
            values: ['center 0.490', 'min 0.400', 'max 0.620'],
            basis: 'sampled 12 training examples across 4 parameter positions on a 2 by 2 grid',
            bestDirection: 'best direction W1[0,0] +0.100, W1[1,0] -0.100',
        });
    });

    it('keeps zero-spread landscape cells finite and visible', () => {
        const model = createInspectionPanelDisplayModel(baseInput({
            landscapeResult: {
                modelStep: 1,
                modelEpoch: 1,
                summary: 'Flat surface.',
                gridSize: 2,
                sampleCount: 2,
                parameterPositionCount: 4,
                objectives: [0.5, 0.5, 0.5, 0.5],
                centerObjective: 0.5,
                minObjective: 0.5,
                maxObjective: 0.5,
                axisALabel: 'a',
                axisBLabel: 'b',
                bestOffsetA: 0,
                bestOffsetB: 0,
            },
        }));

        expect(model.landscape.result?.cells).toEqual([
            { key: '0-0.5', opacity: 1 },
            { key: '1-0.5', opacity: 1 },
            { key: '2-0.5', opacity: 1 },
            { key: '3-0.5', opacity: 1 },
        ]);
        expect(model.landscape.result?.summary).toBe(
            'Training-objective parameter grid: center 0.500, min 0.500, max 0.500. Best direction a 0.000, b 0.000.',
        );
    });
});
