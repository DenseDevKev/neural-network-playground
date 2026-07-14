export type InspectionTraceSource = 'train' | 'test';

export interface InspectionPanelCommands {
    selectHistogramLayer(index: number): void;
    selectTraceSource(source: InspectionTraceSource): void;
    selectSampleIndex(index: number): void;
    requestTrace(): Promise<void>;
    requestBackprop(): Promise<void>;
    requestLandscape(): Promise<void>;
}

interface InspectionLayerStatsInput {
    readonly meanActivation: number;
    readonly activationStd: number;
    readonly meanAbsWeight: number;
    readonly meanAbsGradient: number;
}

interface InspectionActivationBasisInput {
    readonly sampleCount: number;
    readonly populationCount: number;
    readonly modelRevision: number;
    readonly gradientRevision: number | null;
}

interface InspectionHistogramLayerInput {
    readonly layerIndex: number;
    readonly binCount: number;
    readonly minActivation: number;
    readonly maxActivation: number;
    readonly nearZeroCount: number;
    readonly saturatedCount: number;
    readonly totalCount: number;
}

interface InspectionHistogramInput {
    readonly bins: ArrayLike<number>;
    readonly layers: readonly InspectionHistogramLayerInput[];
}

interface InspectionTraceResultInput {
    readonly source: InspectionTraceSource;
    readonly sampleIndex: number;
    readonly modelStep: number;
    readonly modelRevision: number;
    readonly output: readonly number[];
    readonly sampleDataLoss: number;
    readonly regularizationPenalty: number;
    readonly layers: readonly {
        readonly layerIndex: number;
        readonly activations: readonly number[];
    }[];
}

interface InspectionBackpropResultInput {
    readonly modelStep: number;
    readonly modelEpoch: number;
    readonly summary: string;
    readonly batchSize: number;
    readonly learningRate: number;
    readonly dataLoss: number;
    readonly regularizationPenalty: number;
    readonly totalObjective: number;
    readonly totalGradientNorm: number;
    readonly clippedGradientNorm: number;
    readonly clipScale: number;
    readonly layers: readonly {
        readonly layerIndex: number;
        readonly status: string;
        readonly note: string;
        readonly meanAbsUpdate: number;
        readonly meanAbsGradient: number;
        readonly meanAbsErrorSignal: number;
    }[];
}

interface InspectionLandscapeResultInput {
    readonly modelStep: number;
    readonly modelEpoch: number;
    readonly summary: string;
    readonly gridSize: number;
    readonly sampleCount: number;
    readonly parameterPositionCount: number;
    readonly objectives: readonly number[];
    readonly centerObjective: number;
    readonly minObjective: number;
    readonly maxObjective: number;
    readonly axisALabel: string;
    readonly axisBLabel: string;
    readonly bestOffsetA: number;
    readonly bestOffsetB: number;
}

export interface InspectionPanelModelInput {
    readonly hiddenLayerCount: number;
    readonly layerStats: readonly InspectionLayerStatsInput[] | null;
    readonly activationBasis: InspectionActivationBasisInput | null;
    readonly histogram: InspectionHistogramInput | null;
    readonly selectedHistogramLayer: number;
    readonly traceSource: InspectionTraceSource;
    readonly sampleIndex: number;
    readonly trainPointCount: number;
    readonly testPointCount: number;
    readonly hasCurrentModel: boolean;
    readonly traceLoading: boolean;
    readonly traceError: string | null;
    readonly traceResult: InspectionTraceResultInput | null;
    readonly backpropLoading: boolean;
    readonly backpropError: string | null;
    readonly backpropResult: InspectionBackpropResultInput | null;
    readonly landscapeLoading: boolean;
    readonly landscapeError: string | null;
    readonly landscapeResult: InspectionLandscapeResultInput | null;
}

interface InspectionPanelLayerDisplay {
    readonly key: number;
    readonly name: string;
    readonly gradientWidth: string;
    readonly weightWidth: string;
    readonly gradientValue: string;
    readonly weightValue: string;
    readonly meanActivation: string;
    readonly activationStd: string;
}

interface InspectionPanelHistogramDisplay {
    readonly selectedLayerIndex: number;
    readonly options: readonly {
        readonly key: number;
        readonly value: number;
        readonly label: string;
    }[];
    readonly title: string;
    readonly nearZeroText: string;
    readonly saturatedText: string;
    readonly rangeText: string;
    readonly summary: string;
    readonly bins: readonly {
        readonly key: number;
        readonly height: string;
    }[];
}

interface InspectionPanelTraceDisplay {
    readonly source: InspectionTraceSource;
    readonly sampleIndex: number;
    readonly effectiveSampleIndex: number;
    readonly maxSampleIndex: number;
    readonly canRequest: boolean;
    readonly buttonDisabled: boolean;
    readonly buttonLabel: string;
    readonly emptyMessage: string | null;
    readonly errorMessage: string | null;
    readonly result: {
        readonly provenance: string;
        readonly output: string;
        readonly sampleDataLoss: string;
        readonly regularizationPenalty: string;
        readonly layers: readonly {
            readonly key: number;
            readonly label: string;
            readonly activations: string;
        }[];
    } | null;
}

interface InspectionPanelBackpropDisplay {
    readonly buttonDisabled: boolean;
    readonly buttonLabel: string;
    readonly statusText: string;
    readonly result: {
        readonly provenance: string;
        readonly objective: readonly string[];
        readonly gradient: readonly [string, string];
        readonly layers: readonly {
            readonly key: number;
            readonly name: string;
            readonly status: string;
            readonly metrics: readonly [string, string, string];
            readonly note: string;
        }[];
    } | null;
}

interface InspectionPanelLandscapeDisplay {
    readonly buttonDisabled: boolean;
    readonly buttonLabel: string;
    readonly statusText: string;
    readonly result: {
        readonly provenance: string;
        readonly summary: string;
        readonly gridTemplateColumns: string;
        readonly cells: readonly {
            readonly key: string;
            readonly opacity: number;
        }[];
        readonly title: string;
        readonly values: readonly [string, string, string];
        readonly basis: string;
        readonly bestDirection: string;
    } | null;
}

export interface InspectionPanelDisplayModel {
    readonly activationBasis: {
        readonly label: string;
        readonly suffix: string;
    } | null;
    readonly layers: readonly InspectionPanelLayerDisplay[];
    readonly histogramSelection: number;
    readonly histogram: InspectionPanelHistogramDisplay | null;
    readonly trace: InspectionPanelTraceDisplay;
    readonly backprop: InspectionPanelBackpropDisplay;
    readonly landscape: InspectionPanelLandscapeDisplay;
}

function layerNames(hiddenLayerCount: number): string[] {
    return [
        ...Array.from({ length: hiddenLayerCount }, (_, index) => `Hidden ${index + 1}`),
        'Output',
    ];
}

function formatPercent(count: number, total: number): string {
    return total > 0 ? `${((count / total) * 100).toFixed(1)}%` : '0.0%';
}

function formatRange(value: number): string {
    if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {
        return value.toExponential(1);
    }
    return value.toFixed(3);
}

function formatBackpropMetric(value: number): string {
    if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.0001)) {
        return value.toExponential(1);
    }
    return value.toFixed(3);
}

function formatSignedOffset(value: number): string {
    const formatted = formatBackpropMetric(value);
    return value > 0 ? `+${formatted}` : formatted;
}

function describeActivationShape(
    nearZeroCount: number,
    saturatedCount: number,
    totalCount: number,
): string {
    if (totalCount === 0) return 'No activation samples yet';
    const nearZeroRatio = nearZeroCount / totalCount;
    const saturatedRatio = saturatedCount / totalCount;
    if (saturatedRatio >= 0.45) return 'Many activations are near the activation limits';
    if (nearZeroRatio >= 0.45) return 'Many activations are near zero or inactive';
    return 'Activations are spread across the sampled range';
}

export function normalizeInspectionSampleIndex(value: number): number {
    return Number.isFinite(value) ? Math.max(0, Math.trunc(value)) : 0;
}

export function createInspectionPanelDisplayModel(
    input: InspectionPanelModelInput,
): InspectionPanelDisplayModel {
    const names = layerNames(input.hiddenLayerCount);
    let maxGradient = 0.001;
    let maxWeight = 0.001;
    for (const stats of input.layerStats ?? []) {
        maxGradient = Math.max(maxGradient, stats.meanAbsGradient);
        maxWeight = Math.max(maxWeight, stats.meanAbsWeight);
    }

    const layers = (input.layerStats ?? []).map((stats, index) => ({
        key: index,
        name: names[index] ?? `Layer ${index + 1}`,
        gradientWidth: `${Math.max(2, (stats.meanAbsGradient / maxGradient) * 100)}%`,
        weightWidth: `${Math.max(2, (stats.meanAbsWeight / maxWeight) * 100)}%`,
        gradientValue: stats.meanAbsGradient < 0.0001
            ? stats.meanAbsGradient.toExponential(1)
            : stats.meanAbsGradient.toFixed(4),
        weightValue: stats.meanAbsWeight.toFixed(4),
        meanActivation: stats.meanActivation.toFixed(4),
        activationStd: stats.activationStd.toFixed(4),
    }));

    const selectedHistogram = input.histogram?.layers[
        Math.min(input.selectedHistogramLayer, Math.max(0, input.histogram.layers.length - 1))
    ];
    const selectedHistogramIndex = selectedHistogram?.layerIndex ?? 0;
    const selectedHistogramBins = input.histogram && selectedHistogram
        ? Array.from(input.histogram.bins).slice(
            selectedHistogramIndex * selectedHistogram.binCount,
            selectedHistogramIndex * selectedHistogram.binCount + selectedHistogram.binCount,
        )
        : null;
    const maxHistogramCount = selectedHistogramBins
        ? Math.max(1, ...selectedHistogramBins)
        : 1;
    const histogramName = names[selectedHistogramIndex] ?? `Layer ${selectedHistogramIndex + 1}`;
    const nearZeroText = selectedHistogram
        ? `${formatPercent(selectedHistogram.nearZeroCount, selectedHistogram.totalCount)} near zero`
        : '';
    const saturatedText = selectedHistogram
        ? `${formatPercent(selectedHistogram.saturatedCount, selectedHistogram.totalCount)} near activation limits`
        : '';
    const histogram = input.histogram && selectedHistogram
        ? {
            selectedLayerIndex: selectedHistogramIndex,
            options: input.histogram.layers.map((layer) => ({
                key: layer.layerIndex,
                value: layer.layerIndex,
                label: names[layer.layerIndex] ?? `Layer ${layer.layerIndex + 1}`,
            })),
            title: `${histogramName} activations`,
            nearZeroText,
            saturatedText,
            rangeText: `range ${formatRange(selectedHistogram.minActivation)} to ${formatRange(selectedHistogram.maxActivation)}`,
            summary: `${histogramName} activations: ${nearZeroText}, ${saturatedText}. ${describeActivationShape(selectedHistogram.nearZeroCount, selectedHistogram.saturatedCount, selectedHistogram.totalCount)}.`,
            bins: (selectedHistogramBins ?? []).map((count, index) => ({
                key: index,
                height: `${Math.max(6, (count / maxHistogramCount) * 100)}%`,
            })),
        }
        : null;

    const pointCount = input.traceSource === 'test'
        ? input.testPointCount
        : input.trainPointCount;
    const maxSampleIndex = Math.max(0, pointCount - 1);
    const effectiveSampleIndex = Math.min(input.sampleIndex, maxSampleIndex);
    const canRequestTrace = pointCount > 0 && input.hasCurrentModel;
    const traceResult = input.traceResult
        ? {
            provenance: `Trace from ${input.traceResult.source === 'train' ? 'training' : 'test'} sample ${input.traceResult.sampleIndex.toLocaleString()} · model step ${input.traceResult.modelStep.toLocaleString()} · revision ${input.traceResult.modelRevision.toLocaleString()}`,
            output: input.traceResult.output.map((value) => value.toFixed(4)).join(', '),
            sampleDataLoss: input.traceResult.sampleDataLoss.toFixed(4),
            regularizationPenalty: input.traceResult.regularizationPenalty.toFixed(4),
            layers: input.traceResult.layers.map((layer) => ({
                key: layer.layerIndex,
                label: `Layer ${layer.layerIndex + 1}`,
                activations: layer.activations.map((value) => value.toFixed(3)).join(', '),
            })),
        }
        : null;

    const backpropResult = input.backpropResult
        ? {
            provenance: `Preview from step ${input.backpropResult.modelStep} / epoch ${input.backpropResult.modelEpoch}`,
            objective: [
                `batch ${input.backpropResult.batchSize}`,
                `data loss ${formatBackpropMetric(input.backpropResult.dataLoss)}`,
                `penalty ${formatBackpropMetric(input.backpropResult.regularizationPenalty)}`,
                `training objective ${formatBackpropMetric(input.backpropResult.totalObjective)}`,
                `lr ${formatBackpropMetric(input.backpropResult.learningRate)}`,
            ],
            gradient: [
                `complete objective gradient ${formatBackpropMetric(input.backpropResult.totalGradientNorm)}`,
                input.backpropResult.clipScale < 1
                    ? `clipped ${formatBackpropMetric(input.backpropResult.clipScale)}x to ${formatBackpropMetric(input.backpropResult.clippedGradientNorm)}`
                    : 'not clipped',
            ] as const,
            layers: input.backpropResult.layers.map((layer) => ({
                key: layer.layerIndex,
                name: names[layer.layerIndex] ?? `Layer ${layer.layerIndex + 1}`,
                status: layer.status,
                metrics: [
                    `mean update ${formatBackpropMetric(layer.meanAbsUpdate)}`,
                    `mean gradient ${formatBackpropMetric(layer.meanAbsGradient)}`,
                    `error signal ${formatBackpropMetric(layer.meanAbsErrorSignal)}`,
                ] as const,
                note: layer.note,
            })),
        }
        : null;

    const landscapeResult = input.landscapeResult
        ? (() => {
            const result = input.landscapeResult;
            const spread = Math.max(0.000001, result.maxObjective - result.minObjective);
            const center = formatBackpropMetric(result.centerObjective);
            const min = formatBackpropMetric(result.minObjective);
            const max = formatBackpropMetric(result.maxObjective);
            const offsetA = formatSignedOffset(result.bestOffsetA);
            const offsetB = formatSignedOffset(result.bestOffsetB);
            return {
                provenance: `Probe from step ${result.modelStep} / epoch ${result.modelEpoch}`,
                summary: `Training-objective parameter grid: center ${center}, min ${min}, max ${max}. Best direction ${result.axisALabel} ${offsetA}, ${result.axisBLabel} ${offsetB}.`,
                gridTemplateColumns: `repeat(${result.gridSize}, minmax(0, 1fr))`,
                cells: result.objectives.map((objective, index) => {
                    const intensity = 1 - ((objective - result.minObjective) / spread);
                    return {
                        key: `${index}-${objective}`,
                        opacity: 0.28 + Math.max(0, Math.min(1, intensity)) * 0.72,
                    };
                }),
                title: 'Training objective on a parameter grid',
                values: [`center ${center}`, `min ${min}`, `max ${max}`] as const,
                basis: `sampled ${result.sampleCount} training examples across ${result.parameterPositionCount} parameter positions on a ${result.gridSize} by ${result.gridSize} grid`,
                bestDirection: `best direction ${result.axisALabel} ${offsetA}, ${result.axisBLabel} ${offsetB}`,
            };
        })()
        : null;

    return {
        activationBasis: input.activationBasis
            ? {
                label: `Activation statistics across ${input.activationBasis.sampleCount.toLocaleString()} of ${input.activationBasis.populationCount.toLocaleString()} training examples`,
                suffix: input.activationBasis.gradientRevision !== null
                    && input.activationBasis.gradientRevision !== input.activationBasis.modelRevision
                    ? `; gradient summary comes from model revision ${input.activationBasis.gradientRevision.toLocaleString()}.`
                    : '.',
            }
            : null,
        layers,
        histogramSelection: input.selectedHistogramLayer,
        histogram,
        trace: {
            source: input.traceSource,
            sampleIndex: input.sampleIndex,
            effectiveSampleIndex,
            maxSampleIndex,
            canRequest: canRequestTrace,
            buttonDisabled: !canRequestTrace || input.traceLoading,
            buttonLabel: input.traceLoading ? 'Tracing…' : 'Trace prediction',
            emptyMessage: canRequestTrace
                ? null
                : `No ${input.traceSource === 'test' ? 'test' : 'training'} samples are available yet.`,
            errorMessage: input.traceError
                ? input.traceError.startsWith('Trace cleared')
                    ? input.traceError
                    : `Trace failed: ${input.traceError}`
                : null,
            result: traceResult,
        },
        backprop: {
            buttonDisabled: input.backpropLoading,
            buttonLabel: input.backpropLoading ? 'Previewing backprop' : 'Preview backprop',
            statusText: input.backpropLoading
                ? 'Preparing backprop preview...'
                : input.backpropError
                    ? `Backprop preview failed: ${input.backpropError}`
                    : input.backpropResult?.summary ?? '',
            result: backpropResult,
        },
        landscape: {
            buttonDisabled: input.landscapeLoading,
            buttonLabel: input.landscapeLoading ? 'Probing loss surface' : 'Probe loss surface',
            statusText: input.landscapeLoading
                ? 'Probing a bounded local loss surface...'
                : input.landscapeError
                    ? `Loss landscape probe failed: ${input.landscapeError}`
                    : input.landscapeResult?.summary ?? '',
            result: landscapeResult,
        },
    };
}
