import { getDatasetContract, type CompiledTaskContract, type DataPoint } from '@nn-playground/engine';
import type { ArtifactProvenance, MulticlassBoundaryLayout } from '@nn-playground/shared';

const MULTICLASS_LOW_CONFIDENCE_THRESHOLD = 0.6;

export type DecisionOverlayMode = 'none' | 'uncertainty' | 'misclassification' | 'split';

export interface DecisionOverlayCopy {
    label: string;
    description: string;
}

export interface DecisionBoundaryFrameSnapshot {
    readonly decisionBoundaryProvenance?: ArtifactProvenance | null;
    readonly outputGrid: Float32Array | null;
    readonly gridSize: number;
    readonly multiclassClassGrid: Uint8Array | null;
    readonly multiclassConfidenceGrid: Float32Array | null;
    readonly multiclassBoundaryLayout: MulticlassBoundaryLayout | null;
}

export interface DecisionBoundaryModelInput {
    frame: DecisionBoundaryFrameSnapshot;
    task: CompiledTaskContract | null;
    trainPoints: DataPoint[];
    testPoints: DataPoint[];
    showTestData: boolean;
    discretize: boolean;
    overlayMode: DecisionOverlayMode;
    noise?: number;
}

export interface MulticlassBoundarySummary {
    dominantLabel: string;
    dominantShare: number;
    averageConfidence: number;
    lowConfidenceShare: number;
    description: string;
}

interface DecisionBoundaryEmptyModel {
    kind: 'empty';
    title: string;
    description: string;
}

interface DecisionBoundaryUnavailableModel {
    kind: 'unavailable';
    title: string;
    description: string;
}

interface DecisionBoundaryVisibleData {
    provenance?: ArtifactProvenance | null;
    trainPoints: DataPoint[];
    visibleTestPoints: DataPoint[];
}

export interface DecisionBoundaryScalarModel extends DecisionBoundaryVisibleData {
    kind: 'scalar';
    taskKind?: CompiledTaskContract['kind'];
    valueDomain?: readonly [number,number];
    grid: Float32Array | null;
    gridSize: number;
    discretize: boolean;
    overlayMode: DecisionOverlayMode;
    overlayCopy: DecisionOverlayCopy;
    accessibleDescription: string;
    misclassificationTestPoints: DataPoint[];
}

export interface DecisionBoundaryMulticlassModel extends DecisionBoundaryVisibleData {
    kind: 'multiclass';
    overlayMode?: DecisionOverlayMode;
    classGrid: Uint8Array;
    confidenceGrid: Float32Array;
    layout: MulticlassBoundaryLayout;
    summary: MulticlassBoundarySummary;
    accessibleDescription: string;
}

export type DecisionBoundaryDisplayModel =
    | DecisionBoundaryEmptyModel
    | DecisionBoundaryUnavailableModel
    | DecisionBoundaryScalarModel
    | DecisionBoundaryMulticlassModel;

export function getDecisionOverlayCopy(
    mode: DecisionOverlayMode,
    showTestData: boolean,
    discretize: boolean,
): DecisionOverlayCopy {
    switch (mode) {
        case 'uncertainty':
            return {
                label: 'Uncertainty',
                description:
                    'Uncertainty mode brightens regions near 50% probability, where the model is least sure which class to predict.',
            };
        case 'misclassification':
            return {
                label: 'Misclassified',
                description: showTestData
                    ? 'Errors mode marks training and visible test points whose predicted class does not match the label.'
                    : 'Errors mode marks training points whose predicted class does not match the label.',
            };
        case 'split':
            return {
                label: 'Train/test split',
                description:
                    'Split mode keeps held-out test points visible beside training points so you can compare fit against generalization.',
            };
        case 'none':
        default:
            return {
                label: 'Output',
                description: discretize
                    ? 'Output mode shows hard class regions: blue for negative predictions and orange for positive predictions.'
                    : 'Output mode shows predicted probability as a smooth field from negative blue to positive orange.',
            };
    }
}

function hasNonBinaryLabels(points: DataPoint[]): boolean {
    return points.some((point) => point.label !== 0 && point.label !== 1);
}

function formatPercent(value: number): string {
    return `${Math.round(value * 100)}%`;
}

function summarizeMulticlassBoundary(
    classGrid: Uint8Array,
    confidenceGrid: Float32Array,
    layout: MulticlassBoundaryLayout,
): MulticlassBoundarySummary | null {
    const expectedLength = layout.gridSize * layout.gridSize;
    if (
        !Number.isSafeInteger(layout.gridSize)
        || layout.gridSize < 1
        || layout.classCount !== 3
        || layout.classLabels.length !== layout.classCount
        || layout.classLabels[0] !== 0
        || layout.classLabels[1] !== 1
        || layout.classLabels[2] !== 2
        || classGrid.length !== expectedLength
        || confidenceGrid.length !== expectedLength
    ) {
        return null;
    }

    const counts = new Array(layout.classCount).fill(0) as number[];
    let confidenceTotal = 0;
    let lowConfidenceCount = 0;

    for (let index = 0; index < expectedLength; index++) {
        const classIndex = classGrid[index];
        const confidence = confidenceGrid[index];
        if (
            classIndex >= layout.classCount
            || !Number.isFinite(confidence)
            || confidence < 0
            || confidence > 1
        ) {
            return null;
        }
        counts[classIndex]++;
        confidenceTotal += confidence;
        if (confidence < MULTICLASS_LOW_CONFIDENCE_THRESHOLD) lowConfidenceCount++;
    }

    let dominantIndex = 0;
    for (let index = 1; index < counts.length; index++) {
        if (counts[index] > counts[dominantIndex]) dominantIndex = index;
    }

    const dominantLabel = `Class ${layout.classLabels[dominantIndex]}`;
    const dominantShare = counts[dominantIndex] / expectedLength;
    const averageConfidence = confidenceTotal / expectedLength;
    const lowConfidenceShare = lowConfidenceCount / expectedLength;

    return {
        dominantLabel,
        dominantShare,
        averageConfidence,
        lowConfidenceShare,
        description: `Multiclass decision boundary. ${dominantLabel} covers ${formatPercent(dominantShare)} of sampled cells with ${formatPercent(averageConfidence)} average winning confidence. ${formatPercent(lowConfidenceShare)} of cells are below ${formatPercent(MULTICLASS_LOW_CONFIDENCE_THRESHOLD)} confidence.`,
    };
}

export function deriveDecisionBoundaryModel({
    frame,
    task,
    trainPoints,
    testPoints,
    showTestData,
    discretize,
    overlayMode,
    noise = 0,
}: DecisionBoundaryModelInput): DecisionBoundaryDisplayModel {
    if (trainPoints.length === 0) {
        return {
            kind: 'empty',
            title: 'No training data',
            description: 'Generate data or reset the playground to populate the decision boundary.',
        };
    }

    const isMulticlassTask = task?.kind === 'multiclass-classification';
    const multiclassSummary = frame.multiclassClassGrid
        && frame.multiclassConfidenceGrid
        && frame.multiclassBoundaryLayout
        && frame.gridSize === frame.multiclassBoundaryLayout.gridSize
        ? summarizeMulticlassBoundary(
            frame.multiclassClassGrid,
            frame.multiclassConfidenceGrid,
            frame.multiclassBoundaryLayout,
        )
        : null;

    if (isMulticlassTask && multiclassSummary) {
        return {
            kind: 'multiclass',
            provenance: frame.decisionBoundaryProvenance,
            overlayMode,
            classGrid: frame.multiclassClassGrid!,
            confidenceGrid: frame.multiclassConfidenceGrid!,
            layout: frame.multiclassBoundaryLayout!,
            summary: multiclassSummary,
            accessibleDescription: multiclassSummary.description,
            trainPoints,
            visibleTestPoints: (showTestData || overlayMode === 'split') ? testPoints : [],
        };
    }

    const isNonBinaryClassification = task?.kind !== 'regression' && (
        isMulticlassTask
        || hasNonBinaryLabels(trainPoints)
        || hasNonBinaryLabels(testPoints)
    );
    if (isNonBinaryClassification) {
        return {
            kind: 'unavailable',
            title: 'Binary decision boundary unavailable',
            description: 'This visualization supports two-class outputs unless bounded multiclass boundary data is available from the worker.',
        };
    }

    const regression = task?.kind === 'regression';
    const effectiveOverlay = regression && overlayMode !== 'split' ? 'none' : overlayMode;
    const overlayCopy = regression ? {
        label: effectiveOverlay === 'split' ? 'Train/test split' : 'Prediction',
        description: 'Continuous model predictions. Point colors show numerical targets; the legend gives the displayed value range.',
    } : getDecisionOverlayCopy(effectiveOverlay, showTestData, discretize);
    const validScalarGrid = frame.outputGrid
        && Number.isSafeInteger(frame.gridSize)
        && frame.gridSize > 0
        && frame.outputGrid.length === frame.gridSize * frame.gridSize
        ? frame.outputGrid
        : null;

    let valueDomain: readonly [number,number] = [0,1];
    if (regression) {
        const domain = getDatasetContract(task.dataset).targetDomain;
        if (domain.kind === 'continuous') valueDomain = domain.boundsForNoise(noise);
        let [minimum,maximum] = valueDomain;
        for (const value of validScalarGrid ?? []) if (Number.isFinite(value)) {
            minimum = Math.min(minimum,value); maximum = Math.max(maximum,value);
        }
        valueDomain = [minimum,maximum];
    }
    return {
        kind: 'scalar',
        provenance: frame.decisionBoundaryProvenance,
        taskKind: task?.kind,
        valueDomain,
        grid: validScalarGrid,
        gridSize: validScalarGrid ? frame.gridSize : 0,
        discretize: regression ? false : discretize,
        overlayMode: effectiveOverlay,
        overlayCopy,
        accessibleDescription: overlayCopy.description,
        trainPoints,
        visibleTestPoints: (showTestData || overlayMode === 'split') ? testPoints : [],
        misclassificationTestPoints: showTestData ? testPoints : [],
    };
}
