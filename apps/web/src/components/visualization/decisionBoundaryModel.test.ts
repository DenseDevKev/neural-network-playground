import { describe, expect, it } from 'vitest';
import type { CompiledTaskContract, DataPoint } from '@nn-playground/engine';
import type { MulticlassBoundaryLayout } from '@nn-playground/shared';
import {
    deriveDecisionBoundaryModel,
    type DecisionBoundaryFrameSnapshot,
} from './decisionBoundaryModel.ts';

const BINARY_TASK: CompiledTaskContract = {
    kind: 'binary-classification',
    dataset: 'circle',
    outputSize: 1,
    outputActivation: 'sigmoid',
    target: { kind: 'scalar', values: [0, 1] },
};

const MULTICLASS_TASK: CompiledTaskContract = {
    kind: 'multiclass-classification',
    dataset: 'three-class-clusters',
    outputSize: 3,
    outputActivation: 'softmax',
    target: { kind: 'one-hot', length: 3 },
};

const REGRESSION_TASK: CompiledTaskContract = {
    kind: 'regression',
    dataset: 'reg-plane',
    outputSize: 1,
    outputActivation: 'linear',
    target: { kind: 'scalar', finite: true },
};

const EMPTY_FRAME: DecisionBoundaryFrameSnapshot = {
    outputGrid: null,
    gridSize: 0,
    multiclassClassGrid: null,
    multiclassConfidenceGrid: null,
    multiclassBoundaryLayout: null,
};

const TRAIN_POINTS: DataPoint[] = [{ x: -0.5, y: 0.5, label: 0 }];
const TEST_POINTS: DataPoint[] = [{ x: 0.5, y: -0.5, label: 1 }];

function derive(overrides: Partial<Parameters<typeof deriveDecisionBoundaryModel>[0]> = {}) {
    return deriveDecisionBoundaryModel({
        frame: EMPTY_FRAME,
        task: BINARY_TASK,
        trainPoints: TRAIN_POINTS,
        testPoints: TEST_POINTS,
        showTestData: false,
        discretize: false,
        overlayMode: 'none',
        ...overrides,
    });
}

describe('deriveDecisionBoundaryModel', () => {
    it('returns an empty state when there are no training points', () => {
        expect(derive({ trainPoints: [] })).toEqual({
            kind: 'empty',
            title: 'No training data',
            description: 'Generate data or reset the playground to populate the decision boundary.',
        });
    });

    it('prepares a scalar state without copying the prediction grid', () => {
        const outputGrid = new Float32Array([0, 0.25, 0.75, 1]);

        const model = derive({
            frame: { ...EMPTY_FRAME, outputGrid, gridSize: 2 },
            showTestData: true,
            overlayMode: 'uncertainty',
        });

        expect(model.kind).toBe('scalar');
        if (model.kind !== 'scalar') return;
        expect(model.grid).toBe(outputGrid);
        expect(model.visibleTestPoints).toBe(TEST_POINTS);
        expect(model.overlayCopy.description).toMatch(/least sure/i);
        expect(model.accessibleDescription).toBe(model.overlayCopy.description);
    });

    it.each([
        ['binary classification', BINARY_TASK],
        ['regression', REGRESSION_TASK],
    ] as const)('selects scalar display data for %s', (_name, task) => {
        expect(derive({ task }).kind).toBe('scalar');
    });

    it('uses a continuous regression scale without classification overlays or discrete output', () => {
        const grid = new Float32Array([-4, -1, 2, 5]);
        const model = derive({ task: REGRESSION_TASK, discretize: true, overlayMode: 'uncertainty', frame: { ...EMPTY_FRAME, outputGrid: grid, gridSize: 2 } });
        expect(model.kind).toBe('scalar');
        if (model.kind !== 'scalar') return;
        expect(model.taskKind).toBe('regression');
        expect(model.discretize).toBe(false);
        expect(model.overlayMode).toBe('none');
        expect(model.valueDomain![0]).toBeLessThanOrEqual(-4);
        expect(model.valueDomain![1]).toBeGreaterThanOrEqual(5);
        expect(model.accessibleDescription).toMatch(/continuous/i);
        expect(model.grid).toBe(grid);
    });

    it('selects multiclass display data and keeps both grids by identity', () => {
        const classGrid = new Uint8Array([0, 1, 2, 2]);
        const confidenceGrid = new Float32Array([0.9, 0.62, 0.74, 0.58]);
        const layout: MulticlassBoundaryLayout = {
            gridSize: 2,
            classCount: 3,
            classLabels: [0, 1, 2],
        };

        const model = derive({
            task: MULTICLASS_TASK,
            frame: {
                ...EMPTY_FRAME,
                gridSize: 2,
                multiclassClassGrid: classGrid,
                multiclassConfidenceGrid: confidenceGrid,
                multiclassBoundaryLayout: layout,
            },
        });

        expect(model.kind).toBe('multiclass');
        if (model.kind !== 'multiclass') return;
        expect(model.classGrid).toBe(classGrid);
        expect(model.confidenceGrid).toBe(confidenceGrid);
        expect(model.summary).toMatchObject({
            dominantLabel: 'Class 2',
            dominantShare: 0.5,
            lowConfidenceShare: 0.25,
        });
        expect(model.accessibleDescription).toMatch(/Average winning confidence/i);
    });

    it.each([
        {
            name: 'partial payload',
            frame: { ...EMPTY_FRAME, multiclassClassGrid: new Uint8Array([0, 1, 2, 2]) },
        },
        {
            name: 'wrong grid length',
            frame: {
                ...EMPTY_FRAME,
                gridSize: 2,
                multiclassClassGrid: new Uint8Array([0, 1, 2]),
                multiclassConfidenceGrid: new Float32Array([0.9, 0.8, 0.7]),
                multiclassBoundaryLayout: { gridSize: 2, classCount: 3, classLabels: [0, 1, 2] } as MulticlassBoundaryLayout,
            },
        },
        {
            name: 'out-of-range class index',
            frame: {
                ...EMPTY_FRAME,
                gridSize: 2,
                multiclassClassGrid: new Uint8Array([0, 1, 2, 3]),
                multiclassConfidenceGrid: new Float32Array([0.9, 0.8, 0.7, 0.6]),
                multiclassBoundaryLayout: { gridSize: 2, classCount: 3, classLabels: [0, 1, 2] } as MulticlassBoundaryLayout,
            },
        },
        {
            name: 'invalid confidence',
            frame: {
                ...EMPTY_FRAME,
                gridSize: 2,
                multiclassClassGrid: new Uint8Array([0, 1, 2, 2]),
                multiclassConfidenceGrid: new Float32Array([0.9, Number.NaN, 0.7, 0.6]),
                multiclassBoundaryLayout: { gridSize: 2, classCount: 3, classLabels: [0, 1, 2] } as MulticlassBoundaryLayout,
            },
        },
        {
            name: 'non-integer layout size',
            frame: {
                ...EMPTY_FRAME,
                gridSize: 2,
                multiclassClassGrid: new Uint8Array([0, 1, 2, 2]),
                multiclassConfidenceGrid: new Float32Array([0.9, 0.8, 0.7, 0.6]),
                multiclassBoundaryLayout: {
                    gridSize: 1.5,
                    classCount: 3,
                    classLabels: [0, 1, 2],
                } as MulticlassBoundaryLayout,
            },
        },
        {
            name: 'unsupported labels',
            frame: {
                ...EMPTY_FRAME,
                gridSize: 2,
                multiclassClassGrid: new Uint8Array([0, 1, 2, 2]),
                multiclassConfidenceGrid: new Float32Array([0.9, 0.8, 0.7, 0.6]),
                multiclassBoundaryLayout: {
                    gridSize: 2,
                    classCount: 3,
                    classLabels: [0, 1, 9],
                } as unknown as MulticlassBoundaryLayout,
            },
        },
        {
            name: 'layout and frame size mismatch',
            frame: {
                ...EMPTY_FRAME,
                gridSize: 3,
                multiclassClassGrid: new Uint8Array([0, 1, 2, 2]),
                multiclassConfidenceGrid: new Float32Array([0.9, 0.8, 0.7, 0.6]),
                multiclassBoundaryLayout: {
                    gridSize: 2,
                    classCount: 3,
                    classLabels: [0, 1, 2],
                },
            },
        },
    ] satisfies readonly { name: string; frame: DecisionBoundaryFrameSnapshot }[])(
        'returns unavailable for malformed multiclass data: $name', ({ frame }) => {
            const model = derive({ task: MULTICLASS_TASK, frame });

            expect(model).toMatchObject({
                kind: 'unavailable',
                title: 'Binary decision boundary unavailable',
            });
        },
    );

    it('returns unavailable when a training classification point has a non-binary label', () => {
        const model = derive({
            trainPoints: [{ x: 0, y: 0, label: 2 }],
        });

        expect(model.kind).toBe('unavailable');
    });
});
