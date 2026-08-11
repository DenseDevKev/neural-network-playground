import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { PairedEvaluation } from '@nn-playground/shared';
import { PREPARED_PRESETS } from '@nn-playground/shared';
import { ConfusionMatrix } from './ConfusionMatrix.tsx';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { createScientificTrustFixtures } from '../../test/scientificTrustFixtures.ts';
import { installLegacyTrainingSnapshotForTest } from '../../test/playgroundStoreTestUtils.ts';
import { resetFrameBuffer, updateFrameBuffer } from '../../worker/frameBuffer.ts';

function prepared(id: string) {
    const result = PREPARED_PRESETS.find((entry) => entry.id === id)?.prepared;
    if (!result) throw new Error(`missing ${id}`);
    return result;
}

function binaryPair(base: PairedEvaluation): PairedEvaluation {
    return {
        ...base,
        evaluationId: 31,
        model: { ...base.model, revision: 1230, step: 1230, epoch: 12 },
        train: {
            ...base.train,
            values: {
                dataLoss: 0.2,
                accuracy: 0.9,
                confusionMatrix: { tn: 90, fp: 10, fn: 11, tp: 99 },
            },
        },
        test: {
            ...base.test,
            basis: { ...base.test.basis, sampleCount: 90, populationCount: 90 },
            values: {
                dataLoss: 0.3,
                accuracy: 0.85,
                confusionMatrix: { tn: 40, fp: 10, fn: 5, tp: 35 },
            },
        },
    };
}

describe('ConfusionMatrix paired evaluation provenance', () => {
    let basePair: PairedEvaluation;

    beforeEach(async () => {
        basePair = (await createScientificTrustFixtures()).evaluation;
        const binary = prepared('xor-hidden');
        usePlaygroundStore.setState({
            access: { status: 'ready', prepared: binary },
        });
        useTrainingStore.setState({
            latestEvaluation: null,
            trainPoints: [],
            testPoints: [],
        });
        resetFrameBuffer();
    });

    it('renders a neutral state until a paired evaluation includes test confusion', () => {
        render(<ConfusionMatrix />);

        expect(screen.getByText('Confusion matrix unavailable')).toBeInTheDocument();
        expect(screen.getByText(/paired full test evaluation/i)).toBeInTheDocument();
    });

    it('binds only to paired evaluation despite contradictory legacy and frame matrices', () => {
        updateFrameBuffer({ confusionMatrix: { tn: 1, fp: 2, fn: 3, tp: 4 } });
        useTrainingStore.setState({
            latestEvaluation: binaryPair(basePair),
            testPoints: [{ x: 0, y: 0, label: 1 }],
        });
        const removeLegacySnapshot = installLegacyTrainingSnapshotForTest({
            testMetrics: { confusionMatrix: { tn: 9, fp: 9, fn: 9, tp: 9 } },
        });
        try {
            render(<ConfusionMatrix />);

            expect(screen.getByLabelText('TN cell')).toHaveTextContent('40');
            expect(screen.getByLabelText('TP cell')).toHaveTextContent('35');
            expect(screen.queryByText('9')).not.toBeInTheDocument();
            expect(screen.getByText(/Evaluation 31 · model step 1,230 · all 90 test samples/i))
                .toBeInTheDocument();
        } finally {
            removeLegacySnapshot();
        }
    });

    it('renders a worker-authored multiclass matrix from the paired test evaluation', () => {
        const multiclass = prepared('three-class-clusters');
        usePlaygroundStore.setState({
            access: { status: 'ready', prepared: multiclass },
        });
        useTrainingStore.setState({
            latestEvaluation: {
                ...binaryPair(basePair),
                test: {
                    ...basePair.test,
                    basis: { ...basePair.test.basis, sampleCount: 90, populationCount: 90 },
                    values: {
                        dataLoss: 0.4,
                        accuracy: 80 / 90,
                        confusionMatrix: {
                            classCount: 3,
                            classLabels: [0, 1, 2],
                            counts: [25, 3, 2, 2, 27, 1, 1, 1, 28],
                        },
                    },
                },
            },
        });

        render(<ConfusionMatrix />);

        expect(screen.getByText('Multiclass Confusion Matrix (Full Test Split)')).toBeInTheDocument();
        expect(screen.getByLabelText(/actual Class 2 predicted Class 2/i)).toHaveTextContent('28');
        expect(screen.getByText(/Evaluation 31 · model step 1,230 · all 90 test samples/i))
            .toBeInTheDocument();
    });

    it('does not render for regression recipes', () => {
        const regression = prepared('regression-plane');
        usePlaygroundStore.setState({
            access: { status: 'ready', prepared: regression },
        });
        useTrainingStore.setState({ latestEvaluation: binaryPair(basePair) });

        const { container } = render(<ConfusionMatrix />);
        expect(container).toBeEmptyDOMElement();
    });
});
