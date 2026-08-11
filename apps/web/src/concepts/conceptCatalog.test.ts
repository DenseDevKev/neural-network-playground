import { describe, expect, it } from 'vitest';
import { AUDIENCE_MODES } from '../productShell/audienceProfiles.ts';
import {
    CONCEPTS,
    CONCEPT_IDS,
    findConcept,
    getConceptById,
    getConceptsForProfile,
    type ConceptId,
} from './conceptCatalog.ts';

const EXPECTED_ORDER: readonly ConceptId[] = [
    'data-loss',
    'training-objective',
    'decision-boundary',
    'activation',
    'gradient',
    'checkpoint',
    'learning-rate',
    'train-test-split',
    'epoch',
];

describe('concept catalog', () => {
    it('keeps the nine stable concept IDs in editorial order', () => {
        expect(CONCEPT_IDS).toEqual(EXPECTED_ORDER);
        expect(CONCEPTS.map((entry) => entry.id)).toEqual(EXPECTED_ORDER);
    });

    it('returns exact entries by ID and undefined for missing IDs', () => {
        expect(getConceptById('decision-boundary')?.canonicalTerm).toBe('Decision boundary');
        expect(getConceptById('not-a-concept')).toBeUndefined();
        expect(getConceptById('')).toBeUndefined();
    });

    it('finds canonical terms and aliases with case-insensitive trimmed lookup', () => {
        expect(findConcept('  DATA LOSS  ')?.id).toBe('data-loss');
        expect(findConcept(' Objective Function ')?.id).toBe('training-objective');
        expect(findConcept('BACKPROP GRADIENT')?.id).toBe('gradient');
        expect(findConcept('unknown concept')).toBeUndefined();
        expect(findConcept('   ')).toBeUndefined();
    });

    it('filters concepts by profile without changing stable order', () => {
        for (const mode of AUDIENCE_MODES) {
            expect(getConceptsForProfile(mode).map((entry) => entry.id)).toEqual(EXPECTED_ORDER);
        }
    });

    it('contains the scientifically reviewed copy and metadata', () => {
        expect(getConceptById('data-loss')).toMatchObject({
            canonicalTerm: 'Data loss',
            plainDefinition:
                'How much the model’s predictions disagree with the targets under the selected loss rule. Lower is better when the data and rule are unchanged.',
            extendedExplanation:
                'The playground reports data loss separately from regularization. Full-split train and test values are averaged at the same model step, while the batch trend is an exponential moving average of mini-batch data loss. Classification uses cross-entropy; regression uses mean squared error or Huber loss.',
            aliases: ['prediction loss', 'predictive loss', 'fit loss'],
            related: ['training-objective', 'decision-boundary'],
            profiles: ['beginner', 'explore', 'lab'],
            difficulty: 'beginner',
            examples: [
                'On the same test split with the same loss rule, data loss 0.30 is better than 0.60.',
            ],
            uiTarget: 'loss',
        });

        expect(getConceptById('training-objective')).toMatchObject({
            canonicalTerm: 'Training objective',
            plainDefinition:
                'The quantity training minimizes: data loss on the training batch plus any regularization penalty.',
            extendedExplanation:
                'During an update, the objective is the mean mini-batch data loss plus one model-wide penalty. In full-split evidence, the shown training objective is train data loss plus the model penalty. Test data loss stays separate because it measures generalization.',
            aliases: ['total objective', 'regularized objective', 'objective function'],
            related: ['data-loss', 'gradient'],
            difficulty: 'intermediate',
            examples: [
                'If train data loss is 0.40 and the regularization penalty is 0.05, the training objective is 0.45.',
            ],
            uiTarget: 'loss',
        });

        expect(getConceptById('decision-boundary')).toMatchObject({
            canonicalTerm: 'Decision boundary',
            plainDefinition:
                'For classification, the border between input regions where the model predicts different classes.',
            extendedExplanation:
                'In this 2D playground, the colored field samples the model across the input plane. The boundary lies where the winning class changes. Smooth shading can show probability or confidence; discretized output shows hard class regions. Data points are evidence overlaid on the model’s field, not the boundary itself.',
            aliases: ['classification boundary', 'class boundary', 'decision surface'],
            related: ['data-loss', 'activation'],
            difficulty: 'beginner',
            examples: [
                'On the Circle dataset, a useful decision boundary can curve around the inner class.',
            ],
            uiTarget: 'boundary',
        });

        expect(getConceptById('activation')).toMatchObject({
            canonicalTerm: 'Activation',
            plainDefinition:
                'A neuron’s output after an activation function transforms its weighted input.',
            extendedExplanation:
                'Nonlinear hidden activations let stacked layers learn patterns that a purely linear network cannot. The recipe chooses the hidden activation, while the task fixes the output activation. Inspection statistics summarize sampled activation values; they are not raw values left by one arbitrary forward pass.',
            aliases: ['activation function', 'neuron activation', 'hidden activation'],
            related: ['decision-boundary', 'gradient'],
            difficulty: 'beginner',
            examples: [
                'ReLU returns zero for a negative input and passes a positive input through.',
            ],
            uiTarget: 'network',
        });

        expect(getConceptById('gradient')).toMatchObject({
            canonicalTerm: 'Gradient',
            plainDefinition:
                'The slopes that show how a small change to each model parameter would change the training objective.',
            extendedExplanation:
                'Backpropagation computes gradients for weights and biases, and the optimizer uses them to update the model. This playground reports data, penalty, and total gradient norms. Global-norm clipping, when enabled, rescales the complete objective gradient before the optimizer receives it.',
            aliases: ['objective gradient', 'parameter gradient', 'backprop gradient'],
            related: ['training-objective', 'activation'],
            difficulty: 'intermediate',
            examples: [
                'With plain SGD, a positive gradient moves that parameter downward; the learning rate controls how far.',
            ],
            uiTarget: 'inspection',
        });

        expect(getConceptById('checkpoint')).toMatchObject({
            canonicalTerm: 'Checkpoint',
            plainDefinition:
                'A restorable in-session record of the model and optimizer at a training step.',
            extendedExplanation:
                'The worker keeps a bounded set of checkpoints with model parameters, optimizer state, and paired evaluation metadata. Restoring pauses at that state and refreshes its evidence. A checkpoint is not a saved run, does not persist across reloads, and does not guarantee the same future shuffle sequence.',
            aliases: ['training checkpoint', 'model checkpoint', 'restore point'],
            related: ['data-loss', 'training-objective'],
            difficulty: 'beginner',
            examples: [
                'Restore Step 500 to inspect that model again, then resume from its parameters and optimizer state.',
            ],
        });
        expect(getConceptById('checkpoint')?.uiTarget).toBeUndefined();
        expect(getConceptById('learning-rate')).toMatchObject({
            canonicalTerm: 'Learning rate',
            plainDefinition:
                'The learning rate sets the scale of each optimizer update to the model’s parameters.',
            extendedExplanation:
                'The optimizer uses the learning rate to scale parameter updates. Larger values can move faster but may overshoot or make loss unstable; smaller values can be steadier but slower. A schedule can change the rate as training proceeds.',
            aliases: ['step size', 'optimizer learning rate', 'update scale'],
            related: ['gradient', 'training-objective'],
            profiles: ['beginner', 'explore', 'lab'],
            difficulty: 'beginner',
            examples: [
                'With plain SGD, learning rate 0.1 moves a parameter ten times as far as 0.01 for the same gradient.',
            ],
            uiTarget: 'hyperparams',
        });
        expect(getConceptById('train-test-split')).toMatchObject({
            canonicalTerm: 'Train/test split',
            plainDefinition:
                'The train/test split assigns generated examples to a training set used for fitting and a held-out test set used only for evaluation.',
            extendedExplanation:
                'Membership is deterministic and stays fixed while the current prepared experiment trains. Data-recipe changes or Reshuffle split rebuild membership. Test examples never drive weight updates; full-split test evidence measures held-out performance at the same model step.',
            aliases: ['data split', 'training test split', 'held-out split'],
            related: ['data-loss', 'training-objective'],
            profiles: ['beginner', 'explore', 'lab'],
            difficulty: 'beginner',
            examples: [
                'With 200 generated examples and a 70% train ratio, 140 train and 60 are held out for test.',
            ],
            uiTarget: 'data',
        });
        expect(getConceptById('epoch')).toMatchObject({
            canonicalTerm: 'Epoch',
            plainDefinition: 'One epoch is one complete pass through the current training set.',
            extendedExplanation:
                'The worker shuffles the training examples, processes each one once in mini-batches, and increments Epoch only after the full training set is consumed. The final batch may be smaller than the configured batch size. Epoch counts data passes, not convergence or model quality.',
            aliases: ['training epoch', 'data pass', 'full training pass'],
            related: ['learning-rate', 'checkpoint'],
            profiles: ['beginner', 'explore', 'lab'],
            difficulty: 'beginner',
            examples: [
                'With 150 training examples and batch size 64, one epoch completes after batches of 64, 64, and 22 examples.',
            ],
        });
        expect(getConceptById('epoch')?.uiTarget).toBeUndefined();
        expect(CONCEPTS.every((entry) => entry.documentationUrl === undefined)).toBe(true);
    });

    it('has unique IDs and normalized canonical terms and aliases', () => {
        expect(new Set(CONCEPTS.map((entry) => entry.id)).size).toBe(CONCEPTS.length);

        const normalizedLookupTerms = CONCEPTS.flatMap((entry) => [
            entry.canonicalTerm,
            ...entry.aliases,
        ]).map((term) => term.trim().toLowerCase());

        expect(new Set(normalizedLookupTerms).size).toBe(normalizedLookupTerms.length);
    });

    it('references only valid related IDs and includes every profile once', () => {
        const validIds = new Set(CONCEPT_IDS);

        for (const entry of CONCEPTS) {
            expect(entry.related.every((id) => validIds.has(id))).toBe(true);
            expect(entry.profiles).toEqual(AUDIENCE_MODES);
            expect(new Set(entry.profiles).size).toBe(AUDIENCE_MODES.length);
        }
    });

    it('deep-freezes the stable catalog entry structures', () => {
        expect(Object.isFrozen(CONCEPT_IDS)).toBe(true);
        expect(Object.isFrozen(CONCEPTS)).toBe(true);

        for (const entry of CONCEPTS) {
            expect(Object.isFrozen(entry)).toBe(true);
            expect(Object.isFrozen(entry.aliases)).toBe(true);
            expect(Object.isFrozen(entry.related)).toBe(true);
            expect(Object.isFrozen(entry.profiles)).toBe(true);
            expect(Object.isFrozen(entry.examples)).toBe(true);
        }
    });
});
