import type { AudienceMode } from '../productShell/audienceProfiles.ts';
import { AUDIENCE_MODES } from '../productShell/audienceProfiles.ts';
import type { EvidenceViewId, RecipeSectionId } from '../productShell/shellTypes.ts';

export const CONCEPT_IDS = Object.freeze([
    'data-loss',
    'training-objective',
    'decision-boundary',
    'activation',
    'gradient',
    'checkpoint',
    'learning-rate',
    'train-test-split',
    'epoch',
] as const);

export type ConceptId = (typeof CONCEPT_IDS)[number];
export type ConceptDifficulty = 'beginner' | 'intermediate' | 'advanced';
export type ConceptUiTarget = RecipeSectionId | EvidenceViewId;

export interface ConceptEntry {
    readonly id: ConceptId;
    readonly canonicalTerm: string;
    readonly plainDefinition: string;
    readonly extendedExplanation?: string;
    readonly aliases: readonly string[];
    readonly related: readonly ConceptId[];
    readonly profiles: readonly AudienceMode[];
    readonly difficulty: ConceptDifficulty;
    readonly examples?: readonly string[];
    readonly uiTarget?: ConceptUiTarget;
    readonly documentationUrl?: string;
}

function freezeEntry(entry: ConceptEntry): ConceptEntry {
    return Object.freeze({
        ...entry,
        aliases: Object.freeze([...entry.aliases]),
        related: Object.freeze([...entry.related]),
        profiles: Object.freeze([...entry.profiles]),
        examples: entry.examples ? Object.freeze([...entry.examples]) : undefined,
    });
}

const ALL_PROFILES = Object.freeze([...AUDIENCE_MODES]);

export const CONCEPTS: readonly ConceptEntry[] = Object.freeze([
    freezeEntry({
        id: 'data-loss',
        canonicalTerm: 'Data loss',
        plainDefinition:
            'How much the model’s predictions disagree with the targets under the selected loss rule. Lower is better when the data and rule are unchanged.',
        extendedExplanation:
            'The playground reports data loss separately from regularization. Full-split train and test values are averaged at the same model step, while the batch trend is an exponential moving average of mini-batch data loss. Classification uses cross-entropy; regression uses mean squared error or Huber loss.',
        aliases: ['prediction loss', 'predictive loss', 'fit loss'],
        related: ['training-objective', 'decision-boundary'],
        profiles: ALL_PROFILES,
        difficulty: 'beginner',
        examples: [
            'On the same test split with the same loss rule, data loss 0.30 is better than 0.60.',
        ],
        uiTarget: 'loss',
    }),
    freezeEntry({
        id: 'training-objective',
        canonicalTerm: 'Training objective',
        plainDefinition:
            'The quantity training minimizes: data loss on the training batch plus any regularization penalty.',
        extendedExplanation:
            'During an update, the objective is the mean mini-batch data loss plus one model-wide penalty. In full-split evidence, the shown training objective is train data loss plus the model penalty. Test data loss stays separate because it measures generalization.',
        aliases: ['total objective', 'regularized objective', 'objective function'],
        related: ['data-loss', 'gradient'],
        profiles: ALL_PROFILES,
        difficulty: 'intermediate',
        examples: [
            'If train data loss is 0.40 and the regularization penalty is 0.05, the training objective is 0.45.',
        ],
        uiTarget: 'loss',
    }),
    freezeEntry({
        id: 'decision-boundary',
        canonicalTerm: 'Decision boundary',
        plainDefinition:
            'For classification, the border between input regions where the model predicts different classes.',
        extendedExplanation:
            'In this 2D playground, the colored field samples the model across the input plane. The boundary lies where the winning class changes. Smooth shading can show probability or confidence; discretized output shows hard class regions. Data points are evidence overlaid on the model’s field, not the boundary itself.',
        aliases: ['classification boundary', 'class boundary', 'decision surface'],
        related: ['data-loss', 'activation'],
        profiles: ALL_PROFILES,
        difficulty: 'beginner',
        examples: [
            'On the Circle dataset, a useful decision boundary can curve around the inner class.',
        ],
        uiTarget: 'boundary',
    }),
    freezeEntry({
        id: 'activation',
        canonicalTerm: 'Activation',
        plainDefinition:
            'A neuron’s output after an activation function transforms its weighted input.',
        extendedExplanation:
            'Nonlinear hidden activations let stacked layers learn patterns that a purely linear network cannot. The recipe chooses the hidden activation, while the task fixes the output activation. Inspection statistics summarize sampled activation values; they are not raw values left by one arbitrary forward pass.',
        aliases: ['activation function', 'neuron activation', 'hidden activation'],
        related: ['decision-boundary', 'gradient'],
        profiles: ALL_PROFILES,
        difficulty: 'beginner',
        examples: [
            'ReLU returns zero for a negative input and passes a positive input through.',
        ],
        uiTarget: 'network',
    }),
    freezeEntry({
        id: 'gradient',
        canonicalTerm: 'Gradient',
        plainDefinition:
            'The slopes that show how a small change to each model parameter would change the training objective.',
        extendedExplanation:
            'Backpropagation computes gradients for weights and biases, and the optimizer uses them to update the model. This playground reports data, penalty, and total gradient norms. Global-norm clipping, when enabled, rescales the complete objective gradient before the optimizer receives it.',
        aliases: ['objective gradient', 'parameter gradient', 'backprop gradient'],
        related: ['training-objective', 'activation'],
        profiles: ALL_PROFILES,
        difficulty: 'intermediate',
        examples: [
            'With plain SGD, a positive gradient moves that parameter downward; the learning rate controls how far.',
        ],
        uiTarget: 'inspection',
    }),
    freezeEntry({
        id: 'checkpoint',
        canonicalTerm: 'Checkpoint',
        plainDefinition:
            'A restorable in-session record of the model and optimizer at a training step.',
        extendedExplanation:
            'The worker keeps a bounded set of checkpoints with model parameters, optimizer state, and paired evaluation metadata. Restoring pauses at that state and refreshes its evidence. A checkpoint is not a saved run, does not persist across reloads, and does not guarantee the same future shuffle sequence.',
        aliases: ['training checkpoint', 'model checkpoint', 'restore point'],
        related: ['data-loss', 'training-objective'],
        profiles: ALL_PROFILES,
        difficulty: 'beginner',
        examples: [
            'Restore Step 500 to inspect that model again, then resume from its parameters and optimizer state.',
        ],
    }),
    freezeEntry({
        id: 'learning-rate',
        canonicalTerm: 'Learning rate',
        plainDefinition:
            'The learning rate sets the scale of each optimizer update to the model’s parameters.',
        extendedExplanation:
            'The optimizer uses the learning rate to scale parameter updates. Larger values can move faster but may overshoot or make loss unstable; smaller values can be steadier but slower. A schedule can change the rate as training proceeds.',
        aliases: ['step size', 'optimizer learning rate', 'update scale'],
        related: ['gradient', 'training-objective'],
        profiles: ALL_PROFILES,
        difficulty: 'beginner',
        examples: [
            'With plain SGD, learning rate 0.1 moves a parameter ten times as far as 0.01 for the same gradient.',
        ],
        uiTarget: 'hyperparams',
    }),
    freezeEntry({
        id: 'train-test-split',
        canonicalTerm: 'Train/test split',
        plainDefinition:
            'The train/test split assigns generated examples to a training set used for fitting and a held-out test set used only for evaluation.',
        extendedExplanation:
            'Membership is deterministic and stays fixed while the current prepared experiment trains. Data-recipe changes or Reshuffle split rebuild membership. Test examples never drive weight updates; full-split test evidence measures held-out performance at the same model step.',
        aliases: ['data split', 'training test split', 'held-out split'],
        related: ['data-loss', 'training-objective'],
        profiles: ALL_PROFILES,
        difficulty: 'beginner',
        examples: [
            'With 200 generated examples and a 70% train ratio, 140 train and 60 are held out for test.',
        ],
        uiTarget: 'data',
    }),
    freezeEntry({
        id: 'epoch',
        canonicalTerm: 'Epoch',
        plainDefinition: 'One epoch is one complete pass through the current training set.',
        extendedExplanation:
            'The worker shuffles the training examples, processes each one once in mini-batches, and increments Epoch only after the full training set is consumed. The final batch may be smaller than the configured batch size. Epoch counts data passes, not convergence or model quality.',
        aliases: ['training epoch', 'data pass', 'full training pass'],
        related: ['learning-rate', 'checkpoint'],
        profiles: ALL_PROFILES,
        difficulty: 'beginner',
        examples: [
            'With 150 training examples and batch size 64, one epoch completes after batches of 64, 64, and 22 examples.',
        ],
    }),
]);

const conceptsById = new Map<string, ConceptEntry>(
    CONCEPTS.map((entry) => [entry.id, entry]),
);

function normalizeLookupTerm(term: string): string {
    return term.trim().toLowerCase();
}

const conceptsByTerm = new Map<string, ConceptEntry>();
for (const entry of CONCEPTS) {
    conceptsByTerm.set(normalizeLookupTerm(entry.canonicalTerm), entry);
    for (const alias of entry.aliases) {
        conceptsByTerm.set(normalizeLookupTerm(alias), entry);
    }
}

export function getConceptById(id: string): ConceptEntry | undefined {
    return conceptsById.get(id);
}

export function findConcept(term: string): ConceptEntry | undefined {
    const normalizedTerm = normalizeLookupTerm(term);
    return normalizedTerm ? conceptsByTerm.get(normalizedTerm) : undefined;
}

export function getConceptsForProfile(profile: AudienceMode): readonly ConceptEntry[] {
    return CONCEPTS.filter((entry) => entry.profiles.includes(profile));
}
