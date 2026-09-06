export type PrecisionLabRecipeTone =
    | 'ready'
    | 'updating'
    | 'drift'
    | 'stale'
    | 'unavailable';

export interface PrecisionLabRecipeModelInput {
    readonly dataset: string;
    readonly architecture: string;
    readonly hiddenActivation: string;
    readonly output: string;
    readonly seed: number;
    readonly hasRecipeDrift: boolean;
    readonly pendingConfiguration: boolean;
    readonly evaluationAgeSteps: number | null;
}

export interface PrecisionLabRecipeModel {
    readonly dataset: string;
    readonly architecture: string;
    readonly hiddenActivation: string;
    readonly output: string;
    readonly seed: string;
    readonly evaluationLabel: string;
    readonly tone: PrecisionLabRecipeTone;
}

const EMPTY_MODEL: PrecisionLabRecipeModel = Object.freeze({
    dataset: 'Unavailable',
    architecture: 'Unavailable',
    hiddenActivation: 'Unavailable',
    output: 'Unavailable',
    seed: 'Unavailable',
    evaluationLabel: 'No evaluation',
    tone: 'unavailable',
});

export function derivePrecisionLabRecipeModel(
    input: PrecisionLabRecipeModelInput | null,
): PrecisionLabRecipeModel {
    if (input === null) return EMPTY_MODEL;

    const tone: PrecisionLabRecipeTone = input.pendingConfiguration
        ? 'updating'
        : input.hasRecipeDrift
            ? 'drift'
            : (input.evaluationAgeSteps ?? 0) > 0
                ? 'stale'
                : 'ready';

    return Object.freeze({
        dataset: input.dataset,
        architecture: input.architecture,
        hiddenActivation: input.hiddenActivation,
        output: input.output,
        seed: String(input.seed),
        evaluationLabel: tone === 'updating'
            ? 'Updating'
            : tone === 'drift'
                ? 'Recipe differs from trained model'
                : tone === 'stale'
                    ? `Evaluation ${input.evaluationAgeSteps} steps behind`
                    : 'Evaluation fresh',
        tone,
    });
}
