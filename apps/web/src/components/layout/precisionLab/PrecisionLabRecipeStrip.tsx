import { memo } from 'react';
import type { PrecisionLabRecipeModel } from './precisionLabRecipeModel.ts';

export interface PrecisionLabRecipeStripProps {
    readonly model: PrecisionLabRecipeModel;
    readonly onEditRecipe: () => void;
}

export const PrecisionLabRecipeStrip = memo(function PrecisionLabRecipeStrip({
    model,
    onEditRecipe,
}: PrecisionLabRecipeStripProps) {
    return (
        <section
            className="precision-recipe"
            role="region"
            aria-label="Recipe summary"
            data-tone={model.tone}
            data-precision-region="recipe"
        >
            <dl className="precision-recipe__facts">
                <div><dt>Data</dt><dd>{model.dataset}</dd></div>
                <div><dt>Architecture</dt><dd>{model.architecture}</dd></div>
                <div><dt>Activation</dt><dd>{model.hiddenActivation}</dd></div>
                <div><dt>Output</dt><dd>{model.output}</dd></div>
                <div><dt>Seed</dt><dd>{model.seed}</dd></div>
                <div><dt>Evaluation</dt><dd>{model.evaluationLabel}</dd></div>
            </dl>
            <button type="button" onClick={onEditRecipe}>Edit recipe</button>
        </section>
    );
});
