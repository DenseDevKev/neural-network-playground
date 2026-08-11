import { useId } from 'react';
import type { RecipeCatalogEntry } from '@nn-playground/shared';
import { Tooltip } from '../common/Tooltip.tsx';
import { STATE_EFFECTS } from '../../copy/stateEffects.ts';

interface PresetCardProps {
    preset: RecipeCatalogEntry;
    isSelected: boolean;
    disabled?: boolean;
    onSelect: (preset: RecipeCatalogEntry) => void;
}

function formatDifficulty(difficulty: NonNullable<RecipeCatalogEntry['difficulty']>) {
    return difficulty.charAt(0).toUpperCase() + difficulty.slice(1);
}

export function PresetCard({ preset, isSelected, disabled = false, onSelect }: PresetCardProps) {
    const effectsId = `${useId()}-preset-apply-effects`;

    return (
        <>
            <Tooltip content={STATE_EFFECTS['preset-apply']} block>
                <button
                    type="button"
                    className={`preset-card${isSelected ? ' preset-card--selected' : ''}`}
                    onClick={() => onSelect(preset)}
                    aria-pressed={isSelected}
                    aria-label={`Apply preset: ${preset.title}`}
                    aria-describedby={effectsId}
                    disabled={disabled}
                >
                    <div className="preset-card__content">
                        <div className="preset-card__header">
                            <span className="preset-card__title">{preset.title}</span>
                            {preset.difficulty && (
                                <span className={`preset-card__badge preset-card__badge--${preset.difficulty}`}>
                                    {formatDifficulty(preset.difficulty)}
                                </span>
                            )}
                        </div>
                        <p className="preset-card__description">{preset.description}</p>
                        {preset.learningGoal && (
                            <p className="preset-card__goal">
                                <span className="preset-card__goal-icon" aria-hidden="true">💡</span>
                                <span>{preset.learningGoal}</span>
                            </p>
                        )}
                    </div>
                </button>
            </Tooltip>
            <span id={effectsId} className="sr-only">
                {STATE_EFFECTS['preset-apply']}
            </span>
        </>
    );
}
