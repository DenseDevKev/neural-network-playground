import type { Preset } from '@nn-playground/shared';
import { Tooltip } from '../common/Tooltip.tsx';

interface PresetCardProps {
    preset: Preset;
    isSelected: boolean;
    onSelect: (preset: Preset) => void;
}

function formatDifficulty(difficulty: NonNullable<Preset['difficulty']>) {
    return difficulty.charAt(0).toUpperCase() + difficulty.slice(1);
}

export function PresetCard({ preset, isSelected, onSelect }: PresetCardProps) {
    return (
        <Tooltip content="Apply preset and reset training" block>
            <button
                type="button"
                className={`preset-card${isSelected ? ' preset-card--selected' : ''}`}
                onClick={() => onSelect(preset)}
                aria-pressed={isSelected}
                aria-label={`Apply preset: ${preset.title}`}
            >
                {preset.thumbnail && (
                    <div className="preset-card__thumbnail" aria-hidden="true">
                        <img src={preset.thumbnail} alt="" />
                    </div>
                )}
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
    );
}
