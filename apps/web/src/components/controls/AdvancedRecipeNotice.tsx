import { memo, useMemo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import { isRecipeSectionVisible } from '../../productShell/visibleShell.ts';
import { deriveAdvancedRecipeSettings } from '../../productShell/advancedRecipeSettings.ts';
import { scheduleExplanationPanelFocus } from '../../explanations/explanationActionFocus.ts';

export const AdvancedRecipeNotice = memo(function AdvancedRecipeNotice() {
    const currentRecipe = usePlaygroundStore((s) => s.access.status === 'ready' ? s.access.prepared.document.recipe : null);
    const audienceMode = useLayoutStore((s) => s.audienceMode);
    const advancedToolsOpen = useLayoutStore((s) => s.advancedToolsOpen);
    const openAdvancedRecipeSection = useLayoutStore((s) => s.openAdvancedRecipeSection);
    const advancedSettings = useMemo(() => currentRecipe === null ? [] : deriveAdvancedRecipeSettings(currentRecipe), [currentRecipe]);
    const showAdvancedSettings = advancedSettings.length > 0 && !isRecipeSectionVisible(audienceMode, advancedToolsOpen, 'hyperparams');
    return <>            {showAdvancedSettings ? (
                <aside
                    className="forge-advanced-settings-note"
                    role="note"
                    aria-label="Advanced settings active"
                >
                    <strong>Advanced settings active</strong>
                    <ul>
                        {advancedSettings.map((setting) => (
                            <li key={setting.id}>
                                <span>{setting.label}</span>
                                <strong>{setting.value}</strong>
                            </li>
                        ))}
                    </ul>
                    <button
                        type="button"
                        className="btn btn--ghost"
                        onClick={() => {
                            openAdvancedRecipeSection('hyperparams');
                            scheduleExplanationPanelFocus('hyperparams');
                        }}
                    >
                        Open Advanced Tools
                    </button>
                </aside>
            ) : null}
    </>;
});
