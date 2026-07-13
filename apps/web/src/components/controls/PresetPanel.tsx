// ── Preset Panel ──
// Card grid to quickly apply curated experiment presets.

import { memo, useCallback, useEffect, useRef } from 'react';
import {
    PREPARED_PRESETS,
    type ExperimentSchemaIssue,
    type RecipeCatalogEntry,
} from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { LoadingState } from '../common/LoadingState.tsx';
import { PresetCard } from './PresetCard.tsx';

interface PresetPanelProps {
    onReset: () => void;
    onApplied?: () => void;
}

function formatPreparationIssues(issues: readonly ExperimentSchemaIssue[]): string {
    return issues.map((issue) => `${issue.path}: ${issue.message}`).join('; ');
}

export const PresetPanel = memo(function PresetPanel({ onReset, onApplied }: PresetPanelProps) {
    const applyRecipe = usePlaygroundStore((s) => s.applyRecipe);
    const prepared = usePlaygroundStore((s) => s.access.status === 'ready'
        ? s.access.prepared
        : null);
    const isLoading = useTrainingStore((s) => s.presetConfigLoading);
    const configError = useTrainingStore((s) => s.configError);
    const configErrorSource = useTrainingStore((s) => s.configErrorSource);
    const pendingSelection = useRef(false);
    const mounted = useRef(true);

    useEffect(() => {
        mounted.current = true;
        return () => {
            mounted.current = false;
        };
    }, []);

    const handleSelect = useCallback(
        async (entry: RecipeCatalogEntry) => {
            if (pendingSelection.current) return;
            const currentAccess = usePlaygroundStore.getState().access;
            const currentKey = currentAccess.status === 'ready'
                ? currentAccess.prepared.identities.canonicalRecipeKey
                : undefined;
            if (currentKey === entry.prepared.identities.canonicalRecipeKey) return;

            pendingSelection.current = true;
            const trainingStore = useTrainingStore.getState();
            trainingStore.beginConfigChange('preset');
            let requestId = usePlaygroundStore.getState().preparation.requestId;
            try {
                const pendingResult = applyRecipe(entry);
                requestId = usePlaygroundStore.getState().preparation.requestId;
                const result = await pendingResult;
                const playground = usePlaygroundStore.getState();
                if (playground.preparation.requestId !== requestId) return;

                if (!result.ok) {
                    trainingStore.failConfigChange(formatPreparationIssues(result.issues));
                    return;
                }

                if (playground.access.status !== 'ready'
                    || playground.access.prepared !== result.value
                    || !mounted.current) return;
                onReset();
                if (mounted.current) onApplied?.();
            } catch (error) {
                if (usePlaygroundStore.getState().preparation.requestId === requestId) {
                    trainingStore.failConfigChange(
                        error instanceof Error ? error.message : 'Failed to apply preset',
                    );
                }
            } finally {
                pendingSelection.current = false;
            }
        },
        [applyRecipe, onApplied, onReset],
    );
    const retryPresetChange = () => useTrainingStore.getState().retryConfigSync();

    return (
        <div>
            <LoadingState isLoading={isLoading} inline message="Applying preset..." />
            {configError && configErrorSource === 'preset' && (
                <div className="config-feedback config-feedback--error" role="alert">
                    <span>{configError}</span>
                    <button type="button" className="btn btn--ghost btn--sm" onClick={retryPresetChange}>
                        Retry
                    </button>
                </div>
            )}
            <div className="preset-grid" role="list" aria-label="Available presets" aria-busy={isLoading}>
                {PREPARED_PRESETS.map((preset) => {
                    const isSelected = prepared?.identities.canonicalRecipeKey
                        === preset.prepared.identities.canonicalRecipeKey;
                    return (
                    <div key={`${preset.id}@${preset.revision}`} role="listitem">
                        <PresetCard
                            preset={preset}
                            isSelected={isSelected}
                            disabled={isLoading}
                            onSelect={handleSelect}
                        />
                    </div>
                    );
                })}
            </div>
        </div>
    );
});
