import type { EvidenceViewId, RecipeSectionId } from './shellTypes.ts';
import {
    ALL_BUILD_MODULES,
    ALL_EVIDENCE_VIEWS,
    getAudienceProfile,
    type AudienceMode,
    type BuildModuleId,
    type ShellEvidenceViewId,
} from './audienceProfiles.ts';

export const BUILD_FALLBACK = 'data' as const satisfies BuildModuleId;
export const EVIDENCE_FALLBACK = 'boundary' as const satisfies ShellEvidenceViewId;

export function getVisibleBuildModules(
    mode: AudienceMode,
    advancedToolsOpen: boolean,
): readonly BuildModuleId[] {
    return advancedToolsOpen ? ALL_BUILD_MODULES : getAudienceProfile(mode).coreBuildModules;
}

export function getVisibleEvidenceViews(
    mode: AudienceMode,
    advancedToolsOpen: boolean,
): readonly ShellEvidenceViewId[] {
    return advancedToolsOpen ? ALL_EVIDENCE_VIEWS : getAudienceProfile(mode).coreEvidenceViews;
}

export function resolveEvidenceAlias(view: EvidenceViewId): ShellEvidenceViewId {
    return view === 'history' ? EVIDENCE_FALLBACK : view;
}

export function isRecipeSectionVisible(
    mode: AudienceMode,
    advancedToolsOpen: boolean,
    section: RecipeSectionId,
): boolean {
    return section === 'presets' || getVisibleBuildModules(mode, advancedToolsOpen).includes(section);
}

export function isEvidenceViewVisible(
    mode: AudienceMode,
    advancedToolsOpen: boolean,
    view: EvidenceViewId,
): boolean {
    return getVisibleEvidenceViews(mode, advancedToolsOpen).includes(resolveEvidenceAlias(view));
}

export function resolveVisibleRecipeSection(
    mode: AudienceMode,
    advancedToolsOpen: boolean,
    section: RecipeSectionId,
): RecipeSectionId {
    return isRecipeSectionVisible(mode, advancedToolsOpen, section) ? section : BUILD_FALLBACK;
}

export function resolveVisibleEvidenceView(
    mode: AudienceMode,
    advancedToolsOpen: boolean,
    view: EvidenceViewId,
): ShellEvidenceViewId {
    const resolvedView = resolveEvidenceAlias(view);
    return isEvidenceViewVisible(mode, advancedToolsOpen, resolvedView)
        ? resolvedView
        : EVIDENCE_FALLBACK;
}
