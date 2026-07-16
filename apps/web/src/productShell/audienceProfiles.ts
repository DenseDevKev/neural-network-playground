import type { EvidenceViewId, RecipeSectionId } from '../store/useLayoutStore.ts';

export const AUDIENCE_MODES = Object.freeze(['beginner', 'explore', 'lab'] as const);

export type AudienceMode = (typeof AUDIENCE_MODES)[number];
export type GuidanceLevel = 'high' | 'standard' | 'compact';
export type BuildModuleId = Exclude<RecipeSectionId, 'presets'>;
export type ShellEvidenceViewId = Exclude<EvidenceViewId, 'history'>;

export interface AudienceProfile {
    readonly label: string;
    readonly description: string;
    readonly coreBuildModules: readonly BuildModuleId[];
    readonly coreEvidenceViews: readonly ShellEvidenceViewId[];
    readonly advancedDefaultOpen: boolean;
    readonly guidanceLevel: GuidanceLevel;
}

export const ADVANCED_BUILD_MODULES = Object.freeze([
    'features',
    'hyperparams',
    'config',
] as const satisfies readonly BuildModuleId[]);

export const ADVANCED_EVIDENCE_VIEWS = Object.freeze([
    'confusion',
    'inspection',
    'code',
] as const satisfies readonly ShellEvidenceViewId[]);

export const ALL_BUILD_MODULES = Object.freeze([
    'data',
    'network',
    'features',
    'hyperparams',
    'config',
] as const satisfies readonly BuildModuleId[]);

export const ALL_EVIDENCE_VIEWS = Object.freeze([
    'boundary',
    'loss',
    'confusion',
    'inspection',
    'code',
] as const satisfies readonly ShellEvidenceViewId[]);

const beginnerProfile: AudienceProfile = Object.freeze({
    label: 'Beginner',
    description: 'Keeps the core data, network, boundary, and loss tools visible with more guidance.',
    coreBuildModules: Object.freeze(['data', 'network'] as const),
    coreEvidenceViews: Object.freeze(['boundary', 'loss'] as const),
    advancedDefaultOpen: false,
    guidanceLevel: 'high',
});

const exploreProfile: AudienceProfile = Object.freeze({
    label: 'Explore',
    description: 'Adds feature, hyperparameter, and confusion tools for guided experimentation.',
    coreBuildModules: Object.freeze(['data', 'network', 'features', 'hyperparams'] as const),
    coreEvidenceViews: Object.freeze(['boundary', 'loss', 'confusion'] as const),
    advancedDefaultOpen: false,
    guidanceLevel: 'standard',
});

const labProfile: AudienceProfile = Object.freeze({
    label: 'Lab',
    description: 'Opens the full workspace with compact guidance for detailed analysis.',
    coreBuildModules: Object.freeze(['data', 'network', 'features', 'hyperparams'] as const),
    coreEvidenceViews: Object.freeze(['boundary', 'loss', 'confusion'] as const),
    advancedDefaultOpen: true,
    guidanceLevel: 'compact',
});

export const AUDIENCE_PROFILES: Readonly<Record<AudienceMode, AudienceProfile>> = Object.freeze({
    beginner: beginnerProfile,
    explore: exploreProfile,
    lab: labProfile,
});

export function isAudienceMode(value: unknown): value is AudienceMode {
    return typeof value === 'string' && AUDIENCE_MODES.includes(value as AudienceMode);
}

export function getAudienceProfile(mode: AudienceMode): AudienceProfile {
    return AUDIENCE_PROFILES[mode];
}
