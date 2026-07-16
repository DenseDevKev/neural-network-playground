export const WORKSPACE_VIEWS = Object.freeze(['build', 'run'] as const);
export const LAYOUT_VARIANTS = Object.freeze(['dock', 'focus', 'grid', 'split'] as const);
export const RECIPE_SECTION_IDS = Object.freeze([
    'presets',
    'data',
    'features',
    'network',
    'hyperparams',
    'config',
] as const);
export const EVIDENCE_VIEW_IDS = Object.freeze([
    'boundary',
    'loss',
    'confusion',
    'inspection',
    'code',
    'history',
] as const);
export const CODE_EXPORT_TABS = Object.freeze(['pseudocode', 'numpy', 'tfjs'] as const);
export const DRAWER_SURFACE_IDS = Object.freeze(['presets', 'lessons', 'history'] as const);

export const ADVANCED_TOOLS_TRIGGER_ID = 'forge-advanced-tools-trigger';
export const ADVANCED_TOOLS_REGION_ID = 'forge-advanced-tools-region';

export type WorkspaceView = (typeof WORKSPACE_VIEWS)[number];
export type LayoutVariant = (typeof LAYOUT_VARIANTS)[number];
export type PhaseMode = WorkspaceView;
export type RecipeSectionId = (typeof RECIPE_SECTION_IDS)[number];
export type EvidenceViewId = (typeof EVIDENCE_VIEW_IDS)[number];
export type LeftTabId = RecipeSectionId;
export type RightTabId = EvidenceViewId;
export type CodeExportTab = (typeof CODE_EXPORT_TABS)[number];
export type DrawerSurfaceId = (typeof DRAWER_SURFACE_IDS)[number];
