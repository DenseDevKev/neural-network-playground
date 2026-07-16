import { describe, expect, expectTypeOf, it } from 'vitest';
import {
    CODE_EXPORT_TABS,
    EVIDENCE_VIEW_IDS,
    LAYOUT_VARIANTS,
    RECIPE_SECTION_IDS,
    WORKSPACE_VIEWS,
    type CodeExportTab,
    type EvidenceViewId,
    type LayoutVariant,
    type LeftTabId,
    type PhaseMode,
    type RecipeSectionId,
    type RightTabId,
    type WorkspaceView,
} from './shellTypes.ts';

describe('shell types', () => {
    it('owns the immutable runtime ID lists and their inferred unions', () => {
        expect(WORKSPACE_VIEWS).toEqual(['build', 'run']);
        expect(RECIPE_SECTION_IDS).toEqual([
            'presets',
            'data',
            'features',
            'network',
            'hyperparams',
            'config',
        ]);
        expect(EVIDENCE_VIEW_IDS).toEqual([
            'boundary',
            'loss',
            'confusion',
            'inspection',
            'code',
            'history',
        ]);
        expect(CODE_EXPORT_TABS).toEqual(['pseudocode', 'numpy', 'tfjs']);
        expect(LAYOUT_VARIANTS).toEqual(['dock', 'focus', 'grid', 'split']);
        expect(Object.isFrozen(WORKSPACE_VIEWS)).toBe(true);
        expect(Object.isFrozen(RECIPE_SECTION_IDS)).toBe(true);
        expect(Object.isFrozen(EVIDENCE_VIEW_IDS)).toBe(true);
        expect(Object.isFrozen(CODE_EXPORT_TABS)).toBe(true);
        expect(Object.isFrozen(LAYOUT_VARIANTS)).toBe(true);

        expectTypeOf<(typeof WORKSPACE_VIEWS)[number]>().toEqualTypeOf<WorkspaceView>();
        expectTypeOf<(typeof RECIPE_SECTION_IDS)[number]>().toEqualTypeOf<RecipeSectionId>();
        expectTypeOf<(typeof EVIDENCE_VIEW_IDS)[number]>().toEqualTypeOf<EvidenceViewId>();
        expectTypeOf<(typeof CODE_EXPORT_TABS)[number]>().toEqualTypeOf<CodeExportTab>();
        expectTypeOf<(typeof LAYOUT_VARIANTS)[number]>().toEqualTypeOf<LayoutVariant>();
        expectTypeOf<PhaseMode>().toEqualTypeOf<WorkspaceView>();
        expectTypeOf<LeftTabId>().toEqualTypeOf<RecipeSectionId>();
        expectTypeOf<RightTabId>().toEqualTypeOf<EvidenceViewId>();
    });
});
