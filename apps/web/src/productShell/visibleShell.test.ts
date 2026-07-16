import { describe, expect, it } from 'vitest';
import {
    BUILD_FALLBACK,
    EVIDENCE_FALLBACK,
    getVisibleBuildModules,
    getVisibleEvidenceViews,
    isRecipeSectionVisible,
    isEvidenceViewVisible,
    resolveEvidenceAlias,
    resolveVisibleEvidenceView,
    resolveVisibleRecipeSection,
} from './visibleShell.ts';

describe('visible shell model', () => {
    it.each([
        ['beginner', false, ['data', 'network']],
        ['beginner', true, ['data', 'network', 'features', 'hyperparams', 'config']],
        ['explore', false, ['data', 'network', 'features', 'hyperparams']],
        ['explore', true, ['data', 'network', 'features', 'hyperparams', 'config']],
        ['lab', false, ['data', 'network', 'features', 'hyperparams']],
        ['lab', true, ['data', 'network', 'features', 'hyperparams', 'config']],
    ] as const)('resolves %s Build visibility when Advanced Tools is %s', (mode, open, expected) => {
        expect(getVisibleBuildModules(mode, open)).toEqual(expected);
    });

    it.each([
        ['beginner', false, ['boundary', 'loss']],
        ['beginner', true, ['boundary', 'loss', 'confusion', 'inspection', 'code']],
        ['explore', false, ['boundary', 'loss', 'confusion']],
        ['explore', true, ['boundary', 'loss', 'confusion', 'inspection', 'code']],
        ['lab', false, ['boundary', 'loss', 'confusion']],
        ['lab', true, ['boundary', 'loss', 'confusion', 'inspection', 'code']],
    ] as const)('resolves %s Run visibility when Advanced Tools is %s', (mode, open, expected) => {
        expect(getVisibleEvidenceViews(mode, open)).toEqual(expected);
    });

    it('maps legacy History to Boundary before visibility checks', () => {
        expect(resolveEvidenceAlias('history')).toBe('boundary');
        expect(resolveEvidenceAlias('loss')).toBe('loss');
        expect(isEvidenceViewVisible('beginner', false, 'history')).toBe(true);
        expect(resolveVisibleEvidenceView('beginner', false, 'history')).toBe('boundary');
    });

    it('detects hidden targets without treating Presets as a profile module', () => {
        expect(isRecipeSectionVisible('beginner', false, 'presets')).toBe(true);
        expect(isRecipeSectionVisible('beginner', false, 'features')).toBe(false);
        expect(isRecipeSectionVisible('beginner', true, 'features')).toBe(true);
        expect(isEvidenceViewVisible('beginner', false, 'confusion')).toBe(false);
        expect(isEvidenceViewVisible('beginner', true, 'confusion')).toBe(true);
    });

    it('uses Data and Boundary as deterministic fallbacks for hidden targets', () => {
        expect(BUILD_FALLBACK).toBe('data');
        expect(EVIDENCE_FALLBACK).toBe('boundary');
        expect(resolveVisibleRecipeSection('beginner', false, 'hyperparams')).toBe('data');
        expect(resolveVisibleRecipeSection('beginner', false, 'network')).toBe('network');
        expect(resolveVisibleEvidenceView('beginner', false, 'inspection')).toBe('boundary');
        expect(resolveVisibleEvidenceView('beginner', false, 'loss')).toBe('loss');
    });

    it('returns frozen canonical visibility lists instead of rebuilding display state', () => {
        const first = getVisibleEvidenceViews('explore', false);
        const second = getVisibleEvidenceViews('explore', false);
        expect(second).toBe(first);
        expect(Object.isFrozen(first)).toBe(true);

        const openBuild = getVisibleBuildModules('beginner', true);
        expect(Object.isFrozen(openBuild)).toBe(true);
    });
});
