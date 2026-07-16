import { describe, expect, it } from 'vitest';
import {
    ADVANCED_BUILD_MODULES,
    ADVANCED_EVIDENCE_VIEWS,
    AUDIENCE_MODES,
    AUDIENCE_PROFILES,
    getAudienceProfile,
    isAudienceMode,
} from './audienceProfiles.ts';

describe('audience profiles', () => {
    it('defines the exact Beginner, Explore, and Lab capabilities', () => {
        expect(AUDIENCE_MODES).toEqual(['beginner', 'explore', 'lab']);
        expect(AUDIENCE_PROFILES).toEqual({
            beginner: {
                label: 'Beginner',
                description: 'Keeps the core data, network, boundary, and loss tools visible with more guidance.',
                coreBuildModules: ['data', 'network'],
                coreEvidenceViews: ['boundary', 'loss'],
                advancedDefaultOpen: false,
                guidanceLevel: 'high',
            },
            explore: {
                label: 'Explore',
                description: 'Adds feature, hyperparameter, and confusion tools for guided experimentation.',
                coreBuildModules: ['data', 'network', 'features', 'hyperparams'],
                coreEvidenceViews: ['boundary', 'loss', 'confusion'],
                advancedDefaultOpen: false,
                guidanceLevel: 'standard',
            },
            lab: {
                label: 'Lab',
                description: 'Opens the full workspace with compact guidance for detailed analysis.',
                coreBuildModules: ['data', 'network', 'features', 'hyperparams'],
                coreEvidenceViews: ['boundary', 'loss', 'confusion'],
                advancedDefaultOpen: true,
                guidanceLevel: 'compact',
            },
        });
    });

    it('defines the complete Advanced Tools union in stable display order', () => {
        expect(ADVANCED_BUILD_MODULES).toEqual(['features', 'hyperparams', 'config']);
        expect(ADVANCED_EVIDENCE_VIEWS).toEqual(['confusion', 'inspection', 'code']);
    });

    it('returns the canonical immutable profile objects', () => {
        for (const mode of AUDIENCE_MODES) {
            const profile = getAudienceProfile(mode);
            expect(profile).toBe(AUDIENCE_PROFILES[mode]);
            expect(Object.isFrozen(profile)).toBe(true);
            expect(Object.isFrozen(profile.coreBuildModules)).toBe(true);
            expect(Object.isFrozen(profile.coreEvidenceViews)).toBe(true);
        }
        expect(Object.isFrozen(AUDIENCE_MODES)).toBe(true);
        expect(Object.isFrozen(AUDIENCE_PROFILES)).toBe(true);
        expect(Object.isFrozen(ADVANCED_BUILD_MODULES)).toBe(true);
        expect(Object.isFrozen(ADVANCED_EVIDENCE_VIEWS)).toBe(true);
    });

    it('accepts only supported audience mode values', () => {
        expect(isAudienceMode('beginner')).toBe(true);
        expect(isAudienceMode('explore')).toBe(true);
        expect(isAudienceMode('lab')).toBe(true);
        expect(isAudienceMode('expert')).toBe(false);
        expect(isAudienceMode(null)).toBe(false);
    });
});
