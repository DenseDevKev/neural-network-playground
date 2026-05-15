import { describe, expect, it } from 'vitest';
import { PRESETS } from '@nn-playground/shared';
import {
    DEFAULT_LESSON_ID,
    getLessonDefinition,
    getLessonPreset,
    LESSON_DEFINITIONS,
    VALID_LESSON_TARGETS,
} from './lessonRegistry.ts';

function hasFunctionValue(value: unknown): boolean {
    if (typeof value === 'function') return true;
    if (!value || typeof value !== 'object') return false;

    return Object.values(value as Record<string, unknown>).some(hasFunctionValue);
}

describe('lesson registry invariants', () => {
    it('exposes the default XOR hidden-layer lesson', () => {
        const lesson = getLessonDefinition();

        expect(DEFAULT_LESSON_ID).toBe('lesson-xor-hidden-layers');
        expect(lesson).toMatchObject({
            id: DEFAULT_LESSON_ID,
            presetId: 'xor-hidden',
            title: 'XOR Needs Hidden Layers',
        });
    });

    it('keeps the expanded lesson library explicit', () => {
        expect(LESSON_DEFINITIONS.map((lesson) => lesson.id)).toEqual([
            'lesson-xor-hidden-layers',
            'lesson-single-neuron-linear-separator',
            'lesson-regression-plane-baseline',
            'lesson-circle-hidden-layer',
            'lesson-feature-engineering-circle',
            'lesson-spiral-depth',
            'lesson-learning-rate-tuning',
            'lesson-regularization-overfitting',
            'lesson-noisy-data-robustness',
        ]);
    });

    it('keeps lesson and step ids stable and unique', () => {
        const lessonIds = new Set<string>();

        for (const lesson of LESSON_DEFINITIONS) {
            expect(lesson.id).toMatch(/^[a-z0-9]+(?:-[a-z0-9]+)*$/);
            expect(lessonIds.has(lesson.id), lesson.id).toBe(false);
            lessonIds.add(lesson.id);

            const stepIds = new Set<string>();
            expect(lesson.steps.length, lesson.id).toBeGreaterThan(0);
            for (const step of lesson.steps) {
                expect(step.id).toMatch(/^[a-z0-9]+(?:-[a-z0-9]+)*$/);
                expect(stepIds.has(step.id), `${lesson.id}:${step.id}`).toBe(false);
                stepIds.add(step.id);
            }
        }
    });

    it('references only existing presets', () => {
        const presetIds = new Set(PRESETS.map((preset) => preset.id));

        for (const lesson of LESSON_DEFINITIONS) {
            expect(presetIds.has(lesson.presetId), lesson.id).toBe(true);
            expect(getLessonPreset(lesson).id).toBe(lesson.presetId);
        }
    });

    it('keeps lesson presets on public scalar dataset contracts', () => {
        for (const lesson of LESSON_DEFINITIONS) {
            const preset = getLessonPreset(lesson);

            expect(preset.config.network?.outputSize, lesson.id).toBe(1);
            expect(preset.config.network?.outputActivation, lesson.id).not.toBe('softmax');
            expect(preset.config.training?.lossType, lesson.id).not.toBe('categoricalCrossEntropy');
        }
    });

    it('keeps every step text complete and targets valid', () => {
        const validTargets = new Set<string>(VALID_LESSON_TARGETS);

        for (const lesson of LESSON_DEFINITIONS) {
            expect(lesson.title.trim(), lesson.id).not.toBe('');
            expect(lesson.summary.trim(), lesson.id).not.toBe('');
            for (const step of lesson.steps) {
                expect(step.title.trim(), `${lesson.id}:${step.id}`).not.toBe('');
                expect(step.body.trim(), `${lesson.id}:${step.id}`).not.toBe('');
                expect(validTargets.has(step.target), `${lesson.id}:${step.id}`).toBe(true);
                if (step.phase) {
                    expect(['build', 'run']).toContain(step.phase);
                }
            }
        }
    });

    it('keeps highlighted lesson targets aligned with the opened build tab', () => {
        const highlightedTabs = new Set(['data', 'features', 'network', 'hyperparams']);

        for (const lesson of LESSON_DEFINITIONS) {
            for (const step of lesson.steps) {
                if (step.tab && highlightedTabs.has(step.tab)) {
                    expect(step.target, `${lesson.id}:${step.id}`).toBe(step.tab);
                }
            }
        }
    });

    it('does not embed functions or config snapshots in lesson content', () => {
        for (const lesson of LESSON_DEFINITIONS) {
            expect(hasFunctionValue(lesson), lesson.id).toBe(false);
            expect('config' in lesson, lesson.id).toBe(false);
            for (const step of lesson.steps) {
                expect('config' in step, `${lesson.id}:${step.id}`).toBe(false);
            }
        }
    });
});
