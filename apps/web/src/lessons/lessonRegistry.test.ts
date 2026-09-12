import { describe, expect, it } from 'vitest';
import { PREPARED_PRESETS, resolveRecipe } from '@nn-playground/shared';
import {
    DEFAULT_LESSON_ID,
    getLessonDefinition,
    getLessonRecipe,
    LESSON_DEFINITIONS,
    type LessonStep,
    VALID_LESSON_TARGETS,
} from './lessonRegistry.ts';

const APPROVED_MULTICLASS_LESSON_ID = 'lesson-three-class-softmax';

const RUN_OBSERVATION_SURFACES = new Map<string, 'boundary' | 'loss'>([
    ['train-in-small-moves', 'boundary'],
    ['train-linear-boundary', 'boundary'],
    ['train-plane-fit', 'loss'],
    ['train-circle-boundary', 'boundary'],
    ['train-feature-model', 'boundary'],
    ['train-spiral-boundary', 'boundary'],
    ['connect-rate-to-loss', 'loss'],
    ['compare-training-and-holdout', 'loss'],
    ['train-and-read-uncertainty', 'boundary'],
    ['train-class-regions', 'boundary'],
]);

function hasFunctionValue(value: unknown): boolean {
    if (typeof value === 'function') return true;
    if (!value || typeof value !== 'object') return false;

    return Object.values(value as Record<string, unknown>).some(hasFunctionValue);
}

describe('lesson registry invariants', () => {
    it('exposes the default lesson with an exact revision-pinned XOR recipe', () => {
        const lesson = getLessonDefinition();

        expect(DEFAULT_LESSON_ID).toBe('lesson-xor-hidden-layers');
        expect(lesson).toMatchObject({
            id: DEFAULT_LESSON_ID,
            recipeRef: { id: 'xor-hidden', revision: 1 },
            title: 'XOR Needs Hidden Layers',
        });
        expect(lesson).not.toHaveProperty('presetId');
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
            APPROVED_MULTICLASS_LESSON_ID,
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

    it('resolves every lesson through its exact catalog id and revision pair', () => {
        for (const lesson of LESSON_DEFINITIONS) {
            const entry = getLessonRecipe(lesson);

            expect(entry).toBe(resolveRecipe(lesson.recipeRef));
            expect(entry.id).toBe(lesson.recipeRef.id);
            expect(entry.revision).toBe(lesson.recipeRef.revision);
            expect(entry.recipe).toBe(entry.prepared.document.recipe);
            expect(entry.prepared.compiled.task.dataset).toBe(entry.recipe.task.dataset);
            expect(entry.prepared.identities.canonicalRecipeKey).not.toBe('');
            expect(entry.prepared.identities.recipeFingerprint).toMatch(/^r2\.1\./);
        }
    });

    it('reports both id and revision when a lesson reference is missing', () => {
        const lesson = getLessonDefinition()!;
        const missingRevision = {
            ...lesson,
            recipeRef: { id: lesson.recipeRef.id, revision: 999 },
        };

        expect(() => getLessonRecipe(missingRevision))
            .toThrow('Missing guided lesson recipe: xor-hidden@999');
    });

    it('exposes exactly one approved multiclass lesson recipe', () => {
        const multiclassLessons = LESSON_DEFINITIONS.filter(
            (lesson) => getLessonRecipe(lesson).recipe.task.kind === 'multiclass-classification',
        );

        expect(multiclassLessons.map((lesson) => lesson.id)).toEqual([APPROVED_MULTICLASS_LESSON_ID]);

        const entry = getLessonRecipe(multiclassLessons[0]);
        expect(entry.recipe.task).toEqual({
            kind: 'multiclass-classification',
            dataset: 'three-class-clusters',
        });
        expect(entry.recipe.objective.dataLoss.kind).toBe('categorical-cross-entropy-with-logits');
        expect(entry.prepared.compiled.task).toMatchObject({
            kind: 'multiclass-classification',
            dataset: 'three-class-clusters',
            outputSize: 3,
            outputActivation: 'softmax',
        });
    });

    it('keeps the approved multiclass lesson on visible multiclass surfaces', () => {
        const lesson = getLessonDefinition(APPROVED_MULTICLASS_LESSON_ID)!;

        expect(lesson.steps.map((step) => step.target)).toEqual([
            'data',
            'network',
            'network',
            'transport',
        ]);
        expect(lesson.steps[2].body).toContain('three outputs, softmax, and categorical cross-entropy');
    });

    it('keeps every other lesson on a scalar task contract', () => {
        for (const lesson of LESSON_DEFINITIONS) {
            const compiled = getLessonRecipe(lesson).prepared.compiled;

            if (lesson.id === APPROVED_MULTICLASS_LESSON_ID) {
                expect(compiled.task.outputSize, lesson.id).toBe(3);
                expect(compiled.task.outputActivation, lesson.id).toBe('softmax');
                continue;
            }

            expect(compiled.task.outputSize, lesson.id).toBe(1);
            expect(compiled.task.outputActivation, lesson.id).not.toBe('softmax');
        }
    });

    it('keeps every step text complete and targets valid', () => {
        const validTargets = new Set<string>(VALID_LESSON_TARGETS);
        const tryThisCopy = new Set<string>();
        const actionLead = /^(Change|Check|Compare|Count|Find|Keep|Look|Move|Notice|Open|Read|Select|Set|Step|Toggle|Use|Watch)\b/;
        const genericFiller = /\b(?:continue when ready|explore the controls|try this step)\b/i;

        for (const lesson of LESSON_DEFINITIONS) {
            expect(lesson.title.trim(), lesson.id).not.toBe('');
            expect(lesson.summary.trim(), lesson.id).not.toBe('');
            for (const step of lesson.steps) {
                const stepName = `${lesson.id}:${step.id}`;
                expect(step.title.trim(), `${lesson.id}:${step.id}`).not.toBe('');
                expect(step.body.trim(), `${lesson.id}:${step.id}`).not.toBe('');
                expect(step.tryThis.trim(), stepName).not.toBe('');
                expect(step.tryThis.length, stepName).toBeLessThanOrEqual(210);
                expect(step.tryThis, stepName).toMatch(actionLead);
                expect(step.tryThis, stepName).not.toMatch(genericFiller);
                expect(tryThisCopy.has(step.tryThis), stepName).toBe(false);
                tryThisCopy.add(step.tryThis);
                expect(validTargets.has(step.target), `${lesson.id}:${step.id}`).toBe(true);
                if ('phase' in step && step.phase) {
                    expect(['build', 'run']).toContain(step.phase);
                }
            }
        }
    });

    it('uses only state-provable closed completion rules while leaving optional steps', () => {
        const completionKinds = new Set<string>();
        let stepsWithoutCompletion = 0;

        for (const lesson of LESSON_DEFINITIONS) {
            for (const step of lesson.steps) {
                if (!('completion' in step) || !step.completion) {
                    stepsWithoutCompletion++;
                    continue;
                }

                completionKinds.add(step.completion.kind);
                if (step.completion.kind === 'training-step-at-least') {
                    expect(Object.keys(step.completion).sort()).toEqual(['kind', 'step']);
                    expect(Number.isSafeInteger(step.completion.step)).toBe(true);
                    expect(step.completion.step).toBeGreaterThan(0);
                } else {
                    expect(Object.keys(step.completion).sort()).toEqual(['kind', 'tab']);
                    expect(['dataset', 'network', 'training']).toContain(step.completion.tab);
                }
            }
        }

        expect(completionKinds).toEqual(new Set(['training-step-at-least', 'setup-tab-is']));
        expect(stepsWithoutCompletion).toBeGreaterThan(0);
    });

    it('keeps observable actions aligned with the controls and state on their target surface', () => {
        const observableTermsByTarget = {
            data: /\b(?:class|cluster|data|noise|point|quadrant|ring|split|spiral)\b/i,
            features: /\b(?:feature|x²|y²)\b/i,
            network: /\b(?:activation|hidden|layer|model|network|neuron|output|topology)\b/i,
            hyperparams: /\b(?:batch|clipping|learning rate|loss|optimizer|regularization|penalty|setting)\b/i,
            transport: /\b(?:boundary|evaluation|loss|play|run|step|train|training)\b/i,
        } as const;

        for (const lesson of LESSON_DEFINITIONS) {
            for (const step of lesson.steps) {
                expect(step.tryThis, `${lesson.id}:${step.id}`)
                    .toMatch(observableTermsByTarget[step.target]);
            }
        }
    });

    it('keeps Build actions on controls visible in their selected module', () => {
        const visibleBuildActions = new Map<string, RegExp>([
            ['read-xor-pattern', /XOR.*samples.*noise.*train\/test/i],
            ['inspect-gaussian-data', /Gaussian.*samples.*noise.*train\/test/i],
            ['switch-to-regression', /Problem.*Regression.*Plane/i],
            ['read-circle-shape', /Circle.*Samples.*Train ratio.*Noise/i],
            ['read-spiral-twist', /Spiral.*Samples.*Train ratio.*Noise/i],
            ['inspect-noise-controls', /Noise.*noise value.*Setup → Dataset/i],
            ['prefer-smooth-boundaries', /hidden layers.*neuron widths.*Hidden activation/i],
            ['read-three-clusters', /Problem.*Multiclass classification.*Three-Class/i],
        ]);
        const unavailablePlotTerms = /\b(?:class colors?|clusters?|continuous target values?|outer ring|point pattern|quadrants?|spiral arms?)\b/i;

        for (const lesson of LESSON_DEFINITIONS) {
            for (const step of lesson.steps) {
                const expected = visibleBuildActions.get(step.id);
                if (!expected) continue;
                expect(step.tryThis, `${lesson.id}:${step.id}`).toMatch(expected);
                expect(step.tryThis, `${lesson.id}:${step.id}`).not.toMatch(unavailablePlotTerms);
            }
        }

        expect(visibleBuildActions.size).toBe(8);
    });

    it('routes every Run observation to its required evidence surface', () => {
        let runSteps = 0;

        for (const lesson of LESSON_DEFINITIONS) {
            for (const step of lesson.steps) {
                if (!('phase' in step) || step.phase !== 'run') continue;
                runSteps++;
                const expectedSurface = RUN_OBSERVATION_SURFACES.get(step.id);
                expect(expectedSurface, `${lesson.id}:${step.id}`).toBeDefined();
                expect('evidenceView' in step ? step.evidenceView : undefined)
                    .toBe(expectedSurface);
                expect(step.tryThis, `${lesson.id}:${step.id}`)
                    .toMatch(expectedSurface === 'boundary' ? /\bBoundary\b/ : /Learning/);
            }
        }

        expect(runSteps).toBe(RUN_OBSERVATION_SURFACES.size);
    });

    it('routes edit-reset-compare actions through Build Hyperparameters and preselects Loss', () => {
        const editSteps: LessonStep[] = [];
        for (const lesson of LESSON_DEFINITIONS) {
            for (const step of lesson.steps) {
                if (['retry-with-smaller-steps', 'simplify-or-regularize'].includes(step.id)) {
                    editSteps.push(step);
                }
            }
        }

        expect(editSteps).toHaveLength(2);
        for (const step of editSteps) {
            expect(step).toMatchObject({
                target: 'hyperparams',
                tab: 'hyperparams',
                phase: 'build',
                evidenceView: 'loss',
            });
            expect(step.tryThis).toMatch(/Change .*Apply changes.*run training.*Learning/);
        }
    });

    it('keeps highlighted lesson targets aligned with the opened build tab', () => {
        const highlightedTabs = new Set(['data', 'features', 'network', 'hyperparams']);

        for (const lesson of LESSON_DEFINITIONS) {
            for (const step of lesson.steps) {
                if ('tab' in step && step.tab && highlightedTabs.has(step.tab)) {
                    expect(step.target, `${lesson.id}:${step.id}`).toBe(step.tab);
                }
            }
        }
    });

    it('stores only content and a catalog reference, never functions or recipe snapshots', () => {
        expect(PREPARED_PRESETS).toHaveLength(7);
        for (const lesson of LESSON_DEFINITIONS) {
            expect(hasFunctionValue(lesson), lesson.id).toBe(false);
            expect(lesson).not.toHaveProperty('config');
            expect(lesson).not.toHaveProperty('presetId');
            expect(Object.keys(lesson.recipeRef).sort()).toEqual(['id', 'revision']);
            for (const step of lesson.steps) {
                expect(step).not.toHaveProperty('config');
            }
        }
    });
});
