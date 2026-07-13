import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ExperimentSchemaIssue } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import {
    DEFAULT_LESSON_ID,
    LESSON_DEFINITIONS,
    getLessonDefinition,
    getLessonRecipe,
    type LessonStep,
    type LessonTarget,
} from '../../lessons/lessonRegistry.ts';

export type { LessonTarget } from '../../lessons/lessonRegistry.ts';

interface GuidedLessonPanelProps {
    onReset: () => void;
    onHighlightChange?: (target: LessonTarget | null) => void;
}

const COMPACT_DRAWER_QUERY = '(max-width: 900px)';

function getInitialDrawerOpen(): boolean {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
        return true;
    }
    return !window.matchMedia(COMPACT_DRAWER_QUERY).matches;
}

function formatPreparationIssues(issues: readonly ExperimentSchemaIssue[]): string {
    return issues.map((issue) => `${issue.path}: ${issue.message}`).join('; ');
}

export const GuidedLessonPanel = memo(function GuidedLessonPanel({
    onReset,
    onHighlightChange,
}: GuidedLessonPanelProps) {
    const [selectedLessonId, setSelectedLessonId] = useState(DEFAULT_LESSON_ID);
    const [activeStepIndex, setActiveStepIndex] = useState<number | null>(null);
    const [isDrawerOpen, setIsDrawerOpen] = useState(getInitialDrawerOpen);
    const [isStarting, setIsStarting] = useState(false);
    const [lessonError, setLessonError] = useState<string | null>(null);
    const startInFlight = useRef(false);
    const mounted = useRef(true);
    const applyRecipe = usePlaygroundStore((s) => s.applyRecipe);
    const setActiveRecipeSection = useLayoutStore((s) => s.setActiveRecipeSection);
    const setView = useLayoutStore((s) => s.setView);
    const setActiveLessonStep = useLayoutStore((s) => s.setActiveLessonStep);
    const clearActiveLessonStep = useLayoutStore((s) => s.clearActiveLessonStep);
    const selectedLesson = useMemo(
        () => getLessonDefinition(selectedLessonId) ?? getLessonDefinition(DEFAULT_LESSON_ID)!,
        [selectedLessonId],
    );
    const lessonRecipe = useMemo(() => getLessonRecipe(selectedLesson), [selectedLesson]);
    const activeStep = activeStepIndex === null ? null : selectedLesson.steps[activeStepIndex];
    const lessonStateClass = activeStep ? 'guided-lesson--active' : '';

    useEffect(() => {
        mounted.current = true;
        return () => {
            mounted.current = false;
        };
    }, []);

    const focusStep = useCallback(
        (step: LessonStep) => {
            if (step.tab) setActiveRecipeSection(step.tab);
            if (step.phase) setView(step.phase);
            onHighlightChange?.(step.target);
        },
        [onHighlightChange, setActiveRecipeSection, setView],
    );

    useEffect(() => {
        return () => {
            onHighlightChange?.(null);
            clearActiveLessonStep();
        };
    }, [clearActiveLessonStep, onHighlightChange]);

    useEffect(() => {
        if (typeof window.matchMedia !== 'function') return;
        const media = window.matchMedia(COMPACT_DRAWER_QUERY);
        const syncDrawerDefault = () => setIsDrawerOpen(!media.matches);

        syncDrawerDefault();
        media.addEventListener?.('change', syncDrawerDefault);
        return () => media.removeEventListener?.('change', syncDrawerDefault);
    }, []);

    const startLesson = async () => {
        if (startInFlight.current) return;
        startInFlight.current = true;
        setIsStarting(true);
        setLessonError(null);
        const trainingStore = useTrainingStore.getState();
        trainingStore.beginConfigChange('preset');
        let requestId = usePlaygroundStore.getState().preparation.requestId;

        try {
            const pendingResult = applyRecipe(lessonRecipe);
            requestId = usePlaygroundStore.getState().preparation.requestId;
            const result = await pendingResult;
            const playground = usePlaygroundStore.getState();
            if (playground.preparation.requestId !== requestId) return;

            if (!result.ok) {
                const message = formatPreparationIssues(result.issues)
                    || 'Failed to start guided lesson';
                trainingStore.failConfigChange(message);
                if (mounted.current) setLessonError(message);
                return;
            }

            if (playground.access.status !== 'ready'
                || playground.access.prepared !== result.value
                || !mounted.current) return;
            onReset();
            if (!mounted.current) return;
            setActiveStepIndex(0);
            setActiveLessonStep(selectedLesson.id, 0);
            focusStep(selectedLesson.steps[0]);
        } catch (error) {
            if (usePlaygroundStore.getState().preparation.requestId === requestId) {
                const message = error instanceof Error
                    ? error.message
                    : 'Failed to start guided lesson';
                trainingStore.failConfigChange(message);
                if (mounted.current) setLessonError(message);
            }
        } finally {
            startInFlight.current = false;
            if (mounted.current) setIsStarting(false);
        }
    };

    const goToStep = (nextIndex: number) => {
        setActiveStepIndex(nextIndex);
        setActiveLessonStep(selectedLesson.id, nextIndex);
        focusStep(selectedLesson.steps[nextIndex]);
    };

    const finishLesson = () => {
        setActiveStepIndex(null);
        clearActiveLessonStep();
        onHighlightChange?.(null);
    };

    const selectLesson = (lessonId: string) => {
        setSelectedLessonId(lessonId);
        setActiveStepIndex(null);
        setLessonError(null);
        clearActiveLessonStep();
        onHighlightChange?.(null);
    };

    return (
        <aside
            className={`guided-lesson ${lessonStateClass} ${isDrawerOpen ? 'guided-lesson--open' : 'guided-lesson--collapsed'}`}
            aria-label="Guided lesson mode"
            aria-busy={isStarting}
        >
            <div className="guided-lesson__header">
                <div className="guided-lesson__identity">
                    <div className="guided-lesson__eyebrow">Guided lesson</div>
                    <div className="guided-lesson__title">{selectedLesson.title}</div>
                </div>
                {activeStep && (
                    <div className="guided-lesson__step-chip" aria-live="polite">
                        {activeStepIndex! + 1}/{selectedLesson.steps.length}
                    </div>
                )}
                <button
                    type="button"
                    className="guided-lesson__toggle"
                    aria-label={isDrawerOpen ? 'Collapse guided lesson drawer' : 'Expand guided lesson drawer'}
                    aria-expanded={isDrawerOpen}
                    onClick={() => setIsDrawerOpen((open) => !open)}
                >
                    {isDrawerOpen ? '▾' : '▴'}
                </button>
            </div>

            {isDrawerOpen && (
                <div className="guided-lesson__content">
                    {lessonError && (
                        <div className="config-feedback config-feedback--error" role="alert">
                            {lessonError}
                        </div>
                    )}
                    {activeStep ? (
                        <>
                            <div className="guided-lesson__progress" aria-live="polite">
                                Step {activeStepIndex! + 1} of {selectedLesson.steps.length}
                            </div>
                            <h2 className="guided-lesson__step-title">{activeStep.title}</h2>
                            <p className="guided-lesson__body">{activeStep.body}</p>
                            <div className="guided-lesson__actions">
                                <button
                                    type="button"
                                    className="btn btn--ghost btn--sm"
                                    onClick={() => activeStepIndex! > 0 && goToStep(activeStepIndex! - 1)}
                                    disabled={activeStepIndex === 0}
                                >
                                    Back
                                </button>
                                {activeStepIndex === selectedLesson.steps.length - 1 ? (
                                    <button
                                        type="button"
                                        className="btn btn--accent btn--sm"
                                        onClick={finishLesson}
                                        aria-label="Finish guided lesson"
                                    >
                                        Finish
                                    </button>
                                ) : (
                                    <button
                                        type="button"
                                        className="btn btn--accent btn--sm"
                                        onClick={() => goToStep(activeStepIndex! + 1)}
                                        aria-label="Next lesson step"
                                    >
                                        Next
                                    </button>
                                )}
                            </div>
                        </>
                    ) : (
                        <>
                            <label className="guided-lesson__selector">
                                <span className="guided-lesson__selector-label">Lesson</span>
                                <select
                                    className="select guided-lesson__select"
                                    value={selectedLesson.id}
                                    onChange={(event) => selectLesson(event.target.value)}
                                    aria-label="Guided lesson"
                                    disabled={isStarting}
                                >
                                    {LESSON_DEFINITIONS.map((lesson) => (
                                        <option key={lesson.id} value={lesson.id}>
                                            {lesson.title}
                                        </option>
                                    ))}
                                </select>
                            </label>
                            <p className="guided-lesson__body">{selectedLesson.summary}</p>
                            {selectedLesson.estimatedMinutes && (
                                <div className="guided-lesson__meta">
                                    About {selectedLesson.estimatedMinutes} min
                                </div>
                            )}
                            <button
                                type="button"
                                className="btn btn--accent btn--sm guided-lesson__start"
                                onClick={startLesson}
                                aria-label="Start guided lesson"
                                disabled={isStarting}
                            >
                                {isStarting ? 'Starting...' : 'Start'}
                            </button>
                        </>
                    )}
                </div>
            )}
        </aside>
    );
});
