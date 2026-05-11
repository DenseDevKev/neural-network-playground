import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import {
    DEFAULT_LESSON_ID,
    LESSON_DEFINITIONS,
    getLessonDefinition,
    getLessonPreset,
    type LessonStep,
    type LessonTarget,
} from '../../lessons/lessonRegistry.ts';

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

export const GuidedLessonPanel = memo(function GuidedLessonPanel({
    onReset,
    onHighlightChange,
}: GuidedLessonPanelProps) {
    const [selectedLessonId, setSelectedLessonId] = useState(DEFAULT_LESSON_ID);
    const [activeStepIndex, setActiveStepIndex] = useState<number | null>(null);
    const [isDrawerOpen, setIsDrawerOpen] = useState(getInitialDrawerOpen);
    const applyPreset = usePlaygroundStore((s) => s.applyPreset);
    const setActiveTabLeft = useLayoutStore((s) => s.setActiveTabLeft);
    const setPhase = useLayoutStore((s) => s.setPhase);
    const setActiveLessonStep = useLayoutStore((s) => s.setActiveLessonStep);
    const clearActiveLessonStep = useLayoutStore((s) => s.clearActiveLessonStep);
    const selectedLesson = useMemo(
        () => getLessonDefinition(selectedLessonId) ?? getLessonDefinition(DEFAULT_LESSON_ID)!,
        [selectedLessonId],
    );
    const lessonPreset = useMemo(() => getLessonPreset(selectedLesson), [selectedLesson]);
    const activeStep = activeStepIndex === null ? null : selectedLesson.steps[activeStepIndex];

    const focusStep = useCallback(
        (step: LessonStep) => {
            if (step.tab) setActiveTabLeft(step.tab);
            if (step.phase) setPhase(step.phase);
            onHighlightChange?.(step.target);
        },
        [onHighlightChange, setActiveTabLeft, setPhase],
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

    const startLesson = () => {
        applyPreset(lessonPreset);
        onReset();
        setActiveStepIndex(0);
        setActiveLessonStep(selectedLesson.id, 0);
        focusStep(selectedLesson.steps[0]);
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
        clearActiveLessonStep();
        onHighlightChange?.(null);
    };

    return (
        <aside
            className={`guided-lesson ${isDrawerOpen ? 'guided-lesson--open' : 'guided-lesson--collapsed'}`}
            aria-label="Guided lesson mode"
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
                    {activeStep ? (
                        <>
                            <div className="guided-lesson__progress" aria-live="polite">
                                Step {activeStepIndex! + 1} of {selectedLesson.steps.length}
                            </div>
                            <h2 className="guided-lesson__step-title">{activeStep.title}</h2>
                            <p className="guided-lesson__body">{activeStep.body}</p>
                            <div className="guided-lesson__actions">
                                <button
                                    className="btn btn--ghost btn--sm"
                                    onClick={() => activeStepIndex! > 0 && goToStep(activeStepIndex! - 1)}
                                    disabled={activeStepIndex === 0}
                                >
                                    Back
                                </button>
                                {activeStepIndex === selectedLesson.steps.length - 1 ? (
                                    <button
                                        className="btn btn--accent btn--sm"
                                        onClick={finishLesson}
                                        aria-label="Finish guided lesson"
                                    >
                                        Finish
                                    </button>
                                ) : (
                                    <button
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
                                className="btn btn--accent btn--sm guided-lesson__start"
                                onClick={startLesson}
                                aria-label="Start guided lesson"
                            >
                                Start
                            </button>
                        </>
                    )}
                </div>
            )}
        </aside>
    );
});
