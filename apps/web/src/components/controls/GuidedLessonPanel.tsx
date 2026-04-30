import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import {
    DEFAULT_LESSON_ID,
    getLessonDefinition,
    getLessonPreset,
    LESSON_DEFINITIONS,
    type LessonDefinition,
    type LessonStep,
    type LessonTarget,
} from '../../lessons/lessonRegistry.ts';

export type { LessonTarget } from '../../lessons/lessonRegistry.ts';

interface GuidedLessonPanelProps {
    onReset: () => void;
    onHighlightChange?: (target: LessonTarget | null) => void;
}

function getRequiredLesson(id: string): LessonDefinition {
    const lesson = getLessonDefinition(id);
    if (!lesson) {
        throw new Error(`Missing guided lesson: ${id}`);
    }
    return lesson;
}

export const GuidedLessonPanel = memo(function GuidedLessonPanel({
    onReset,
    onHighlightChange,
}: GuidedLessonPanelProps) {
    const [selectedLessonId, setSelectedLessonId] = useState(DEFAULT_LESSON_ID);
    const [activeStepIndex, setActiveStepIndex] = useState<number | null>(null);
    const applyPreset = usePlaygroundStore((s) => s.applyPreset);
    const setActiveTabLeft = useLayoutStore((s) => s.setActiveTabLeft);
    const setPhase = useLayoutStore((s) => s.setPhase);
    const activeLesson = useMemo(() => getRequiredLesson(selectedLessonId), [selectedLessonId]);
    const lessonPreset = useMemo(() => getLessonPreset(activeLesson), [activeLesson]);
    const activeStep = activeStepIndex === null ? null : activeLesson.steps[activeStepIndex];

    const focusStep = useCallback(
        (step: LessonStep) => {
            if (step.tab) setActiveTabLeft(step.tab);
            if (step.phase) setPhase(step.phase);
            onHighlightChange?.(step.target);
        },
        [onHighlightChange, setActiveTabLeft, setPhase],
    );

    useEffect(() => {
        return () => onHighlightChange?.(null);
    }, [onHighlightChange]);

    const startLesson = () => {
        applyPreset(lessonPreset);
        onReset();
        setActiveStepIndex(0);
        focusStep(activeLesson.steps[0]);
    };

    const goToStep = (nextIndex: number) => {
        setActiveStepIndex(nextIndex);
        focusStep(activeLesson.steps[nextIndex]);
    };

    const finishLesson = () => {
        setActiveStepIndex(null);
        onHighlightChange?.(null);
    };

    return (
        <aside className="guided-lesson" aria-label="Guided lesson mode">
            <div className="guided-lesson__eyebrow">Guided lesson</div>
            <div className="guided-lesson__title">{lessonPreset.title}</div>

            {activeStep ? (
                <>
                    <div className="guided-lesson__progress" aria-live="polite">
                        Step {activeStepIndex! + 1} of {activeLesson.steps.length}
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
                        {activeStepIndex === activeLesson.steps.length - 1 ? (
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
                    <p className="guided-lesson__body">{activeLesson.summary}</p>
                    {LESSON_DEFINITIONS.length > 1 && (
                        <div
                            role="group"
                            aria-label="Available lessons"
                            style={{
                                display: 'grid',
                                gap: 6,
                                marginTop: 12,
                            }}
                        >
                            {LESSON_DEFINITIONS.map((lesson) => (
                                <button
                                    key={lesson.id}
                                    type="button"
                                    className="btn btn--ghost btn--sm"
                                    style={{
                                        justifyContent: 'flex-start',
                                        minWidth: 0,
                                        whiteSpace: 'normal',
                                        textAlign: 'left',
                                    }}
                                    aria-pressed={lesson.id === selectedLessonId}
                                    aria-label={`Select lesson: ${lesson.title}`}
                                    onClick={() => setSelectedLessonId(lesson.id)}
                                >
                                    {lesson.title}
                                </button>
                            ))}
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
        </aside>
    );
});
