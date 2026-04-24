import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { PRESETS, type Preset } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useLayoutStore, type LeftTabId, type PhaseMode } from '../../store/useLayoutStore.ts';

export type LessonTarget = 'data' | 'network' | 'hyperparams' | 'transport';

interface GuidedLessonPanelProps {
    onReset: () => void;
    onHighlightChange?: (target: LessonTarget | null) => void;
}

interface LessonStep {
    title: string;
    target: LessonTarget;
    tab?: LeftTabId;
    phase?: PhaseMode;
    body: string;
}

const XOR_LESSON_PRESET_ID = 'xor-hidden';

const XOR_LESSON_STEPS: LessonStep[] = [
    {
        title: 'Read the XOR pattern',
        target: 'data',
        tab: 'data',
        phase: 'build',
        body: 'The XOR preset alternates labels by quadrant, so no single straight line can separate every point.',
    },
    {
        title: 'Give the model capacity',
        target: 'network',
        tab: 'network',
        phase: 'build',
        body: 'Two hidden layers let the network combine simple bends into the corners needed for XOR.',
    },
    {
        title: 'Use steady updates',
        target: 'hyperparams',
        tab: 'hyperparams',
        phase: 'build',
        body: 'A moderate learning rate and small batches make the loss react without bouncing wildly.',
    },
    {
        title: 'Train in small moves',
        target: 'transport',
        phase: 'run',
        body: 'Step or play from the transport controls and watch the boundary change as weights update.',
    },
];

function findLessonPreset(): Preset {
    const preset = PRESETS.find((item) => item.id === XOR_LESSON_PRESET_ID);
    if (!preset) {
        throw new Error(`Missing guided lesson preset: ${XOR_LESSON_PRESET_ID}`);
    }
    return preset;
}

export const GuidedLessonPanel = memo(function GuidedLessonPanel({
    onReset,
    onHighlightChange,
}: GuidedLessonPanelProps) {
    const [activeStepIndex, setActiveStepIndex] = useState<number | null>(null);
    const applyPreset = usePlaygroundStore((s) => s.applyPreset);
    const setActiveTabLeft = useLayoutStore((s) => s.setActiveTabLeft);
    const setPhase = useLayoutStore((s) => s.setPhase);
    const lessonPreset = useMemo(findLessonPreset, []);
    const activeStep = activeStepIndex === null ? null : XOR_LESSON_STEPS[activeStepIndex];

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
        focusStep(XOR_LESSON_STEPS[0]);
    };

    const goToStep = (nextIndex: number) => {
        setActiveStepIndex(nextIndex);
        focusStep(XOR_LESSON_STEPS[nextIndex]);
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
                        Step {activeStepIndex! + 1} of {XOR_LESSON_STEPS.length}
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
                        {activeStepIndex === XOR_LESSON_STEPS.length - 1 ? (
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
                    <p className="guided-lesson__body">
                        Load a preset-backed walkthrough that shows why XOR needs hidden layers.
                    </p>
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
