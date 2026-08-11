interface FirstVisitLessonCueProps {
    historyReady: boolean;
    lessonCueDismissed: boolean;
    hasSavedRuns: boolean;
    hasStartedLesson: boolean;
    hasActiveLesson: boolean;
    onOpenLessons: () => void;
    onDismiss: () => void;
}

export function FirstVisitLessonCue({
    historyReady,
    lessonCueDismissed,
    hasSavedRuns,
    hasStartedLesson,
    hasActiveLesson,
    onOpenLessons,
    onDismiss,
}: FirstVisitLessonCueProps) {
    if (
        !historyReady
        || lessonCueDismissed
        || hasSavedRuns
        || hasStartedLesson
        || hasActiveLesson
    ) {
        return null;
    }

    return (
        <aside className="forge-instrument-module" role="region" aria-label="Getting started">
            <div className="forge-instrument-module__head">
                <span className="forge-instrument-module__grip" aria-hidden />
                <span className="forge-instrument-module__title">New here?</span>
            </div>
            <div className="forge-instrument-module__body">
                <p>See how a small network learns in one guided lesson.</p>
                <div className="guided-lesson__actions">
                    <button
                        type="button"
                        className="btn btn--accent btn--sm"
                        onClick={onOpenLessons}
                    >
                        Start a 3-minute lesson
                    </button>
                    <button
                        type="button"
                        className="btn btn--ghost btn--sm"
                        onClick={onDismiss}
                        aria-label="Dismiss lesson suggestion"
                    >
                        Not now
                    </button>
                </div>
            </div>
        </aside>
    );
}
