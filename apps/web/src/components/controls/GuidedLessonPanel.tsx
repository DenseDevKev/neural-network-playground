import { memo, useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import type { ExperimentSchemaIssue } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import {
    DEFAULT_LESSON_ID,
    LESSON_DEFINITIONS,
    getLessonDefinition,
    getLessonRecipe,
    type LessonCompletionRule,
    type LessonStep,
    type LessonTarget,
} from '../../lessons/lessonRegistry.ts';
import '../../lessons/lessons.css';
import { STATE_EFFECTS } from '../../copy/stateEffects.ts';

export type { LessonTarget } from '../../lessons/lessonRegistry.ts';

interface GuidedLessonPanelProps {
    onReset: () => void;
    onNavigate?: (action: () => void) => void;
    onHighlightChange?: (target: LessonTarget | null) => void;
}

const COMPACT_DRAWER_QUERY = '(max-width: 900px)';

interface LessonModelIdentity {
    generationId: number;
    revision: number;
    step: number;
}

function snapshotModelIdentity(model: LessonModelIdentity | null): LessonModelIdentity | null {
    return model && {
        generationId: model.generationId,
        revision: model.revision,
        step: model.step,
    };
}

function hasEvidenceAfterLessonStart(
    current: LessonModelIdentity | null,
    activation: LessonModelIdentity | null,
): boolean {
    if (!current) return false;
    if (!activation) return true;
    if (current.generationId !== activation.generationId) {
        return current.generationId > activation.generationId;
    }
    return false; // The pre-reset generation is never evidence for this lesson.
}

function isLessonCompletionSatisfied(
    rule: LessonCompletionRule,
    state: {
        workspaceTab: string;
        setupTab: string;
        destination: string;
        currentModel: LessonModelIdentity | null;
        activationModel: LessonModelIdentity | null;
    },
): boolean {
    switch (rule.kind) {
        case 'training-step-at-least': {
            if (!state.currentModel) return false;
            return hasEvidenceAfterLessonStart(state.currentModel, state.activationModel)
                && state.currentModel.step >= rule.step;
        }
        case 'setup-tab-is':
            return state.destination === 'playground' && state.workspaceTab === 'setup' && state.setupTab === rule.tab;
    }
}

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
    onNavigate = (action) => action(),
    onHighlightChange,
}: GuidedLessonPanelProps) {
    const [selectedLessonId, setSelectedLessonId] = useState(DEFAULT_LESSON_ID);
    const [activeLessonId, setActiveLessonId] = useState<string | null>(null);
    const [activeStepIndex, setActiveStepIndex] = useState<number | null>(null);
    const [isDrawerOpen, setIsDrawerOpen] = useState(getInitialDrawerOpen);
    const [isStarting, setIsStarting] = useState(false);
    const [lessonError, setLessonError] = useState<string | null>(null);
    const lessonEffectsId = `${useId()}-lesson-start-effects`;
    const startInFlight = useRef(false);
    const lastStartLesson = useRef(selectedLessonId);
    const mounted = useRef(true);
    const activationModel = useRef<LessonModelIdentity | null>(null);
    const applyRecipe = usePlaygroundStore((s) => s.applyRecipe);
    const currentModel = useTrainingStore((s) => s.latestLiveSignal?.model ?? null);
    const destination = useLayoutStore((s) => s.destination);
    const workspaceTab = useLayoutStore((s) => s.workspaceTab);
    const setupTab = useLayoutStore((s) => s.setupTab);
    const setActiveLessonStep = useLayoutStore((s) => s.setActiveLessonStep);
    const clearActiveLessonStep = useLayoutStore((s) => s.clearActiveLessonStep);
    const selectedLesson = useMemo(
        () => getLessonDefinition(selectedLessonId) ?? getLessonDefinition(DEFAULT_LESSON_ID)!,
        [selectedLessonId],
    );
    const lessonRecipe = useMemo(() => getLessonRecipe(selectedLesson), [selectedLesson]);
    const activeLesson = activeLessonId ? getLessonDefinition(activeLessonId) : null;
    const activeStep = activeStepIndex === null ? null : activeLesson?.steps[activeStepIndex];
    const showLibrary = destination === 'lessons' || !activeStep;
    const completionSatisfied = activeStep?.completion
        ? isLessonCompletionSatisfied(activeStep.completion, {
            destination, workspaceTab, setupTab,
            currentModel,
            activationModel: activationModel.current,
        })
        : false;

    useEffect(() => {
        mounted.current = true;
        return () => {
            mounted.current = false;
        };
    }, []);

    const focusStep = useCallback(
        (step: LessonStep) => {
            const layout = useLayoutStore.getState();
            if (step.evidenceView) layout.setResultsTab(step.evidenceView === 'loss' ? 'learning' : step.evidenceView === 'confusion' ? 'errors' : 'boundary');
            if (step.tab) layout.openSetup(step.tab === 'hyperparams' ? 'training' : step.tab === 'data' ? 'dataset' : 'network');
            else layout.navigate('playground', step.evidenceView === 'inspection' ? 'inspect' : 'results');
            onHighlightChange?.(step.target);
        },
        [onHighlightChange],
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

    const startLesson = async (lesson = selectedLesson) => {
        if (startInFlight.current) return;
        startInFlight.current = true;
        lastStartLesson.current = lesson.id;
        setIsStarting(true);
        setLessonError(null);
        const trainingStore = useTrainingStore.getState();
        trainingStore.beginConfigChange('preset');
        let requestId = usePlaygroundStore.getState().preparation.requestId;

        try {
            const pendingResult = applyRecipe(getLessonRecipe(lesson));
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
            activationModel.current = snapshotModelIdentity(
                useTrainingStore.getState().latestLiveSignal?.model ?? null,
            );
            onReset();
            if (!mounted.current) return;
            setActiveLessonId(lesson.id);
            setActiveStepIndex(0);
            setActiveLessonStep(lesson.id, 0);
            focusStep(lesson.steps[0]);
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

    const goToStep = (nextIndex: number) => onNavigate(() => {
        if (!activeLesson || startInFlight.current) return;
        setActiveStepIndex(nextIndex);
        setActiveLessonStep(activeLesson.id, nextIndex);
        focusStep(activeLesson.steps[nextIndex]);
    });

    const finishLesson = () => onNavigate(() => {
        if (startInFlight.current) return;
        setActiveLessonId(null);
        activationModel.current = null;
        setActiveStepIndex(null);
        clearActiveLessonStep();
        onHighlightChange?.(null);
        useLayoutStore.getState().navigate('lessons');
    });

    const selectLesson = (lessonId: string) => {
        setSelectedLessonId(lessonId);
        setLessonError(null);
    };

    return (
        <aside className={`guided-lesson atelier-lessons ${activeStep ? 'guided-lesson--active' : ''} ${showLibrary ? 'atelier-lessons--library' : 'atelier-lessons--panel'}`} aria-label="Guided lesson mode" aria-busy={isStarting}>
            <header className="lesson-heading">
                <div>{!showLibrary && <><small>Active lesson</small><h2>{activeLesson!.title}</h2></>}</div>
                <button type="button" aria-label={isDrawerOpen ? 'Collapse guided lesson drawer' : 'Expand guided lesson drawer'} aria-expanded={isDrawerOpen} onClick={() => setIsDrawerOpen(!isDrawerOpen)}>{isDrawerOpen ? '−' : '+'}</button>
            </header>
            {isDrawerOpen && <>
                {isStarting && <p role="status">Preparing lesson. Navigation resumes when preparation finishes.</p>}
                {lessonError && <div role="alert">{lessonError}<button type="button" disabled={isStarting} onClick={() => onNavigate(() => { void startLesson(getLessonDefinition(lastStartLesson.current)!); })}>Retry lesson</button></div>}
                {showLibrary ? <div className="lesson-library">
                    <div>
                        <p>Small experiments that make neural networks tangible.</p>
                        <label className="lesson-mobile-select">Lesson<select aria-label="Guided lesson" value={selectedLessonId} disabled={isStarting} onChange={(event) => selectLesson(event.target.value)}>{LESSON_DEFINITIONS.map((lesson) => <option key={lesson.id} value={lesson.id}>{lesson.title}</option>)}</select></label>
                        <div className="lesson-list">{LESSON_DEFINITIONS.map((lesson, index) => <button key={lesson.id} type="button" aria-pressed={lesson.id === selectedLessonId} disabled={isStarting} onClick={() => selectLesson(lesson.id)}><span className="lesson-number">{String(index + 1).padStart(2, '0')}</span><span><strong>{lesson.title}</strong><small>{lesson.summary}</small></span><span aria-hidden="true">↗</span></button>)}</div>
                    </div>
                    <section className="lesson-detail" aria-label="Selected lesson details">
                        <small>About {selectedLesson.estimatedMinutes} min · {selectedLesson.steps.length} steps</small>
                        <h3>{selectedLesson.title}</h3><p>{selectedLesson.summary}</p>
                        <p className="lesson-recipe">{lessonRecipe.recipe.task.dataset} · {lessonRecipe.recipe.model.hiddenLayers.length ? `Hidden layers: ${lessonRecipe.recipe.model.hiddenLayers.join(' → ')}` : 'No hidden layers'} · {lessonRecipe.recipe.model.hiddenActivation}<br />Dataset seed {lessonRecipe.recipe.data.seed} · Model seed {lessonRecipe.recipe.model.seed}</p>
                        <ol>{selectedLesson.steps.map((step) => <li key={step.id}><strong>{step.title}</strong><p>{step.body}</p></li>)}</ol>
                        <p id={lessonEffectsId} className="guided-lesson__consequence">{STATE_EFFECTS['lesson-start']}</p>
                        <button type="button" className="atelier-primary" aria-label="Start lesson and reset" aria-describedby={lessonEffectsId} disabled={isStarting} onClick={() => onNavigate(() => { void startLesson(); })}>{isStarting ? 'Starting…' : activeLesson ? 'Replace active lesson and reset' : 'Start lesson and reset'}</button>
                        {activeLesson && <button type="button" disabled={isStarting} onClick={() => goToStep(activeStepIndex!)}>Resume {activeLesson.title}</button>}
                    </section>
                </div> : <div className="lesson-active">
                    <div className="lesson-actions"><button type="button" disabled={isStarting} onClick={() => onNavigate(() => { if (!startInFlight.current) useLayoutStore.getState().navigate('lessons'); })}>All lessons</button><button type="button" disabled={isStarting} onClick={finishLesson}>Exit lesson</button></div>
                    <p aria-live="polite">Step {activeStepIndex! + 1} of {activeLesson!.steps.length}</p>
                    <progress aria-label="Lesson progress" value={activeStepIndex! + 1} max={activeLesson!.steps.length} />
                    <h3>{activeStep!.title}</h3><p>{activeStep!.body}</p>
                    <h4>Try this</h4><p>{activeStep!.tryThis}</p>
                    {activeStep!.tab && <p className="lesson-note">Edit in Setup, then select Apply changes before leaving. Dataset seed changes the samples; Model seed changes initial weights.</p>}
                    {completionSatisfied && <p role="status">Done</p>}
                    <button type="button" disabled={isStarting} onClick={() => goToStep(activeStepIndex!)}>Show me →</button>
                    <div className="lesson-actions"><button type="button" disabled={isStarting || activeStepIndex === 0} onClick={() => goToStep(activeStepIndex! - 1)}>Previous</button>{activeStepIndex === activeLesson!.steps.length - 1 ? <button type="button" className="atelier-primary" aria-label="Finish guided lesson" disabled={isStarting} onClick={finishLesson}>Finish</button> : <button type="button" className="atelier-primary" aria-label="Next lesson step" disabled={isStarting} onClick={() => goToStep(activeStepIndex! + 1)}>Continue</button>}</div>
                    <details><summary>Restart lesson</summary><p id={lessonEffectsId}>{STATE_EFFECTS['lesson-start']}</p><button type="button" disabled={isStarting} aria-describedby={lessonEffectsId} onClick={() => onNavigate(() => { void startLesson(activeLesson!); })}>Restart lesson and reset</button></details>
                </div>}
            </>}
        </aside>
    );
});
