import type { RecipeRef } from '@nn-playground/shared';
import type { EvidenceViewId, LeftTabId, PhaseMode } from '../store/useLayoutStore.ts';

export type LessonTarget = 'data' | 'features' | 'network' | 'hyperparams' | 'transport';

export type LessonCompletionRule =
    | { kind: 'training-step-at-least'; step: number }
    | { kind: 'view-is'; view: 'build' | 'run' };

export interface LessonStep {
    id: string;
    title: string;
    body: string;
    tryThis: string;
    target: LessonTarget;
    tab?: LeftTabId;
    phase?: PhaseMode;
    evidenceView?: EvidenceViewId;
    expectedOutcome?: string;
    explanationRuleIds?: readonly string[];
    completion?: LessonCompletionRule;
}

export interface LessonDefinition {
    id: string;
    title: string;
    summary: string;
    recipeRef: RecipeRef;
    estimatedMinutes?: number;
    steps: readonly LessonStep[];
}
