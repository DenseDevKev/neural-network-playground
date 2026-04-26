import type { Preset } from '@nn-playground/shared';
import type { LeftTabId, PhaseMode } from '../store/useLayoutStore.ts';

export type LessonTarget = 'data' | 'network' | 'hyperparams' | 'transport';

export interface LessonStep {
    id: string;
    title: string;
    body: string;
    target: LessonTarget;
    tab?: LeftTabId;
    phase?: PhaseMode;
    expectedOutcome?: string;
    explanationRuleIds?: readonly string[];
}

export interface LessonDefinition {
    id: string;
    title: string;
    summary: string;
    presetId: Preset['id'];
    estimatedMinutes?: number;
    steps: readonly LessonStep[];
}
