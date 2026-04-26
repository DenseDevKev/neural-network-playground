// ── Shared types for the application layer ──
import type { NetworkConfig, TrainingConfig, DataConfig, FeatureFlags } from '@nn-playground/engine';

export interface UIConfig {
    showTestData: boolean;
    discretizeOutput: boolean;
}

export interface AppConfig {
    network: NetworkConfig;
    training: TrainingConfig;
    data: DataConfig;
    features: FeatureFlags;
    ui: UIConfig;
}

export interface Preset {
    id: string;
    title: string;
    description: string;
    learningGoal?: string;
    thumbnail?: string;
    difficulty?: 'beginner' | 'intermediate' | 'advanced';
    config: Partial<AppConfig>;
}

export type TrainingStatus = 'idle' | 'running' | 'paused';

export const PAUSE_REASONS = [
    'target-loss-reached',
    'target-accuracy-reached',
    'plateau',
    'diverged',
    'max-steps',
    'manual',
    'error',
] as const;

export type PauseReason = typeof PAUSE_REASONS[number];

export function isPauseReason(value: unknown): value is PauseReason {
    return typeof value === 'string' && (PAUSE_REASONS as readonly string[]).includes(value);
}
