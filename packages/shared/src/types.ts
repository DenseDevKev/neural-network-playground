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
