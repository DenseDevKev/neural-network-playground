// ── Shared types for the application layer ──
import type { NetworkConfig, TrainingConfig, DataConfig, FeatureFlags, NetworkSnapshot, HistoryPoint } from '@nn-playground/engine';

export interface UIConfig {
    showTestData: boolean;
    discretizeOutput: boolean;
    /** @reserved Not wired to UI. Training speed is controlled via `stepsPerFrame` in the Zustand store. */
    animationSpeed: number;
    /** @reserved Placeholder for the upcoming Guided Lesson Mode feature. */
    lessonMode?: boolean;
    /** @reserved Placeholder for the upcoming Guided Lesson Mode feature. */
    hiddenControls?: string[];
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

/** Full app state used by the Zustand store. */
export interface PlaygroundState {
    config: AppConfig;
    status: TrainingStatus;
    snapshot: NetworkSnapshot | null;
    history: HistoryPoint[];
    isInitialized: boolean;
}
