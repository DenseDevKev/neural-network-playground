import type { PauseReason, TrainingStatus } from '@nn-playground/shared';
import type { ConfigChangeSource } from '../../store/useTrainingStore.ts';

export interface LifecycleUi {
    primaryLabel: 'Start' | 'Pause' | 'Resume';
    primaryAriaLabel: string;
    statusText: string | null;
    disabledReason: string | null;
    isBlocked: boolean;
}

const CONFIG_SOURCE_LABELS: Record<Exclude<ConfigChangeSource, null>, string> = {
    data: 'data',
    network: 'network',
    features: 'features',
    training: 'training',
    preset: 'preset',
};

function formatPauseReason(reason: PauseReason): string {
    if (reason === 'manual') return 'Paused manually';
    if (reason === 'error') return 'Stopped by an error';
    return `Stopped: ${reason}`;
}

export function getTrainingLifecycleUi({
    status,
    pauseReason,
    pendingConfigSource,
}: {
    status: TrainingStatus;
    pauseReason: PauseReason | null;
    pendingConfigSource: ConfigChangeSource;
}): LifecycleUi {
    const isRunning = status === 'running';
    const isPaused = status === 'paused';
    const isBlocked = pendingConfigSource !== null;
    const primaryLabel = isRunning ? 'Pause' : isPaused ? 'Resume' : 'Start';
    const disabledReason = isBlocked
        ? `Updating ${CONFIG_SOURCE_LABELS[pendingConfigSource]} config...`
        : null;
    const statusText = disabledReason
        ?? (isPaused && pauseReason ? formatPauseReason(pauseReason) : null);

    return {
        primaryLabel,
        primaryAriaLabel: `${primaryLabel} training`,
        statusText,
        disabledReason,
        isBlocked,
    };
}
