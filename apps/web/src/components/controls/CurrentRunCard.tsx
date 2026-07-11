import { memo, useMemo } from 'react';
import type { AppConfig, PauseReason, TrainingStatus } from '@nn-playground/shared';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore, type ConfigChangeSource } from '../../store/useTrainingStore.ts';
import { getRecipeDrift } from '../../store/recipeIdentity.ts';

function formatMetric(value: number | undefined): string {
    return value === undefined || !Number.isFinite(value) ? 'n/a' : value.toFixed(4);
}

function formatSignedMetric(value: number | undefined): string {
    if (value === undefined || !Number.isFinite(value)) return 'n/a';
    return `${value > 0 ? '+' : ''}${value.toFixed(4)}`;
}

function formatAccuracy(value: number | undefined): string {
    return value === undefined || !Number.isFinite(value) ? 'n/a' : `${(value * 100).toFixed(1)}%`;
}

function sourceLabel(source: Exclude<ConfigChangeSource, null>): string {
    switch (source) {
        case 'data':
            return 'data';
        case 'network':
            return 'network';
        case 'features':
            return 'features';
        case 'training':
            return 'training';
        case 'preset':
            return 'preset';
    }
}

function pauseReasonCopy(reason: PauseReason | null): string {
    switch (reason) {
        case 'target-loss-reached':
            return 'Paused after reaching the target loss.';
        case 'target-accuracy-reached':
            return 'Paused after reaching the target accuracy.';
        case 'plateau':
            return 'Paused because learning plateaued.';
        case 'diverged':
            return 'Paused because training diverged.';
        case 'max-steps':
            return 'Paused at the configured step limit.';
        case 'error':
            return 'Paused because the run needs attention.';
        case 'manual':
            return 'Paused manually.';
        default:
            return 'Paused and ready to resume.';
    }
}

function useCurrentRecipeConfig(): AppConfig {
    const data = usePlaygroundStore((s) => s.data);
    const features = usePlaygroundStore((s) => s.features);
    const network = usePlaygroundStore((s) => s.network);
    const training = usePlaygroundStore((s) => s.training);
    const ui = usePlaygroundStore((s) => s.ui);

    return useMemo(() => ({ data, features, network, training, ui }), [data, features, network, training, ui]);
}

function getRunStateCopy(args: {
    status: TrainingStatus;
    step: number | null;
    pendingConfigSource: ConfigChangeSource;
    workerError: string | null;
    configError: string | null;
    pauseReason: PauseReason | null;
    staleMetrics: boolean;
    hasSnapshot: boolean;
    hasDrift: boolean;
}) {
    if (args.workerError) {
        return {
            tone: 'error',
            title: 'Worker connection lost',
            detail: `${args.workerError} Refresh the page to restart the playground.`,
        };
    }

    if (args.configError) {
        return {
            tone: 'error',
            title: 'Recipe update failed',
            detail: args.configError,
        };
    }

    if (args.pendingConfigSource) {
        return {
            tone: 'pending',
            title: `Updating ${sourceLabel(args.pendingConfigSource)} config`,
            detail: 'Evidence still reflects the last trained snapshot until the update completes.',
        };
    }

    if (!args.hasSnapshot) {
        return {
            tone: 'idle',
            title: 'Ready to train',
            detail: 'No trained snapshot exists yet.',
        };
    }

    if (args.status === 'running') {
        return {
            tone: 'live',
            title: 'Live run',
            detail: `Training is updating the model${args.step === null ? '.' : ` at step ${args.step}.`}`,
        };
    }

    if (args.hasDrift) {
        return {
            tone: 'drift',
            title: 'Recipe drift',
            detail: 'Current recipe differs from trained snapshot.',
        };
    }

    if (args.staleMetrics) {
        return {
            tone: 'stale',
            title: 'Stale metrics',
            detail: 'The latest evidence is reusing cached test metrics until a fresh pass completes.',
        };
    }

    if (args.status === 'paused') {
        return {
            tone: 'paused',
            title: 'Paused run',
            detail: pauseReasonCopy(args.pauseReason),
        };
    }

    return {
        tone: 'idle',
        title: 'Trained snapshot',
        detail: 'Evidence reflects the last accepted recipe.',
    };
}

export const CurrentRunCard = memo(function CurrentRunCard() {
    const currentConfig = useCurrentRecipeConfig();
    const status = useTrainingStore((s) => s.status);
    const snapshot = useTrainingStore((s) => s.snapshot);
    const trainedRecipeConfig = useTrainingStore((s) => s.trainedRecipeConfig);
    const trainedRecipeFingerprint = useTrainingStore((s) => s.trainedRecipeFingerprint);
    const currentRecipeFingerprint = usePlaygroundStore(
        (s) => s.prepared?.identities.recipeFingerprint ?? null,
    );
    const pendingConfigSource = useTrainingStore((s) => s.pendingConfigSource);
    const workerError = useTrainingStore((s) => s.workerError);
    const configError = useTrainingStore((s) => s.configError);
    const pauseReason = useTrainingStore((s) => s.pauseReason);
    const testMetricsStale = useTrainingStore((s) => s.testMetricsStale);

    const drift = useMemo(
        () => getRecipeDrift(
            trainedRecipeConfig,
            currentConfig,
            3,
            trainedRecipeFingerprint === null ? undefined : {
                trainedRecipeFingerprint,
                currentRecipeFingerprint,
            },
        ),
        [trainedRecipeConfig, currentConfig, trainedRecipeFingerprint, currentRecipeFingerprint],
    );
    const state = getRunStateCopy({
        status,
        step: snapshot?.step ?? null,
        pendingConfigSource,
        workerError,
        configError,
        pauseReason,
        staleMetrics: testMetricsStale,
        hasSnapshot: snapshot !== null,
        hasDrift: drift.hasDrift,
    });

    return (
        <section
            className={`forge-context-card forge-run-card forge-run-card--${state.tone}`}
            role="region"
            aria-label="Current run"
        >
            <div className="forge-context-card__head">
                <span className="forge-context-card__eyebrow">Current run</span>
                <span className="forge-context-pill">{status}</span>
            </div>
            <div className="forge-run-card__state">
                <strong>{state.title}</strong>
                <p>{state.detail}</p>
            </div>
            <div className="forge-run-card__meta" aria-label="Run snapshot metadata">
                <span>{snapshot ? `Snapshot step ${snapshot.step.toLocaleString()}` : 'No snapshot'}</span>
                <span>{snapshot ? `Epoch ${snapshot.epoch.toLocaleString()}` : 'Epoch 0'}</span>
                {testMetricsStale && <span>metrics stale</span>}
                {drift.groupLabels.length > 0 && <span>{drift.groupLabels.join(', ')}</span>}
            </div>
            {snapshot && (
                <div className="forge-run-card__metrics" aria-label="Run metrics">
                    <span>{`train ${formatMetric(snapshot.trainLoss)}`}</span>
                    <span>{`test ${formatMetric(snapshot.testLoss)}`}</span>
                    <span>{`gap ${formatSignedMetric(snapshot.testLoss - snapshot.trainLoss)}`}</span>
                    <span>{`accuracy ${formatAccuracy(snapshot.testMetrics?.accuracy)}`}</span>
                </div>
            )}
        </section>
    );
});
