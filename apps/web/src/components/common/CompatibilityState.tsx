import { useState } from 'react';
import type { ExperimentAccessState } from '../../store/usePlaygroundStore.ts';

interface CompatibilityStateProps {
    access: Extract<ExperimentAccessState, { status: 'incompatible' }>;
    onStartFresh: () => void | Promise<unknown>;
}

export function CompatibilityState({ access, onStartFresh }: CompatibilityStateProps) {
    const [isStarting, setIsStarting] = useState(false);
    const [recoveryError, setRecoveryError] = useState<string | null>(null);
    const sourceTitle = access.source.kind === 'url'
        ? 'Original URL fragment'
        : 'Original import file';

    const startFresh = async () => {
        if (isStarting) return;
        setIsStarting(true);
        setRecoveryError(null);
        try {
            await onStartFresh();
        } catch (error) {
            setRecoveryError(error instanceof Error
                ? error.message
                : 'The default experiment could not be prepared.');
        } finally {
            setIsStarting(false);
        }
    };

    return (
        <main
            className="compatibility-state"
            aria-label="Experiment compatibility"
            tabIndex={-1}
        >
            <section aria-labelledby="compatibility-state-title">
                <h1 id="compatibility-state-title">Experiment is incompatible</h1>
                <p>
                    This input was preserved exactly and was not loaded or repaired.
                    Training has not started.
                </p>
                <dl>
                    <div>
                        <dt>{sourceTitle}</dt>
                        <dd>
                            {access.source.kind === 'url' ? (
                                <code>{access.source.rawHash}</code>
                            ) : (
                                <>
                                    <span>{access.source.file.name}</span>{' '}
                                    <span>{access.source.file.size} bytes</span>
                                </>
                            )}
                        </dd>
                    </div>
                </dl>
                <h2>Compatibility issues</h2>
                <ul aria-label="Compatibility issues">
                    {access.issues.map((issue, index) => (
                        <li key={`${issue.path}-${issue.code}-${index}`}>
                            <code>{issue.path}</code>: {issue.message}
                        </li>
                    ))}
                </ul>
                {recoveryError && <p role="alert">{recoveryError}</p>}
                <button
                    type="button"
                    className="btn btn--primary"
                    onClick={() => { void startFresh(); }}
                    disabled={isStarting}
                    aria-busy={isStarting}
                >
                    {isStarting ? 'Starting fresh…' : 'Start fresh'}
                </button>
                <p>
                    Start fresh installs the default version-2 experiment and replaces
                    the incompatible URL only after validation succeeds.
                </p>
            </section>
        </main>
    );
}
