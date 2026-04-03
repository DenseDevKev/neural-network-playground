import { useEffect, useRef, useState } from 'react';
import type { TrainingStatus } from '@nn-playground/shared';

interface AccessibilityAnnouncerProps {
    status: TrainingStatus;
    dataConfigLoading: boolean;
    networkConfigLoading: boolean;
    configError: string | null;
    configErrorSource: 'data' | 'network' | null;
}

export function AccessibilityAnnouncer({
    status,
    dataConfigLoading,
    networkConfigLoading,
    configError,
    configErrorSource,
}: AccessibilityAnnouncerProps) {
    const [message, setMessage] = useState('');
    const previousStatusRef = useRef(status);
    const previousDataLoadingRef = useRef(dataConfigLoading);
    const previousNetworkLoadingRef = useRef(networkConfigLoading);
    const previousErrorRef = useRef<string | null>(configError);

    useEffect(() => {
        if (previousStatusRef.current === status) {
            return;
        }

        if (status === 'running') {
            setMessage('Training started');
        } else if (status === 'paused') {
            setMessage('Training paused');
        } else if (status === 'idle' && previousStatusRef.current !== 'idle') {
            setMessage('Training reset');
        }

        previousStatusRef.current = status;
    }, [status]);

    useEffect(() => {
        if (dataConfigLoading && !previousDataLoadingRef.current) {
            setMessage('Generating data');
        }

        previousDataLoadingRef.current = dataConfigLoading;
    }, [dataConfigLoading]);

    useEffect(() => {
        if (networkConfigLoading && !previousNetworkLoadingRef.current) {
            setMessage('Initializing network');
        }

        previousNetworkLoadingRef.current = networkConfigLoading;
    }, [networkConfigLoading]);

    useEffect(() => {
        if (configError && configError !== previousErrorRef.current) {
            const scope = configErrorSource === 'network' ? 'Network' : 'Data';
            setMessage(`${scope} error: ${configError}`);
        }

        previousErrorRef.current = configError;
    }, [configError, configErrorSource]);

    return (
        <div className="sr-only" aria-live="polite" aria-atomic="true" role="status">
            {message}
        </div>
    );
}
