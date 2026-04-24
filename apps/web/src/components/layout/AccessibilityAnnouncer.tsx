import { useEffect, useRef, useState } from 'react';
import type { TrainingStatus } from '@nn-playground/shared';
import { useTrainingStore, type ConfigChangeSource } from '../../store/useTrainingStore.ts';

interface AccessibilityAnnouncerProps {
    status: TrainingStatus;
    dataConfigLoading: boolean;
    networkConfigLoading: boolean;
    featuresConfigLoading?: boolean;
    trainingConfigLoading?: boolean;
    configError: string | null;
    configErrorSource: ConfigChangeSource;
}

export function AccessibilityAnnouncer({
    status,
    dataConfigLoading,
    networkConfigLoading,
    featuresConfigLoading,
    trainingConfigLoading,
    configError,
    configErrorSource,
}: AccessibilityAnnouncerProps) {
    const storeFeaturesConfigLoading = useTrainingStore((s) => s.featuresConfigLoading);
    const storeTrainingConfigLoading = useTrainingStore((s) => s.trainingConfigLoading);
    const activeFeaturesConfigLoading = featuresConfigLoading ?? storeFeaturesConfigLoading;
    const activeTrainingConfigLoading = trainingConfigLoading ?? storeTrainingConfigLoading;
    const [message, setMessage] = useState('');
    const previousStatusRef = useRef(status);
    const previousDataLoadingRef = useRef(dataConfigLoading);
    const previousNetworkLoadingRef = useRef(networkConfigLoading);
    const previousFeaturesLoadingRef = useRef(activeFeaturesConfigLoading);
    const previousTrainingLoadingRef = useRef(activeTrainingConfigLoading);
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
        if (activeFeaturesConfigLoading && !previousFeaturesLoadingRef.current) {
            setMessage('Updating features');
        }

        previousFeaturesLoadingRef.current = activeFeaturesConfigLoading;
    }, [activeFeaturesConfigLoading]);

    useEffect(() => {
        if (activeTrainingConfigLoading && !previousTrainingLoadingRef.current) {
            setMessage('Updating training');
        }

        previousTrainingLoadingRef.current = activeTrainingConfigLoading;
    }, [activeTrainingConfigLoading]);

    useEffect(() => {
        if (configError && configError !== previousErrorRef.current) {
            const scope = {
                data: 'Data',
                network: 'Network',
                features: 'Features',
                training: 'Training',
            }[configErrorSource ?? 'data'];
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
