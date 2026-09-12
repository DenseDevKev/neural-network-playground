import { useEffect, useRef } from 'react';
import type {
    PauseReason,
    TrainingStatus,
    ValidatedStandardExperimentRecipeV2,
} from '@nn-playground/shared';
import type {
    ConfigChangeSource,
    TrainedRecipeSource,
} from '../../store/useTrainingStore.ts';

interface AccessibilityAnnouncerProps {
    status: TrainingStatus;
    pauseReason: PauseReason | null;
    workerError: string | null;
    pendingConfigSource: ConfigChangeSource;
    configError: string | null;
    configErrorSource: ConfigChangeSource;
    evidenceGenerationId: number | null;
    trainedRecipe: ValidatedStandardExperimentRecipeV2 | null;
    trainedRecipeSource: TrainedRecipeSource | null;
}

type AnnouncementSnapshot = AccessibilityAnnouncerProps;

const CONFIG_START_MESSAGES = {
    data: 'Generating data',
    network: 'Initializing network',
    features: 'Updating features',
    training: 'Updating training',
    preset: 'Applying preset',
    setup: 'Applying setup',
} as const;

const CONFIG_SCOPE_LABELS = {
    data: 'Data',
    network: 'Network',
    features: 'Features',
    training: 'Training',
    preset: 'Preset',
    setup: 'Setup',
} as const;

export function AccessibilityAnnouncer(props: AccessibilityAnnouncerProps) {
    const previousRef = useRef<AnnouncementSnapshot>(props);
    const handledResetGenerationRef = useRef<number | null>(null);
    const pendingPublicationGenerationRef = useRef<number | null>(null);
    const regionRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const previous = previousRef.current;
        const {
            status,
            pauseReason,
            pendingConfigSource,
            configError,
            configErrorSource,
            evidenceGenerationId,
            trainedRecipe,
            trainedRecipeSource,
        } = props;

        const hasNewError = configError !== null
            && configErrorSource !== null
            && (configError !== previous.configError
                || configErrorSource !== previous.configErrorSource);
        const hasNewConfigStart = pendingConfigSource !== null
            && pendingConfigSource !== previous.pendingConfigSource;
        const hasConfigCompletion = previous.pendingConfigSource !== null
            && pendingConfigSource === null
            && !(configError !== null && configErrorSource === previous.pendingConfigSource);
        const generationChanged = evidenceGenerationId !== previous.evidenceGenerationId;
        if (generationChanged) {
            pendingPublicationGenerationRef.current = evidenceGenerationId;
        }
        const hasNewTrainedRecipePublication = trainedRecipe !== previous.trainedRecipe;
        const hasResetPublication = evidenceGenerationId !== null
            && pendingPublicationGenerationRef.current === evidenceGenerationId
            && hasNewTrainedRecipePublication
            && trainedRecipeSource === 'reset'
            && evidenceGenerationId !== handledResetGenerationRef.current;
        const hasNonErrorPause = status === 'paused'
            && pauseReason !== null
            && pauseReason !== 'error'
            && (previous.status !== 'paused' || previous.pauseReason !== pauseReason);
        const hasTrainingStart = status === 'running'
            && (previous.status === 'idle' || previous.status === 'paused');

        if (hasNewTrainedRecipePublication
            && pendingPublicationGenerationRef.current === evidenceGenerationId) {
            pendingPublicationGenerationRef.current = null;
            if (hasResetPublication) {
                handledResetGenerationRef.current = evidenceGenerationId;
            }
        }

        let nextMessage: string | null = null;
        if (hasNewError) {
            nextMessage = `${CONFIG_SCOPE_LABELS[configErrorSource]} error: ${configError}`;
        } else if (hasNewConfigStart) {
            nextMessage = CONFIG_START_MESSAGES[pendingConfigSource];
        } else if (hasConfigCompletion) {
            nextMessage = `${CONFIG_SCOPE_LABELS[previous.pendingConfigSource!]} update complete`;
        } else if (hasResetPublication) {
            nextMessage = 'Training reset';
        } else if (hasNonErrorPause) {
            nextMessage = 'Training paused';
        } else if (hasTrainingStart) {
            nextMessage = 'Training started';
        }

        previousRef.current = props;

        if (nextMessage !== null && regionRef.current) {
            regionRef.current.replaceChildren(nextMessage);
        }
    }, [props]);

    return (
        <div
            ref={regionRef}
            className="sr-only"
            role="status"
            aria-label="Training and configuration announcements"
            aria-live="polite"
            aria-atomic="true"
        />
    );
}
