// ── Code Export Panel ──
// Tabbed panel that generates pseudocode, NumPy, and TF.js code from the current network.

import { useMemo, useCallback, useId, useRef, memo, type KeyboardEvent } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useLayoutStore, type CodeExportTab } from '../../store/useLayoutStore.ts';
import { generatePseudocode, generateNumPy, generateTFJS } from '@nn-playground/shared';
import { Tooltip } from '../common/Tooltip.tsx';
import { getFrameBuffer } from '../../worker/frameBuffer.ts';
import { unflattenBiases, unflattenWeights } from '../../worker/frameBufferLayout.ts';
import { useTimedState } from '../../hooks/useTimedState.ts';
import type { TrainingConfig } from '@nn-playground/engine';
import type { ValidatedStandardExperimentRecipeV2 } from '@nn-playground/shared';
import { selectScientificEvidence } from '../../store/evidenceSelectors.ts';

const TABS: { id: CodeExportTab; label: string }[] = [
    { id: 'pseudocode', label: 'Pseudocode' },
    { id: 'numpy', label: 'NumPy' },
    { id: 'tfjs', label: 'TF.js' },
];

function toCodeExportTraining(
    recipe: ValidatedStandardExperimentRecipeV2,
): TrainingConfig {
    const optimizer = recipe.training.optimizer;
    const dataLoss = recipe.objective.dataLoss;
    const penalty = recipe.objective.penalty;
    const clipping = recipe.training.gradientClipping;
    const schedule = recipe.training.schedule;
    return {
        learningRate: recipe.training.learningRate,
        batchSize: recipe.training.batchSize,
        lossType: dataLoss.kind === 'binary-cross-entropy-with-logits'
            ? 'crossEntropy'
            : dataLoss.kind === 'categorical-cross-entropy-with-logits'
                ? 'categoricalCrossEntropy'
                : dataLoss.kind === 'mean-squared-error'
                    ? 'mse'
                    : 'huber',
        optimizer: optimizer.kind === 'sgd-momentum' ? 'sgdMomentum' : optimizer.kind,
        momentum: optimizer.kind === 'sgd-momentum' ? optimizer.momentum : 0.9,
        regularization: penalty.kind,
        regularizationRate: penalty.kind === 'none' ? 0 : penalty.coefficient,
        gradientClip: clipping.kind === 'none' ? null : clipping.maximumNorm,
        ...(optimizer.kind === 'adam' ? {
            adamBeta1: optimizer.beta1,
            adamBeta2: optimizer.beta2,
            adamEps: optimizer.epsilon,
        } : {}),
        ...(dataLoss.kind === 'huber' ? { huberDelta: dataLoss.delta } : {}),
        ...(schedule.kind === 'step' ? {
            lrSchedule: {
                type: 'step' as const,
                stepSize: schedule.interval,
                gamma: schedule.gamma,
            },
        } : schedule.kind === 'cosine' ? {
            lrSchedule: {
                type: 'cosine' as const,
                totalSteps: schedule.totalSteps,
                minLr: schedule.minimumRate,
            },
        } : {}),
    };
}

export const CodeExportPanel = memo(function CodeExportPanel() {
    const activeTab = useLayoutStore((s) => s.codeExportTab);
    const setActiveTab = useLayoutStore((s) => s.setCodeExportTab);
    const [copied, setCopied] = useTimedState(false, 2000);
    const [copyError, setCopyError] = useTimedState<string | null>(null, 2000);
    const tabIdBase = useId();
    const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);

    const prepared = usePlaygroundStore((s) => (
        s.access.status === 'ready' ? s.access.prepared : null
    ));
    const trainedRecipeFingerprint = useTrainingStore((s) => s.trainedRecipeFingerprint);
    const latestLiveSignal = useTrainingStore((s) => s.latestLiveSignal);
    const latestEvaluation = useTrainingStore((s) => s.latestEvaluation);
    const paramsVersion = useTrainingStore((s) => s.paramsVersion);
    const evidence = useMemo(
        () => selectScientificEvidence({ latestLiveSignal, latestEvaluation }),
        [latestEvaluation, latestLiveSignal],
    );

    const exportSnapshot = useMemo(() => {
        // This version selector intentionally drives the mutable frame-buffer read.
        void paramsVersion;
        if (!prepared
            || !evidence.currentModel
            || trainedRecipeFingerprint === null
            || trainedRecipeFingerprint !== prepared.identities.recipeFingerprint) {
            return null;
        }

        const frameBuffer = getFrameBuffer();
        const parameterProvenance = frameBuffer.parameterProvenance;
        if (frameBuffer.weights
            && frameBuffer.biases
            && frameBuffer.weightLayout
            && parameterProvenance
            && parameterProvenance.recipeFingerprint
                === prepared.identities.recipeFingerprint
            && parameterProvenance.recipeFingerprint === trainedRecipeFingerprint
            && parameterProvenance.model.generationId
                === evidence.currentModel.generationId) {
            const expectedLayerSizes = [
                prepared.compiled.network.inputSize,
                ...prepared.compiled.network.hiddenLayers,
                prepared.compiled.task.outputSize,
            ];
            if (frameBuffer.weightLayout.layerSizes.length !== expectedLayerSizes.length
                || frameBuffer.weightLayout.layerSizes.some((size, index) => (
                    size !== expectedLayerSizes[index]
                ))) {
                return null;
            }
            return {
                step: parameterProvenance.model.step,
                weights: unflattenWeights(frameBuffer.weights, frameBuffer.weightLayout.layerSizes),
                biases: unflattenBiases(frameBuffer.biases, frameBuffer.weightLayout.layerSizes),
            };
        }

        return null;
    }, [evidence.currentModel, paramsVersion, prepared, trainedRecipeFingerprint]);

    const code = useMemo(() => {
        if (!prepared) return '# No compatible version-2 experiment is active.';
        const config = prepared.compiled.network;
        const training = toCodeExportTraining(prepared.document.recipe);
        const features = prepared.compiled.features;
        switch (activeTab) {
            case 'pseudocode':
                return generatePseudocode(config, training, features, exportSnapshot);
            case 'numpy':
                return generateNumPy(config, training, features, exportSnapshot);
            case 'tfjs':
                return generateTFJS(config, training, features, exportSnapshot);
        }
    }, [activeTab, exportSnapshot, prepared]);

    const handleCopy = useCallback(async () => {
        try {
            if (!navigator.clipboard?.writeText) {
                throw new Error('Clipboard API unavailable');
            }

            await navigator.clipboard.writeText(code);
            setCopyError(null);
            setCopied(true);
        } catch {
            setCopied(false);
            setCopyError('Could not copy code.');
        }
    }, [code, setCopied, setCopyError]);

    const handleTabKeyDown = useCallback((event: KeyboardEvent<HTMLButtonElement>, index: number) => {
        let nextIndex: number | null = null;
        if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
            nextIndex = (index + 1) % TABS.length;
        } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
            nextIndex = (index - 1 + TABS.length) % TABS.length;
        } else if (event.key === 'Home') {
            nextIndex = 0;
        } else if (event.key === 'End') {
            nextIndex = TABS.length - 1;
        }

        if (nextIndex === null) return;
        event.preventDefault();
        setActiveTab(TABS[nextIndex].id);
        tabRefs.current[nextIndex]?.focus();
    }, [setActiveTab]);

    return (
        <div className="code-export-panel">
            <div className="code-export__tabs" role="tablist" aria-label="Code format">
                {TABS.map((tab, index) => (
                    <button
                        type="button"
                        role="tab"
                        key={tab.id}
                        id={`${tabIdBase}-${tab.id}`}
                        aria-controls={`${tabIdBase}-panel`}
                        aria-selected={activeTab === tab.id}
                        tabIndex={activeTab === tab.id ? 0 : -1}
                        ref={(element) => {
                            tabRefs.current[index] = element;
                        }}
                        className={`chip ${activeTab === tab.id ? 'active' : ''}`}
                        onClick={() => setActiveTab(tab.id)}
                        onKeyDown={(event) => handleTabKeyDown(event, index)}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>

            <div
                className="code-export__code-container"
                role="tabpanel"
                id={`${tabIdBase}-panel`}
                aria-labelledby={`${tabIdBase}-${activeTab}`}
                tabIndex={0}
            >
                <pre className="code-export__code">{code}</pre>
            </div>

            <Tooltip content="Cause: copy exports the code generated from the visible configuration. Effect: you can inspect or reuse the learned setup outside the playground." block>
                <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    style={{ marginTop: 6, width: '100%' }}
                    onClick={handleCopy}
                >
                    {copied ? '✓ Copied!' : '📋 Copy Code'}
                </button>
            </Tooltip>
            {copyError && (
                <div className="config-feedback config-feedback--error" role="alert">
                    {copyError}
                </div>
            )}
        </div>
    );
});
