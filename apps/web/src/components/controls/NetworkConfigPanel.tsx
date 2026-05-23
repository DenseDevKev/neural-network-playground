// ── Network Configuration Panel ──
import { memo, useEffect, useState } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { ACTIVATION_LABELS } from '@nn-playground/engine';
import type { ActivationType } from '@nn-playground/engine';
import { MAX_HIDDEN_LAYERS } from '@nn-playground/shared';
import { LoadingState } from '../common/LoadingState.tsx';
import { Tooltip } from '../common/Tooltip.tsx';

const ACTIVATIONS: ActivationType[] = ['relu', 'tanh', 'sigmoid', 'linear', 'leakyRelu', 'elu', 'swish', 'softplus'];
const MIN_NEURONS_PER_LAYER = 1;
const MAX_NEURONS_PER_LAYER_UI = 16;

function clampNeuronCount(value: number): number {
    if (!Number.isFinite(value)) return MIN_NEURONS_PER_LAYER;
    return Math.max(MIN_NEURONS_PER_LAYER, Math.min(MAX_NEURONS_PER_LAYER_UI, Math.trunc(value)));
}

function NeuronCountControl({
    layer,
    value,
    onChange,
}: {
    layer: number;
    value: number;
    onChange: (nextValue: number) => void;
}) {
    const [draft, setDraft] = useState(String(value));

    useEffect(() => {
        setDraft(String(value));
    }, [value]);

    const commit = (nextValue: number) => {
        const clamped = clampNeuronCount(nextValue);
        setDraft(String(clamped));
        if (clamped !== value) onChange(clamped);
    };

    return (
        <div className="neuron-stepper" role="group" aria-label={`Layer ${layer} neuron count`}>
            <button
                type="button"
                className="forge-stepper__btn neuron-stepper__btn"
                onClick={() => commit(value - 1)}
                disabled={value <= MIN_NEURONS_PER_LAYER}
                aria-label={`Decrease neurons in layer ${layer}`}
            >
                −
            </button>
            <input
                className="neuron-stepper__input"
                type="number"
                min={MIN_NEURONS_PER_LAYER}
                max={MAX_NEURONS_PER_LAYER_UI}
                step={1}
                inputMode="numeric"
                value={draft}
                onChange={(event) => {
                    const nextDraft = event.target.value;
                    setDraft(nextDraft);
                    if (nextDraft.trim() === '') return;
                    commit(Number(nextDraft));
                }}
                onBlur={() => {
                    if (draft.trim() === '') {
                        setDraft(String(value));
                        return;
                    }
                    commit(Number(draft));
                }}
                aria-label={`Neuron count for layer ${layer}`}
            />
            <button
                type="button"
                className="forge-stepper__btn neuron-stepper__btn"
                onClick={() => commit(value + 1)}
                disabled={value >= MAX_NEURONS_PER_LAYER_UI}
                aria-label={`Increase neurons in layer ${layer}`}
            >
                +
            </button>
        </div>
    );
}

export const NetworkConfigPanel = memo(function NetworkConfigPanel() {
    const hiddenLayers = usePlaygroundStore((s) => s.network.hiddenLayers);
    const activation = usePlaygroundStore((s) => s.network.activation);
    const isLoading = useTrainingStore((s) => s.networkConfigLoading);
    const configError = useTrainingStore((s) => s.configError);
    const configErrorSource = useTrainingStore((s) => s.configErrorSource);
    const store = usePlaygroundStore;

    const beginNetworkChange = () => useTrainingStore.getState().beginConfigChange('network');
    const retryNetworkChange = () => useTrainingStore.getState().retryConfigSync();
    const setLayerNeuronCount = (idx: number, count: number) => {
        const nextCount = clampNeuronCount(count);
        if (hiddenLayers[idx] === nextCount) return;
        beginNetworkChange();
        store.getState().setNeuronsInLayer(idx, nextCount);
    };

    return (
        <div>
            <LoadingState isLoading={isLoading} inline message="Initializing network..." />
            {configError && configErrorSource === 'network' && (
                <div className="config-feedback config-feedback--error" role="alert">
                    <span>{configError}</span>
                    <button type="button" className="btn btn--ghost btn--sm" onClick={retryNetworkChange}>
                        Retry
                    </button>
                </div>
            )}

            {/* Hidden layers +/- */}
            <div className="control-row" style={{ marginBottom: 8 }}>
                <span className="control-label">Hidden Layers</span>
                <div className="layer-controls">
                    <Tooltip content="Cause: removing a hidden layer lowers model capacity. Effect: the boundary becomes simpler and may underfit curved data.">
                        <button
                            type="button"
                            className="forge-stepper__btn"
                            onClick={() => {
                                beginNetworkChange();
                                store.getState().removeLayer();
                            }}
                            disabled={hiddenLayers.length === 0}
                            aria-label="Remove hidden layer"
                        >
                            −
                        </button>
                    </Tooltip>
                    <span className="layer-controls__count">{hiddenLayers.length}</span>
                    <Tooltip content="Cause: adding a hidden layer adds another learned transformation. Effect: the boundary can bend more, but training may take longer.">
                        <button
                            type="button"
                            className="forge-stepper__btn"
                            onClick={() => {
                                beginNetworkChange();
                                store.getState().addLayer();
                            }}
                            disabled={hiddenLayers.length >= MAX_HIDDEN_LAYERS}
                            aria-label="Add hidden layer"
                        >
                            +
                        </button>
                    </Tooltip>
                </div>
            </div>

            {/* Neurons per layer */}
            {hiddenLayers.length > 0 && (
                <div className="forge-section__label" style={{ marginTop: 8, marginBottom: 6 }}>
                    Neurons per layer
                </div>
            )}
            {hiddenLayers.map((count, idx) => (
                <div key={idx} className="neuron-row">
                    <span className="control-label" style={{ minWidth: 60 }}>Layer {idx + 1}</span>
                    <Tooltip content={`Cause: layer ${idx + 1} has ${count} neurons to detect intermediate patterns. Effect: more neurons can model finer bends, but too many can overfit noisy samples.`}>
                        <NeuronCountControl
                            layer={idx + 1}
                            value={count}
                            onChange={(nextCount) => setLayerNeuronCount(idx, nextCount)}
                        />
                    </Tooltip>
                </div>
            ))}

            {/* Activation */}
            <div className="control-row" style={{ marginTop: 8 }}>
                <span className="control-label">Activation</span>
                <Tooltip content="Cause: activation functions decide when neurons pass signal forward. Effect: tanh/sigmoid smooth the boundary, while ReLU-family choices make sharper bends.">
                    <select
                        className="select"
                        aria-label="Activation"
                        value={activation}
                        onChange={(e) => {
                            beginNetworkChange();
                            store.getState().setActivation(e.target.value as ActivationType);
                        }}
                    >
                        {ACTIVATIONS.map((a) => (
                            <option key={a} value={a}>{ACTIVATION_LABELS[a]}</option>
                        ))}
                    </select>
                </Tooltip>
            </div>
        </div>
    );
});
