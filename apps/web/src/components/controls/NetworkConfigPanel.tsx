// ── Network Configuration Panel ──
import { memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { ACTIVATION_LABELS } from '@nn-playground/engine';
import type { ActivationType } from '@nn-playground/engine';
import { MAX_HIDDEN_LAYERS } from '@nn-playground/shared';
import { LoadingState } from '../common/LoadingState.tsx';
import { Tooltip } from '../common/Tooltip.tsx';

const ACTIVATIONS: ActivationType[] = ['relu', 'tanh', 'sigmoid', 'linear', 'leakyRelu', 'elu', 'swish', 'softplus'];

export const NetworkConfigPanel = memo(function NetworkConfigPanel() {
    const hiddenLayers = usePlaygroundStore((s) => s.network.hiddenLayers);
    const activation = usePlaygroundStore((s) => s.network.activation);
    const isLoading = useTrainingStore((s) => s.networkConfigLoading);
    const configError = useTrainingStore((s) => s.configError);
    const configErrorSource = useTrainingStore((s) => s.configErrorSource);
    const store = usePlaygroundStore;

    const beginNetworkChange = () => useTrainingStore.getState().beginConfigChange('network');
    const retryNetworkChange = () => useTrainingStore.getState().retryConfigSync();

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
                <span className="control-label">Hidden layers</span>
                <div
                    className="forge-stepper"
                    role="group"
                    aria-label="Hidden layer count"
                >
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
                    <span className="forge-stepper__value" aria-live="polite">
                        {hiddenLayers.length}
                    </span>
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
                    <input
                        type="range"
                        min="1"
                        max="16"
                        value={count}
                        onChange={(e) => {
                            beginNetworkChange();
                            store.getState().setNeuronsInLayer(idx, Number(e.target.value));
                        }}
                        aria-label={`Neurons in layer ${idx + 1}`}
                        style={{ flex: 1 }}
                    />
                    <Tooltip content={`Cause: layer ${idx + 1} has ${count} neurons to detect intermediate patterns. Effect: more neurons can model finer bends, but too many can overfit noisy samples.`}>
                        <span className="neuron-badge">{count}</span>
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
