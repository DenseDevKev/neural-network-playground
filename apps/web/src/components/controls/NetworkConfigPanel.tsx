// ── Network Configuration Panel ──
import { memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { ACTIVATION_LABELS } from '@nn-playground/engine';
import type { ActivationType } from '@nn-playground/engine';
import { MAX_HIDDEN_LAYERS } from '@nn-playground/shared';

const ACTIVATIONS: ActivationType[] = ['relu', 'tanh', 'sigmoid', 'linear', 'leakyRelu', 'elu', 'swish', 'softplus'];

export const NetworkConfigPanel = memo(function NetworkConfigPanel() {
    const hiddenLayers = usePlaygroundStore((s) => s.network.hiddenLayers);
    const activation = usePlaygroundStore((s) => s.network.activation);
    const store = usePlaygroundStore;

    return (
        <div className="panel">
            <div className="panel__title">Network</div>

            {/* Hidden layers +/- */}
            <div className="control-row" style={{ marginBottom: 8 }}>
                <span className="control-label">Hidden Layers</span>
                <div className="layer-controls">
                    <button
                        className="btn btn--ghost btn--icon btn--sm"
                        onClick={() => store.getState().removeLayer()}
                        disabled={hiddenLayers.length === 0}
                        aria-label="Remove hidden layer"
                    >
                        −
                    </button>
                    <span className="layer-controls__count">{hiddenLayers.length}</span>
                    <button
                        className="btn btn--ghost btn--icon btn--sm"
                        onClick={() => store.getState().addLayer()}
                        disabled={hiddenLayers.length >= MAX_HIDDEN_LAYERS}
                        aria-label="Add hidden layer"
                    >
                        +
                    </button>
                </div>
            </div>

            {/* Neurons per layer */}
            {hiddenLayers.map((count, idx) => (
                <div key={idx} className="neuron-row">
                    <span className="control-label" style={{ minWidth: 60 }}>Layer {idx + 1}</span>
                    <input
                        type="range"
                        min="1"
                        max="16"
                        value={count}
                        onChange={(e) => store.getState().setNeuronsInLayer(idx, Number(e.target.value))}
                        aria-label={`Neurons in layer ${idx + 1}`}
                        style={{ flex: 1 }}
                    />
                    <span className="neuron-badge">{count}</span>
                </div>
            ))}

            {/* Activation */}
            <div className="control-row" style={{ marginTop: 8 }}>
                <span className="control-label">Activation</span>
                <select
                    className="select"
                    value={activation}
                    onChange={(e) => store.getState().setActivation(e.target.value as ActivationType)}
                >
                    {ACTIVATIONS.map((a) => (
                        <option key={a} value={a}>{ACTIVATION_LABELS[a]}</option>
                    ))}
                </select>
            </div>
        </div>
    );
});
