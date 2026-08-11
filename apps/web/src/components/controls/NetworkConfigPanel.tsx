// ── Canonical V2 Network Configuration Panel ──
import { memo, useEffect, useState } from 'react';
import {
    ACTIVATION_LABELS,
    type ScalarActivationType,
    type WeightInitType,
} from '@nn-playground/engine';
import { MAX_HIDDEN_LAYERS } from '@nn-playground/shared';
import { commitRecipeEdit } from '../../store/commitRecipeEdit.ts';
import {
    setHiddenActivation,
    setHiddenLayers,
    setHiddenLayerWidth,
    setInitialization,
} from '../../store/recipeEdits.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { LoadingState } from '../common/LoadingState.tsx';
import { Tooltip } from '../common/Tooltip.tsx';

const HIDDEN_ACTIVATIONS: readonly ScalarActivationType[] = [
    'relu',
    'tanh',
    'sigmoid',
    'linear',
    'leakyRelu',
    'elu',
    'swish',
    'softplus',
];
const INITIALIZATIONS: ReadonlyArray<{ value: WeightInitType; label: string }> = [
    { value: 'xavier', label: 'Xavier' },
    { value: 'he', label: 'He' },
    { value: 'uniform', label: 'Uniform' },
    { value: 'zeros', label: 'Zeros' },
];
const MIN_NEURONS_PER_LAYER = 1;
const MAX_NEURONS_PER_LAYER = 16;
const DEFAULT_NEW_LAYER_WIDTH = 4;

function NeuronCountControl({
    layer,
    value,
    onCommit,
}: {
    layer: number;
    value: number;
    onCommit: (nextValue: number) => Promise<boolean>;
}) {
    const [draft, setDraft] = useState(String(value));

    useEffect(() => {
        setDraft(String(value));
    }, [value]);

    const commitDraft = async () => {
        if (draft.trim() === '') {
            setDraft(String(value));
            return;
        }
        const nextValue = Number(draft);
        if (nextValue === value) return;
        await onCommit(nextValue);
    };

    return (
        <div className="neuron-stepper" role="group" aria-label={`Layer ${layer} neuron count`}>
            <button
                type="button"
                className="forge-stepper__btn neuron-stepper__btn"
                onClick={() => void onCommit(value - 1)}
                disabled={value <= MIN_NEURONS_PER_LAYER}
                aria-label={`Decrease neurons in layer ${layer}`}
            >
                −
            </button>
            <input
                className="neuron-stepper__input"
                type="number"
                min={MIN_NEURONS_PER_LAYER}
                max={MAX_NEURONS_PER_LAYER}
                step={1}
                inputMode="numeric"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onBlur={() => void commitDraft()}
                onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                        event.currentTarget.blur();
                    }
                }}
                aria-label={`Neuron count for layer ${layer}`}
            />
            <button
                type="button"
                className="forge-stepper__btn neuron-stepper__btn"
                onClick={() => void onCommit(value + 1)}
                disabled={value >= MAX_NEURONS_PER_LAYER}
                aria-label={`Increase neurons in layer ${layer}`}
            >
                +
            </button>
        </div>
    );
}

export const NetworkConfigPanel = memo(function NetworkConfigPanel() {
    const prepared = usePlaygroundStore((state) => state.access.status === 'ready'
        ? state.access.prepared
        : null);
    const isLoading = useTrainingStore((state) => state.networkConfigLoading);
    const configError = useTrainingStore((state) => state.configError);
    const configErrorSource = useTrainingStore((state) => state.configErrorSource);

    if (!prepared) {
        return (
            <div className="config-feedback config-feedback--error" role="alert">
                No validated experiment recipe is available.
            </div>
        );
    }

    const { model } = prepared.document.recipe;
    const { hiddenLayers, hiddenActivation, initialization } = model;
    const retryNetworkChange = () => useTrainingStore.getState().retryConfigSync();
    const commit = (edit: Parameters<typeof commitRecipeEdit>[1]) => (
        commitRecipeEdit('network', edit)
    );

    return (
        <div aria-busy={isLoading}>
            <LoadingState isLoading={isLoading} inline announce={false} message="Initializing network..." />
            {configError && configErrorSource === 'network' && (
                <div className="config-feedback config-feedback--error">
                    <span>{configError}</span>
                    <button type="button" className="btn btn--ghost btn--sm" onClick={retryNetworkChange}>
                        Retry
                    </button>
                </div>
            )}

            <div className="control-row" aria-label="Derived network dimensions">
                <span className="control-label">Task-derived shape</span>
                <span>{prepared.compiled.network.inputSize} inputs</span>
                <span>{prepared.compiled.task.outputSize} {prepared.compiled.task.outputSize === 1 ? 'output' : 'outputs'}</span>
                <span>{prepared.compiled.task.outputActivation}</span>
            </div>

            <div className="control-row" style={{ marginBottom: 8 }}>
                <span className="control-label">Hidden Layers</span>
                <div className="layer-controls">
                    <Tooltip content="Cause: removing a hidden layer lowers model capacity. Effect: the boundary becomes simpler and may underfit curved data.">
                        <button
                            type="button"
                            className="forge-stepper__btn"
                            onClick={() => void commit((recipe) => (
                                setHiddenLayers(recipe, recipe.model.hiddenLayers.slice(0, -1))
                            ))}
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
                            onClick={() => void commit((recipe) => setHiddenLayers(
                                recipe,
                                [...recipe.model.hiddenLayers, DEFAULT_NEW_LAYER_WIDTH],
                            ))}
                            disabled={hiddenLayers.length >= MAX_HIDDEN_LAYERS}
                            aria-label="Add hidden layer"
                        >
                            +
                        </button>
                    </Tooltip>
                </div>
            </div>

            {hiddenLayers.length > 0 && (
                <div className="forge-section__label" style={{ marginTop: 8, marginBottom: 6 }}>
                    Neurons per layer
                </div>
            )}
            {hiddenLayers.map((count, index) => (
                <div key={index} className="neuron-row">
                    <span className="control-label" style={{ minWidth: 60 }}>Layer {index + 1}</span>
                    <Tooltip content={`Cause: layer ${index + 1} has ${count} neurons to detect intermediate patterns. Effect: more neurons can model finer bends, but too many can overfit noisy samples.`}>
                        <NeuronCountControl
                            layer={index + 1}
                            value={count}
                            onCommit={(width) => commit((recipe) => (
                                setHiddenLayerWidth(recipe, index, width)
                            ))}
                        />
                    </Tooltip>
                </div>
            ))}

            <div className="control-row" style={{ marginTop: 8 }}>
                <span className="control-label">Hidden activation</span>
                <Tooltip content="Cause: activation functions decide when neurons pass signal forward. Effect: tanh/sigmoid smooth the boundary, while ReLU-family choices make sharper bends.">
                    <select
                        className="select"
                        aria-label="Hidden activation"
                        value={hiddenActivation}
                        onChange={(event) => void commit((recipe) => (
                            setHiddenActivation(recipe, event.target.value as ScalarActivationType)
                        ))}
                    >
                        {HIDDEN_ACTIVATIONS.map((activation) => (
                            <option key={activation} value={activation}>
                                {ACTIVATION_LABELS[activation]}
                            </option>
                        ))}
                    </select>
                </Tooltip>
            </div>

            <div className="control-row">
                <span className="control-label">Weight init</span>
                <Tooltip content="Cause: initialization sets the model's starting weights. Effect: a suitable scale keeps early gradients useful and training stable.">
                    <select
                        className="select"
                        aria-label="Weight initialization"
                        value={initialization}
                        onChange={(event) => void commit((recipe) => (
                            setInitialization(recipe, event.target.value as WeightInitType)
                        ))}
                    >
                        {INITIALIZATIONS.map(({ value, label }) => (
                            <option key={value} value={value}>{label}</option>
                        ))}
                    </select>
                </Tooltip>
            </div>
        </div>
    );
});
