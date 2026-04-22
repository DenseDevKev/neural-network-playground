// ── Zustand store — stable configuration state ──
// This store holds ONLY configuration that changes on user interaction.
// Volatile runtime state (snapshot, history, status) lives in useTrainingStore.
import { create } from 'zustand';
import type {
    NetworkConfig,
    TrainingConfig,
    DataConfig,
    FeatureFlags,
    DatasetType,
    ActivationType,
    LossType,
    OptimizerType,
    RegularizationType,
    DataSplit,
} from '@nn-playground/engine';
import {
    countActiveFeatures,
    generateDataset,
    getDefaultProblemType,
    isLossCompatible,
} from '@nn-playground/engine';
import type { UIConfig, AppConfig, VisualizationDemand } from '@nn-playground/shared';
import type { Preset } from '@nn-playground/shared';
import {
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    DEFAULT_DATA,
    DEFAULT_DEMAND,
    MAX_HIDDEN_LAYERS,
    MAX_NEURONS_PER_LAYER,
    encodeUrlState,
    decodeUrlState,
} from '@nn-playground/shared';

/**
 * UI-only renderer feature flags. These do not affect the engine, only how
 * the React UI draws certain visualizations. Each flag controls a runtime
 * pick between an optimized implementation and a known-good fallback so a
 * regression in either renderer can be flipped off without a redeploy.
 */
export interface FeaturesUI {
    /** Use the canvas-based NetworkGraph renderer (AS-5). When false, the
     *  legacy SVG implementation renders. */
    canvasNetworkGraph: boolean;
    /** Use the WebGPU decision-boundary grid predictor (AS-4). When false,
     *  the worker uses the CPU `Network.predictGridInto` path. Capability
     *  detection still gates this — flipping it on does not bypass the
     *  device check. */
    webgpuGrid: boolean;
}

const DEFAULT_FEATURES_UI: FeaturesUI = {
    canvasNetworkGraph: true,
    webgpuGrid: true,
};

export interface PlaygroundStore {
    // ── Config ──
    network: NetworkConfig;
    training: TrainingConfig;
    data: DataConfig;
    features: FeatureFlags;
    ui: UIConfig;
    featuresUI: FeaturesUI;

    // ── Visualization Demand ──
    demand: VisualizationDemand;

    // ── Transient ──
    dataset: DataSplit | null;

    // ── Actions ──
    setDataset: (type: DatasetType) => void;
    setNoise: (noise: number) => void;
    setTrainTestRatio: (ratio: number) => void;
    setNumSamples: (n: number) => void;
    toggleFeature: (feature: keyof FeatureFlags) => void;
    setHiddenLayers: (layers: number[]) => void;
    addLayer: () => void;
    removeLayer: () => void;
    setNeuronsInLayer: (layerIndex: number, count: number) => void;
    setActivation: (act: ActivationType) => void;
    setLearningRate: (lr: number) => void;
    setBatchSize: (bs: number) => void;
    setLossType: (loss: LossType) => void;
    setOptimizer: (opt: OptimizerType) => void;
    setRegularization: (reg: RegularizationType) => void;
    setRegularizationRate: (rate: number) => void;
    setShowTestData: (show: boolean) => void;
    setDiscretize: (d: boolean) => void;
    setDemand: (demand: VisualizationDemand) => void;
    regenerateData: () => void;
    applyPreset: (preset: Preset) => void;
    getConfig: () => AppConfig;
    syncToUrl: () => void;
    loadFromUrl: () => void;
}

function getCompatibleOutputActivation(
    lossType: LossType,
    currentOutputActivation: ActivationType,
): ActivationType {
    if (isLossCompatible(lossType, currentOutputActivation)) {
        return currentOutputActivation;
    }
    return lossType === 'crossEntropy' ? 'sigmoid' : 'linear';
}

function normalizeLossOutputCompatibility(config: AppConfig): AppConfig {
    const outputActivation = getCompatibleOutputActivation(
        config.training.lossType,
        config.network.outputActivation,
    );
    if (outputActivation === config.network.outputActivation) {
        return config;
    }
    return {
        ...config,
        network: {
            ...config.network,
            outputActivation,
        },
    };
}

function buildInitialState() {
    // Try to load from URL hash
    const hash = window.location.hash.slice(1);
    if (hash) {
        try {
            const config = decodeUrlState(hash);
            return normalizeLossOutputCompatibility(config);
        } catch {
            // fall through to defaults
        }
    }

    const inputSize = countActiveFeatures(DEFAULT_FEATURES);
    return {
        network: { ...DEFAULT_NETWORK, inputSize, seed: DEFAULT_DATA.seed } as NetworkConfig,
        training: { ...DEFAULT_TRAINING },
        data: { ...DEFAULT_DATA },
        features: { ...DEFAULT_FEATURES },
        ui: { showTestData: false, discretizeOutput: false } as UIConfig,
    };
}

export const usePlaygroundStore = create<PlaygroundStore>((set, get) => {
    const initial = buildInitialState();

    return {
        // Config
        network: initial.network,
        training: initial.training,
        data: initial.data,
        features: initial.features,
        ui: initial.ui,
        // UI-only renderer toggles. Persisted only in memory — not encoded
        // into the URL hash, since they describe browser capability not
        // shared playground state.
        featuresUI: { ...DEFAULT_FEATURES_UI },

        // Visualization demand
        demand: { ...DEFAULT_DEMAND },

        // Transient
        dataset: null,

        // Actions
        setDataset: (dataset) => {
            const problemType = getDefaultProblemType(dataset);
            const outputActivation = problemType === 'regression' ? 'linear' : 'sigmoid';
            const lossType = problemType === 'regression' ? 'mse' : 'crossEntropy';
            set((s) => ({
                data: { ...s.data, dataset, problemType },
                network: { ...s.network, outputActivation: outputActivation as ActivationType },
                training: { ...s.training, lossType: lossType as LossType },
            }));
        },

        setNoise: (noise) => set((s) => ({ data: { ...s.data, noise } })),
        setTrainTestRatio: (trainTestRatio) => set((s) => ({ data: { ...s.data, trainTestRatio } })),
        setNumSamples: (numSamples) => set((s) => ({ data: { ...s.data, numSamples } })),

        toggleFeature: (feature) => {
            set((s) => {
                const newFeatures = { ...s.features, [feature]: !s.features[feature] };
                const inputSize = countActiveFeatures(newFeatures);
                if (inputSize === 0) return s; // Prevent zero-feature state
                return {
                    features: newFeatures,
                    network: { ...s.network, inputSize },
                };
            });
        },

        setHiddenLayers: (layers) => set((s) => ({
            network: { ...s.network, hiddenLayers: layers },
        })),

        addLayer: () => set((s) => {
            if (s.network.hiddenLayers.length >= MAX_HIDDEN_LAYERS) return s;
            return {
                network: {
                    ...s.network,
                    hiddenLayers: [...s.network.hiddenLayers, 4],
                },
            };
        }),

        removeLayer: () => set((s) => {
            if (s.network.hiddenLayers.length === 0) return s;
            return {
                network: {
                    ...s.network,
                    hiddenLayers: s.network.hiddenLayers.slice(0, -1),
                },
            };
        }),

        setNeuronsInLayer: (idx, count) => set((s) => {
            const clamped = Math.max(1, Math.min(MAX_NEURONS_PER_LAYER, count));
            const layers = [...s.network.hiddenLayers];
            layers[idx] = clamped;
            return { network: { ...s.network, hiddenLayers: layers } };
        }),

        setActivation: (activation) => set((s) => ({
            network: { ...s.network, activation },
        })),

        setLearningRate: (learningRate) => set((s) => ({
            training: { ...s.training, learningRate },
        })),

        setBatchSize: (batchSize) => set((s) => ({
            training: { ...s.training, batchSize },
        })),

        setLossType: (lossType) => set((s) => ({
            training: { ...s.training, lossType },
            network: {
                ...s.network,
                outputActivation: getCompatibleOutputActivation(lossType, s.network.outputActivation),
            },
        })),

        setOptimizer: (optimizer) => set((s) => ({
            training: { ...s.training, optimizer },
        })),

        setRegularization: (regularization) => set((s) => ({
            training: { ...s.training, regularization },
        })),

        setRegularizationRate: (regularizationRate) => set((s) => ({
            training: { ...s.training, regularizationRate },
        })),

        setShowTestData: (showTestData) => set((s) => ({
            ui: { ...s.ui, showTestData },
        })),

        setDiscretize: (discretizeOutput) => set((s) => ({
            ui: { ...s.ui, discretizeOutput },
        })),

        setDemand: (demand) => set({ demand }),

        regenerateData: () => {
            const s = get();
            const ds = generateDataset(
                s.data.dataset,
                s.data.numSamples,
                s.data.noise,
                s.data.trainTestRatio,
                s.data.seed,
            );
            set({ dataset: ds });
        },

        getConfig: () => {
            const s = get();
            return {
                network: s.network,
                training: s.training,
                data: s.data,
                features: s.features,
                ui: s.ui,
            };
        },

        syncToUrl: () => {
            const config = get().getConfig();
            const hash = encodeUrlState(config);
            window.history.replaceState(null, '', '#' + hash);
        },

        loadFromUrl: () => {
            const hash = window.location.hash.slice(1);
            if (!hash) return;
            try {
                const config = decodeUrlState(hash);
                const normalized = normalizeLossOutputCompatibility(config);
                set({
                    network: normalized.network,
                    training: normalized.training,
                    data: normalized.data,
                    features: normalized.features,
                    ui: normalized.ui,
                });
            } catch {
                // ignore invalid hashes
            }
        },

        applyPreset: (preset: Preset) => {
            const c = preset.config;
            const updates: Partial<PlaygroundStore> = {};
            if (c.data) updates.data = { ...get().data, ...c.data };
            if (c.network) updates.network = { ...get().network, ...c.network };
            if (c.features) updates.features = { ...get().features, ...c.features };
            if (c.training) updates.training = { ...get().training, ...c.training };
            if (c.ui) updates.ui = { ...get().ui, ...c.ui };
            const nextTraining = updates.training ?? get().training;
            const nextNetwork = updates.network ?? get().network;
            updates.network = {
                ...nextNetwork,
                outputActivation: getCompatibleOutputActivation(
                    nextTraining.lossType,
                    nextNetwork.outputActivation,
                ),
            };
            set(updates);
            get().syncToUrl();
        },
    };
});
