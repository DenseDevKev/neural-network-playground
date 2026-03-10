// ── Zustand store — single source of truth for app state ──
import { create } from 'zustand';
import type {
    NetworkConfig,
    TrainingConfig,
    DataConfig,
    FeatureFlags,
    NetworkSnapshot,
    HistoryPoint,
    DatasetType,
    ActivationType,
    LossType,
    OptimizerType,
    RegularizationType,
    DataPoint,
    DataSplit,
} from '@nn-playground/engine';
import {
    countActiveFeatures,
    generateDataset,
    getDefaultProblemType,
} from '@nn-playground/engine';
import type { UIConfig, TrainingStatus, AppConfig } from '@nn-playground/shared';
import type { Preset } from '@nn-playground/shared';
import {
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    DEFAULT_DATA,
    MAX_HIDDEN_LAYERS,
    MAX_NEURONS_PER_LAYER,
    encodeUrlState,
    decodeUrlState,
} from '@nn-playground/shared';

export interface PlaygroundStore {
    // ── Config ──
    network: NetworkConfig;
    training: TrainingConfig;
    data: DataConfig;
    features: FeatureFlags;
    ui: UIConfig;

    // ── Runtime ──
    status: TrainingStatus;
    snapshot: NetworkSnapshot | null;
    history: HistoryPoint[];
    dataset: DataSplit | null;
    trainPoints: DataPoint[];
    testPoints: DataPoint[];
    /** Steps of training to run per animation frame. Not persisted to URL. */
    stepsPerFrame: number;

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
    setStepsPerFrame: (n: number) => void;
    setStatus: (s: TrainingStatus) => void;
    setSnapshot: (snap: NetworkSnapshot) => void;
    addHistoryPoint: (point: HistoryPoint) => void;
    regenerateData: () => void;
    resetHistory: () => void;
    setTrainPoints: (pts: DataPoint[]) => void;
    setTestPoints: (pts: DataPoint[]) => void;
    applyPreset: (preset: Preset) => void;
    getConfig: () => AppConfig;
    syncToUrl: () => void;
    loadFromUrl: () => void;
}

function buildInitialState() {
    // Try to load from URL hash
    const hash = window.location.hash.slice(1);
    if (hash) {
        try {
            const config = decodeUrlState(hash);
            return config;
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
        ui: { showTestData: false, discretizeOutput: false, animationSpeed: 1 } as UIConfig,
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

        // Runtime
        status: 'idle',
        snapshot: null,
        history: [],
        dataset: null,
        trainPoints: [],
        testPoints: [],
        stepsPerFrame: 5,

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

        setStepsPerFrame: (n) => set({ stepsPerFrame: Math.max(1, Math.min(100, n)) }),

        setStatus: (status) => set({ status }),
        setSnapshot: (snapshot) => set({ snapshot }),
        addHistoryPoint: (point) => set((s) => ({ history: [...s.history, point] })),
        resetHistory: () => set({ history: [], snapshot: null }),
        setTrainPoints: (trainPoints) => set({ trainPoints }),
        setTestPoints: (testPoints) => set({ testPoints }),

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
                set({
                    network: config.network,
                    training: config.training,
                    data: config.data,
                    features: config.features,
                    ui: config.ui,
                });
            } catch {
                // ignore invalid hashes
            }
        },

        applyPreset: (preset: Preset) => {
            const c = preset.config;
            const updates: Partial<PlaygroundStore> = {
                snapshot: null,
                history: [],
                status: 'idle' as TrainingStatus,
            };
            if (c.data) updates.data = { ...get().data, ...c.data };
            if (c.network) updates.network = { ...get().network, ...c.network };
            if (c.features) updates.features = { ...get().features, ...c.features };
            if (c.training) updates.training = { ...get().training, ...c.training };
            if (c.ui) updates.ui = { ...get().ui, ...c.ui };
            set(updates);
            get().syncToUrl();
        },
    };
});
