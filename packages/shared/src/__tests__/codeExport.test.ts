import { describe, expect, it } from "vitest";
import {
  generatePseudocode,
  generateNumPy,
  generateTFJS,
  DEFAULT_NETWORK,
  DEFAULT_TRAINING,
  DEFAULT_FEATURES,
} from "../index.js";
import type {
  NetworkSnapshot,
  NetworkConfig,
  FeatureFlags,
} from "@nn-playground/engine";

const mockConfig: NetworkConfig = {
  ...DEFAULT_NETWORK,
  inputSize: 2,
  hiddenLayers: [2], // 2 -> 2 -> 1
};

const mockSnapshot: NetworkSnapshot = {
  step: 100,
  epoch: 1,
  weights: [
    [
      [0.1, 0.2],
      [0.3, 0.4],
    ], // Layer 1: 2 neurons, 2 inputs
    [[0.5, 0.6]], // Layer 2: 1 neuron, 2 inputs
  ],
  biases: [[0.1, 0.2], [0.3]],
  trainLoss: 0.1,
  testLoss: 0.12,
  trainMetrics: { loss: 0.1, accuracy: 0.9 },
  testMetrics: { loss: 0.12, accuracy: 0.88 },
  outputGrid: [],
  gridSize: 0,
  historyPoint: { step: 100, trainLoss: 0.1, testLoss: 0.12 },
};

describe("codeExport", () => {
  describe("generatePseudocode", () => {
    it("generates pseudocode without snapshot", () => {
      const output = generatePseudocode(
        mockConfig,
        DEFAULT_TRAINING,
        DEFAULT_FEATURES,
        null,
      );
      expect(output).toContain("Neural Network — Pseudocode");
      expect(output).toContain("Architecture: 2 → 2 → 1");
      expect(output).toContain("INPUT features = [x, y]");
      expect(output).not.toContain("Trained weights");
    });

    it("generates pseudocode with snapshot", () => {
      const output = generatePseudocode(
        mockConfig,
        DEFAULT_TRAINING,
        DEFAULT_FEATURES,
        mockSnapshot,
      );
      expect(output).toContain("Trained weights (step 100)");
      expect(output).toContain(
        "neuron 0: bias=0.1000  weights=[0.1000, 0.2000]",
      );
    });
  });

  describe("generateNumPy", () => {
    it("generates NumPy code without snapshot", () => {
      const output = generateNumPy(
        mockConfig,
        DEFAULT_TRAINING,
        DEFAULT_FEATURES,
        null,
      );
      expect(output).toContain("import numpy as np");
      expect(output).toContain("def tanh(x):");
      expect(output).toContain("Train the model first");
    });

    it("generates NumPy code with snapshot", () => {
      const output = generateNumPy(
        mockConfig,
        DEFAULT_TRAINING,
        DEFAULT_FEATURES,
        mockSnapshot,
      );
      expect(output).toContain("W1 = np.array([");
      expect(output).toContain("[0.100000, 0.200000]");
      expect(output).toContain("def predict(x):");
    });

    it("includes correct definition for relu activation", () => {
      const config = { ...mockConfig, activation: "relu" as const };
      const output = generateNumPy(
        config,
        DEFAULT_TRAINING,
        DEFAULT_FEATURES,
        null,
      );
      expect(output).toContain("def relu(x):");
      expect(output).toContain("return np.maximum(0, x)");
    });

    it("includes correct definition for sigmoid output activation when hidden is different", () => {
      const config = {
        ...mockConfig,
        activation: "relu" as const,
        outputActivation: "sigmoid" as const,
      };
      const output = generateNumPy(
        config,
        DEFAULT_TRAINING,
        DEFAULT_FEATURES,
        null,
      );
      expect(output).toContain("def relu(x):");
      expect(output).toContain("def sigmoid(x):");
      expect(output).toContain("return 1 / (1 + np.exp(-x))  # sigmoid");
    });

    it("does not duplicate activation definition if output and hidden match", () => {
      const config = {
        ...mockConfig,
        activation: "sigmoid" as const,
        outputActivation: "sigmoid" as const,
      };
      const output = generateNumPy(
        config,
        DEFAULT_TRAINING,
        DEFAULT_FEATURES,
        null,
      );
      expect(output).toContain("def sigmoid(x):");
      // 'def sigmoid(x):' should appear exactly once
      expect(output.match(/def sigmoid\(x\):/g)?.length).toBe(1);
    });

    it("handles linear output activation gracefully", () => {
      const config = {
        ...mockConfig,
        activation: "relu" as const,
        outputActivation: "linear" as const,
      };
      const output = generateNumPy(
        config,
        DEFAULT_TRAINING,
        DEFAULT_FEATURES,
        mockSnapshot,
      );
      // It should define relu
      expect(output).toContain("def relu(x):");
      // It should NOT define linear as a standalone func since we don't define linear explicitly for outAct if it's 'linear' unless it's main act
      // Actually let's check what it does for the prediction code
      // The code has: const actFn = isOutput ? (outAct === 'linear' ? '' : outAct) : act;
      expect(output).toContain("h = W2 @ h + b2\n"); // No activation applied
    });

    it("handles 0 hidden layers correctly", () => {
      const config = { ...mockConfig, hiddenLayers: [] };
      const snapshot: NetworkSnapshot = {
        ...mockSnapshot,
        weights: [[[0.1, 0.2]]], // Layer 1 (Output): 1 neuron, 2 inputs
        biases: [[0.3]],
      };
      const output = generateNumPy(
        config,
        DEFAULT_TRAINING,
        DEFAULT_FEATURES,
        snapshot,
      );

      // Check architecture string
      expect(output).toContain("Neural Network: 2 → 1");

      // Should apply output activation directly
      expect(output).toContain("h = sigmoid(W1 @ h + b1)");
    });

    it("formats weight matrices with 6 decimal places", () => {
      const output = generateNumPy(
        mockConfig,
        DEFAULT_TRAINING,
        DEFAULT_FEATURES,
        mockSnapshot,
      );
      expect(output).toContain(
        "W1 = np.array([\n    [0.100000, 0.200000],\n    [0.300000, 0.400000],\n])",
      );
      expect(output).toContain("b1 = np.array([0.100000, 0.200000])");
    });

    it("maps feature flags to the correct feature string", () => {
      const features: FeatureFlags = {
        x: true,
        y: true,
        xSquared: true,
        ySquared: false,
        xy: true,
        sinX: false,
        sinY: false,
        cosX: false,
        cosY: false,
      };
      const output = generateNumPy(
        mockConfig,
        DEFAULT_TRAINING,
        features,
        null,
      );
      expect(output).toContain("Features: [x, y, x², x·y]");
    });
  });

  describe("generateTFJS", () => {
    it("generates TFJS code without snapshot", () => {
      const output = generateTFJS(
        mockConfig,
        DEFAULT_TRAINING,
        DEFAULT_FEATURES,
        null,
      );
      expect(output).toContain("import * as tf from '@tensorflow/tfjs'");
      expect(output).toContain("model.add(tf.layers.dense");
      expect(output).toContain("loss: 'binaryCrossentropy'");
      expect(output).not.toContain("Load trained weights");
    });

    it("generates TFJS code with snapshot", () => {
      const output = generateTFJS(
        mockConfig,
        DEFAULT_TRAINING,
        DEFAULT_FEATURES,
        mockSnapshot,
      );
      expect(output).toContain("Load trained weights");
      expect(output).toContain("Layer 1: 2×2, bias: 2");
    });
  });
});
