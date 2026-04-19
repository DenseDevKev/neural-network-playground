// ── Finite-difference gradient check ──
// For each valid (loss, output activation) combination, verify that the
// analytic gradients produced by backward() agree with a numerical
// ∂L/∂θ ≈ (L(θ+ε) − L(θ−ε)) / (2ε) estimate for every weight and bias in the
// network. This locks in the correctness of the core autodiff against any
// future regression in activations, losses, or backprop bookkeeping.

import { describe, it, expect } from 'vitest';
import { Network } from '../network.js';
import { getLoss } from '../losses.js';
import type { LossType, ActivationType, NetworkConfig } from '../types.js';

interface Combo {
    loss: LossType;
    outputActivation: ActivationType;
    hiddenActivation: ActivationType;
}

// Representative cross-section. Cross-entropy is only valid with sigmoid
// output; MSE/Huber are exercised against one unbounded and one squashing
// activation each. Hidden-activation variety guards different derivative
// formulas in backward's chain rule.
const COMBOS: Combo[] = [
    { loss: 'crossEntropy', outputActivation: 'sigmoid', hiddenActivation: 'relu' },
    { loss: 'crossEntropy', outputActivation: 'sigmoid', hiddenActivation: 'tanh' },
    { loss: 'mse', outputActivation: 'linear', hiddenActivation: 'relu' },
    { loss: 'mse', outputActivation: 'tanh', hiddenActivation: 'swish' },
    { loss: 'huber', outputActivation: 'linear', hiddenActivation: 'leakyRelu' },
    { loss: 'huber', outputActivation: 'tanh', hiddenActivation: 'elu' },
];

function makeConfig(combo: Combo): NetworkConfig {
    return {
        inputSize: 2,
        hiddenLayers: [3],
        outputSize: 1,
        activation: combo.hiddenActivation,
        outputActivation: combo.outputActivation,
        weightInit: 'xavier',
        seed: 7,
    };
}

function sampleLoss(net: Network, input: number[], target: number[], lossType: LossType): number {
    const fn = getLoss(lossType);
    const out = net.forward(input);
    let s = 0;
    for (let i = 0; i < out.length; i++) s += fn.loss(out[i], target[i]);
    return s;
}

describe('backward gradient check (finite differences)', () => {
    const input = [0.4, -0.25];

    for (const combo of COMBOS) {
        it(`matches numerical gradients for ${combo.loss} / output=${combo.outputActivation} / hidden=${combo.hiddenActivation}`, () => {
            const net = new Network(makeConfig(combo));
            // Target inside the activation's support so gradients are informative.
            const target = combo.outputActivation === 'sigmoid' ? [0.8]
                : combo.outputActivation === 'tanh' ? [0.3]
                    : [1.2];

            // Analytic gradients — run backward once, snapshot the
            // accumulated grads. backward() accumulates, so re-running per
            // parameter would double-count. We read the grads through the
            // network's public accessors, which materialise nested views of
            // the packed Float32Array storage.
            net.forward(input);
            net.backward(target, combo.loss);
            const analyticWeightGrads = net.getWeightGrads();
            const analyticBiasGrads = net.getBiasGrads();

            const weights = net.getWeights();
            const biases = net.getBiases();

            const eps = 1e-5;
            const tol = 1e-3;

            // Weights — perturb via setWeight so the change is reflected in
            // the packed buffer used by forward(); restore after sampling.
            for (let l = 0; l < weights.length; l++) {
                for (let n = 0; n < weights[l].length; n++) {
                    for (let w = 0; w < weights[l][n].length; w++) {
                        const original = weights[l][n][w];

                        net.setWeight(l, n, w, original + eps);
                        const lPlus = sampleLoss(net, input, target, combo.loss);

                        net.setWeight(l, n, w, original - eps);
                        const lMinus = sampleLoss(net, input, target, combo.loss);

                        net.setWeight(l, n, w, original);

                        const numerical = (lPlus - lMinus) / (2 * eps);
                        const analytic = analyticWeightGrads[l][n][w];

                        expect(
                            Math.abs(analytic - numerical),
                            `w[${l}][${n}][${w}] analytic=${analytic} numerical=${numerical}`,
                        ).toBeLessThan(tol);
                    }
                }
            }

            // Biases — same pattern via setBias.
            for (let l = 0; l < biases.length; l++) {
                for (let n = 0; n < biases[l].length; n++) {
                    const original = biases[l][n];

                    net.setBias(l, n, original + eps);
                    const lPlus = sampleLoss(net, input, target, combo.loss);

                    net.setBias(l, n, original - eps);
                    const lMinus = sampleLoss(net, input, target, combo.loss);

                    net.setBias(l, n, original);

                    const numerical = (lPlus - lMinus) / (2 * eps);
                    const analytic = analyticBiasGrads[l][n];

                    expect(
                        Math.abs(analytic - numerical),
                        `b[${l}][${n}] analytic=${analytic} numerical=${numerical}`,
                    ).toBeLessThan(tol);
                }
            }
        });
    }
});
