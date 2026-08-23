// ── Code Export Generators ──
// Generate code representations of the current network for learning purposes.

import type { NetworkConfig, TrainingConfig, FeatureFlags } from '@nn-playground/engine';

export interface ExportedModelParameters {
    readonly step: number;
    readonly weights: readonly (readonly (readonly number[])[])[];
    readonly biases: readonly (readonly number[])[];
}

/**
 * Names of feature transforms for code generation.
 */
function getFeatureList(features: FeatureFlags): string[] {
    const list: string[] = [];
    if (features.x) list.push('x');
    if (features.y) list.push('y');
    if (features.xSquared) list.push('x²');
    if (features.ySquared) list.push('y²');
    if (features.xy) list.push('x·y');
    if (features.sinX) list.push('sin(x)');
    if (features.sinY) list.push('sin(y)');
    if (features.cosX) list.push('cos(x)');
    if (features.cosY) list.push('cos(y)');
    return list;
}

function activationStr(act: string): string {
    const map: Record<string, string> = {
        relu: 'ReLU',
        tanh: 'Tanh',
        sigmoid: 'Sigmoid',
        linear: 'Linear',
        leakyRelu: 'LeakyReLU',
        elu: 'ELU',
        swish: 'Swish',
        softplus: 'Softplus',
        softmax: 'Softmax',
    };
    return map[act] || act;
}

function lossStr(loss: TrainingConfig['lossType']): string {
    const map: Record<TrainingConfig['lossType'], string> = {
        mse: 'MSE',
        crossEntropy: 'Cross-Entropy',
        categoricalCrossEntropy: 'Categorical Cross-Entropy',
        huber: 'Huber',
    };
    return map[loss];
}

function lrScheduleStr(training: TrainingConfig): string | null {
    const schedule = training.lrSchedule;
    if (!schedule || schedule.type === 'constant') return null;
    if (schedule.type === 'step') {
        return `step(step_size=${schedule.stepSize}, gamma=${schedule.gamma})`;
    }
    return `cosine(total_steps=${schedule.totalSteps}, min_lr=${schedule.minLr})`;
}

/** Human-readable regularization summary; null when the objective is unpenalized. */
function regularizationSummary(training: TrainingConfig): string | null {
    if (training.regularization === 'none' || !(training.regularizationRate > 0)) return null;
    return training.regularization === 'l2'
        ? `L2(rate=${training.regularizationRate})`
        : `L1(rate=${training.regularizationRate})`;
}

/**
 * TensorFlow.js kernel regularizer reproducing the engine's weight penalty
 * (L1: rate·Σ|w|, L2: rate·Σw²/2, applied to kernels only); null when off.
 */
function tfjsKernelRegularizer(training: TrainingConfig): string | null {
    if (training.regularization === 'none' || !(training.regularizationRate > 0)) return null;
    return training.regularization === 'l2'
        ? `tf.regularizers.l2({ l2: ${training.regularizationRate} })`
        : `tf.regularizers.l1({ l1: ${training.regularizationRate} })`;
}

/**
 * Emit one dense layer (plus its activation when TensorFlow.js has no matching
 * built-in identifier, i.e. LeakyReLU must be a separate layer).
 */
function appendTfjsDenseLayer(
    code: string,
    opts: {
        units: number;
        activation: string;
        inputShape: string;
        kernelRegularizer: string | null;
    },
): string {
    const regularizerLine = opts.kernelRegularizer
        ? `  kernelRegularizer: ${opts.kernelRegularizer},\n`
        : '';
    if (opts.activation === 'leakyRelu') {
        code += `model.add(tf.layers.dense({\n`;
        code += `  units: ${opts.units},\n`;
        code += `  activation: 'linear',${opts.inputShape}\n`;
        code += regularizerLine;
        code += `}));\n`;
        code += `model.add(tf.layers.leakyReLU({ alpha: 0.01 }));\n\n`;
    } else {
        code += `model.add(tf.layers.dense({\n`;
        code += `  units: ${opts.units},\n`;
        code += `  activation: '${opts.activation}',${opts.inputShape}\n`;
        code += regularizerLine;
        code += `}));\n\n`;
    }
    return code;
}

/**
 * Generate pseudocode description of the network.
 */
export function generatePseudocode(
    config: NetworkConfig,
    training: TrainingConfig,
    features: FeatureFlags,
    snapshot: ExportedModelParameters | null,
): string {
    const feats = getFeatureList(features);
    const layers = [config.inputSize, ...config.hiddenLayers, config.outputSize];

    let code = `# Neural Network — Pseudocode\n`;
    code += `# Architecture: ${layers.join(' → ')}\n\n`;
    code += `INPUT features = [${feats.join(', ')}]  # ${feats.length} features\n\n`;

    for (let l = 0; l < config.hiddenLayers.length; l++) {
        code += `LAYER hidden_${l + 1}:\n`;
        code += `  neurons = ${config.hiddenLayers[l]}\n`;
        code += `  activation = ${activationStr(config.activation)}\n`;
        code += `  FOR each neuron i:\n`;
        code += `    z[i] = bias[i] + SUM(w[i][j] * input[j] for j in prev_layer)\n`;
        code += `    output[i] = ${activationStr(config.activation)}(z[i])\n\n`;
    }

    code += `LAYER output:\n`;
    code += `  neurons = ${config.outputSize}\n`;
    code += `  activation = ${activationStr(config.outputActivation)}\n`;
    if (config.outputSize > 1) {
        code += `  FOR each class c:\n`;
        code += `    logits[c] = bias[c] + SUM(w[c][j] * hidden[j] for j in prev_layer)\n`;
        code += `  prediction = ${activationStr(config.outputActivation)}(logits)\n\n`;
    } else {
        code += `  prediction = ${activationStr(config.outputActivation)}(bias + SUM(w[j] * hidden[j]))\n\n`;
    }

    code += `TRAINING:\n`;
    code += `  loss = ${lossStr(training.lossType)}\n`;
    const regularization = regularizationSummary(training);
    if (regularization) code += `  regularization = ${regularization}\n`;
    code += `  optimizer = ${training.optimizer === 'sgd' ? 'SGD' : training.optimizer === 'sgdMomentum' ? 'SGD+Momentum' : 'Adam'}\n`;
    code += `  learning_rate = ${training.learningRate}\n`;
    code += `  batch_size = ${training.batchSize}\n`;
    if (training.optimizer === 'sgdMomentum') {
        code += `  momentum = ${training.momentum}\n`;
    }
    code += `  gradient_clip = ${training.gradientClip ?? 'off'}\n`;
    if (training.optimizer === 'adam') {
        code += `  adam_beta1 = ${training.adamBeta1 ?? 0.9}\n`;
        code += `  adam_beta2 = ${training.adamBeta2 ?? 0.999}\n`;
    }
    if (training.lossType === 'huber') {
        code += `  huber_delta = ${training.huberDelta ?? 1}\n`;
    }
    const schedule = lrScheduleStr(training);
    if (schedule) code += `  lr_schedule = ${schedule}\n`;

    if (snapshot) {
        code += `\n# Trained weights (step ${snapshot.step}):\n`;
        for (let l = 0; l < snapshot.weights.length; l++) {
            code += `# Layer ${l + 1}: ${snapshot.weights[l].length} neurons × ${snapshot.weights[l][0]?.length ?? 0} inputs\n`;
            for (let n = 0; n < snapshot.weights[l].length; n++) {
                const w = snapshot.weights[l][n].map((v) => v.toFixed(4)).join(', ');
                const b = snapshot.biases[l][n].toFixed(4);
                code += `#   neuron ${n}: bias=${b}  weights=[${w}]\n`;
            }
        }
    }

    return code;
}

function numpyActivationDefinition(act: string): string {
    switch (act) {
        case 'relu': return `def relu(x):\n    return np.maximum(0, x)\n\n`;
        case 'tanh': return `def tanh(x):\n    return np.tanh(x)\n\n`;
        case 'sigmoid': return `def sigmoid(x):\n    return 1 / (1 + np.exp(-x))  # sigmoid\n\n`;
        case 'leakyRelu': return `def leakyRelu(x):\n    return np.where(x > 0, x, 0.01 * x)\n\n`;
        case 'elu': return `def elu(x):\n    return np.where(x > 0, x, np.exp(x) - 1)\n\n`;
        case 'swish': return `def swish(x):\n    return x / (1 + np.exp(-x))\n\n`;
        case 'softplus': return `def softplus(x):\n    return np.log(1 + np.exp(x))\n\n`;
        case 'softmax': return `def softmax(x):\n    e = np.exp(x - np.max(x))\n    return e / np.sum(e)\n\n`;
        default: return `def ${act}(x):\n    return x  # linear\n\n`;
    }
}

/**
 * Generate NumPy-compatible Python code.
 */
export function generateNumPy(
    config: NetworkConfig,
    _training: TrainingConfig,
    features: FeatureFlags,
    snapshot: ExportedModelParameters | null,
): string {
    const feats = getFeatureList(features);
    const layers = [config.inputSize, ...config.hiddenLayers, config.outputSize];
    const act = config.activation;
    const outAct = config.outputActivation;

    let code = `import numpy as np\n\n`;
    code += `# Neural Network: ${layers.join(' → ')}\n`;
    code += `# Features: [${feats.join(', ')}]\n\n`;

    // Activation function
    code += numpyActivationDefinition(act);

    if (outAct !== act && outAct !== 'linear') {
        code += numpyActivationDefinition(outAct);
    }

    if (snapshot) {
        // Output actual weights
        code += `# Trained weights\n`;
        for (let l = 0; l < snapshot.weights.length; l++) {
            const w = snapshot.weights[l];
            const b = snapshot.biases[l];
            code += `W${l + 1} = np.array([\n`;
            for (const row of w) {
                code += `    [${row.map((v) => v.toFixed(6)).join(', ')}],\n`;
            }
            code += `])\n`;
            code += `b${l + 1} = np.array([${b.map((v) => v.toFixed(6)).join(', ')}])\n\n`;
        }

        code += `def predict(x):\n`;
        code += `    """Forward pass through the network."""\n`;
        code += `    h = x\n`;
        for (let l = 0; l < snapshot.weights.length; l++) {
            const isOutput = l === snapshot.weights.length - 1;
            const actFn = isOutput ? (outAct === 'linear' ? '' : outAct) : act;
            if (actFn) {
                code += `    h = ${actFn}(W${l + 1} @ h + b${l + 1})\n`;
            } else {
                code += `    h = W${l + 1} @ h + b${l + 1}\n`;
            }
        }
        code += `    return h\n`;
    } else {
        code += `# Train the model first to generate weights\n`;
    }

    return code;
}

/**
 * Generate TensorFlow.js code.
 */
export function generateTFJS(
    config: NetworkConfig,
    training: TrainingConfig,
    features: FeatureFlags,
    snapshot: ExportedModelParameters | null,
): string {
    const feats = getFeatureList(features);
    const layers = [config.inputSize, ...config.hiddenLayers, config.outputSize];
    const act = config.activation;
    const outAct = config.outputActivation;

    let code = `import * as tf from '@tensorflow/tfjs';\n\n`;
    code += `// Neural Network: ${layers.join(' → ')}\n`;
    code += `// Features: [${feats.join(', ')}]\n\n`;
    code += `const model = tf.sequential();\n\n`;

    const kernelRegularizer = tfjsKernelRegularizer(training);
    for (let l = 0; l < config.hiddenLayers.length; l++) {
        const inputShape = l === 0 ? ` inputShape: [${config.inputSize}],` : '';
        code = appendTfjsDenseLayer(code, {
            units: config.hiddenLayers[l],
            activation: act,
            inputShape,
            kernelRegularizer,
        });
    }

    // Output layer
    const outputInputShape = config.hiddenLayers.length === 0 ? ` inputShape: [${config.inputSize}],` : '';
    code = appendTfjsDenseLayer(code, {
        units: config.outputSize,
        activation: outAct,
        inputShape: outputInputShape,
        kernelRegularizer,
    });

    // Compile
    const lossMap: Record<string, string> = {
        mse: 'meanSquaredError',
        crossEntropy: 'binaryCrossentropy',
        categoricalCrossEntropy: 'categoricalCrossentropy',
        huber: 'huberLoss',
    };
    // TensorFlow.js compiled-loss strings pin Huber's transition point to 1;
    // a custom closure is the only way to honor a recipe-tuned delta while
    // matching the engine objective (0.5·a² below δ, δ·(a−δ/2) above).
    if (training.lossType === 'huber') {
        code += `const huberDelta = ${training.huberDelta ?? 1};\n\n`;
    }
    const optMap: Record<string, string> = {
        sgd: `tf.train.sgd(${training.learningRate})`,
        sgdMomentum: `tf.train.momentum(${training.learningRate}, ${training.momentum})`,
        adam: `tf.train.adam(${training.learningRate}, ${training.adamBeta1 ?? 0.9}, ${training.adamBeta2 ?? 0.999}, ${training.adamEps ?? 1e-8})`,
    };

    code += `model.compile({\n`;
    code += `  optimizer: ${optMap[training.optimizer] || 'tf.train.sgd(0.03)'},\n`;
    if (training.lossType === 'huber') {
        code += `  loss: (yTrue, yPred) => {\n`;
        code += `    const err = yTrue.sub(yPred);\n`;
        code += `    const absErr = err.abs();\n`;
        code += `    return absErr.lessEqual(huberDelta)\n`;
        code += `        .where(err.square().mul(0.5), absErr.mul(huberDelta).sub(0.5 * huberDelta * huberDelta))\n`;
        code += `        .mean();\n`;
        code += `  },\n`;
    } else {
        code += `  loss: '${lossMap[training.lossType] || 'meanSquaredError'}',\n`;
    }
    code += `  metrics: ['accuracy'],\n`;
    code += `});\n\n`;

    code += `// Training:\n`;
    code += `// await model.fit(xTrain, yTrain, {\n`;
    code += `//   epochs: 100,\n`;
    code += `//   batchSize: ${training.batchSize},\n`;
    if (training.gradientClip != null) {
        code += `//   // Gradient clipping threshold: ${training.gradientClip}\n`;
    }
    const schedule = lrScheduleStr(training);
    if (schedule) {
        code += `//   // Learning-rate schedule: ${schedule}\n`;
    }
    code += `// });\n`;

    if (snapshot) {
        code += `\n// Load trained weights:\n`;
        for (let l = 0; l < snapshot.weights.length; l++) {
            const w = snapshot.weights[l];
            const b = snapshot.biases[l];
            code += `// Layer ${l + 1}: ${w.length}×${w[0]?.length ?? 0}, bias: ${b.length}\n`;
        }
        code += `// Use model.setWeights() to load the weight tensors.\n`;
    }

    return code;
}
