// ── Code Export Generators ──
// Generate code representations of the current network for learning purposes.

import type { NetworkConfig, TrainingConfig, FeatureFlags, NetworkSnapshot } from '@nn-playground/engine';

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
    };
    return map[act] || act;
}

/**
 * Generate pseudocode description of the network.
 */
export function generatePseudocode(
    config: NetworkConfig,
    training: TrainingConfig,
    features: FeatureFlags,
    snapshot: NetworkSnapshot | null,
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
    code += `  prediction = ${activationStr(config.outputActivation)}(bias + SUM(w[j] * hidden[j]))\n\n`;

    code += `TRAINING:\n`;
    code += `  loss = ${training.lossType === 'crossEntropy' ? 'Cross-Entropy' : training.lossType === 'mse' ? 'MSE' : 'Huber'}\n`;
    code += `  optimizer = ${training.optimizer === 'sgd' ? 'SGD' : training.optimizer === 'sgdMomentum' ? 'SGD+Momentum' : 'Adam'}\n`;
    code += `  learning_rate = ${training.learningRate}\n`;
    code += `  batch_size = ${training.batchSize}\n`;

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

/**
 * Generate NumPy-compatible Python code.
 */
export function generateNumPy(
    config: NetworkConfig,
    _training: TrainingConfig,
    features: FeatureFlags,
    snapshot: NetworkSnapshot | null,
): string {
    const feats = getFeatureList(features);
    const layers = [config.inputSize, ...config.hiddenLayers, config.outputSize];
    const act = config.activation;
    const outAct = config.outputActivation;

    let code = `import numpy as np\n\n`;
    code += `# Neural Network: ${layers.join(' → ')}\n`;
    code += `# Features: [${feats.join(', ')}]\n\n`;

    // Activation function
    code += `def ${act}(x):\n`;
    switch (act) {
        case 'relu': code += `    return np.maximum(0, x)\n\n`; break;
        case 'tanh': code += `    return np.tanh(x)\n\n`; break;
        case 'sigmoid': code += `    return 1 / (1 + np.exp(-x))\n\n`; break;
        case 'leakyRelu': code += `    return np.where(x > 0, x, 0.01 * x)\n\n`; break;
        case 'elu': code += `    return np.where(x > 0, x, np.exp(x) - 1)\n\n`; break;
        case 'swish': code += `    return x / (1 + np.exp(-x))\n\n`; break;
        case 'softplus': code += `    return np.log(1 + np.exp(x))\n\n`; break;
        default: code += `    return x  # linear\n\n`;
    }

    if (outAct !== act && outAct !== 'linear') {
        code += `def ${outAct}(x):\n`;
        code += `    return 1 / (1 + np.exp(-x))  # sigmoid\n\n`;
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
    snapshot: NetworkSnapshot | null,
): string {
    const feats = getFeatureList(features);
    const layers = [config.inputSize, ...config.hiddenLayers, config.outputSize];
    const act = config.activation;
    const outAct = config.outputActivation;

    let code = `import * as tf from '@tensorflow/tfjs';\n\n`;
    code += `// Neural Network: ${layers.join(' → ')}\n`;
    code += `// Features: [${feats.join(', ')}]\n\n`;
    code += `const model = tf.sequential();\n\n`;

    for (let l = 0; l < config.hiddenLayers.length; l++) {
        const inputShape = l === 0 ? `, inputShape: [${config.inputSize}]` : '';
        code += `model.add(tf.layers.dense({\n`;
        code += `  units: ${config.hiddenLayers[l]},\n`;
        code += `  activation: '${act}'${inputShape},\n`;
        code += `}));\n\n`;
    }

    // Output layer
    const outputInputShape = config.hiddenLayers.length === 0 ? `, inputShape: [${config.inputSize}]` : '';
    code += `model.add(tf.layers.dense({\n`;
    code += `  units: ${config.outputSize},\n`;
    code += `  activation: '${outAct}'${outputInputShape},\n`;
    code += `}));\n\n`;

    // Compile
    const lossMap: Record<string, string> = {
        mse: 'meanSquaredError',
        crossEntropy: 'binaryCrossentropy',
        huber: 'huberLoss',
    };
    const optMap: Record<string, string> = {
        sgd: `tf.train.sgd(${training.learningRate})`,
        sgdMomentum: `tf.train.momentum(${training.learningRate}, 0.9)`,
        adam: `tf.train.adam(${training.learningRate})`,
    };

    code += `model.compile({\n`;
    code += `  optimizer: ${optMap[training.optimizer] || 'tf.train.sgd(0.03)'},\n`;
    code += `  loss: '${lossMap[training.lossType] || 'meanSquaredError'}',\n`;
    code += `  metrics: ['accuracy'],\n`;
    code += `});\n\n`;

    code += `// Training:\n`;
    code += `// await model.fit(xTrain, yTrain, {\n`;
    code += `//   epochs: 100,\n`;
    code += `//   batchSize: ${training.batchSize},\n`;
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
