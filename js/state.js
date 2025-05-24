/**
 * Application state management
 * @module State
 */

/**
 * Application state object
 * @type {Object}
 */
export const appState = {
  model: null,
  training: false,
  intervalId: null,
  epochCount: 0,
  hiddenLayers: 3,
  neuronsPerLayer: 7,
  showTestData: false,
  discretizeOutput: false,
  networkData: null,
  xs: null,
  ys: null,
  xsTest: null,
  ysTest: null,
  rawXs: [],
  rawYs: [],
  rawXsTest: [],
  rawYsTest: [],
  batchSize: 30,
  optimizer: null,
  dataGenerator: null
};

/**
 * Default configuration
 * @type {Object}
 */
export const defaultConfig = {
  hiddenLayers: 3,
  neuronsPerLayer: 7,
  activation: 'relu',
  learningRate: 0.03,
  regularization: 'none',
  regularizationRate: 0,
  ratio: 50,
  noise: 0,
  batchSize: 30,
  dataset: 'circle',
  features: {
    x1: true,
    x2: true,
    x1Squared: false,
    x2Squared: false,
    x1x2: false,
    sinX1: false,
    sinX2: false,
  },
};
