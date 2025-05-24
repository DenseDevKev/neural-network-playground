/**
 * Main Application Entry Point
 * @module Main
 */

import * as tf from '@tensorflow/tfjs';
import { DataGenerator } from './data-generator.js';
import { nn_vis } from './visualization.js';
import { appState, defaultConfig } from './state.js';
import { initUI, updateEpochCounter, updateLossDisplay, updateLayerCountDisplay } from './ui.js';
import { toggleTraining, trainStep, resetTraining, startTraining, pauseTraining } from './training.js';

// Initialize the data generator
appState.dataGenerator = new DataGenerator();

// Make available globally for debugging
window.app = {
  tf,
  appState,
  toggleTraining,
  trainStep,
  resetTraining
};

// Cache DOM elements
const trainingLossDisplay = document.getElementById('trainingLoss');
const layerCountDisplay = document.getElementById('layerCountDisplay');
const ratioSlider = document.getElementById('ratioSlider');
const ratioValue = document.getElementById('ratioValue');
const noiseSlider = document.getElementById('noiseSlider');
const noiseValue = document.getElementById('noiseValue');
const regenerateBtn = document.getElementById('regenerateBtn');
const showTestCheckbox = document.getElementById('showTestData');
const discretizeCheckbox = document.getElementById('discretizeOutput');
const neuronCountSlider = document.getElementById('neuronCountSlider');
const neuronCountInput = document.getElementById('neuronCountInput');
const activationSelect = document.getElementById('activation');
const learningRateInput = document.getElementById('learningRate');
const regularizationSelect = document.getElementById('regularization');
const regularizationRateInput = document.getElementById('regularizationRate');
const batchSizeInput = document.getElementById('batchSize');
const datasetSelect = document.getElementById('dataset');
const playBtn = document.getElementById('playBtn');
const stepBtn = document.getElementById('stepBtn');
const resetBtn = document.getElementById('resetBtn');
const datasetOptions = document.querySelectorAll('.dataset-option');

// Additional DOM elements
const addLayerBtn = document.getElementById('addLayerBtn');
const removeLayerBtn = document.getElementById('removeLayerBtn');
const activationSelectEl = document.getElementById('activationSelect');
const learningRateSelect = document.getElementById('learningRateSelect');
const regularizationSelectEl = document.getElementById('regularizationSelect');
const regularizationRateSelect = document.getElementById('regularizationRateSelect');
const batchSizeSlider = document.getElementById('batchSizeSlider');
const batchSizeValue = document.getElementById('batchSizeValue');

// Feature checkboxes
const featureCheckboxElements = {
  x1: document.getElementById('feature-x1'),
  x2: document.getElementById('feature-x2'),
  x1Squared: document.getElementById('feature-x1sq'),
  x2Squared: document.getElementById('feature-x2sq'),
  x1x2: document.getElementById('feature-x1x2'),
  sinX1: document.getElementById('feature-sinx1'),
  sinX2: document.getElementById('feature-sinx2')
};

// Error display helpers
function showError(message) {
  const errorDiv = document.getElementById('errorMessage');
  if (errorDiv) {
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
  }
}
function hideError() {
  const errorDiv = document.getElementById('errorMessage');
  if (errorDiv) {
    errorDiv.textContent = '';
    errorDiv.style.display = 'none';
  }
}

/**
 * Generate data and update the model
 */
async function generateData() {
  try {
    hideError();
    // Reset training state
    resetTraining();
    
    // Dispose old tensors before overwriting
    if (appState.xs) { appState.xs.dispose(); appState.xs = null; }
    if (appState.ys) { appState.ys.dispose(); appState.ys = null; }
    if (appState.xsTest) { appState.xsTest.dispose(); appState.xsTest = null; }
    if (appState.ysTest) { appState.ysTest.dispose(); appState.ysTest = null; }
    
    // Generate new data using the data generator
    const data = appState.dataGenerator.generateData();
    
    // Check for no features enabled
    if (appState.dataGenerator.getEnabledFeatureCount() === 0) {
      showError('Please enable at least one input feature.');
      return;
    }
    
    // Update app state with new data
    appState.rawXs = data.rawXs;
    appState.rawYs = data.rawYs;
    appState.rawXsTest = data.rawXsTest;
    appState.rawYsTest = data.rawYsTest;
    
    // Use tensors directly from data generator
    appState.xs = data.xs;
    appState.ys = data.ys;
    appState.xsTest = data.xsTest;
    appState.ysTest = data.ysTest;
    
    // Create a new model with the current configuration
    await createModel();
    
    // Update visualizations
    updateVisualization();
    updateDecisionBoundary();
    
    hideError();
    console.log('Data and model updated successfully');
  } catch (error) {
    showError('An error occurred while generating data or creating the model. See console for details.');
    console.error('Error generating data:', error);
  }
}

/**
 * Create a new model with the current configuration
 */
async function createModel() {
  try {
    // Clean up previous model if it exists
    if (appState.model) {
      appState.model.dispose();
    }
    
    // Check for no features enabled
    if (appState.dataGenerator.getEnabledFeatureCount() === 0) {
      showError('Please enable at least one input feature.');
      return;
    }
    
    // Create a new sequential model
    appState.model = tf.sequential();
    
    // Add input layer
    const inputShape = [appState.dataGenerator.getEnabledFeatureCount()];
    appState.model.add(tf.layers.dense({
      units: appState.neuronsPerLayer,
      inputShape,
      activation: appState.activation,
      kernelInitializer: 'glorotNormal',
      biasInitializer: 'zeros'
    }));
    
    // Add hidden layers
    for (let i = 0; i < appState.hiddenLayers - 1; i++) {
      appState.model.add(tf.layers.dense({
        units: appState.neuronsPerLayer,
        activation: appState.activation,
        kernelInitializer: 'glorotNormal',
        biasInitializer: 'zeros'
      }));
    }
    
    // Add output layer
    appState.model.add(tf.layers.dense({
      units: 1,
      activation: 'sigmoid',
      kernelInitializer: 'glorotNormal',
      biasInitializer: 'zeros'
    }));
    
    // Compile the model
    const optimizer = tf.train.adam(appState.learningRate);
    appState.model.compile({
      optimizer,
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });
    
    hideError();
    console.log('Model created and compiled successfully');
  } catch (error) {
    showError('An error occurred while creating the model. See console for details.');
    console.error('Error creating model:', error);
    throw error;
  }
}

/**
 * Update the visualization
 */
function updateVisualization() {
  if (!appState.model) return;
  
  const featureNames = appState.dataGenerator.getEnabledFeatureNames();
  appState.networkData = nn_vis.visualizeNetwork(
    appState.model,
    featureNames,
    appState.rawXs,
    appState.rawYs,
    appState.rawXsTest,
    appState.rawYsTest,
    appState.showTestData,
    null // transformationPreviews
  );
}

/**
 * Update the decision boundary visualization
 */
function updateDecisionBoundary() {
  if (!appState.model) return;
  
  nn_vis.visualizeDecisionBoundary(
    appState.model,
    appState.dataGenerator,
    appState.rawXs,
    appState.rawYs,
    appState.rawXsTest,
    appState.rawYsTest,
    appState.showTestData,
    appState.discretizeOutput
  );
}

/**
 * Initialize the application
 */
async function init() {
  try {
    // Initialize UI
    initUI();
    
    // Initialize data and model
    await generateData();
    
    // Initial visualization update
    updateVisualization();
    updateDecisionBoundary();
    
    console.log('Application initialized successfully');
  } catch (error) {
    console.error('Error initializing application:', error);
  }
}

// Start the application when the DOM is fully loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    init().then(() => setupAllEventListeners());
  });
} else {
  init().then(() => setupAllEventListeners());
}

// Export public API
export {
  toggleTraining,
  trainStep,
  resetTraining,
  startTraining,
  pauseTraining,
  generateData,
  createModel,
  updateVisualization,
  updateDecisionBoundary
};

  /**
 * Set up all event listeners
 */
function setupAllEventListeners() {
  // Set up event listeners for controls
  setupEventListeners();

  // Set up event listeners for collapse icons
  setupCollapseIcons();

  // Set up neuron count controls
  setupNeuronCountControls();

  // Generate initial data
  generateData();
}

/**
 * Set up event listeners for all UI controls
 */
function setupEventListeners() {
  // Dataset selection
  document.querySelectorAll('.dataset-option').forEach(option => {
    option.addEventListener('click', () => {
      document.querySelectorAll('.dataset-option').forEach(opt => opt.classList.remove('active'));
      option.classList.add('active');
      dataGenerator.setDataset(option.dataset.dataset);
      generateData(); // Regenerate data and model on dataset change
    });
  });

  // Sliders
  document.getElementById('ratioSlider').addEventListener('input', e => {
    const value = e.target.value;
    document.getElementById('ratioValue').textContent = `${value}%`;
    dataGenerator.setTrainTestRatio(value);
    generateData(); // Regenerate data on split change
  });

  document.getElementById('noiseSlider').addEventListener('input', e => {
    const value = e.target.value;
    document.getElementById('noiseValue').textContent = value;
    dataGenerator.setNoise(value);
    generateData(); // Regenerate data on noise change
  });

  document.getElementById('batchSizeSlider').addEventListener('input', e => {
    const value = e.target.value;
    document.getElementById('batchSizeValue').textContent = value;
    appState.batchSize = parseInt(value, 10);
  });

  // Feature checkboxes: enable/disable features and regenerate data
  document.querySelectorAll('.feature-checkboxes input[type="checkbox"]').forEach(cb => {
    cb.addEventListener('change', e => {
      // Map checkbox id to feature key
      const id = e.target.id.replace('feature-', '');
      let featureKey = id;
      if (id === 'x1sq') featureKey = 'x1Squared';
      else if (id === 'x2sq') featureKey = 'x2Squared';
      else if (id === 'x1x2') featureKey = 'x1x2';
      else if (id === 'sinx1') featureKey = 'sinX1';
      else if (id === 'sinx2') featureKey = 'sinX2';
      // x1 and x2 are already correct
      dataGenerator.setFeatureEnabled(featureKey, e.target.checked);
      generateData();
    });
  });

  // Network configuration controls: recreate model on change
  ['activationSelect','learningRateSelect','regularizationSelect','regularizationRateSelect'].forEach(id => {
    document.getElementById(id).addEventListener('change', () => generateData());
  });

  // Discretize output toggle: update decision boundary
  document.getElementById('discretizeOutput').addEventListener('change', e => {
    appState.discretizeOutput = e.target.checked;
    updateDecisionBoundary();
  });

  // Training controls
  document.getElementById('playBtn').addEventListener('click', toggleTraining);
  document.getElementById('stepBtn').addEventListener('click', () => { pauseTraining(); trainStep(appState.batchSize); });
  document.getElementById('resetBtn').addEventListener('click', fullReset);

  document.getElementById('addLayerBtn').addEventListener('click', () => {
    if (appState.hiddenLayers < 6) { // Use appState
      appState.hiddenLayers++; // Use appState
      updateLayerCountDisplay();
      createModel(); // Recreate model
      updateVisualization(); // Update visuals
      updateDecisionBoundary();
    }
  });

  document.getElementById('removeLayerBtn').addEventListener('click', () => {
    if (appState.hiddenLayers > 1) { // Use appState
      appState.hiddenLayers--; // Use appState
      updateLayerCountDisplay();
      createModel(); // Recreate model
      updateVisualization(); // Update visuals
      updateDecisionBoundary();
    }
  });

  // Regenerate data with current settings
  document.getElementById('regenerateBtn').addEventListener('click', () => {
    generateData();
  });

  // Toggle display of test data on decision boundary
  document.getElementById('showTestData').addEventListener('change', e => {
    appState.showTestData = e.target.checked;
    updateDecisionBoundary();
  });
}

/**
 * Set up neuron count controls with synchronized slider and numeric input
 */
function setupNeuronCountControls() {
  const neuronCountSlider = document.getElementById('neuronCountSlider');
  const neuronCountInput = document.getElementById('neuronCountInput');

  if (!neuronCountSlider || !neuronCountInput) return;

  // State reducer for neuron count updates
  const neuronCountReducer = (prev, next) => {
    const clamped = Math.max(1, Math.min(32, next)); // Ensure within bounds
    if (clamped === prev || clamped === appState.neuronsPerLayer) return prev; // No change

    // Preserve weights where possible when architecture changes (best effort)
    const oldWeights = appState.model?.getWeights(); // Use appState
    appState.neuronsPerLayer = clamped; // Use appState
    resetTraining(); // Reset epoch count, etc.
    createModel(); // Create the new model structure

    // Try to restore compatible weights (might fail if layer count changed)
    if (oldWeights && appState.model) { // Use appState
      try {
        const newWeights = appState.model.getWeights();
        // Basic check: only transfer if layer count is the same
        if (oldWeights.length === newWeights.length) {
          // More robust check: ensure shapes match for each weight tensor
          let shapesMatch = true;
          for(let i=0; i<oldWeights.length; i++) {
            if (!tf.util.arraysEqual(oldWeights[i].shape, newWeights[i].shape)) {
              shapesMatch = false;
              break;
            }
          }
          if (shapesMatch) {
            appState.model.setWeights(oldWeights); // Use appState, transfer weights
            // console.info('Transferred weights to new model structure.'); // For debugging, uncomment if needed
          } else {
            console.warn('Weight shapes mismatch after neuron count change. Using new initialization.');
          }
        } else {
          console.warn('Layer count changed. Using new initialization.');
        }
      } catch (e) {
        console.warn('Weight transfer/preservation failed:', e);
      }
    }

    // Batch visualization updates
    requestAnimationFrame(() => {
      updateVisualization();
      updateDecisionBoundary();
    });

    return clamped;
  };

  // Debounced state update with reducer
  let currentNeurons = appState.neuronsPerLayer; // Use appState
  const handleNeuronCountChange = debounce(() => {
    // Use the input field's value as the source of truth when debounced function runs
    const newValue = parseInt(neuronCountInput.value, 10);
    if (!isNaN(newValue)) { // Check if parsing was successful
      currentNeurons = neuronCountReducer(currentNeurons, newValue);
    } else {
      // Handle invalid input, maybe reset to slider value or previous valid value
      neuronCountInput.value = String(currentNeurons);
    }
  }, 250); // Increased debounce time slightly

  // Sync controls: Slider updates Input
  neuronCountSlider.addEventListener('input', function() {
    neuronCountInput.value = this.value;
    // Trigger the debounced update immediately on slider input for responsiveness
    handleNeuronCountChange();
  });

  // Sync controls: Input updates Slider (and triggers debounced update on blur/enter)
  neuronCountInput.addEventListener('input', function() {
    const value = parseInt(this.value, 10);
    if (!isNaN(value)) {
      // Clamp value for slider update
      neuronCountSlider.value = String(Math.max(1, Math.min(32, value)));
    }
  });
  neuronCountInput.addEventListener('blur', handleNeuronCountChange); // Update on blur
  neuronCountInput.addEventListener('keyup', function(e) {
    if (e.key === 'Enter') {
      handleNeuronCountChange(); // Trigger update on Enter
      this.blur(); // Optional: remove focus
    }
  });

  // Initialize UI elements with current state values
  neuronCountSlider.value = String(appState.neuronsPerLayer); // Use appState, ensure string
  neuronCountInput.value = String(appState.neuronsPerLayer); // Use appState, ensure string
}

/**
 * Full reset of the network and UI to default configuration
 */
function fullReset() {
  // console.info('Performing full reset...'); // For debugging, uncomment if needed
  // Reset internal state variables to defaults
  appState.hiddenLayers = defaultConfig.hiddenLayers;
  appState.neuronsPerLayer = defaultConfig.neuronsPerLayer;
  appState.showTestData = false;
  appState.discretizeOutput = false;

  // Stop training and reset epoch/loss display
  resetTraining();

  // Reset UI elements to match default config
  document.getElementById('activationSelect').value = defaultConfig.activation;
  document.getElementById('learningRateSelect').value = String(defaultConfig.learningRate);
  document.getElementById('regularizationSelect').value = defaultConfig.regularization;
  document.getElementById('regularizationRateSelect').value = String(defaultConfig.regularizationRate);
  document.getElementById('ratioSlider').value = String(defaultConfig.ratio);
  document.getElementById('ratioValue').textContent = `${defaultConfig.ratio}%`;
  document.getElementById('noiseSlider').value = String(defaultConfig.noise);
  document.getElementById('noiseValue').textContent = String(defaultConfig.noise);
  document.getElementById('batchSizeSlider').value = String(defaultConfig.batchSize);
  document.getElementById('batchSizeValue').textContent = String(defaultConfig.batchSize);
  neuronCountSlider.value = String(appState.neuronsPerLayer); // Reset neuron slider/input
  neuronCountInput.value = String(appState.neuronsPerLayer);

  // Reset checkboxes
  document.getElementById('showTestData').checked = appState.showTestData;
  document.getElementById('discretizeOutput').checked = appState.discretizeOutput;
  document.getElementById('feature-x1').checked = defaultConfig.features.x1;
  document.getElementById('feature-x2').checked = defaultConfig.features.x2;
  document.getElementById('feature-x1sq').checked = defaultConfig.features.x1Squared;
  document.getElementById('feature-x2sq').checked = defaultConfig.features.x2Squared;
  document.getElementById('feature-x1x2').checked = defaultConfig.features.x1x2;
  document.getElementById('feature-sinx1').checked = defaultConfig.features.sinX1;
  document.getElementById('feature-sinx2').checked = defaultConfig.features.sinX2;

  // Reset dataset selection UI
  const datasetOptions = document.querySelectorAll('.dataset-option');
  datasetOptions.forEach(opt => opt.classList.remove('active'));
  const defaultDataset = document.querySelector(`.dataset-option[data-dataset="${defaultConfig.dataset}"]`);
  if (defaultDataset) defaultDataset.classList.add('active');

  // Reset data generator state to defaults
  dataGenerator.setDataset(defaultConfig.dataset);
  dataGenerator.setTrainTestRatio(defaultConfig.ratio);
  dataGenerator.setNoise(defaultConfig.noise);
  for (const [feature, enabled] of Object.entries(defaultConfig.features)) {
    dataGenerator.setFeatureEnabled(feature, enabled);
  }

  // Update layer count display
  updateLayerCountDisplay();

  // Generate new data and create a fresh model (disposes old one)
  generateData();
}

/**
 * Update the layer count display
 */
function updateLayerCountDisplay() {
  layerCountDisplay.textContent = `${appState.hiddenLayers} HIDDEN LAYERS`; // Use appState
}


/**
 * Update the visualization (Network Graph)
 */
// Visualization update queue and batcher
const visualizationQueue = [];
let visualizationFrameId = null; // ID for scheduled animation frame

function batchVisualizationUpdates() {
  visualizationFrameId = null; // Reset frame ID since it's running now

  if (!appState.model || visualizationQueue.length === 0) return;

  // Use the latest state from the queue (only feature names needed currently)
  const { featureNames } = visualizationQueue[visualizationQueue.length - 1];
  visualizationQueue.length = 0; // Clear the queue

  try {
    // visualizeNetwork returns plain JS data, no tensors to dispose here
    const vizData = window.nn_vis.visualizeNetwork(
      appState.model,
      featureNames,
      appState.rawXs,
      appState.rawYs,
      appState.rawXsTest,
      appState.rawYsTest,
      appState.showTestData,
      null, // transformationPreviews placeholder
    );
    appState.networkData = vizData; // Store the plain data

  } catch (e) {
    console.error('Visualization error:', e);
    appState.networkData = null; // Clear potentially invalid data
  }
}

function updateVisualization() {
  if (!appState.model) return; // Need a model to visualize

  // Always push the latest feature names to the queue
  visualizationQueue.push({
    featureNames: dataGenerator.getEnabledFeatureNames(),
  });

  // Schedule batch update only if one isn't already pending
  if (visualizationFrameId === null) {
    visualizationFrameId = requestAnimationFrame(batchVisualizationUpdates);
  }
}


/**
 * Update the decision boundary visualization
 */
function updateDecisionBoundary() {
  if (!appState.model) return; // Need a model

  // This function uses tf.tidy internally for its predictions
  window.nn_vis.visualizeDecisionBoundary(
    appState.model,
    dataGenerator, // Pass the global instance
    appState.rawXs,
    appState.rawYs,
    appState.rawXsTest,
    appState.rawYsTest,
    appState.showTestData,
    appState.discretizeOutput,
  );
}

/**
 * Setup collapse icon event listeners to toggle panel content visibility.
 */
function setupCollapseIcons() {
  const headers = document.querySelectorAll('.panel .panel-header'); // Select headers directly
  headers.forEach(header => {
    const icon = header.querySelector('.collapse-icon');
    const panelContent = header.nextElementSibling;

    if (icon && panelContent && panelContent.classList.contains('panel-content')) {
      // Initial state setup
      const isCollapsed = panelContent.classList.contains('collapsed');
      icon.textContent = isCollapsed ? '▶' : '▼';
      icon.classList.toggle('collapsed', isCollapsed); // Ensure class matches state

      // Click listener
      header.addEventListener('click', () => {
        const currentlyCollapsed = panelContent.classList.toggle('collapsed');
        icon.textContent = currentlyCollapsed ? '▶' : '▼';
        icon.classList.toggle('collapsed', currentlyCollapsed);
        // Optional: Save state to localStorage if needed
        // const panelTitle = header.querySelector('.panel-title')?.textContent || 'panel';
        // Storage.save(`${panelTitle}-collapsed`, currentlyCollapsed);
      });
    }
  });
}
