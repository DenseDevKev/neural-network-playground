/**
 * UI Management Module
 * Handles all UI-related functionality
 * @module UI
 */

import { appState } from './state.js';
import { updateVisualization, updateDecisionBoundary } from './visualization.js';
import { toggleTraining, trainStep, resetTraining } from './training.js';

// DOM elements
let epochCounter;
let testLossDisplay;
let trainingLossDisplay;
let layerCountDisplay;

/**
 * Initialize UI elements
 */
export function initUI() {
  // Cache DOM elements
  epochCounter = document.getElementById('epochCounter');
  testLossDisplay = document.getElementById('testLoss');
  trainingLossDisplay = document.getElementById('trainingLoss');
  layerCountDisplay = document.getElementById('layerCountDisplay');
  
  // Setup event listeners
  setupEventListeners();
  setupNeuronCountControls();
  setupCollapseIcons();
}

/**
 * Set up event listeners for UI controls
 */
function setupEventListeners() {
  document.getElementById('playBtn').addEventListener('click', toggleTraining);
  document.getElementById('stepBtn').addEventListener('click', trainStep);
  document.getElementById('resetBtn').addEventListener('click', resetTraining);
  document.getElementById('regenerateBtn').addEventListener('click', () => window.app.generateData());

  // Dataset selection
  document.querySelectorAll('.dataset-option').forEach(option => {
    option.addEventListener('click', () => {
      document.querySelectorAll('.dataset-option').forEach(opt => opt.classList.remove('active'));
      option.classList.add('active');
      appState.dataGenerator.setDataset(option.dataset.dataset);
      window.app.generateData();
    });
  });

  // Sliders
  document.getElementById('ratioSlider').addEventListener('input', e => {
    const value = e.target.value;
    document.getElementById('ratioValue').textContent = `${value}%`;
    appState.dataGenerator.setTrainTestRatio(value);
    window.app.generateData();
  });

  document.getElementById('noiseSlider').addEventListener('input', e => {
    const value = e.target.value;
    document.getElementById('noiseValue').textContent = value;
    appState.dataGenerator.setNoise(value);
    window.app.generateData();
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
      appState.dataGenerator.setFeatureEnabled(featureKey, e.target.checked);
      window.app.generateData();
    });
  });

  // Network configuration controls: recreate model on change
  ['activationSelect','learningRateSelect','regularizationSelect','regularizationRateSelect'].forEach(id => {
    document.getElementById(id).addEventListener('change', () => window.app.generateData());
  });

  // Discretize output toggle: update decision boundary
  document.getElementById('discretizeOutput').addEventListener('change', e => {
    appState.discretizeOutput = e.target.checked;
    window.app.updateDecisionBoundary();
  });

  // Add/remove layer buttons
  document.getElementById('addLayerBtn').addEventListener('click', () => {
    if (appState.hiddenLayers < 6) {
      appState.hiddenLayers++;
      updateLayerCountDisplay();
      window.app.createModel();
      window.app.updateVisualization();
      window.app.updateDecisionBoundary();
    }
  });
  document.getElementById('removeLayerBtn').addEventListener('click', () => {
    if (appState.hiddenLayers > 1) {
      appState.hiddenLayers--;
      updateLayerCountDisplay();
      window.app.createModel();
      window.app.updateVisualization();
      window.app.updateDecisionBoundary();
    }
  });

  // Toggle display of test data on decision boundary
  document.getElementById('showTestData').addEventListener('change', e => {
    appState.showTestData = e.target.checked;
    window.app.updateDecisionBoundary();
  });
}

/**
 * Set up neuron count controls
 */
function setupNeuronCountControls() {
  const neuronCountSlider = document.getElementById('neuronCountSlider');
  const neuronCountInput = document.getElementById('neuronCountInput');
  if (!neuronCountSlider || !neuronCountInput) return;
  let currentNeurons = appState.neuronsPerLayer;
  const neuronCountReducer = (prev, next) => {
    const clamped = Math.max(1, Math.min(32, next));
    if (clamped === prev || clamped === appState.neuronsPerLayer) return prev;
    const oldWeights = appState.model?.getWeights();
    appState.neuronsPerLayer = clamped;
    resetTraining();
    window.app.createModel();
    if (oldWeights && appState.model) {
      try {
        const newWeights = appState.model.getWeights();
        if (oldWeights.length === newWeights.length) {
          let shapesMatch = true;
          for(let i=0; i<oldWeights.length; i++) {
            if (!tf.util.arraysEqual(oldWeights[i].shape, newWeights[i].shape)) {
              shapesMatch = false;
              break;
            }
          }
          if (shapesMatch) {
            appState.model.setWeights(oldWeights);
          }
        }
      } catch (e) {
        // Ignore weight transfer errors
      }
    }
    requestAnimationFrame(() => {
      window.app.updateVisualization();
      window.app.updateDecisionBoundary();
    });
    return clamped;
  };
  const handleNeuronCountChange = () => {
    const newValue = parseInt(neuronCountInput.value, 10);
    if (!isNaN(newValue)) {
      currentNeurons = neuronCountReducer(currentNeurons, newValue);
    } else {
      neuronCountInput.value = String(currentNeurons);
    }
  };
  neuronCountSlider.addEventListener('input', function() {
    neuronCountInput.value = this.value;
    handleNeuronCountChange();
  });
  neuronCountInput.addEventListener('input', function() {
    const value = parseInt(this.value, 10);
    if (!isNaN(value)) {
      neuronCountSlider.value = String(Math.max(1, Math.min(32, value)));
    }
  });
  neuronCountInput.addEventListener('blur', handleNeuronCountChange);
  neuronCountInput.addEventListener('keyup', function(e) {
    if (e.key === 'Enter') {
      handleNeuronCountChange();
      this.blur();
    }
  });
  neuronCountSlider.value = String(appState.neuronsPerLayer);
  neuronCountInput.value = String(appState.neuronsPerLayer);
}

/**
 * Setup collapse icons for collapsible panels
 */
function setupCollapseIcons() {
  const collapseIcons = document.querySelectorAll('.collapse-icon');
  collapseIcons.forEach(icon => {
    icon.addEventListener('click', (e) => {
      const panel = e.target.closest('.panel');
      const content = panel.querySelector('.panel-content');
      content.style.display = content.style.display === 'none' ? 'block' : 'none';
      e.target.textContent = content.style.display === 'none' ? '⊕' : '⊖';
    });
  });
}

/**
 * Update the epoch counter display
 * @param {number} count - Current epoch count
 */
export function updateEpochCounter(count) {
  if (epochCounter) {
    epochCounter.textContent = `Epoch: ${count.toString().padStart(6, '0')}`;
  }
}

/**
 * Update the loss display
 * @param {number} testLoss - Current test loss
 * @param {number} trainingLoss - Current training loss
 */
export function updateLossDisplay(testLoss, trainingLoss) {
  if (testLossDisplay && trainingLossDisplay) {
    testLossDisplay.textContent = testLoss ? testLoss.toFixed(3) : '0.000';
    trainingLossDisplay.textContent = trainingLoss ? trainingLoss.toFixed(3) : '0.000';
  }
}

/**
 * Update the layer count display
 */
export function updateLayerCountDisplay() {
  if (layerCountDisplay) {
    layerCountDisplay.textContent = `${appState.hiddenLayers} layers, ${appState.neuronsPerLayer} neurons`;
  }
}
