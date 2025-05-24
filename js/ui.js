/**
 * UI Management Module
 * Handles all UI-related functionality
 * @module UI
 */

import { appState } from './state.js';
import { updateVisualization, updateDecisionBoundary } from './visualization.js';

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
  layerCountDisplay = document.getElementById('layerCount');
  
  // Setup event listeners
  setupEventListeners();
  setupNeuronCountControls();
  setupCollapseIcons();
}

/**
 * Set up event listeners for UI controls
 */
function setupEventListeners() {
  // Add your event listeners here
  // This is a simplified version - you'll need to implement the actual event listeners
  document.getElementById('playBtn').addEventListener('click', toggleTraining);
  document.getElementById('stepBtn').addEventListener('click', trainStep);
  document.getElementById('resetBtn').addEventListener('click', fullReset);
  document.getElementById('regenerateBtn').addEventListener('click', generateData);
  
  // Add other event listeners as needed
}

/**
 * Set up neuron count controls
 */
function setupNeuronCountControls() {
  // Implementation for neuron count controls
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
