/**
 * Training Module
 * Handles model training and optimization
 * @module Training
 */

import * as tf from '@tensorflow/tfjs';
import { appState, defaultConfig } from './state.js';
import { updateEpochCounter, updateLossDisplay } from './ui.js';

// Visualization queue for batching updates
const visualizationQueue = [];
let visualizationFrameId = null;

/**
 * Toggle training on/off
 */
export function toggleTraining() {
  if (appState.training) {
    pauseTraining();
  } else {
    startTraining();
  }
}

/**
 * Start the training loop using requestAnimationFrame
 */
export function startTraining() {
  if (!appState.training) {
    appState.training = true;
    document.getElementById('playBtn').textContent = '⏸';
    trainingLoop();
  }
}

/**
 * Pause training
 */
export function pauseTraining() {
  appState.training = false;
  document.getElementById('playBtn').textContent = '▶';
  
  if (appState.intervalId) {
    cancelAnimationFrame(appState.intervalId);
    appState.intervalId = null;
  }
}

/**
 * Training loop using requestAnimationFrame
 */
async function trainingLoop() {
  if (!appState.training) return;
  
  await trainStep();
  
  // Schedule next frame if still training
  if (appState.training) {
    appState.intervalId = requestAnimationFrame(trainingLoop);
  }
}

/**
 * Perform a single training step
 * @param {number} batchSize - Size of the batch to use for training
 */
export async function trainStep(batchSize = appState.batchSize) {
  if (!appState.model || !appState.xs || !appState.ys) {
    console.warn('Model or training data not ready');
    return;
  }

  try {
    // Perform training step
    const history = await appState.model.fit(appState.xs, appState.ys, {
      batchSize,
      epochs: 1,
      shuffle: true,
      validationData: appState.showTestData ? [appState.xsTest, appState.ysTest] : null,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          appState.epochCount++;
          updateEpochCounter(appState.epochCount);
          
          // Update loss display
          const testLoss = appState.showTestData ? logs.val_loss : null;
          updateLossDisplay(testLoss, logs.loss);
          
          // Queue visualization update
          updateVisualization();
          updateDecisionBoundary();
        }
      }
    });
    
    return history;
  } catch (error) {
    console.error('Error during training step:', error);
    pauseTraining();
    throw error;
  }
}

/**
 * Reset training state
 */
export function resetTraining() {
  appState.epochCount = 0;
  updateEpochCounter(0);
  updateLossDisplay(0, 0);
  
  if (appState.intervalId) {
    cancelAnimationFrame(appState.intervalId);
    appState.intervalId = null;
  }
  
  appState.training = false;
}
