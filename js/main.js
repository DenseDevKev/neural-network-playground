// Application State
const appState = {
  model: null,
  training: false,
  intervalId: null, // Used for requestAnimationFrame ID
  epochCount: 0,
  hiddenLayers: 3,
  neuronsPerLayer: 7,
  showTestData: false,
  discretizeOutput: false,
  networkData: null, // Result from visualizeNetwork (plain JS objects/arrays)
  // Data Tensors (managed with tf.keep/dispose)
  xs: null,
  ys: null,
  xsTest: null,
  ysTest: null,
  // Raw Data (for visualization)
  rawXs: [],
  rawYs: [],
  rawXsTest: [],
  rawYsTest: [],
  batchSize: 30, // New property
  optimizer: null, // New property
};

// Store initial configuration for reset
const defaultConfig = {
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

// DOM elements (Assigned once on load)
let epochCounter;
let testLossDisplay;
let trainingLossDisplay;
let layerCountDisplay;
let ratioSlider, ratioValue, noiseSlider, noiseValue;
let regenerateBtn, showTestData;
let featureCheckboxes;
let addLayerBtn, removeLayerBtn;
let neuronCountSlider, neuronCountInput;
let activationSelect, learningRateSelect, regularizationSelect, regularizationRateSelect;
let batchSizeSlider, batchSizeValue;
let discretizeOutput;
let playBtn, stepBtn, resetBtn;
let datasetOptions;

// Create global data generator instance
const dataGenerator = new DataGenerator();

// Initialize the application
window.addEventListener('load', () => {
  // Handle window resize
  window.addEventListener('resize', debounce(() => {
    if (appState.model) { // Use appState
      updateVisualization();
      updateDecisionBoundary();
    }
  }, 250));

  // Get DOM elements
  epochCounter = document.getElementById('epochCounter');
  testLossDisplay = document.getElementById('testLoss');
  trainingLossDisplay = document.getElementById('trainingLoss');
  layerCountDisplay = document.getElementById('layerCountDisplay');
  ratioSlider = document.getElementById('ratioSlider');
  ratioValue = document.getElementById('ratioValue');
  noiseSlider = document.getElementById('noiseSlider');
  noiseValue = document.getElementById('noiseValue');
  regenerateBtn = document.getElementById('regenerateBtn');
  showTestData = document.getElementById('showTestData');
  featureCheckboxes = [
    document.getElementById('feature-x1'),
    document.getElementById('feature-x2'),
    document.getElementById('feature-x1sq'),
    document.getElementById('feature-x2sq'),
    document.getElementById('feature-x1x2'),
    document.getElementById('feature-sinx1'),
    document.getElementById('feature-sinx2'),
  ];
  addLayerBtn = document.getElementById('addLayerBtn');
  removeLayerBtn = document.getElementById('removeLayerBtn');
  neuronCountSlider = document.getElementById('neuronCountSlider');
  neuronCountInput = document.getElementById('neuronCountInput');
  activationSelect = document.getElementById('activationSelect');
  learningRateSelect = document.getElementById('learningRateSelect');
  regularizationSelect = document.getElementById('regularizationSelect');
  regularizationRateSelect = document.getElementById('regularizationRateSelect');
  batchSizeSlider = document.getElementById('batchSizeSlider');
  batchSizeValue = document.getElementById('batchSizeValue');
  discretizeOutput = document.getElementById('discretizeOutput');
  playBtn = document.getElementById('playBtn');
  stepBtn = document.getElementById('stepBtn');
  resetBtn = document.getElementById('resetBtn');
  datasetOptions = Array.from(document.querySelectorAll('.dataset-option'));

  // Set up event listeners for controls
  setupEventListeners();

  // Set up event listeners for collapse icons
  setupCollapseIcons();

  // Set up neuron count controls
  setupNeuronCountControls();

  // Generate initial data
  generateData();
});

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
      const id = e.target.id.replace('feature-','');
      let featureKey = id;
      if (id === 'x1sq') featureKey = 'x1Squared';
      else if (id === 'x2sq') featureKey = 'x2Squared';
      else if (id === 'sinx1') featureKey = 'sinX1';
      else if (id === 'sinx2') featureKey = 'sinX2';
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
 * Generate data using the DataGenerator and create a new model
 */
function generateData() {
  // Use tf.tidy to ensure proper cleanup of tensors
  tf.tidy(() => {
    try {
      // Generate new data
      const data = dataGenerator.generateData();
      
      // Clean up previous data if it exists
      if (appState.xs) appState.xs.dispose();
      if (appState.ys) appState.ys.dispose();
      if (appState.xsTest) appState.xsTest.dispose();
      if (appState.ysTest) appState.ysTest.dispose();
      
      // Store new data with proper memory management
      appState.xs = tf.keep(data.xs);
      appState.ys = tf.keep(data.ys);
      appState.xsTest = tf.keep(data.xsTest);
      appState.ysTest = tf.keep(data.ysTest);
      appState.rawXs = data.rawXs;
      appState.rawYs = data.rawYs;
      appState.rawXsTest = data.rawXsTest;
      appState.rawYsTest = data.rawYsTest;
      
      // Reset training state
      resetTraining();
      
      // Create a new model for the new data
      createModel();
      
      // Update visualizations
      updateVisualization();
      updateDecisionBoundary();
      updateLossDisplay('N/A', 'N/A');
      
    } catch (error) {
      console.error('Error generating data:', error);
      // Fallback to default data if generation fails
      dataGenerator.setDataset('circle');
      this.generateData();
    }
  });
  
  // Log memory usage for debugging
  if (typeof tf.memory !== 'undefined') {
    // console.info('Memory after data generation:', tf.memory()); // For debugging, uncomment if needed
  }
}

/**
 * Creates a new sequential model based on user input from the UI
 */
function createModel() {
  pauseTraining(); // Ensure training is stopped

  // Dispose previous model if it exists
  if (appState.model) {
    appState.model.dispose();
    appState.model = null;
  }

  // Get parameters from UI
  const activation = document.getElementById('activationSelect').value;
  const learningRate = parseFloat(document.getElementById('learningRateSelect').value);
  const regularizationType = document.getElementById('regularizationSelect').value;
  const regularizationRate = parseFloat(document.getElementById('regularizationRateSelect').value);
  const inputShape = dataGenerator.getEnabledFeatureCount();

  if (inputShape === 0) {
    console.error('No features enabled. Cannot create model.');
    // Optionally disable UI elements or show error message
    return;
  }

  // Create model within a tidy scope
  tf.tidy(() => {
    appState.model = tf.sequential(); // Assign to appState

    let regularizer = null;
    if (regularizationType === 'l1') {
      regularizer = tf.regularizers.l1({ l1: regularizationRate });
    } else if (regularizationType === 'l2') {
      regularizer = tf.regularizers.l2({ l2: regularizationRate });
    }

    // Input layer
    appState.model.add(tf.layers.dense({
      inputShape: [inputShape],
      units: appState.neuronsPerLayer,
      activation: activation,
      kernelRegularizer: regularizer,
      kernelInitializer: 'glorotUniform',
    }));

    // Hidden layers
    for (let i = 1; i < appState.hiddenLayers; i++) {
      appState.model.add(tf.layers.dense({
        units: appState.neuronsPerLayer,
        activation: activation,
        kernelRegularizer: regularizer,
        kernelInitializer: 'glorotUniform',
      }));
    }

    // Output layer
    appState.model.add(tf.layers.dense({
      units: 1,
      activation: 'sigmoid',
      kernelInitializer: 'glorotUniform',
    }));

    // Compile model
    const optimizer = tf.train.adam(learningRate); // Using Adam optimizer
    appState.optimizer = optimizer; // Store optimizer for custom training loop
    appState.model.compile({
      optimizer: optimizer,
      loss: 'binaryCrossentropy',
      metrics: ['accuracy'],
    });

    // NOTE: Weight caching logic removed for simplification.
    // Always uses new random weights on model creation.

  }); // End tf.tidy for model creation

  // Model structure is created, ready for training/visualization
  // console.info(`Created model: ${appState.hiddenLayers} layers, ${appState.neuronsPerLayer} neurons/layer`); // For debugging, uncomment if needed
}

/**
 * Toggle training on/off
 */
function toggleTraining() {
  if (!appState.model) {
    console.warn('Cannot toggle training: model not available.');
    return;
  }
  if (appState.training) { // Use appState
    pauseTraining();
    document.getElementById('playBtn').textContent = '▶';
  } else {
    startTraining();
    document.getElementById('playBtn').textContent = '⏸';
  }
}

/**
 * Start the training loop using requestAnimationFrame
 */
function startTraining() {
  if (!appState.model || appState.training) return; // Check model and training status
  appState.training = true; // Use appState
  // console.info('startTraining called'); // For debugging, uncomment if needed

  let lastFrameTime = performance.now();
  const targetFPS = 30; // Target FPS for training steps + UI updates
  const interval = 1000 / targetFPS;

  async function trainLoop(currentTime) {
    // Store the frame ID immediately
    appState.intervalId = requestAnimationFrame(trainLoop); // Use appState

    if (!appState.training) { // Check if stopped during the previous frame
      cancelAnimationFrame(appState.intervalId);
      appState.intervalId = null;
      return;
    }

    const deltaTime = currentTime - lastFrameTime;

    // Only train if enough time has passed
    if (deltaTime >= interval) {
      lastFrameTime = currentTime - (deltaTime % interval); // Adjust for precision

      const batchSize = appState.batchSize; // Use appState
      // Perform one training step
      try {
        await trainStep(batchSize); // Await the single step
      } catch (error) {
        console.error('Error during training step:', error);
        pauseTraining(); // Stop training on error
        document.getElementById('playBtn').textContent = '▶';
        // No need to return here, loop is stopped by pauseTraining setting appState.training=false
      }
    }
    // Continue the loop via the requestAnimationFrame at the beginning
  }

  // Start the loop by requesting the first frame
  appState.intervalId = requestAnimationFrame(trainLoop);
}

/**
 * Perform a single training step
 */
async function trainStep(batchSize = appState.batchSize) {
  if (!appState.model || !appState.xs || !appState.ys) return null;

  let trainLoss = 0;
  let valLoss = 0;
  
  // Use tf.tidy to automatically clean up intermediate tensors
  const losses = await tf.tidy(() => {
    try {
      // Sample a random batch
      const numExamples = appState.xs.shape[0];
      const indices = tf.util.createShuffledIndices(numExamples).slice(0, batchSize);
      
      // Get the batch
      const xBatch = appState.xs.gather(indices);
      const yBatch = appState.ys.gather(indices);
      
      // Train on the batch
      const history = appState.optimizer.minimize(() => {
        const preds = appState.model.predict(xBatch);
        return tf.losses.binaryCrossentropy(yBatch, preds).mean();
      }, true);
      
      // Compute training loss
      const trainPreds = appState.model.predict(xBatch);
      trainLoss = tf.losses.binaryCrossentropy(yBatch, trainPreds).mean();
      
      // Compute validation loss if test data is available
      let valLoss = 0;
      if (appState.xsTest && appState.ysTest) {
        const testPreds = appState.model.predict(appState.xsTest);
        valLoss = tf.losses.binaryCrossentropy(appState.ysTest, testPreds).mean();
      }
      
      return { trainLoss, valLoss };
    } catch (error) {
      console.error('Error during training step:', error);
      return { trainLoss: 0, valLoss: 0 };
    }
  });
  
  // Update epoch count and displays
  appState.epochCount++;
  updateEpochCounter(appState.epochCount);
  updateLossDisplay(losses.valLoss, losses.trainLoss);
  
  // Update visualizations
  updateVisualization();
  updateDecisionBoundary();
  
  // Animate data flow with the first training example
  if (appState.networkData && appState.rawXs.length > 0) {
    const activations = tf.tidy(() => {
      const tensor = tf.tensor2d([appState.rawXs[0]]);
      return window.nn_vis.getActivations(appState.model, tensor);
    });
    
    window.nn_vis.animateDataFlow(
      appState.networkData.neurons,
      appState.networkData.edges,
      activations
    );
    
    tf.dispose(activations);
  }
  
  // Force garbage collection occasionally to prevent memory buildup
  if (appState.epochCount % 100 === 0) {
    await tf.nextFrame(); // Allow the browser to breathe
    if (typeof tf.memory !== 'undefined') {
      // console.info('Memory status:', tf.memory()); // For debugging, uncomment if needed
    }
  }
  
  return losses.trainLoss;
}

/**
 * Pause training by cancelling the animation frame
 */
function pauseTraining() {
  if (appState.training) {
    // console.info('Pausing training loop...'); // For debugging, uncomment if needed
    appState.training = false; // Set flag to stop the loop
    if (appState.intervalId) {
      cancelAnimationFrame(appState.intervalId); // Cancel the next scheduled frame
      appState.intervalId = null;
    }
  }
}

/**
 * Reset training state (epoch count, loss display)
 */
function resetTraining() {
  pauseTraining(); // Ensure training loop is stopped
  document.getElementById('playBtn').textContent = '▶'; // Reset button text
  appState.epochCount = 0; // Reset epoch count
  updateEpochCounter(appState.epochCount); // Update display
  updateLossDisplay('N/A', 'N/A'); // Reset loss display
  // Note: Model weights are NOT reset here.
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
 * Update the epoch counter display
 */
function updateEpochCounter(count) {
  epochCounter.textContent = count.toString().padStart(6, '0');
}

/**
 * Update the loss display
 */
function updateLossDisplay(testLoss, trainingLoss) {
  testLossDisplay.textContent = testLoss;
  trainingLossDisplay.textContent = trainingLoss;
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
