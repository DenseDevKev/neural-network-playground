// visualization.js - Minimal robust D3 neural network visualization

// This file handles drawing the network layers, neurons, edges, and the decision boundary using D3.
// It provides visualization functions called from main.js.

// Configuration
const networkWidth = 800;
const networkHeight = 500;
const decisionBoundarySize = 400; // Increased from 200 to 400 for better quality
const neuronSize = 24;
const neuronMargin = 10;
const layerControlSize = 20;
const dataFlowAnimationDuration = 600; // ms for data to flow through network
const tooltipDelay = 300; // ms delay before showing tooltip
const decisionBoundaryResolution = 100; // Number of points to sample in each dimension
const transformationPreviewSize = 30; // Size for transformation preview SVGs

// Global references
let svg = null;
let decisionCanvas = null;
let decisionBoundaryContainer = null;
let dataFlowSvg = null; // SVG layer for data flow animations
let tooltip = null;
const activeEdges = []; // Store active edges for data flow animation

// Color scales using CSS variables
// Helper to get CSS variable value
const getCssVar = (varName) => getComputedStyle(document.documentElement).getPropertyValue(varName).trim();

// Define scales - ensure these are updated if CSS vars change dynamically (e.g., theme switch)
// Note: Using fixed fallbacks in case CSS vars aren't immediately available
let weightColorScale = d3.scaleLinear()
  .domain([-1, 0, 1])
  .range([getCssVar('--md-error') || '#cf6679', 'rgba(255, 255, 255, 0.3)', getCssVar('--md-primary') || '#bb86fc']);

let activationColorScale = d3.scaleLinear()
  .domain([-1, 0, 1])
  .range([getCssVar('--md-error') || '#cf6679', 'rgba(255, 255, 255, 0.3)', getCssVar('--md-primary') || '#bb86fc']);


// Function to update color scales if needed (e.g., after theme change)
function updateColorScales() {
    weightColorScale = d3.scaleLinear()
      .domain([-1, 0, 1])
      .range([getCssVar('--md-error') || '#cf6679', 'rgba(255, 255, 255, 0.3)', getCssVar('--md-primary') || '#bb86fc']);

    activationColorScale = d3.scaleLinear()
      .domain([-1, 0, 1])
      .range([getCssVar('--md-error') || '#cf6679', 'rgba(255, 255, 255, 0.3)', getCssVar('--md-primary') || '#bb86fc']);
}

// Initialize visualization elements
window.addEventListener('load', () => {
  // Prepare SVG for the network
  const networkContainer = document.getElementById('network-container');
if (!networkContainer) {
  console.error('Network container not found');
  return;
}
svg = d3.select(networkContainer)
    .append('svg')
    .attr('width', '100%')
    .attr('height', '100%')
    .attr('viewBox', `0 0 ${networkWidth} ${networkHeight}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  // Create a separate SVG layer for data flow animations
  dataFlowSvg = d3.select(networkContainer)
    .append('svg')
    .attr('width', '100%')
    .attr('height', '100%')
    .attr('viewBox', `0 0 ${networkWidth} ${networkHeight}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('pointer-events', 'none'); // Let events pass through to the layer below
    
  // Create a canvas for the decision boundary
const decisionBoundaryContainerElem = document.getElementById('decision-boundary-container');
if (!decisionBoundaryContainerElem) {
  console.error('Decision boundary container not found');
  return;
}
decisionBoundaryContainer = d3.select(decisionBoundaryContainerElem);
decisionCanvas = decisionBoundaryContainer
  .append('canvas')
  .attr('width', decisionBoundarySize)
  .attr('height', decisionBoundarySize)
  .style('width', '100%')
  .style('height', '100%')
  .style('image-rendering', 'pixelated') // Improve rendering of pixelated data
  .style('border-radius', '12px') // Add rounded corners
  .style('box-shadow', '0 4px 12px rgba(0, 0, 0, 0.15)') // Add subtle shadow
  .node();

// Set the canvas context properties for better rendering
const ctx = decisionCanvas.getContext('2d');
ctx.imageSmoothingEnabled = false; // Disable anti-aliasing for sharper boundaries

// Create tooltip element
tooltip = d3.select('body')
  .append('div')
  .attr('class', 'tooltip')
  .style('opacity', 0);

// Function to handle responsive resizing
function handleResize() {
    const windowWidth = window.innerWidth;
    const isMobile = windowWidth <= 768;
    const isSmallMobile = windowWidth <= 480;
    const isTablet = windowWidth > 768 && windowWidth <= 1024;
    
    // Get container dimensions
    const networkContainer = document.getElementById('network-container');
    const containerWidth = networkContainer.clientWidth;
    const containerHeight = networkContainer.clientHeight;
    
    // Ensure network container has appropriate minimum height based on screen size
    if (isSmallMobile) {
      networkContainer.style.minHeight = '250px';
    } else if (isMobile) {
      networkContainer.style.minHeight = '300px';
    } else if (isTablet) {
      networkContainer.style.minHeight = '350px';
    } else {
      networkContainer.style.minHeight = '400px';
    }
    
    // Add appropriate class to main container for layout adjustments
    const mainContainer = document.getElementById('mainContainer');
    if (mainContainer) {
      mainContainer.classList.remove('mobile-layout', 'tablet-layout', 'desktop-layout');
      if (isMobile) {
        mainContainer.classList.add('mobile-layout');
      } else if (isTablet) {
        mainContainer.classList.add('tablet-layout');
      } else {
        mainContainer.classList.add('desktop-layout');
      }
    }
    
    // Update network layout based on device size
    if (isMobile) {
      // Mobile layout (more compact)
      svg.attr('viewBox', `0 0 ${networkWidth} ${networkHeight*0.8}`);
      dataFlowSvg.attr('viewBox', `0 0 ${networkWidth} ${networkHeight*0.8}`);
    } else if (isTablet) {
      // Tablet layout (slightly adjusted)
      svg.attr('viewBox', `0 0 ${networkWidth} ${networkHeight*0.9}`);
      dataFlowSvg.attr('viewBox', `0 0 ${networkWidth} ${networkHeight*0.9}`);
    } else {
      // Desktop layout
      svg.attr('viewBox', `0 0 ${networkWidth} ${networkHeight}`);
      dataFlowSvg.attr('viewBox', `0 0 ${networkWidth} ${networkHeight}`);
    }
    
    // Reposition and resize the decision boundary based on device size
    if (decisionBoundaryContainer.node()) {
      const boundaryContainer = decisionBoundaryContainer.node();
      
      // Position decision boundary container differently based on screen size
      if (isMobile) {
        // For mobile, make it full width below the network
        decisionBoundaryContainer
          .style('position', 'relative')
          .style('bottom', 'auto')
          .style('right', 'auto')
          .style('width', '100%')
          .style('height', 'auto')
          .style('margin-top', '16px')
          .style('aspect-ratio', '1/1');
      } else {
        // For desktop, keep it as an overlay in the corner
        decisionBoundaryContainer
          .style('position', 'absolute')
          .style('bottom', '20px')
          .style('right', '20px')
          .style('width', isTablet ? '180px' : '220px')
          .style('height', isTablet ? '180px' : '220px');
      }
      
      // Ensure the canvas has proper dimensions
      const containerWidth = boundaryContainer.clientWidth;
      const containerHeight = boundaryContainer.clientHeight || containerWidth; // Fallback to width
      
      // Set canvas size
      decisionCanvas.style.width = '100%';
      decisionCanvas.style.height = '100%';
      
        // Redraw decision boundary if we have data, using requestAnimationFrame
        if (currentDecisionBoundaryData) {
          requestAnimationFrame(() => {
            // Check if data still exists in case of rapid changes
            if (currentDecisionBoundaryData) {
               const { model, dataGenerator, trainX, trainY, testX, testY, showTest, discretize } = currentDecisionBoundaryData;
               visualizeDecisionBoundary(model, dataGenerator, trainX, trainY, testX, testY, showTest, discretize);
            }
          });
        }
      }
  }
    
  // Add layer control buttons for each layer
  addNeuronControls();
});

/**
 * Add neuron control buttons (+/-) for each layer
 */
function addNeuronControls() {
  // We'll add these dynamically when visualizing the network
}

/**
 * Renders a visual representation of the network's layers, neurons, and weights using D3.
 * @param {tf.Sequential} model - The neural network model
 * @param {Array} featureNames - Names of input features
 * @param {Array} trainX - Training data points
 * @param {Array} trainY - Training data labels
 * @param {Array} testX - Test data points
 * @param {Array} testY - Test data labels
 * @param {boolean} showTest - Whether to show test data
 * @param {Object} transformationPreviews - (new) SVG previews for feature transformations
 */
function visualizeNetwork(model, featureNames, trainX, trainY, testX, testY, showTest, transformationPreviews) {
  console.log('[Vis] visualizeNetwork called. Model:', model ? 'Exists' : 'NULL', 'SVG:', svg ? 'Exists' : 'NULL');
  if (!svg) {
    console.error('[Vis] visualizeNetwork: SVG container (svg) not initialized. Aborting.');
    return null; 
  }
  if (!model) {
    console.warn('[Vis] visualizeNetwork: Model not provided. Aborting.');
    return null;
  }
  if (!model || !svg) return;

  // Clear the SVG
  svg.selectAll('*').remove();

  // Get model structure
  const layerSizes = model.layers.map(layer => layer.getConfig().units);
  const inputShape = featureNames.length;
  const allLayers = [inputShape, ...layerSizes];
  const totalLayers = allLayers.length;

  // Calculate layout
  const layerGap = networkWidth / (totalLayers + 1);
  const maxNeurons = Math.max(...allLayers);
  const layerHeight = Math.min(
    networkHeight - 100, // Leave room for controls
    maxNeurons * (neuronSize + neuronMargin)
  );

  // Calculate neuron positions
  const neuronPositions = [];
  const layerCenters = [];

  for (let l = 0; l < totalLayers; l++) {
    const layerSize = allLayers[l];
    const xPos = layerGap * (l + 1);
    layerCenters.push({ x: xPos, y: networkHeight / 2 });
    
    // Calculate vertical spacing for neurons
    const totalLayerHeight = layerSize * (neuronSize + neuronMargin);
    const startY = (networkHeight - totalLayerHeight) / 2;
    
    for (let i = 0; i < layerSize; i++) {
      const yPos = startY + i * (neuronSize + neuronMargin) + neuronSize / 2;
      neuronPositions.push({
        layer: l,
        index: i,
        x: xPos,
        y: yPos,
        label: l === 0 ? featureNames[i] : ''
      });
    }
  }

  // Add layer controls
  for (let l = 1; l < totalLayers - 1; l++) { // Skip input and output layers
    const x = layerCenters[l].x;
    const y = 30; // Top of the network area
    
    // Add neuron count display
    svg.append('text')
      .attr('x', x)
      .attr('y', y - 15)
      .attr('text-anchor', 'middle')
      .attr('class', 'neuron-count')
      .text(`${allLayers[l]} neurons`);
    
    // Add + button
    svg.append('circle')
      .attr('cx', x - 15)
      .attr('cy', y)
      .attr('r', 10)
      .attr('fill', 'rgba(60, 60, 70, 0.8)')
      .attr('stroke', 'rgba(255, 255, 255, 0.1)')
      .attr('class', 'neuron-control')
      .style('cursor', 'pointer')
      .on('click', () => {
        // This will be handled by the main.js event listeners
        // We're just adding the visual elements here
      });
    
    svg.append('text')
      .attr('x', x - 15)
      .attr('y', y + 4)
      .attr('text-anchor', 'middle')
      .attr('fill', 'white')
      .style('font-size', '14px')
      .style('pointer-events', 'none')
      .text('+');
    
    // Add - button
    svg.append('circle')
      .attr('cx', x + 15)
      .attr('cy', y)
      .attr('r', 10)
      .attr('fill', 'rgba(60, 60, 70, 0.8)')
      .attr('stroke', 'rgba(255, 255, 255, 0.1)')
      .attr('class', 'neuron-control')
      .style('cursor', 'pointer')
      .on('click', () => {
        // This will be handled by the main.js event listeners
        // We're just adding the visual elements here
      });
    
    svg.append('text')
      .attr('x', x + 15)
      .attr('y', y + 4)
      .attr('text-anchor', 'middle')
      .attr('fill', 'white')
      .style('font-size', '14px')
      .style('pointer-events', 'none')
      .text('-');
  }

  // Get weights and activations
  const edgesData = [];
  const activations = [];

  // Calculate activations for a sample input
  let sampleActivations = [];
  if (trainX && trainX.length > 0) {
    // Use the first training example to visualize activations
    const sampleInput = transformSampleForModel(trainX[0], featureNames, transformationPreviews);
    sampleActivations = getActivations(model, sampleInput);
  }

  // Process weights and create edge data
  for (let l = 0; l < model.layers.length; l++) {
    const [kernel, bias] = model.layers[l].getWeights();
    const kernelVals = kernel.dataSync();
    const shape = kernel.shape; // [inFeatures, outFeatures]

    // Find the positions for the correct layers
    const sourceLayer = l;     // index in allLayers
    const targetLayer = l + 1; // next layer

    // Get neurons for these layers
    const sourceNeurons = neuronPositions.filter(pos => pos.layer === sourceLayer);
    const targetNeurons = neuronPositions.filter(pos => pos.layer === targetLayer);

    // Create edges
    for (let i = 0; i < shape[0]; i++) {
      for (let j = 0; j < shape[1]; j++) {
        const weight = kernelVals[j + i * shape[1]];
        edgesData.push({
          x1: sourceNeurons[i].x,
          y1: sourceNeurons[i].y,
          x2: targetNeurons[j].x,
          y2: targetNeurons[j].y,
          weight: weight,
          sourceLayer: sourceLayer,
          sourceNeuron: i,
          targetLayer: targetLayer,
          targetNeuron: j
        });
      }
    }
  }

  // Draw edges
  const edgeGroup = svg.append('g').attr('class', 'edges');
  
  edgeGroup.selectAll('.edge')
    .data(edgesData)
    .enter()
    .append('line')
    .attr('class', 'edge')
    .attr('x1', d => d.x1)
    .attr('y1', d => d.y1)
    .attr('x2', d => d.x2)
    .attr('y2', d => d.y2)
    .attr('stroke-width', d => Math.max(0.5, Math.abs(d.weight) * 2))
    .attr('stroke', d => weightColorScale(Math.tanh(d.weight)))
    .attr('opacity', 0.7);

  // Draw neurons
  const neuronGroup = svg.append('g').attr('class', 'neurons');
  
  // Create neuron groups
  const neurons = neuronGroup.selectAll('.neuron-group')
    .data(neuronPositions)
    .enter()
    .append('g')
    .attr('class', 'neuron-group')
    .attr('transform', d => `translate(${d.x - neuronSize/2}, ${d.y - neuronSize/2})`);

  // Add neuron rectangles
  neurons.append('rect')
    .attr('class', 'neuron')
    .attr('width', neuronSize)
    .attr('height', neuronSize)
    .attr('rx', 4)
    .attr('ry', 4)
    .attr('fill', (d, i) => {
      // Color based on activation if available
      if (d.layer < sampleActivations.length && d.index < sampleActivations[d.layer].length) {
        const activation = sampleActivations[d.layer][d.index];
        return activationColorScale(Math.tanh(activation));
      }
      return '#444';
    })
    .attr('stroke', 'rgba(255, 255, 255, 0.3)')
    .attr('stroke-width', 1);

  // Add feature labels and transformation previews for input layer
  const inputNeurons = neurons.filter(d => d.layer === 0);
  
  inputNeurons.append('text')
    .attr('x', -10)
    .attr('y', neuronSize / 2 + 4)
    .attr('text-anchor', 'end')
    .attr('fill', 'rgba(255, 255, 255, 0.7)')
    .style('font-family', 'monospace')
    .style('font-size', '12px')
    .text(d => d.label);
    
  inputNeurons.each(function(d) {
    if (transformationPreviews && transformationPreviews[d.label]) {
      const previewSVG = transformationPreviews[d.label];
      
      // Append the preview SVG to the neuron group
      try {
        d3.select(this)
          .append(() => previewSVG.node()) // Append the actual SVG node
          .attr('x', neuronSize + 5) // Position to the right of the neuron
          .attr('y', 0)
          .attr('width', transformationPreviewSize)
          .attr('height', transformationPreviewSize);
      } catch (e) {
        console.warn('Failed to append transformation preview:', e);
      }
    }
  });
    
  // Add hover effects
  neurons.on('mouseover', function(event, d) {
    // Highlight this neuron
    d3.select(this).select('rect')
      .attr('stroke', 'var(--accent-blue)')
      .attr('stroke-width', 2)
      .transition()
      .duration(200)
      .attr('width', neuronSize + 4)
      .attr('height', neuronSize + 4)
      .attr('transform', `translate(-2, -2)`);
      
    // Highlight connected edges
    edgeGroup.selectAll('line.edge')
      .filter(e => (e.sourceLayer === d.layer && e.sourceNeuron === d.index) ||
                   (e.targetLayer === d.layer && e.targetNeuron === d.index))
      .transition()
      .duration(200)
      .attr('stroke-width', e => Math.max(1.5, Math.abs(e.weight) * 3))
      .attr('opacity', 1);
      
    // Show tooltip with neuron info
    let activationValue = 0;
    let layerName = '';
    
    if (d.layer === 0) {
      layerName = 'Input';
      activationValue = trainX && trainX.length > 0 ? trainX[0][d.index] : 0;
    } else if (d.layer === totalLayers - 1) {
      layerName = 'Output';
      const layerIndex = d.layer - 1; // Adjust for neural network indexing
      activationValue = sampleActivations && sampleActivations.length > layerIndex ? 
                        sampleActivations[layerIndex][d.index] : 0;
    } else {
      layerName = 'Hidden';
      const layerIndex = d.layer - 1; // Adjust for neural network indexing
      activationValue = sampleActivations && sampleActivations.length > layerIndex ? 
                        sampleActivations[layerIndex][d.index] : 0;
    }
    
    // Format the activation value
    const formattedActivation = Math.round(activationValue * 1000) / 1000;
    
    // Show tooltip
    tooltip.transition()
      .delay(tooltipDelay)
      .duration(200)
      .style('opacity', 1);
    
    tooltip.html(`
      <div class="tooltip-title">${layerName} Neuron ${d.index + 1}</div>
      <div class="tooltip-content">
        ${d.label ? `Feature: <strong>${d.label}</strong><br>` : ''}
        Activation: <strong>${formattedActivation}</strong>
      </div>
    `)
    .style('left', (event.pageX + 15) + 'px')
    .style('top', (event.pageY - 40) + 'px');
    
    // Note: We've already highlighted edges above using 'line.edge' selector
  })
  .on('mouseout', function() {
    // Reset neuron highlight
    d3.select(this).select('rect')
      .attr('stroke', 'rgba(255, 255, 255, 0.3)')
      .attr('stroke-width', 1)
      .transition()
      .duration(200)
      .attr('width', neuronSize)
      .attr('height', neuronSize)
      .attr('transform', 'translate(0, 0)');
    
    // Reset edge highlight
    edgeGroup.selectAll('line.edge')
      .transition()
      .duration(200)
      .attr('stroke-width', d => Math.max(0.5, Math.abs(d.weight) * 2))
      .attr('opacity', 0.7);
      
    // Hide tooltip
    tooltip.transition()
      .duration(200)
      .style('opacity', 0);
  });
  
  // Return network data for animation
  return {
    neuronPositions,
    edgesData,
    sampleActivations
  };
}

/**
 * Transform a sample input point for the model based on enabled features
 * Includes transformation previews
 * @param {Array} sample - Raw input point [x1, x2]
 * @param {Array} featureNames - Names of enabled features
 * @param {Object} transformationPreviews - (new) SVG previews for feature transformations
 * @returns {tf.Tensor} Tensor ready for model input
 */
function transformSampleForModel(sample, featureNames, transformationPreviews = null) {
  const x1 = sample[0];
  const x2 = sample[1];
  const features = [];
  
  for (const name of featureNames) {
    if (name === 'X₁') features.push(x1);
    else if (name === 'X₂') features.push(x2);
    else if (name === 'X₁²') features.push(x1 * x1);
    else if (name === 'X₂²') features.push(x2 * x2);
    else if (name === 'X₁X₂') features.push(x1 * x2);
    else if (name === 'sin(X₁)') features.push(Math.sin(x1));
    else if (name === 'sin(X₂)') features.push(Math.sin(x2));
  }
  
  return tf.tensor2d([features]);
}

/**
 * Get activations for all layers in the model for a given input
 * @param {tf.Sequential} model - The neural network model
 * @param {tf.Tensor} input - Input tensor
 * @returns {Array} Array of activation values (as plain JS arrays) for each layer
 */
function getActivations(model, input) {
  return tf.tidy(() => {
    const activations = [];
    let currentInput = input;

    // Input layer activations (already a tensor)
    activations.push(Array.from(currentInput.dataSync()));

    // Hidden layers and output layer activations
    for (let i = 0; i < model.layers.length; i++) {
      const layer = model.layers[i];
      currentInput = layer.apply(currentInput); // Creates intermediate tensor
      activations.push(Array.from(currentInput.dataSync())); // Copy data to JS array
    }
    // Intermediate tensors created by layer.apply() are disposed by tf.tidy()
    return activations; // Return plain JS arrays
  });
}

/**
 * Visualize data flow through the network with animated particles
 * @param {Array} neuronPositions - Array of neuron position data
 * @param {Array} edgesData - Array of edge data
 * @param {Array} activations - Activation values for each layer
 */
function animateDataFlow(neuronPositions, edgesData, activations) {
  // Clear previous animations
  dataFlowSvg.selectAll('*').remove();

  // We'll animate from input layer to output layer
  // For each layer transition, animate particles along edges
  for (let l = 0; l < activations.length - 1; l++) {
    const sourceLayer = l;
    const targetLayer = l + 1;
    const delay = l * (dataFlowAnimationDuration / 3); // Stagger the animations
    
    // Get edges between current layer and next layer
    const layerEdges = edgesData.filter(e => 
      e.sourceLayer === sourceLayer && e.targetLayer === targetLayer
    );
    
    // Animate particles along these edges
    layerEdges.forEach(edge => {
      // Get activation value to determine particle size and opacity
      const sourceActivation = activations[sourceLayer][edge.sourceNeuron];
      const weight = edge.weight;
      
      // Skip tiny activations
      if (Math.abs(sourceActivation) < 0.05) return;
      
      // Determine if the signal is excitatory (positive) or inhibitory (negative)
      const isExcitatory = sourceActivation > 0;
      const signalStrength = Math.min(1, Math.abs(sourceActivation) * 1.5); // Adjusted opacity scaling

      // OPTIMIZATION: Create only one particle per edge, adjust size/opacity
      const particleCount = 1; // Reduced from multiple particles

      for (let i = 0; i < particleCount; i++) {
        // Add small random offset to departure time for more natural look
        const particleDelay = delay + Math.random() * 50; // Reduced random delay

        // Create a particle
        const particle = dataFlowSvg.append('circle')
          .attr('class', 'data-particle')
          .attr('cx', edge.x1)
          .attr('cy', edge.y1)
          .attr('r', Math.min(4, Math.abs(sourceActivation) * 4 + 1.5)) // Adjusted size scaling
          .attr('fill', isExcitatory ? 'var(--accent-orange)' : 'var(--accent-blue)')
          .attr('opacity', signalStrength * 0.8); // Slightly reduced max opacity
          // OPTIMIZATION: Removed SVG filters (glow)

        // Animate the particle along the edge (simplified to straight line)
        const midX = (edge.x1 + edge.x2) / 2;
        const midY = (edge.y1 + edge.y2) / 2;
        // OPTIMIZATION: Removed random offset for curve, using straight line

        particle.transition()
          .delay(particleDelay)
          .duration(dataFlowAnimationDuration) // Simplified to single transition
          .attr('cx', edge.x2)
          .attr('cy', edge.y2)
          .ease(d3.easeLinear) // Use linear easing for simplicity
          .on('end', function() {
            // OPTIMIZATION: Removed ripple and flash effects
            d3.select(this).remove(); // Remove the particle
          });
      }
    });
  }

  // OPTIMIZATION: Removed neuron highlighting based on activation to simplify
}

// Cache for decision boundary data to avoid unnecessary recalculations
let decisionBoundaryCache = {
  modelHash: null,
  discretize: null,
  resolution: null,
  imageData: null
};

/**
 * Visualizes the decision boundary on a canvas
 * @param {tf.Sequential} model - The neural network model
 * @param {Object} dataGenerator - The data generator instance
 * @param {Array} trainX - Training data points
 * @param {Array} trainY - Training data labels
 * @param {Array} testX - Test data points
 * @param {Array} testY - Test data labels
 * @param {boolean} showTest - Whether to show test data
 * @param {boolean} discretize - Whether to discretize the output
 */
function visualizeDecisionBoundary(model, dataGenerator, trainX, trainY, testX, testY, showTest, discretize) {
  console.log('[Vis] visualizeDecisionBoundary called. Model:', model ? 'Exists' : 'NULL', 'Canvas:', decisionCanvas ? 'Exists' : 'NULL');
  if (!decisionCanvas) {
    console.error('[Vis] visualizeDecisionBoundary: Decision boundary canvas (decisionCanvas) not initialized. Aborting.');
    return;
  }
  if (!model) {
    console.warn('[Vis] visualizeDecisionBoundary: Model not provided. Aborting.');
    return;
  }
  if (!decisionCanvas) return;
  
  const ctx = decisionCanvas.getContext('2d');
  const width = decisionCanvas.width;
  const height = decisionCanvas.height;
  
  // Clear the canvas
  ctx.clearRect(0, 0, width, height);
  
  // Create a grid of points to evaluate
  const gridSize = decisionBoundaryResolution;
  const gridStep = width / gridSize;
  
  // Find data bounds for scaling
  let xMin = -6, xMax = 6, yMin = -6, yMax = 6;
  
  // Enable high-quality rendering
  ctx.imageSmoothingEnabled = false;
  
  // Scale function to map from pixel to data coordinates
  const scaleX = d3.scaleLinear().domain([0, width]).range([xMin, xMax]);
  const scaleY = d3.scaleLinear().domain([0, height]).range([yMax, yMin]);
  
  // Create image data for the decision boundary
  const imageData = ctx.createImageData(width, height);
  const data = imageData.data;
  
  // Process in batches for performance
  tf.tidy(() => {
    // Generate grid points and transform them
    const points = [];
    
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const x = scaleX(j * gridStep);
        const y = scaleY(i * gridStep);
        points.push([x, y]);
      }
    }

    // Transform points using the DataGenerator's method
    const transformedPoints = dataGenerator.transformFeatures(points);

    // Run prediction on all points
    const input = tf.tensor2d(transformedPoints);
    const predictions = model.predict(input);
    const values = predictions.dataSync();
    
    // Get Material colors (assuming they are loaded)
    const colorNeg = getCssVar('--md-error') || '#cf6679'; // Reddish for negative
    const colorPos = getCssVar('--md-primary') || '#bb86fc'; // Primary for positive

    // Convert hex/rgb to [r, g, b] arrays
    const parseColor = (colorStr) => {
      if (colorStr.startsWith('#')) {
        const hex = colorStr.substring(1);
        const bigint = parseInt(hex, 16);
        return [(bigint >> 16) & 255, (bigint >> 8) & 255, bigint & 255];
      } else if (colorStr.startsWith('rgb')) {
        return colorStr.match(/\d+/g).map(Number);
      }
      return [128, 128, 128]; // Default gray
    };

    const rgbNeg = parseColor(colorNeg);
    const rgbPos = parseColor(colorPos);

    // Color the image data based on predictions
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const idx = i * gridSize + j;
        let val = values[idx];

        // Discretize if requested
        if (discretize) {
          val = val > 0.5 ? 1.0 : 0.0;
        }

        // Interpolate color based on value (0 -> Neg, 1 -> Pos)
        const r = Math.round(rgbNeg[0] * (1 - val) + rgbPos[0] * val);
        const g = Math.round(rgbNeg[1] * (1 - val) + rgbPos[1] * val);
        const b = Math.round(rgbNeg[2] * (1 - val) + rgbPos[2] * val);

        // Fill the grid cell
        const pixelXBase = Math.floor(j * gridStep);
        const pixelYBase = Math.floor(i * gridStep);

        for (let py = pixelYBase; py < pixelYBase + gridStep && py < height; py++) {
          for (let px = pixelXBase; px < pixelXBase + gridStep && px < width; px++) {
            const offset = (py * width + px) * 4;
            data[offset] = r;
            data[offset + 1] = g;
            data[offset + 2] = b;
            data[offset + 3] = 255; // Alpha
          }
        }
      }
    }

    // Put the image data onto the canvas
    ctx.putImageData(imageData, 0, 0);

    // Function to draw a data point
    const drawPoint = (x, y, label, isTest = false) => {
      const pixelX = ((x - xMin) / (xMax - xMin)) * width;
      const pixelY = height - ((y - yMin) / (yMax - yMin) * height);
      
      ctx.beginPath();
      ctx.arc(pixelX, pixelY, isTest ? 5 : 4, 0, 2 * Math.PI);
      ctx.fillStyle = label > 0.5 ? '#5B46E5' : '#F44336';
      ctx.fill();
      
      if (isTest) {
        ctx.strokeStyle = '#FFD700';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    };

    // Draw training points
    if (trainX && trainY) {
      for (let i = 0; i < trainX.length; i++) {
        drawPoint(trainX[i][0], trainX[i][1], trainY[i]);
      }
    }
    
    // Draw test points if requested
    if (showTest && testX && testY) {
      ctx.globalAlpha = 0.7;
      for (let i = 0; i < testX.length; i++) {
        drawPoint(testX[i][0], testX[i][1], testY[i], true);
      }
      ctx.globalAlpha = 1.0;
    }
  });
}

// Store current decision boundary data for redrawing on resize
let currentDecisionBoundaryData = null;

// Export visualization functions
window.nn_vis = {
  visualizeNetwork,
  visualizeDecisionBoundary: function(model, dataGenerator, trainX, trainY, testX, testY, showTest, discretize) {
    if (!decisionCanvas) {
      console.error('Decision boundary canvas not found');
      return;
    }
    
    // Store the data for resize handling
    currentDecisionBoundaryData = {
      model, dataGenerator, trainX, trainY, testX, testY, showTest, discretize
    };
    
    // Call the visualization function
    return visualizeDecisionBoundary(model, dataGenerator, trainX, trainY, testX, testY, showTest, discretize);
  },
  animateDataFlow,
  getActivations,
  createTransformationPreview
};

/**
 * (new) Function to create SVG previews for feature transformations
 * @param {string} featureName - Name of the feature (e.g., 'X₁²', 'sin(X₁)')
 * @returns {d3.Selection} - D3 selection of the created SVG
 */
function createTransformationPreview(featureName) {
  try {
    // Create a simple SVG element directly without D3
    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("width", transformationPreviewSize.toString());
    svg.setAttribute("height", transformationPreviewSize.toString());
    svg.setAttribute("viewBox", "0 0 100 100");
    
    // Create a path element
    const path = document.createElementNS(svgNS, "path");
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "#4d8bf9"); // Use a direct color value instead of CSS variable
    path.setAttribute("stroke-width", "4");
    path.setAttribute("stroke-linejoin", "round");
    
    // Set path data based on feature type
    if (featureName === 'X₁²' || featureName === 'X₂²') {
      path.setAttribute("d", "M10,90 Q50,10 90,90"); // Simple parabola
    } else if (featureName === 'X₁X₂') {
      path.setAttribute("d", "M10,90 L90,10"); // Simple line for X₁X₂
    } else if (featureName === 'sin(X₁)' || featureName === 'sin(X₂)') {
      path.setAttribute("d", "M10,50 Q25,10 50,50 T90,50"); // Simple sine-like curve
    } else {
      // Default curve
      path.setAttribute("d", "M10,50 H90");
    }
    
    svg.appendChild(path);
    
    // Create a D3 selection from the SVG element
    const selection = d3.select(svg);
    
    return selection;
  } catch (e) {
    console.error("Error creating transformation preview:", e);
    // Return a minimal valid D3 selection with a node
    return d3.select(document.createElementNS("http://www.w3.org/2000/svg", "svg"));
  }
}
