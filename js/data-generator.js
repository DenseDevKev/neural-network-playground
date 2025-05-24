// data-generator.js - minimal dataset generator for demo

//
// This file handles dataset generation and feature transformations.
// It provides functions to create different types of datasets and transform features.

/**
 * Class to handle dataset generation and feature transformations
 */
class DataGenerator {
  constructor() {
    this.rawData = []; // Original 2D points
    this.rawLabels = []; // Original labels
    this.currentDataset = 'spiral';
    this.pointsPerClass = 100;
    this.noise = 0;
    this.trainTestRatio = 60;
    this.enabledFeatures = {
      x1: true,
      x2: true,
      x1Squared: true,
      x2Squared: true,
      x1x2: true,
      sinX1: true,
      sinX2: true
    };
  }

  /**
   * Generate a dataset based on the current settings
   * @returns {Object} Object containing training and test data as TensorFlow tensors
   */
  generateData() {
    // Clear previous data
    this.rawData = [];
    this.rawLabels = [];

    // Generate the appropriate dataset
    switch (this.currentDataset) {
      case 'spiral':
        this.generateSpiralData();
        break;
      case 'circle':
        this.generateCircleData();
        break;
      case 'xor':
        this.generateXORData();
        break;
      case 'gaussian':
        this.generateGaussianData();
        break;
      default:
        this.generateSpiralData();
    }

    // Shuffle and split the data
    return this.processData();
  }

  /**
   * Generate a spiral dataset
   */
  generateSpiralData() {
    const classes = 2;
    
    for (let i = 0; i < classes; i++) {
      const r = i === 0 ? 1 : -1; // direction
      for (let j = 0; j < this.pointsPerClass; j++) {
        const rate = j * 1.5 / this.pointsPerClass;
        const x = rate * Math.sin(2.5 * rate) * r + this.randUniform(-this.noise/100, this.noise/100);
        const y = rate * Math.cos(2.5 * rate) * r + this.randUniform(-this.noise/100, this.noise/100);
        this.rawData.push([x, y]);
        this.rawLabels.push(i);
      }
    }
  }

  /**
   * Generate a circle dataset
   */
  generateCircleData() {
    const innerRadius = 0.3;
    const outerRadius = 0.8;
    
    // Inner circle (class 0)
    for (let i = 0; i < this.pointsPerClass; i++) {
      const angle = this.randUniform(0, 2 * Math.PI);
      const radius = innerRadius * Math.sqrt(this.randUniform(0, 1)) + this.randUniform(-this.noise/100, this.noise/100);
      const x = radius * Math.cos(angle);
      const y = radius * Math.sin(angle);
      this.rawData.push([x, y]);
      this.rawLabels.push(0);
    }
    
    // Outer circle (class 1)
    for (let i = 0; i < this.pointsPerClass; i++) {
      const angle = this.randUniform(0, 2 * Math.PI);
      const radius = outerRadius * Math.sqrt(this.randUniform(0.7, 1)) + this.randUniform(-this.noise/100, this.noise/100);
      const x = radius * Math.cos(angle);
      const y = radius * Math.sin(angle);
      this.rawData.push([x, y]);
      this.rawLabels.push(1);
    }
  }

  /**
   * Generate an XOR dataset
   */
  generateXORData() {
    const radius = 0.3;
    const centerDistance = 0.6;
    
    // Top-left cluster (class 0)
    this.generateGaussianCluster(-centerDistance, -centerDistance, radius, 0);
    
    // Bottom-right cluster (class 0)
    this.generateGaussianCluster(centerDistance, centerDistance, radius, 0);
    
    // Top-right cluster (class 1)
    this.generateGaussianCluster(centerDistance, -centerDistance, radius, 1);
    
    // Bottom-left cluster (class 1)
    this.generateGaussianCluster(-centerDistance, centerDistance, radius, 1);
  }

  /**
   * Generate a dataset with two Gaussian clusters
   */
  generateGaussianData() {
    // Class 0 cluster
    this.generateGaussianCluster(-0.5, -0.5, 0.4, 0);
    
    // Class 1 cluster
    this.generateGaussianCluster(0.5, 0.5, 0.4, 1);
  }

  /**
   * Helper function to generate a Gaussian cluster
   * @param {number} centerX - X coordinate of cluster center
   * @param {number} centerY - Y coordinate of cluster center
   * @param {number} radius - Cluster radius (standard deviation)
   * @param {number} label - Class label for this cluster
   */
  generateGaussianCluster(centerX, centerY, radius, label) {
    const pointsInCluster = Math.floor(this.pointsPerClass / 2); // For XOR we need 4 clusters
    
    for (let i = 0; i < pointsInCluster; i++) {
      // Box-Muller transform for Gaussian distribution
      const u1 = this.randUniform(0, 1);
      const u2 = this.randUniform(0, 1);
      const r = radius * Math.sqrt(-2 * Math.log(u1)) * (1 + this.randUniform(-this.noise/100, this.noise/100));
      const theta = 2 * Math.PI * u2;
      
      const x = centerX + r * Math.cos(theta);
      const y = centerY + r * Math.sin(theta);
      
      this.rawData.push([x, y]);
      this.rawLabels.push(label);
    }
  }

  /**
   * Process the raw data: shuffle, transform features, split into train/test
   * @returns {Object} Object containing training and test data as TensorFlow tensors
   */
  processData() {
    // Shuffle the data
    const shuffleIndices = tf.util.createShuffledIndices(this.rawData.length);
    const shuffledData = [];
    const shuffledLabels = [];
    
    for (let idx of shuffleIndices) {
      shuffledData.push(this.rawData[idx]);
      shuffledLabels.push(this.rawLabels[idx]);
    }
    
    // Apply feature transformations
    const transformedData = this.transformFeatures(shuffledData);
    
    // Split into training and test sets
    const nTrain = Math.floor((this.trainTestRatio / 100) * transformedData.length);
    
    const xTrain = transformedData.slice(0, nTrain);
    const yTrain = shuffledLabels.slice(0, nTrain);
    const xTest = transformedData.slice(nTrain);
    const yTest = shuffledLabels.slice(nTrain);
    
    // Convert to TensorFlow tensors
    return {
      xs: tf.tensor2d(xTrain),
      ys: tf.tensor1d(yTrain, 'int32'),
      xsTest: tf.tensor2d(xTest),
      ysTest: tf.tensor1d(yTest, 'int32'),
      // Also return raw data for visualization
      rawXs: shuffledData.slice(0, nTrain),
      rawYs: shuffledLabels.slice(0, nTrain),
      rawXsTest: shuffledData.slice(nTrain),
      rawYsTest: shuffledLabels.slice(nTrain)
    };
  }

  /**
   * Transform features based on enabled feature flags
   * @param {Array} data - Array of [x, y] points
   * @returns {Array} Array of transformed feature vectors
   */
  transformFeatures(data) {
    return data.map(point => {
      const x1 = point[0];
      const x2 = point[1];
      const features = [];
      
      if (this.enabledFeatures.x1) features.push(x1);
      if (this.enabledFeatures.x2) features.push(x2);
      if (this.enabledFeatures.x1Squared) features.push(x1 * x1);
      if (this.enabledFeatures.x2Squared) features.push(x2 * x2);
      if (this.enabledFeatures.x1x2) features.push(x1 * x2);
      if (this.enabledFeatures.sinX1) features.push(Math.sin(x1));
      if (this.enabledFeatures.sinX2) features.push(Math.sin(x2));
      
      return features;
    });
  }

  /**
   * Get the number of enabled features
   * @returns {number} Count of enabled features
   */
  getEnabledFeatureCount() {
    return Object.values(this.enabledFeatures).filter(Boolean).length;
  }

  /**
   * Get the names of enabled features
   * @returns {Array} Array of enabled feature names
   */
  getEnabledFeatureNames() {
    const names = [];
    if (this.enabledFeatures.x1) names.push('X₁');
    if (this.enabledFeatures.x2) names.push('X₂');
    if (this.enabledFeatures.x1Squared) names.push('X₁²');
    if (this.enabledFeatures.x2Squared) names.push('X₂²');
    if (this.enabledFeatures.x1x2) names.push('X₁X₂');
    if (this.enabledFeatures.sinX1) names.push('sin(X₁)');
    if (this.enabledFeatures.sinX2) names.push('sin(X₂)');
    return names;
  }

  /**
   * Utility for random uniform values in [min, max]
   */
  randUniform(min, max) {
    return Math.random() * (max - min) + min;
  }

  /**
   * Set the current dataset type
   * @param {string} dataset - Dataset type ('spiral', 'circle', 'xor', 'gaussian')
   */
  setDataset(dataset) {
    this.currentDataset = dataset;
  }

  /**
   * Set the noise level
   * @param {number} noise - Noise level (0-50)
   */
  setNoise(noise) {
    this.noise = noise;
  }

  /**
   * Set the train/test ratio
   * @param {number} ratio - Percentage of data to use for training (10-90)
   */
  setTrainTestRatio(ratio) {
    this.trainTestRatio = ratio;
  }

  /**
   * Set the number of points per class
   * @param {number} points - Number of points per class
   */
  setPointsPerClass(points) {
    this.pointsPerClass = points;
  }

  /**
   * Toggle a feature on or off
   * @param {string} feature - Feature name
   * @param {boolean} enabled - Whether the feature is enabled
   */
  setFeatureEnabled(feature, enabled) {
    if (feature in this.enabledFeatures) {
      this.enabledFeatures[feature] = enabled;
    }
  }
}
