// Neural Network Playground - minimal fresh implementation
// Loads TensorFlow.js and D3 via global scripts in index.html

const tf = window.tf;
const d3 = window.d3;

// ---------- State ----------
const state = {
  // Data/config
  dataset: 'circle',
  ratio: 50,
  noise: 0,
  features: { x1: true, x2: true, x1Squared: false, x2Squared: false, x1x2: false, sinX1: false, sinX2: false, cosX1: false, cosX2: false },
  hiddenLayers: 3,
  neuronsPerLayer: 7,
  activation: 'relu',
  learningRate: 0.03,
  regularization: 'none', // 'none' | 'l1' | 'l2'
  regularizationRate: 0,
  discretize: false,
  showTest: false,
  batchSize: 10,
  // Runtime
  epoch: 0,
  model: null,
  xs: null, ys: null, xsTest: null, ysTest: null,
  rawTrain: [], rawTest: [], rawTrainY: [], rawTestY: [],
  lossHistory: [], valLossHistory: [],
};

// ---------- Data generation ----------
function rand(min, max) { return Math.random() * (max - min) + min; }

function genCircle(pointsPerClass, noise) {
  const innerRadius = 0.3;
  const outerRadius = 0.8;
  const X = []; const y = [];
  for (let i = 0; i < pointsPerClass; i++) {
    const a = rand(0, 2 * Math.PI); const r = innerRadius * Math.sqrt(Math.random()) + rand(-noise, noise);
    X.push([r * Math.cos(a), r * Math.sin(a)]); y.push(0);
  }
  for (let i = 0; i < pointsPerClass; i++) {
    const a = rand(0, 2 * Math.PI); const r = outerRadius * Math.sqrt(rand(0.7, 1)) + rand(-noise, noise);
    X.push([r * Math.cos(a), r * Math.sin(a)]); y.push(1);
  }
  return { X, y };
}

function genSpiral(pointsPerClass, noise) {
  const X = []; const y = [];
  const classes = 2; const rateScale = 1.5;
  for (let c = 0; c < classes; c++) {
    const dir = c === 0 ? 1 : -1;
    for (let i = 0; i < pointsPerClass; i++) {
      const t = i * rateScale / pointsPerClass;
      const x = t * Math.sin(2.5 * t) * dir + rand(-noise, noise);
      const yy = t * Math.cos(2.5 * t) * dir + rand(-noise, noise);
      X.push([x, yy]); y.push(c);
    }
  }
  return { X, y };
}

function genXor(pointsPerClass, noise) {
  const X = []; const y = [];
  const r = 0.3; const d = 0.6;
  const clusters = [
    [-d, -d, 0], [d, d, 0], [d, -d, 1], [-d, d, 1],
  ];
  const per = Math.ceil(pointsPerClass / 2);
  for (const [cx, cy, lab] of clusters) {
    for (let i = 0; i < per; i++) {
      const u1 = Math.random(); const u2 = Math.random();
      const rr = r * Math.sqrt(-2 * Math.log(u1)) * (1 + rand(-noise, noise));
      const th = 2 * Math.PI * u2;
      X.push([cx + rr * Math.cos(th), cy + rr * Math.sin(th)]); y.push(lab);
    }
  }
  return { X, y };
}

function genGaussian(pointsPerClass, noise) {
  const X = []; const y = [];
  const clusters = [ [-0.5, -0.5, 0], [0.5, 0.5, 1] ];
  const per = Math.ceil(pointsPerClass / 2);
  for (const [cx, cy, lab] of clusters) {
    for (let i = 0; i < per; i++) {
      const u1 = Math.random(); const u2 = Math.random();
      const rr = 0.4 * Math.sqrt(-2 * Math.log(u1)) * (1 + rand(-noise, noise));
      const th = 2 * Math.PI * u2;
      X.push([cx + rr * Math.cos(th), cy + rr * Math.sin(th)]); y.push(lab);
    }
  }
  return { X, y };
}

function getEnabledFeatureCount() {
  return Object.values(state.features).filter(Boolean).length;
}

function transformFeatures(raw) {
  const enabled = [];
  if (state.features.x1) enabled.push('x1');
  if (state.features.x2) enabled.push('x2');
  if (state.features.x1Squared) enabled.push('x1Squared');
  if (state.features.x2Squared) enabled.push('x2Squared');
  if (state.features.x1x2) enabled.push('x1x2');
  if (state.features.sinX1) enabled.push('sinX1');
  if (state.features.sinX2) enabled.push('sinX2');
  if (state.features.cosX1) enabled.push('cosX1');
  if (state.features.cosX2) enabled.push('cosX2');
  return raw.map(([x1, x2]) => enabled.map(f => {
    switch (f) {
      case 'x1': return x1;
      case 'x2': return x2;
      case 'x1Squared': return x1 * x1;
      case 'x2Squared': return x2 * x2;
      case 'x1x2': return x1 * x2;
      case 'sinX1': return Math.sin(x1);
      case 'sinX2': return Math.sin(x2);
      case 'cosX1': return Math.cos(x1);
      case 'cosX2': return Math.cos(x2);
      default: return 0;
    }
  }));
}

function generateData() {
  const pointsPerClass = 100;
  const noise = state.noise / 100;
  let data;
  if (state.dataset === 'circle') data = genCircle(pointsPerClass, noise);
  else if (state.dataset === 'spiral') data = genSpiral(pointsPerClass, noise);
  else if (state.dataset === 'xor') data = genXor(pointsPerClass, noise);
  else data = genGaussian(pointsPerClass, noise);

  // shuffle
  const idx = tf.util.createShuffledIndices(data.X.length);
  const Xs = []; const ys = [];
  for (const i of idx) { Xs.push(data.X[i]); ys.push(data.y[i]); }
  const nTrain = Math.floor((state.ratio / 100) * Xs.length);
  const trainX = Xs.slice(0, nTrain);
  const trainY = ys.slice(0, nTrain);
  const testX = Xs.slice(nTrain);
  const testY = ys.slice(nTrain);

  // tensors
  const tTrain = transformFeatures(trainX);
  const tTest = transformFeatures(testX);
  state.xs?.dispose(); state.ys?.dispose(); state.xsTest?.dispose(); state.ysTest?.dispose();
  state.xs = tf.tensor2d(tTrain);
  state.ys = tf.tensor2d(trainY.map(v => [v]), [trainY.length, 1], 'float32');
  state.xsTest = tf.tensor2d(tTest);
  state.ysTest = tf.tensor2d(testY.map(v => [v]), [testY.length, 1], 'float32');

  state.rawTrain = trainX; state.rawTest = testX; state.rawTrainY = trainY; state.rawTestY = testY;
}

// ---------- Model ----------
function createModel() {
  state.model?.dispose();
  const model = tf.sequential();
  const inputDim = transformFeatures([[0,0]])[0].length || 1;
  const regularizer = state.regularization === 'l1'
    ? tf.regularizers.l1({ l1: state.regularizationRate })
    : state.regularization === 'l2'
      ? tf.regularizers.l2({ l2: state.regularizationRate })
      : undefined;
  model.add(tf.layers.dense({ units: state.neuronsPerLayer, activation: state.activation, inputShape: [inputDim], kernelRegularizer: regularizer }));
  for (let i = 0; i < state.hiddenLayers - 1; i++) {
    model.add(tf.layers.dense({ units: state.neuronsPerLayer, activation: state.activation, kernelRegularizer: regularizer }));
  }
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: tf.train.adam(state.learningRate), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  state.model = model;
}

// ---------- Visualization ----------
const netContainer = document.getElementById('network-container');
const boundaryContainer = document.getElementById('decision-boundary-container');
const svg = d3.select(netContainer).append('svg').attr('width', '100%').attr('height', '100%')
  .attr('preserveAspectRatio', 'xMidYMid meet');
const canvas = d3.select(boundaryContainer).append('canvas').attr('width', 400).attr('height', 400)
  .style('width', '100%').style('height', '100%').node();
const lossCanvas = document.getElementById('lossChart');

function drawNetwork() {
  if (!state.model) return;
  svg.selectAll('*').remove();
  const neuronSize = 20, gap = 120;
  const inputDim = state.xs.shape[1];
  const layerSizes = [inputDim, ...state.model.layers.map(l => l.getConfig().units)];

  // Dynamic viewBox width based on number of layers
  const viewBoxWidth = gap * (layerSizes.length + 1);
  const height = 500;
  svg.attr('viewBox', `0 0 ${viewBoxWidth} ${height}`);

  const cx = i => gap * (i + 1);
  const layerY = (count) => {
    const total = count * (neuronSize + 8);
    const start = (height - total) / 2;
    return Array.from({ length: count }, (_, k) => start + k * (neuronSize + 8) + neuronSize / 2);
  };

  // build positions
  const positions = [];
  for (let l = 0; l < layerSizes.length; l++) {
    const ys = layerY(layerSizes[l]);
    ys.forEach((y, i) => positions.push({ layer: l, idx: i, x: cx(l), y }));
  }
  // edges
  const edges = [];
  for (let l = 0; l < layerSizes.length - 1; l++) {
    const src = positions.filter(p => p.layer === l);
    const dst = positions.filter(p => p.layer === l + 1);
    for (const s of src) for (const d of dst) edges.push({ x1: s.x, y1: s.y, x2: d.x, y2: d.y });
  }
  svg.append('g').selectAll('line').data(edges).enter().append('line')
    .attr('x1', d => d.x1).attr('y1', d => d.y1).attr('x2', d => d.x2).attr('y2', d => d.y2)
    .attr('stroke', 'rgba(255,255,255,0.2)').attr('stroke-width', 1);
  svg.append('g').selectAll('circle').data(positions).enter().append('circle')
    .attr('cx', d => d.x).attr('cy', d => d.y).attr('r', neuronSize/2)
    .attr('fill', '#444').attr('stroke', 'rgba(255,255,255,0.3)');
}

function drawDecisionBoundary() {
  if (!state.model || !state.xs) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width, height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  const grid = 120; const step = width / grid;
  const xMin = -6, xMax = 6, yMin = -6, yMax = 6;
  const scaleX = d3.scaleLinear().domain([0, width]).range([xMin, xMax]);
  const scaleY = d3.scaleLinear().domain([0, height]).range([yMax, yMin]);

  const points = [];
  for (let i = 0; i < grid; i++) {
    for (let j = 0; j < grid; j++) {
      const x = scaleX(j * step); const y = scaleY(i * step);
      points.push([x, y]);
    }
  }
  const feats = transformFeatures(points);
  tf.tidy(() => {
    const input = tf.tensor2d(feats);
    const pred = state.model.predict(input);
    const vals = pred.dataSync();
    let k = 0;
    for (let i = 0; i < grid; i++) {
      for (let j = 0; j < grid; j++) {
        let v = vals[k++];
        if (state.discretize) v = v > 0.5 ? 1 : 0;
        const r = Math.round(207 * (1 - v) + 187 * v);
        const g = Math.round(102 * (1 - v) + 134 * v);
        const b = Math.round(121 * (1 - v) + 252 * v);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(Math.floor(j * step), Math.floor(i * step), Math.ceil(step), Math.ceil(step));
      }
    }
  });

  // points
  const drawPt = (x, y, lab, isTest) => {
    const px = ((x - xMin) / (xMax - xMin)) * width;
    const py = height - ((y - yMin) / (yMax - yMin) * height);
    ctx.beginPath(); ctx.arc(px, py, isTest ? 4 : 3, 0, 2 * Math.PI);
    ctx.fillStyle = lab ? '#5B46E5' : '#F44336'; ctx.fill();
    if (isTest) { ctx.strokeStyle = '#FFD700'; ctx.lineWidth = 1; ctx.stroke(); }
  };
  state.rawTrain.forEach((p, i) => drawPt(p[0], p[1], state.rawTrainY[i], false));
  if (state.showTest) state.rawTest.forEach((p, i) => drawPt(p[0], p[1], state.rawTestY[i], true));
}

function drawLossChart() {
  if (!lossCanvas) return;
  const ctx = lossCanvas.getContext('2d');
  const w = lossCanvas.width, h = lossCanvas.height;
  ctx.clearRect(0, 0, w, h);
  const pad = 38;
  const padLeft = 48;
  const xs = state.lossHistory.map((_, i) => i);
  const all = state.lossHistory.concat(state.valLossHistory);
  if (all.length === 0) return;
  const minY = Math.min(...all);
  const maxY = Math.max(...all);
  const xScale = t => padLeft + (w - padLeft - pad) * (t / Math.max(1, xs[xs.length - 1] || 1));
  const yScale = v => h - pad - (h - pad - 20) * ((v - minY) / Math.max(1e-6, (maxY - minY) || 1));

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = 20 + (h - pad - 20) * (i / 4);
    ctx.beginPath(); ctx.moveTo(padLeft, y); ctx.lineTo(w - pad, y); ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = 'rgba(255,255,255,0.25)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(padLeft, 20); ctx.lineTo(padLeft, h - pad); ctx.lineTo(w - pad, h - pad); ctx.stroke();

  // Y-axis labels
  ctx.fillStyle = 'rgba(255,255,255,0.5)';
  ctx.font = '10px Roboto, Arial, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i <= 4; i++) {
    const val = maxY - (maxY - minY) * (i / 4);
    const y = 20 + (h - pad - 20) * (i / 4);
    ctx.fillText(val.toFixed(2), padLeft - 4, y);
  }

  // X-axis label
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  ctx.font = '11px Roboto, Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('Epoch', (padLeft + w - pad) / 2, h - pad + 10);

  // Y-axis label
  ctx.save();
  ctx.translate(12, (20 + h - pad) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  ctx.font = '11px Roboto, Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('Loss', 0, 0);
  ctx.restore();

  // Training loss line
  ctx.strokeStyle = '#03dac6'; ctx.lineWidth = 2; ctx.beginPath();
  state.lossHistory.forEach((v, i) => { const x = xScale(i); const y = yScale(v); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
  ctx.stroke();

  // Validation loss line
  ctx.strokeStyle = '#bb86fc'; ctx.lineWidth = 2; ctx.beginPath();
  state.valLossHistory.forEach((v, i) => { const x = xScale(i); const y = yScale(v); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
  ctx.stroke();

  // Legend
  const legendX = w - pad - 110;
  const legendY = 26;
  // Train
  ctx.strokeStyle = '#03dac6'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(legendX, legendY); ctx.lineTo(legendX + 18, legendY); ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,0.7)';
  ctx.font = '10px Roboto, Arial, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText('Train', legendX + 22, legendY);
  // Test
  ctx.strokeStyle = '#bb86fc'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(legendX + 58, legendY); ctx.lineTo(legendX + 76, legendY); ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,0.7)';
  ctx.fillText('Test', legendX + 80, legendY);
}

// ---------- Error display ----------
function showError(msg) {
  const el = document.getElementById('errorMessage');
  if (msg) {
    el.textContent = msg;
    el.style.display = 'block';
  } else {
    el.textContent = '';
    el.style.display = 'none';
  }
}

// ---------- Layer count display ----------
function updateLayerCountDisplay() {
  const el = document.getElementById('layerCountDisplay');
  el.textContent = `${state.hiddenLayers} HIDDEN LAYER${state.hiddenLayers !== 1 ? 'S' : ''}`;
}

// ---------- Training ----------
let rafId = null;
async function trainStep() {
  if (!state.model || !state.xs || !state.ys) return;
  const history = await state.model.fit(state.xs, state.ys, { epochs: 1, batchSize: state.batchSize, shuffle: true });
  const trainLoss = history.history.loss[0];

  // Always compute validation loss (dispose tensors to avoid leak)
  let valLoss = trainLoss;
  if (state.xsTest && state.ysTest) {
    const ev = state.model.evaluate(state.xsTest, state.ysTest, { batchSize: state.batchSize });
    if (Array.isArray(ev)) {
      valLoss = ev[0].dataSync()[0];
      ev.forEach(t => t.dispose());
    } else {
      valLoss = ev.dataSync()[0];
      ev.dispose();
    }
  }

  state.lossHistory.push(trainLoss);
  state.valLossHistory.push(valLoss);
  state.epoch += 1;
  document.getElementById('epochCounter').textContent = `Epoch: ${String(state.epoch).padStart(6, '0')}`;
  document.getElementById('trainingLoss').textContent = (trainLoss ?? 0).toFixed(3);
  document.getElementById('testLoss').textContent = (valLoss ?? 0).toFixed(3);
  drawDecisionBoundary();
  drawLossChart();
}
function loop() {
  trainStep().then(() => { if (rafId) rafId = requestAnimationFrame(loop); });
}

function playPause() {
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; document.getElementById('playBtn').textContent = '▶'; return; }
  document.getElementById('playBtn').textContent = '⏸';
  rafId = requestAnimationFrame(loop);
}

// Reset state helper (shared by resetAll and config changes)
function resetTrainingState() {
  cancelAnimationFrame(rafId); rafId = null; state.epoch = 0;
  document.getElementById('playBtn').textContent = '▶';
  document.getElementById('epochCounter').textContent = 'Epoch: 000000';
  state.lossHistory = []; state.valLossHistory = [];
  document.getElementById('trainingLoss').textContent = '0.000';
  document.getElementById('testLoss').textContent = '0.000';
}

function resetAll() {
  resetTrainingState();
  generateData(); createModel(); drawNetwork(); drawDecisionBoundary(); drawLossChart();
}

// ---------- Zero-feature guard ----------
function safeRebuild() {
  if (getEnabledFeatureCount() === 0) {
    showError('At least one feature must be enabled.');
    return;
  }
  showError(null);
  generateData(); createModel(); drawNetwork(); drawDecisionBoundary();
}

// ---------- UI wiring ----------
document.querySelectorAll('.dataset-option').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.dataset-option').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.dataset = btn.dataset.dataset;
    resetTrainingState();
    generateData(); createModel(); drawNetwork(); drawDecisionBoundary(); drawLossChart();
  });
});

document.getElementById('ratioSlider').addEventListener('input', e => {
  state.ratio = parseInt(e.target.value, 10); document.getElementById('ratioValue').textContent = `${state.ratio}%`;
  resetTrainingState();
  generateData(); createModel(); drawNetwork(); drawDecisionBoundary(); drawLossChart();
});
document.getElementById('noiseSlider').addEventListener('input', e => {
  state.noise = parseInt(e.target.value, 10); document.getElementById('noiseValue').textContent = `${state.noise}`;
  resetTrainingState();
  generateData(); createModel(); drawNetwork(); drawDecisionBoundary(); drawLossChart();
});

document.getElementById('addLayerBtn').addEventListener('click', () => {
  if (state.hiddenLayers < 6) {
    state.hiddenLayers++;
    updateLayerCountDisplay();
    resetTrainingState();
    createModel(); drawNetwork(); drawDecisionBoundary(); drawLossChart();
  }
});
document.getElementById('removeLayerBtn').addEventListener('click', () => {
  if (state.hiddenLayers > 1) {
    state.hiddenLayers--;
    updateLayerCountDisplay();
    resetTrainingState();
    createModel(); drawNetwork(); drawDecisionBoundary(); drawLossChart();
  }
});
document.getElementById('neuronCountSlider').addEventListener('input', e => {
  const v = Math.max(1, Math.min(32, parseInt(e.target.value, 10)));
  state.neuronsPerLayer = v; document.getElementById('neuronCountInput').value = String(v);
  createModel(); drawNetwork(); drawDecisionBoundary();
});
document.getElementById('neuronCountInput').addEventListener('change', e => {
  const v = Math.max(1, Math.min(32, parseInt(e.target.value, 10)));
  state.neuronsPerLayer = v; document.getElementById('neuronCountSlider').value = String(v);
  createModel(); drawNetwork(); drawDecisionBoundary();
});

document.getElementById('activationSelect').addEventListener('change', e => { state.activation = e.target.value; createModel(); drawNetwork(); drawDecisionBoundary(); });
document.getElementById('learningRateSelect').addEventListener('change', e => { state.learningRate = parseFloat(e.target.value); createModel(); drawNetwork(); drawDecisionBoundary(); });

// Regularization dropdowns (were no-ops — now wired)
document.getElementById('regularizationSelect').addEventListener('change', e => {
  state.regularization = e.target.value;
  createModel(); drawNetwork(); drawDecisionBoundary();
});
document.getElementById('regularizationRateSelect').addEventListener('change', e => {
  state.regularizationRate = parseFloat(e.target.value);
  createModel(); drawNetwork(); drawDecisionBoundary();
});

document.getElementById('batchSizeSlider').addEventListener('input', e => { state.batchSize = parseInt(e.target.value, 10); document.getElementById('batchSizeValue').textContent = String(state.batchSize); });

document.getElementById('discretizeOutput').addEventListener('change', e => { state.discretize = e.target.checked; drawDecisionBoundary(); });
document.getElementById('showTestData').addEventListener('change', e => { state.showTest = e.target.checked; drawDecisionBoundary(); });

// Feature checkboxes (with zero-feature guard)
['x1','x2','x1sq','x2sq','x1x2','sinx1','sinx2','cosx1','cosx2'].forEach(id => {
  const el = document.getElementById(`feature-${id}`);
  if (!el) return;
  el.addEventListener('change', e => {
    const map = { x1: 'x1', x2: 'x2', x1sq: 'x1Squared', x2sq: 'x2Squared', x1x2: 'x1x2', sinx1: 'sinX1', sinx2: 'sinX2', cosx1: 'cosX1', cosx2: 'cosX2' };
    const key = map[id];
    state.features[key] = e.target.checked;
    // Guard: prevent unchecking the last feature
    if (getEnabledFeatureCount() === 0) {
      state.features[key] = true;
      e.target.checked = true;
      showError('At least one feature must be enabled.');
      return;
    }
    showError(null);
    resetTrainingState();
    generateData(); createModel(); drawNetwork(); drawDecisionBoundary(); drawLossChart();
  });
});

document.getElementById('playBtn').addEventListener('click', playPause);
document.getElementById('stepBtn').addEventListener('click', () => { cancelAnimationFrame(rafId); rafId = null; trainStep(); });
document.getElementById('resetBtn').addEventListener('click', resetAll);
document.getElementById('regenerateBtn').addEventListener('click', () => { resetTrainingState(); generateData(); createModel(); drawNetwork(); drawDecisionBoundary(); drawLossChart(); });

// ---------- Init ----------
function init() {
  document.getElementById('ratioValue').textContent = `${state.ratio}%`;
  document.getElementById('noiseValue').textContent = `${state.noise}`;
  document.getElementById('batchSizeValue').textContent = `${state.batchSize}`;
  document.getElementById('neuronCountSlider').value = String(state.neuronsPerLayer);
  document.getElementById('neuronCountInput').value = String(state.neuronsPerLayer);
  updateLayerCountDisplay();
  generateData(); createModel(); drawNetwork(); drawDecisionBoundary(); drawLossChart();
  handleResize();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

// ---------- Responsiveness ----------
function handleResize() {
  const container = document.getElementById('decision-boundary-container');
  if (container && canvas) {
    const size = Math.min(container.clientWidth || 400, container.clientHeight || 400, 600);
    if (size > 0) {
      canvas.width = size;
      canvas.height = size;
      drawDecisionBoundary();
    }
  }
}
window.addEventListener('resize', () => requestAnimationFrame(handleResize));
