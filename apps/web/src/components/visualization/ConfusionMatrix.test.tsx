import { beforeEach, describe, expect, it } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import { DEFAULT_NETWORK } from '@nn-playground/shared';
import { ConfusionMatrix } from './ConfusionMatrix';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import {
  getFrameBuffer,
  getFrameVersions,
  resetFrameBuffer,
  updateFrameBuffer,
} from '../../worker/frameBuffer.ts';

function setProblemType(problemType: 'classification' | 'regression') {
  usePlaygroundStore.setState((state) => ({
    data: {
      ...state.data,
      problemType,
    },
  }));
}

function setApprovedMulticlassConfig() {
  usePlaygroundStore.setState((state) => ({
    data: {
      ...state.data,
      dataset: 'three-class-clusters',
      problemType: 'classification',
    },
    network: {
      ...state.network,
      hiddenLayers: [],
      inputSize: 2,
      outputSize: 3,
      outputActivation: 'softmax',
    },
    training: {
      ...state.training,
      lossType: 'categoricalCrossEntropy',
    },
  }));
}

describe('ConfusionMatrix', () => {
  beforeEach(() => {
    resetFrameBuffer();
    setProblemType('classification');
    usePlaygroundStore.setState((state) => ({
      network: {
        ...DEFAULT_NETWORK,
        inputSize: state.network.inputSize,
      },
    }));
    useTrainingStore.getState().resetHistory();
    useTrainingStore.setState({
      status: 'idle',
      snapshot: null,
      trainPoints: [],
      testPoints: [],
      frameVersion: 0,
      paramsVersion: 0,
      stepsPerFrame: 5,
      dataConfigLoading: false,
      networkConfigLoading: false,
      featuresConfigLoading: false,
      trainingConfigLoading: false,
      presetConfigLoading: false,
      pendingConfigSource: null,
    });
  });

  it('should render an empty state when no test data exists', () => {
    render(<ConfusionMatrix />);

    expect(screen.getByText('No test data')).toBeInTheDocument();
    expect(screen.getByText('Train the model to generate test predictions and evaluation metrics.')).toBeInTheDocument();
  });

  it('should render a neutral unavailable state when test data has no confusion matrix yet', () => {
    useTrainingStore.setState({
      testPoints: [{ id: 1 } as any],
      snapshot: {
        testMetrics: {
          loss: 0.4,
          accuracy: 0.6,
        },
      } as any,
    });

    render(<ConfusionMatrix />);

    expect(screen.getByText('Confusion matrix unavailable')).toBeInTheDocument();
    expect(screen.getByText(/metrics are still loading/i)).toBeInTheDocument();
    expect(screen.queryByText(/only renders binary classification matrices/i)).not.toBeInTheDocument();
    expect(screen.queryByText('No test data')).not.toBeInTheDocument();
    expect(screen.queryByText('Pred 1')).not.toBeInTheDocument();
  });

  it('should not render a stale binary matrix for hidden multiclass config', () => {
    usePlaygroundStore.setState((state) => ({
      network: {
        ...state.network,
        outputSize: 3,
        outputActivation: 'softmax',
      },
    }));
    useTrainingStore.setState({
      testPoints: [{ x: 0, y: 0, label: 0 }],
      snapshot: {
        testMetrics: {
          loss: 0.4,
          accuracy: 0.6,
          confusionMatrix: { tn: 40, fp: 10, fn: 5, tp: 45 },
        },
      } as any,
    });

    render(<ConfusionMatrix />);

    expect(screen.getByText('Multiclass readout unavailable')).toBeInTheDocument();
    expect(screen.getByText(/current network parameters are still loading/i)).toBeInTheDocument();
    expect(screen.queryByText(/only renders binary classification matrices/i)).not.toBeInTheDocument();
    expect(screen.queryByText('Pred 1')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('TP cell')).not.toBeInTheDocument();
  });

  it('renders a derived 3-class test-set readout from current frame parameters', () => {
    usePlaygroundStore.setState((state) => ({
      network: {
        ...state.network,
        hiddenLayers: [],
        inputSize: 2,
        outputSize: 3,
        outputActivation: 'softmax',
      },
    }));
    updateFrameBuffer({
      weights: new Float32Array([
        2, 0,
        0, 2,
        -2, -2,
      ]),
      biases: new Float32Array([0, 0, 1]),
      weightLayout: { layerSizes: [2, 3] },
    });
    useTrainingStore.setState({
      ...getFrameVersions(),
      testPoints: [
        { x: 1, y: 0, label: 0 },
        { x: 0, y: 1, label: 1 },
        { x: -1, y: -1, label: 2 },
        { x: 1, y: 0, label: 2 },
      ],
    });

    render(<ConfusionMatrix />);

    expect(screen.getByText('Multiclass Confusion Readout (Test Set)')).toBeInTheDocument();
    expect(screen.getAllByText('C0')[0]).toBeInTheDocument();
    expect(screen.getAllByText('C2')[0]).toBeInTheDocument();
    expect(screen.getByLabelText(/1 test sample .* actual Class 2 predicted Class 0/i)).toHaveTextContent('1');
    expect(screen.getByLabelText(/1 test sample .* actual Class 2 predicted Class 2/i)).toHaveTextContent('1');
    expect(screen.getByLabelText('Actual Class 2 total 2')).toHaveTextContent('2');
    expect(screen.getByLabelText('Predicted Class 0 total 2')).toHaveTextContent('2');
    expect(screen.getByText('Accuracy')).toBeInTheDocument();
    expect(screen.getByText('75.0%')).toBeInTheDocument();
    expect(screen.getByText(/3 of 4 test samples land on the diagonal/i)).toBeInTheDocument();
    expect(screen.getByText(/derived from current frame-buffer parameters/i)).toBeInTheDocument();
    expect(screen.queryByLabelText('TP cell')).not.toBeInTheDocument();
  });

  it('renders a worker-authored 3-class matrix for the approved tuple while training is running', () => {
    setApprovedMulticlassConfig();
    updateFrameBuffer({
      multiclassConfusionMatrix: {
        classCount: 3,
        classLabels: [0, 1, 2],
        counts: [4, 1, 0, 0, 3, 2, 1, 0, 5],
      },
    });
    useTrainingStore.setState({
      frameVersion: getFrameBuffer().version,
      status: 'running',
      testPoints: [
        { x: 0, y: 0, label: 0 },
        { x: 1, y: 0, label: 1 },
        { x: 0, y: 1, label: 2 },
      ],
    });

    render(<ConfusionMatrix />);

    expect(screen.getByText('Multiclass Confusion Readout (Test Set)')).toBeInTheDocument();
    expect(screen.getByLabelText(/2 test samples .* actual Class 1 predicted Class 2/i)).toHaveTextContent('2');
    expect(screen.getByLabelText('Actual Class 2 total 6')).toHaveTextContent('6');
    expect(screen.getByLabelText('Predicted Class 2 total 7')).toHaveTextContent('7');
    expect(screen.getByLabelText('Total test samples 16')).toHaveTextContent('16');
    expect(screen.getByText('75.0%')).toBeInTheDocument();
    expect(screen.getByText(/latest worker-authored test evaluation/i)).toBeInTheDocument();
    expect(screen.queryByText(/pause training to inspect/i)).not.toBeInTheDocument();
  });

  it('prefers worker-authored multiclass data over a contradictory derived fallback', () => {
    setApprovedMulticlassConfig();
    updateFrameBuffer({
      weights: new Float32Array([
        2, 0,
        0, 2,
        -2, -2,
      ]),
      biases: new Float32Array([0, 0, 1]),
      weightLayout: { layerSizes: [2, 3] },
      multiclassConfusionMatrix: {
        classCount: 3,
        classLabels: [0, 1, 2],
        counts: [1, 0, 0, 1, 0, 0, 0, 0, 1],
      },
    });
    useTrainingStore.setState({
      frameVersion: getFrameBuffer().version,
      paramsVersion: getFrameBuffer().paramsVersion,
      status: 'paused',
      testPoints: [
        { x: 1, y: 0, label: 0 },
        { x: 0, y: 1, label: 1 },
        { x: -1, y: -1, label: 2 },
      ],
    });

    render(<ConfusionMatrix />);

    expect(screen.getByLabelText(/1 test sample .* actual Class 1 predicted Class 0/i)).toHaveTextContent('1');
    expect(screen.getByLabelText(/0 test samples .* actual Class 1 predicted Class 1/i)).toHaveTextContent('0');
    expect(screen.getByText('66.7%')).toBeInTheDocument();
    expect(screen.getByText(/latest worker-authored test evaluation/i)).toBeInTheDocument();
    expect(screen.queryByText(/derived from current frame-buffer parameters/i)).not.toBeInTheDocument();
  });

  it('updates the worker-authored matrix from the broad frame version without a dedicated store field', () => {
    setApprovedMulticlassConfig();
    useTrainingStore.setState({
      frameVersion: getFrameBuffer().version,
      status: 'running',
      testPoints: [{ x: 0, y: 0, label: 0 }],
    });

    render(<ConfusionMatrix />);
    expect(screen.getByText('Multiclass readout unavailable')).toBeInTheDocument();

    act(() => {
      updateFrameBuffer({
        multiclassConfusionMatrix: {
          classCount: 3,
          classLabels: [0, 1, 2],
          counts: [1, 0, 0, 0, 1, 0, 0, 0, 1],
        },
      });
      useTrainingStore.setState({ frameVersion: getFrameBuffer().version });
    });

    expect(screen.getByText('Multiclass Confusion Readout (Test Set)')).toBeInTheDocument();
    expect(screen.getByText('100.0%')).toBeInTheDocument();
    expect('multiclassConfusionMatrixVersion' in useTrainingStore.getState()).toBe(false);
  });

  it.each([
    ['wrong dataset', { data: { dataset: 'circle' as const } }, {}],
    ['wrong loss', {}, { training: { lossType: 'crossEntropy' as const } }],
  ])('ignores worker-authored multiclass data for an unapproved tuple: %s', (_label, dataPatch, trainingPatch) => {
    setApprovedMulticlassConfig();
    usePlaygroundStore.setState((state) => ({
      data: { ...state.data, ...dataPatch.data },
      training: { ...state.training, ...trainingPatch.training },
    }));
    updateFrameBuffer({
      multiclassConfusionMatrix: {
        classCount: 3,
        classLabels: [0, 1, 2],
        counts: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      },
    });
    useTrainingStore.setState({
      frameVersion: getFrameBuffer().version,
      status: 'running',
      testPoints: [{ x: 0, y: 0, label: 0 }],
    });

    render(<ConfusionMatrix />);

    expect(screen.getByText('Multiclass readout unavailable')).toBeInTheDocument();
    expect(screen.queryByText('C2')).not.toBeInTheDocument();
    expect(screen.queryByText(/latest worker-authored test evaluation/i)).not.toBeInTheDocument();
  });

  it('derives a 3-class readout through hidden-layer activations without importing the engine network', () => {
    usePlaygroundStore.setState((state) => ({
      network: {
        ...state.network,
        activation: 'relu',
        hiddenLayers: [2],
        inputSize: 2,
        outputSize: 3,
        outputActivation: 'softmax',
      },
    }));
    updateFrameBuffer({
      weights: new Float32Array([
        1, 0,
        0, 1,
        3, 0,
        0, 3,
        -3, -3,
      ]),
      biases: new Float32Array([0, 0, 0, 0, 2]),
      weightLayout: { layerSizes: [2, 2, 3] },
    });
    useTrainingStore.setState({
      ...getFrameVersions(),
      testPoints: [
        { x: 1, y: 0, label: 0 },
        { x: 0, y: 1, label: 1 },
        { x: -1, y: -1, label: 2 },
      ],
    });

    render(<ConfusionMatrix />);

    expect(screen.getByText('Multiclass Confusion Readout (Test Set)')).toBeInTheDocument();
    expect(screen.getByLabelText(/1 test sample .* actual Class 0 predicted Class 0/i)).toHaveTextContent('1');
    expect(screen.getByLabelText(/1 test sample .* actual Class 1 predicted Class 1/i)).toHaveTextContent('1');
    expect(screen.getByLabelText(/1 test sample .* actual Class 2 predicted Class 2/i)).toHaveTextContent('1');
    expect(screen.getByText('100.0%')).toBeInTheDocument();
  });

  it('does not derive a multiclass readout while training is running', () => {
    usePlaygroundStore.setState((state) => ({
      network: {
        ...state.network,
        hiddenLayers: [],
        inputSize: 2,
        outputSize: 3,
        outputActivation: 'softmax',
      },
    }));
    updateFrameBuffer({
      weights: new Float32Array([
        2, 0,
        0, 2,
        -2, -2,
      ]),
      biases: new Float32Array([0, 0, 1]),
      weightLayout: { layerSizes: [2, 3] },
    });
    useTrainingStore.setState({
      ...getFrameVersions(),
      status: 'running',
      testPoints: [
        { x: 1, y: 0, label: 0 },
        { x: 0, y: 1, label: 1 },
        { x: -1, y: -1, label: 2 },
      ],
    });

    render(<ConfusionMatrix />);

    expect(screen.getByText('Multiclass readout unavailable')).toBeInTheDocument();
    expect(screen.getByText(/pause training to inspect/i)).toBeInTheDocument();
    expect(screen.queryByText('C2')).not.toBeInTheDocument();
  });

  it('does not combine frame parameters with a pending config sync', () => {
    usePlaygroundStore.setState((state) => ({
      network: {
        ...state.network,
        hiddenLayers: [],
        inputSize: 2,
        outputSize: 3,
        outputActivation: 'softmax',
      },
    }));
    updateFrameBuffer({
      weights: new Float32Array([
        2, 0,
        0, 2,
        -2, -2,
      ]),
      biases: new Float32Array([0, 0, 1]),
      weightLayout: { layerSizes: [2, 3] },
    });
    useTrainingStore.setState({
      ...getFrameVersions(),
      pendingConfigSource: 'network',
      testPoints: [
        { x: 1, y: 0, label: 0 },
        { x: 0, y: 1, label: 1 },
        { x: -1, y: -1, label: 2 },
      ],
    });

    render(<ConfusionMatrix />);

    expect(screen.getByText('Multiclass readout unavailable')).toBeInTheDocument();
    expect(screen.getByText(/configuration is still syncing/i)).toBeInTheDocument();
    expect(screen.queryByText('C2')).not.toBeInTheDocument();
  });

  it('shows a multiclass unavailable state when frame parameters are missing', () => {
    usePlaygroundStore.setState((state) => ({
      network: {
        ...state.network,
        outputSize: 3,
        outputActivation: 'softmax',
      },
    }));
    useTrainingStore.setState({
      testPoints: [{ x: 0, y: 0, label: 2 }],
    });

    render(<ConfusionMatrix />);

    expect(screen.getByText('Multiclass readout unavailable')).toBeInTheDocument();
    expect(screen.getByText(/current network parameters are still loading/i)).toBeInTheDocument();
    expect(screen.queryByText('C2')).not.toBeInTheDocument();
  });

  it('should not render a stale binary matrix for non-binary test labels', () => {
    useTrainingStore.setState({
      testPoints: [{ x: 0, y: 0, label: 2 }],
      snapshot: {
        testMetrics: {
          loss: 0.4,
          accuracy: 0.6,
          confusionMatrix: { tn: 40, fp: 10, fn: 5, tp: 45 },
        },
      } as any,
    });

    render(<ConfusionMatrix />);

    expect(screen.getByText('Confusion matrix unavailable')).toBeInTheDocument();
    expect(screen.getByText(/only renders binary classification matrices/i)).toBeInTheDocument();
    expect(screen.queryByText('Pred 1')).not.toBeInTheDocument();
  });

  it('should not render a stale binary matrix when train labels reveal multiclass data', () => {
    useTrainingStore.setState({
      trainPoints: [{ x: 0, y: 0, label: 2 }],
      testPoints: [{ x: 1, y: 1, label: 1 }],
      snapshot: {
        testMetrics: {
          loss: 0.4,
          accuracy: 0.6,
          confusionMatrix: { tn: 40, fp: 10, fn: 5, tp: 45 },
        },
      } as any,
    });

    render(<ConfusionMatrix />);

    expect(screen.getByText('Confusion matrix unavailable')).toBeInTheDocument();
    expect(screen.getByText(/only renders binary classification matrices/i)).toBeInTheDocument();
    expect(screen.queryByText('Pred 1')).not.toBeInTheDocument();
  });

  it('should render percentages, totals, and summary metrics', () => {
    useTrainingStore.setState({
      testPoints: [{ id: 1 } as any],
      snapshot: {
        testMetrics: {
          confusionMatrix: { tn: 40, fp: 10, fn: 5, tp: 45 },
        },
      } as any,
    });

    const { container } = render(<ConfusionMatrix />);

    expect(screen.getByText('Predicted')).toBeInTheDocument();
    expect(screen.getByText('Actual')).toBeInTheDocument();
    expect(screen.getByText('40.0%')).toBeInTheDocument();
    expect(screen.getByText('10.0%')).toBeInTheDocument();
    expect(screen.getByText('5.0%')).toBeInTheDocument();
    expect(screen.getByText('45.0%')).toBeInTheDocument();
    expect(screen.getByText('Accuracy')).toBeInTheDocument();
    expect(screen.getByText('85.0%')).toBeInTheDocument();
    expect(screen.getByText('81.8%')).toBeInTheDocument();
    expect(screen.getByText('90.0%')).toBeInTheDocument();

    const totals = Array.from(container.querySelectorAll('.cm-total')).map((node) => node.textContent?.trim());
    expect(totals).toEqual(['50', '50', '45', '55', '100']);
  });

  it('should use semantic colors and handle zero-denominator metric edge cases', () => {
    useTrainingStore.setState({
      testPoints: [{ id: 1 } as any],
      snapshot: {
        testMetrics: {
          confusionMatrix: { tn: 10, fp: 0, fn: 0, tp: 0 },
        },
      } as any,
    });

    render(<ConfusionMatrix />);

    const tnCell = screen.getByLabelText('TN cell');
    const fpCell = screen.getByLabelText('FP cell');

    expect(tnCell.getAttribute('style')).toContain('34, 197, 94');
    expect(fpCell.getAttribute('style')).toContain('239, 68, 68');
    expect(screen.getAllByText('100.0%').length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText('0.0%').length).toBeGreaterThanOrEqual(2);
  });
});
