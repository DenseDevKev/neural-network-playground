import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DEFAULT_NETWORK } from '@nn-playground/shared';
import { ConfusionMatrix } from './ConfusionMatrix';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';

function setProblemType(problemType: 'classification' | 'regression') {
  usePlaygroundStore.setState((state) => ({
    data: {
      ...state.data,
      problemType,
    },
  }));
}

describe('ConfusionMatrix', () => {
  beforeEach(() => {
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
      stepsPerFrame: 5,
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

    expect(screen.getByText('Confusion matrix unavailable')).toBeInTheDocument();
    expect(screen.getByText(/only renders binary classification matrices/i)).toBeInTheDocument();
    expect(screen.queryByText('Pred 1')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('TP cell')).not.toBeInTheDocument();
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
