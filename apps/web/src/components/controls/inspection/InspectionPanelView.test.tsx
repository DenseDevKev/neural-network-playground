import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { InspectionPanelView } from './InspectionPanelView.tsx';
import type {
    InspectionPanelCommands,
    InspectionPanelDisplayModel,
} from './inspectionPanelModel.ts';
import { getConceptById } from '../../../concepts/conceptCatalog.ts';

function displayModel(): InspectionPanelDisplayModel {
    return {
        activationBasis: {
            label: 'Activation statistics across 128 of 210 training examples',
            suffix: '; gradient summary comes from model revision 11.',
        },
        layers: [{
            key: 0,
            name: 'Output',
            gradientWidth: '100%',
            weightWidth: '80%',
            gradientValue: '0.0200',
            weightValue: '0.4000',
            meanActivation: '0.7500',
            activationStd: '0.0500',
        }],
        histogramSelection: 1,
        histogram: {
            selectedLayerIndex: 0,
            options: [{ key: 0, value: 0, label: 'Output' }],
            title: 'Output activations',
            nearZeroText: '20.0% near zero',
            saturatedText: '10.0% near activation limits',
            rangeText: 'range -1.000 to 1.000',
            summary: 'Output activations: 20.0% near zero, 10.0% near activation limits. Activations are spread across the sampled range.',
            bins: [{ key: 0, height: '100%' }, { key: 1, height: '50%' }],
        },
        trace: {
            source: 'train',
            sampleIndex: 4,
            effectiveSampleIndex: 4,
            maxSampleIndex: 9,
            canRequest: true,
            buttonDisabled: false,
            buttonLabel: 'Trace prediction',
            emptyMessage: null,
            errorMessage: null,
            result: {
                provenance: 'Trace from training sample 4 · model step 12 · revision 12',
                output: '0.8200, 0.1800',
                sampleDataLoss: '0.1900',
                regularizationPenalty: '0.0300',
                layers: [{ key: 0, label: 'Layer 1', activations: '0.820, 0.180' }],
            },
        },
        backprop: {
            buttonDisabled: false,
            buttonLabel: 'Preview backprop',
            statusText: 'Backprop preview found 1 healthy layer update.',
            result: {
                provenance: 'Preview from step 12 / epoch 1',
                objective: [
                    'batch 5',
                    'data loss 0.120',
                    'penalty 0.010',
                    'training objective 0.130',
                    'lr 0.030',
                ],
                gradient: ['complete objective gradient 0.006', 'not clipped'],
                layers: [{
                    key: 0,
                    name: 'Output',
                    status: 'healthy',
                    metrics: [
                        'mean update 0.001',
                        'mean gradient 0.003',
                        'error signal 0.012',
                    ],
                    note: 'The previewed update is in a moderate range.',
                }],
            },
        },
        landscape: {
            buttonDisabled: false,
            buttonLabel: 'Probe loss surface',
            statusText: 'Training-objective surface found best objective 0.4000.',
            result: {
                provenance: 'Probe from step 12 / epoch 1',
                summary: 'Training-objective parameter grid: center 0.490, min 0.400, max 0.620. Best direction W1[0,0] +0.100, W1[1,0] -0.100.',
                gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                cells: [{ key: '0-0.62', opacity: 0.28 }, { key: '1-0.4', opacity: 1 }],
                title: 'Training objective on a parameter grid',
                values: ['center 0.490', 'min 0.400', 'max 0.620'],
                basis: 'sampled 12 training examples across 4 parameter positions on a 2 by 2 grid',
                bestDirection: 'best direction W1[0,0] +0.100, W1[1,0] -0.100',
            },
        },
    };
}

function commands(): InspectionPanelCommands {
    return {
        selectHistogramLayer: vi.fn(),
        selectTraceSource: vi.fn(),
        selectSampleIndex: vi.fn(),
        requestTrace: vi.fn(async () => undefined),
        requestBackprop: vi.fn(async () => undefined),
        requestLandscape: vi.fn(async () => undefined),
    };
}

describe('InspectionPanelView', () => {
    it('renders the complete prepared model with semantic labels and live regions', () => {
        const { container } = render(
            <InspectionPanelView model={displayModel()} commands={commands()} guidanceLevel="standard" />,
        );

        expect(screen.getByText('Activation statistics across 128 of 210 training examples'))
            .toBeInTheDocument();
        expect(screen.getByRole('region', { name: 'Activation histogram explorer' }))
            .toBeInTheDocument();
        expect(screen.getByRole('img', { name: /Output activations: 20\.0% near zero/i }))
            .toBeInTheDocument();
        expect(screen.getByLabelText('Prediction trace'))
            .toHaveAttribute('aria-label', 'Prediction trace');
        expect(container.querySelector(
            '[aria-label="Prediction trace"] [aria-live="polite"]',
        )).not.toBeNull();
        expect(screen.getByText('Trace from training sample 4 · model step 12 · revision 12'))
            .toBeInTheDocument();
        expect(screen.getByRole('status', { name: 'Backprop preview status' }))
            .toHaveAttribute('aria-live', 'polite');
        expect(screen.getByRole('status', { name: 'Loss landscape probe status' }))
            .toHaveTextContent('Training-objective surface found best objective 0.4000.');
        expect(screen.getByRole('img', { name: /Training-objective parameter grid/i }))
            .toBeInTheDocument();
    });

    it('renders prepared loading, disabled, and empty states without stores or workers', () => {
        const model = displayModel();
        render(<InspectionPanelView
            model={{
                ...model,
                activationBasis: null,
                layers: [],
                histogram: null,
                trace: {
                    ...model.trace,
                    canRequest: false,
                    buttonDisabled: true,
                    buttonLabel: 'Tracing…',
                    emptyMessage: 'No test samples are available yet.',
                    errorMessage: 'Trace failed: unavailable',
                    result: null,
                },
                backprop: {
                    buttonDisabled: true,
                    buttonLabel: 'Previewing backprop',
                    statusText: 'Preparing backprop preview...',
                    result: null,
                },
                landscape: {
                    buttonDisabled: true,
                    buttonLabel: 'Probing loss surface',
                    statusText: 'Probing a bounded local loss surface...',
                    result: null,
                },
            }}
            commands={commands()}
            guidanceLevel="compact"
        />);

        expect(screen.getByText('Train the model to see stats')).toBeInTheDocument();
        expect(screen.getByText('Open inspection while training to sample layer activations.'))
            .toBeInTheDocument();
        expect(screen.getByText('No test samples are available yet.')).toBeInTheDocument();
        expect(screen.getByText('Trace failed: unavailable')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Tracing…' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Previewing backprop' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Probing loss surface' })).toBeDisabled();
    });

    it('forwards selection and request events to explicit commands', () => {
        const viewCommands = commands();
        render(<InspectionPanelView
            model={displayModel()}
            commands={viewCommands}
            guidanceLevel="standard"
        />);

        fireEvent.change(screen.getByRole('combobox', { name: 'Histogram layer' }), {
            target: { value: '0' },
        });
        fireEvent.change(screen.getByRole('combobox', { name: 'Sample' }), {
            target: { value: 'test' },
        });
        fireEvent.change(screen.getByRole('spinbutton', { name: 'Index' }), {
            target: { value: '-3' },
        });
        fireEvent.click(screen.getByRole('button', { name: 'Trace prediction' }));
        fireEvent.click(screen.getByRole('button', { name: 'Preview backprop' }));
        fireEvent.click(screen.getByRole('button', { name: 'Probe loss surface' }));

        expect(viewCommands.selectHistogramLayer).toHaveBeenCalledWith(0);
        expect(viewCommands.selectTraceSource).toHaveBeenCalledWith('test');
        expect(viewCommands.selectSampleIndex).toHaveBeenCalledWith(-3);
        expect(viewCommands.requestTrace).toHaveBeenCalledTimes(1);
        expect(viewCommands.requestBackprop).toHaveBeenCalledTimes(1);
        expect(viewCommands.requestLandscape).toHaveBeenCalledTimes(1);
    });

    it('renders gradient help from the catalog without requiring a store', async () => {
        const user = userEvent.setup();
        render(<InspectionPanelView
            model={displayModel()}
            commands={commands()}
            guidanceLevel="high"
        />);

        expect(screen.getByText('Slow-Motion Backprop')).toBeInTheDocument();
        await user.click(screen.getByRole('button', { name: 'Learn about Gradient' }));

        expect(screen.getByText(getConceptById('gradient')?.plainDefinition ?? ''))
            .toBeInTheDocument();
        expect(screen.getByText(getConceptById('gradient')?.examples?.[0] ?? ''))
            .toBeInTheDocument();
    });
});
