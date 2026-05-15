import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { ConfigPanel } from './ConfigPanel';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
    type AppConfig,
} from '@nn-playground/shared';

describe('ConfigPanel clipboard feedback', () => {
    beforeEach(() => {
        vi.restoreAllMocks();
        usePlaygroundStore.setState({
            data: { ...DEFAULT_DATA },
            network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, seed: DEFAULT_DATA.seed },
            features: { ...DEFAULT_FEATURES },
            training: { ...DEFAULT_TRAINING },
            ui: { showTestData: false, discretizeOutput: false },
        });
        Object.defineProperty(navigator, 'clipboard', {
            value: { writeText: vi.fn().mockResolvedValue(undefined) },
            configurable: true,
        });
        Object.defineProperty(URL, 'createObjectURL', {
            value: vi.fn(() => 'blob:nn-playground-config'),
            configurable: true,
        });
        Object.defineProperty(URL, 'revokeObjectURL', {
            value: vi.fn(),
            configurable: true,
        });
        vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined);
    });

    it('shows success feedback when the current URL is copied', async () => {
        render(<ConfigPanel onReset={vi.fn()} />);

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /copy url/i }));
        });

        expect(navigator.clipboard.writeText).toHaveBeenCalledWith(window.location.href);
        expect(screen.getByText('URL copied!')).toBeInTheDocument();
    });

    it('shows announced failure feedback when copying the current URL fails', async () => {
        (navigator.clipboard.writeText as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error('Denied'));
        render(<ConfigPanel onReset={vi.fn()} />);

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /copy url/i }));
        });

        expect(screen.getByRole('alert')).toHaveTextContent('Could not copy URL');
    });

    it('shows announced failure feedback when the Clipboard API is unavailable', async () => {
        Object.defineProperty(navigator, 'clipboard', {
            value: undefined,
            configurable: true,
        });
        render(<ConfigPanel onReset={vi.fn()} />);

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /copy url/i }));
        });

        expect(screen.getByRole('alert')).toHaveTextContent('Could not copy URL');
    });

    it.each([
        [
            'multi-output size',
            { network: { outputSize: 3 }, training: {} },
            'Only single-output networks are supported.',
        ],
        [
            'softmax output activation',
            { network: { outputActivation: 'softmax' }, training: {} },
            'Multiclass configurations must use classification data, output size 3, softmax output activation, and categorical cross-entropy loss.',
        ],
        [
            'categorical cross-entropy loss',
            { network: {}, training: { lossType: 'categoricalCrossEntropy' } },
            'Multiclass configurations must use classification data, output size 3, softmax output activation, and categorical cross-entropy loss.',
        ],
    ])('refuses to export hidden unsupported %s configuration JSON', async (_label, overrides, expectedError) => {
        usePlaygroundStore.setState((state) => ({
            network: {
                ...state.network,
                ...overrides.network,
            },
            training: {
                ...state.training,
                ...overrides.training,
            },
        }));

        render(<ConfigPanel onReset={vi.fn()} />);

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /export json/i }));
        });

        expect(screen.getByRole('alert')).toHaveTextContent(expectedError);
        expect(URL.createObjectURL).not.toHaveBeenCalled();
        expect(HTMLAnchorElement.prototype.click).not.toHaveBeenCalled();
    });

    it('exports approved multiclass configuration JSON', async () => {
        usePlaygroundStore.setState((state) => ({
            data: {
                ...state.data,
                dataset: 'three-class-clusters',
                problemType: 'classification',
            },
            network: {
                ...state.network,
                outputSize: 3,
                outputActivation: 'softmax',
            },
            training: {
                ...state.training,
                lossType: 'categoricalCrossEntropy',
            },
        }));

        render(<ConfigPanel onReset={vi.fn()} />);

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /export json/i }));
        });

        expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
        const blob = (URL.createObjectURL as ReturnType<typeof vi.fn>).mock.calls[0][0] as Blob;
        const exported = JSON.parse(await blob.text()) as AppConfig;

        expect(exported.data.dataset).toBe('three-class-clusters');
        expect(exported.data.problemType).toBe('classification');
        expect(exported.network.outputSize).toBe(3);
        expect(exported.network.outputActivation).toBe('softmax');
        expect(exported.training.lossType).toBe('categoricalCrossEntropy');
        expect(HTMLAnchorElement.prototype.click).toHaveBeenCalledTimes(1);
        expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:nn-playground-config');
        expect(screen.getByRole('status')).toHaveTextContent('Exported!');
    });

    it('refuses to import hidden unsupported multiclass configuration JSON', async () => {
        const onReset = vi.fn();
        const hiddenMulticlassConfig = usePlaygroundStore.getState().getConfig();
        hiddenMulticlassConfig.network = {
            ...hiddenMulticlassConfig.network,
            outputSize: 3,
            outputActivation: 'softmax',
        };
        hiddenMulticlassConfig.training = {
            ...hiddenMulticlassConfig.training,
            lossType: 'categoricalCrossEntropy',
        } as typeof hiddenMulticlassConfig.training;

        const { container } = render(<ConfigPanel onReset={onReset} />);
        const input = container.querySelector<HTMLInputElement>('input[type="file"]');
        expect(input).toBeTruthy();

        await act(async () => {
            fireEvent.change(input!, {
                target: {
                    files: [
                        new File([JSON.stringify(hiddenMulticlassConfig)], 'hidden-multiclass.json', {
                            type: 'application/json',
                        }),
                    ],
                },
            });
        });

        expect(await screen.findByRole('alert', undefined, { timeout: 5000 })).toHaveTextContent(/multiclass configurations/i);
        expect(onReset).not.toHaveBeenCalled();
        expect(usePlaygroundStore.getState().network.outputSize).toBe(1);
        expect(usePlaygroundStore.getState().network.outputActivation).not.toBe('softmax');
        expect(usePlaygroundStore.getState().training.lossType).not.toBe('categoricalCrossEntropy');
    });

    it('imports approved multiclass configuration JSON', async () => {
        const onReset = vi.fn();
        const approvedMulticlassConfig = {
            ...usePlaygroundStore.getState().getConfig(),
            data: {
                ...usePlaygroundStore.getState().data,
                dataset: 'three-class-clusters',
                problemType: 'classification',
            },
            network: {
                ...usePlaygroundStore.getState().network,
                outputSize: 3,
                outputActivation: 'softmax',
            },
            training: {
                ...usePlaygroundStore.getState().training,
                lossType: 'categoricalCrossEntropy',
            },
        };

        const { container } = render(<ConfigPanel onReset={onReset} />);
        const input = container.querySelector<HTMLInputElement>('input[type="file"]');
        expect(input).toBeTruthy();

        await act(async () => {
            fireEvent.change(input!, {
                target: {
                    files: [
                        new File([JSON.stringify(approvedMulticlassConfig)], 'approved-multiclass.json', {
                            type: 'application/json',
                        }),
                    ],
                },
            });
        });

        expect(await screen.findByRole('status', undefined, { timeout: 5000 })).toHaveTextContent('Imported!');
        expect(onReset).toHaveBeenCalledTimes(1);
        expect(usePlaygroundStore.getState().data.dataset).toBe('three-class-clusters');
        expect(usePlaygroundStore.getState().data.problemType).toBe('classification');
        expect(usePlaygroundStore.getState().network.outputSize).toBe(3);
        expect(usePlaygroundStore.getState().network.outputActivation).toBe('softmax');
        expect(usePlaygroundStore.getState().training.lossType).toBe('categoricalCrossEntropy');
    });
});
