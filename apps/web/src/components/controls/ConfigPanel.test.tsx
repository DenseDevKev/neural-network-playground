import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { ConfigPanel } from './ConfigPanel';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import {
    DEFAULT_DATA,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
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
            'Multiclass configurations are not runtime-enabled yet.',
        ],
        [
            'categorical cross-entropy loss',
            { network: {}, training: { lossType: 'categoricalCrossEntropy' } },
            'Multiclass configurations are not runtime-enabled yet.',
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

        expect(await screen.findByRole('alert')).toHaveTextContent('Multiclass configurations are not runtime-enabled yet.');
        expect(onReset).not.toHaveBeenCalled();
        expect(usePlaygroundStore.getState().network.outputSize).toBe(1);
        expect(usePlaygroundStore.getState().network.outputActivation).not.toBe('softmax');
        expect(usePlaygroundStore.getState().training.lossType).not.toBe('categoricalCrossEntropy');
    });
});
