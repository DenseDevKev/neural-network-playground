import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { ConfigPanel } from './ConfigPanel';

describe('ConfigPanel clipboard feedback', () => {
    beforeEach(() => {
        Object.defineProperty(navigator, 'clipboard', {
            value: { writeText: vi.fn().mockResolvedValue(undefined) },
            configurable: true,
        });
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
});
