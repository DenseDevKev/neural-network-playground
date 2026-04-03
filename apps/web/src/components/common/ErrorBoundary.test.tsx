import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ErrorBoundary } from './ErrorBoundary';

function Thrower({ shouldThrow }: { shouldThrow: boolean }) {
    if (shouldThrow) {
        throw new Error('Boom');
    }

    return <div>Healthy content</div>;
}

describe('ErrorBoundary', () => {
    beforeEach(() => {
        vi.spyOn(console, 'error').mockImplementation(() => {});
    });

    it('catches render errors and shows the fallback state', () => {
        render(
            <ErrorBoundary title="Workspace unavailable" description="A section failed to render.">
                <Thrower shouldThrow={true} />
            </ErrorBoundary>,
        );

        expect(screen.getByText('Workspace unavailable')).toBeInTheDocument();
        expect(screen.getByText('A section failed to render. Boom')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument();
    });

    it('resets the boundary and calls onRetry', async () => {
        const user = userEvent.setup();
        const onRetry = vi.fn();

        const { rerender } = render(
            <ErrorBoundary
                title="Workspace unavailable"
                description="A section failed to render."
                onRetry={onRetry}
            >
                <Thrower shouldThrow={true} />
            </ErrorBoundary>,
        );

        rerender(
            <ErrorBoundary
                title="Workspace unavailable"
                description="A section failed to render."
                onRetry={onRetry}
            >
                <Thrower shouldThrow={false} />
            </ErrorBoundary>,
        );

        await user.click(screen.getByRole('button', { name: 'Try again' }));

        expect(onRetry).toHaveBeenCalledTimes(1);
        expect(screen.getByText('Healthy content')).toBeInTheDocument();
    });
});
