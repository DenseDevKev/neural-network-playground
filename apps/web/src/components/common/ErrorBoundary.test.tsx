import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ErrorBoundary } from './ErrorBoundary';

function createDeferred<T>() {
    let resolve!: (value: T | PromiseLike<T>) => void;
    let reject!: (reason?: unknown) => void;
    const promise = new Promise<T>((promiseResolve, promiseReject) => {
        resolve = promiseResolve;
        reject = promiseReject;
    });

    return { promise, resolve, reject };
}

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
        expect(screen.getByText(/A section failed to render\. Boom/)).toBeInTheDocument();
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

    it('keeps the fallback visible and exposes retry progress until an async retry succeeds', async () => {
        const user = userEvent.setup();
        const deferred = createDeferred<void>();
        const onRetry = vi.fn(() => deferred.promise);
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
        await user.click(screen.getByRole('button', { name: 'Retrying…' }));

        expect(onRetry).toHaveBeenCalledTimes(1);
        expect(screen.getByText('Workspace unavailable')).toBeInTheDocument();
        expect(screen.getByText('A section failed to render. Boom')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Retrying…' })).toBeInTheDocument();

        deferred.resolve();

        expect(await screen.findByText('Healthy content')).toBeInTheDocument();
    });

    it('shows a safe retry error and leaves Try again actionable when an async retry fails', async () => {
        const user = userEvent.setup();
        const onRetry = vi.fn().mockRejectedValue(new Error('reset failed'));

        render(
            <ErrorBoundary
                title="Workspace unavailable"
                description="A section failed to render."
                onRetry={onRetry}
            >
                <Thrower shouldThrow={true} />
            </ErrorBoundary>,
        );

        await user.click(screen.getByRole('button', { name: 'Try again' }));

        expect(await screen.findByText(/reset failed/i)).toBeInTheDocument();
        expect(screen.getByText(/A section failed to render\. Boom/)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument();
    });
});
