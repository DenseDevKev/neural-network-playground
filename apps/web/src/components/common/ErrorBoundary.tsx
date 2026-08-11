import React from 'react';
import { EmptyState } from './EmptyState.tsx';

interface ErrorBoundaryProps {
    children: React.ReactNode;
    title: string;
    description: string;
    actionLabel?: string;
    onRetry?: () => void | Promise<void>;
    className?: string;
}

interface ErrorBoundaryState {
    error: Error | null;
    retrying: boolean;
    retryError: string | null;
}

function toErrorMessage(error: unknown): string {
    return error instanceof Error && error.message
        ? error.message
        : 'Unable to recover. Please try again.';
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
    state: ErrorBoundaryState = {
        error: null,
        retrying: false,
        retryError: null,
    };

    private isMounted = false;
    private retryInFlight = false;

    static getDerivedStateFromError(error: Error): ErrorBoundaryState {
        return { error, retrying: false, retryError: null };
    }

    componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
        console.error(`${this.props.title} boundary caught an error`, error, errorInfo);
    }

    componentDidMount() {
        this.isMounted = true;
    }

    componentWillUnmount() {
        this.isMounted = false;
    }

    handleRetry = async () => {
        if (this.retryInFlight) {
            return;
        }

        this.retryInFlight = true;
        this.setState({ retrying: true, retryError: null });

        try {
            await this.props.onRetry?.();
            if (this.isMounted) {
                this.setState({ error: null, retrying: false, retryError: null });
            }
        } catch (error) {
            if (this.isMounted) {
                this.setState({ retrying: false, retryError: toErrorMessage(error) });
            }
        } finally {
            this.retryInFlight = false;
        }
    };

    render() {
        if (this.state.error) {
            const retryStatus = this.state.retrying
                ? 'Retrying…'
                : this.state.retryError
                    ? `Retry failed: ${this.state.retryError}`
                    : '';

            return (
                <div className={this.props.className}>
                    <span className="sr-only" role="status" aria-live="polite" aria-atomic="true">
                        {retryStatus}
                    </span>
                    <div aria-busy={this.state.retrying}>
                        <EmptyState
                            icon="⚠"
                            title={this.props.title}
                            description={[
                                this.props.description,
                                this.state.error.message,
                                this.state.retryError,
                            ].filter(Boolean).join(' ')}
                            action={{
                                label: this.state.retrying ? 'Retrying…' : this.props.actionLabel ?? 'Try again',
                                onClick: this.handleRetry,
                            }}
                        />
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}
