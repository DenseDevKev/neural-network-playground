import React from 'react';
import { EmptyState } from './EmptyState.tsx';

interface ErrorBoundaryProps {
    children: React.ReactNode;
    title: string;
    description: string;
    actionLabel?: string;
    onRetry?: () => void;
    className?: string;
}

interface ErrorBoundaryState {
    error: Error | null;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
    state: ErrorBoundaryState = {
        error: null,
    };

    static getDerivedStateFromError(error: Error): ErrorBoundaryState {
        return { error };
    }

    componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
        console.error(`${this.props.title} boundary caught an error`, error, errorInfo);
    }

    handleRetry = () => {
        this.props.onRetry?.();
        this.setState({ error: null });
    };

    render() {
        if (this.state.error) {
            return (
                <div className={this.props.className}>
                    <EmptyState
                        icon="⚠"
                        title={this.props.title}
                        description={`${this.props.description} ${this.state.error.message}`.trim()}
                        action={{
                            label: this.props.actionLabel ?? 'Try again',
                            onClick: this.handleRetry,
                        }}
                    />
                </div>
            );
        }

        return this.props.children;
    }
}
