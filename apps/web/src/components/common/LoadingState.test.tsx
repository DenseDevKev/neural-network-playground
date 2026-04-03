import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LoadingState } from './LoadingState';

describe('LoadingState', () => {
  it('should render nothing when isLoading is false', () => {
    render(<LoadingState isLoading={false} message="Loading" />);

    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    expect(document.body.querySelector('.loading-overlay')).not.toBeInTheDocument();
  });

  it('should render inline loading state with message', () => {
    const { container } = render(
      <LoadingState isLoading={true} inline message="Generating data..." />
    );

    expect(screen.getByRole('status')).toHaveClass('loading-state--inline');
    expect(screen.getByText('Generating data...')).toBeInTheDocument();
    expect(container.querySelector('.loading-spinner')).toBeInTheDocument();
    expect(document.body.querySelector('.loading-overlay')).not.toBeInTheDocument();
  });

  it('should render overlay loading state in a portal', () => {
    render(<LoadingState isLoading={true} message="Initializing network..." />);

    const status = screen.getByRole('status');
    expect(status).toHaveClass('loading-state--overlay');
    expect(status).toHaveTextContent('Initializing network...');
    expect(document.body.querySelector('.loading-overlay')).toBeInTheDocument();
    expect(document.body.querySelector('.loading-spinner--large')).toBeInTheDocument();
  });

  it('should render without a message when one is not provided', () => {
    render(<LoadingState isLoading={true} inline />);

    expect(screen.getByRole('status')).toBeInTheDocument();
    expect(document.body.querySelector('.loading-state__message')).not.toBeInTheDocument();
  });

  it('should update conditional rendering when isLoading changes', () => {
    const { rerender } = render(<LoadingState isLoading={true} inline message="Loading" />);

    expect(screen.getByRole('status')).toBeInTheDocument();

    rerender(<LoadingState isLoading={false} inline message="Loading" />);

    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });
});
