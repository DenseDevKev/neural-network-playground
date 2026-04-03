import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { EmptyState } from './EmptyState';

describe('EmptyState', () => {
  it('should render title', () => {
    render(<EmptyState title="No data available" />);
    expect(screen.getByText('No data available')).toBeInTheDocument();
  });

  it('should render icon when provided', () => {
    render(<EmptyState icon="🎯" title="No data" />);
    expect(screen.getByText('🎯')).toBeInTheDocument();
  });

  it('should render description when provided', () => {
    render(
      <EmptyState
        title="No data"
        description="Configure your dataset to get started"
      />
    );
    expect(screen.getByText('Configure your dataset to get started')).toBeInTheDocument();
  });

  it('should render action button when provided', () => {
    const handleClick = vi.fn();
    render(
      <EmptyState
        title="No data"
        action={{ label: 'Start Training', onClick: handleClick }}
      />
    );
    
    const button = screen.getByRole('button', { name: 'Start Training' });
    expect(button).toBeInTheDocument();
  });

  it('should call action onClick when button is clicked', async () => {
    const user = userEvent.setup();
    const handleClick = vi.fn();
    
    render(
      <EmptyState
        title="No data"
        action={{ label: 'Start Training', onClick: handleClick }}
      />
    );
    
    const button = screen.getByRole('button', { name: 'Start Training' });
    await user.click(button);
    
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('should render all props together', () => {
    const handleClick = vi.fn();
    render(
      <EmptyState
        icon="📊"
        title="No training data"
        description="Click Play to start training"
        action={{ label: 'Play', onClick: handleClick }}
      />
    );
    
    expect(screen.getByText('📊')).toBeInTheDocument();
    expect(screen.getByText('No training data')).toBeInTheDocument();
    expect(screen.getByText('Click Play to start training')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Play' })).toBeInTheDocument();
  });

  it('should not render icon when not provided', () => {
    const { container } = render(<EmptyState title="No data" />);
    expect(container.querySelector('.empty-state__icon')).not.toBeInTheDocument();
  });

  it('should not render description when not provided', () => {
    const { container } = render(<EmptyState title="No data" />);
    expect(container.querySelector('.empty-state__description')).not.toBeInTheDocument();
  });

  it('should not render action button when not provided', () => {
    render(<EmptyState title="No data" />);
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });
});
