import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { Tooltip } from './Tooltip';

describe('Tooltip', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  it('should render children', () => {
    render(
      <Tooltip content="Test tooltip">
        <button>Hover me</button>
      </Tooltip>
    );

    expect(screen.getByRole('button', { name: 'Hover me' })).toBeInTheDocument();
  });

  it('should render tooltip in portal', () => {
    render(
      <Tooltip content="Test tooltip">
        <button>Hover me</button>
      </Tooltip>
    );

    const tooltip = document.querySelector('[role="tooltip"]');
    expect(tooltip).toBeInTheDocument();
    expect(tooltip).toHaveTextContent('Test tooltip');
  });

  describe('Keyboard Shortcut Display', () => {
    it('should display keyboard shortcut when provided', () => {
      render(
        <Tooltip content="Play/Pause training" shortcut="Space">
          <button>Play</button>
        </Tooltip>
      );

      const tooltip = document.querySelector('[role="tooltip"]');
      expect(tooltip).toHaveTextContent('Play/Pause training');
      expect(tooltip).toHaveTextContent('Space');
      expect(tooltip?.querySelector('kbd')).toBeInTheDocument();
    });

    it('should not display shortcut section when shortcut is not provided', () => {
      render(
        <Tooltip content="Test tooltip">
          <button>Hover me</button>
        </Tooltip>
      );

      const tooltip = document.querySelector('[role="tooltip"]');
      expect(tooltip?.querySelector('kbd')).not.toBeInTheDocument();
      expect(tooltip?.querySelector('.tooltip__shortcut')).not.toBeInTheDocument();
    });

    it('should display shortcut in kbd element', () => {
      render(
        <Tooltip content="Reset" shortcut="R">
          <button>Reset</button>
        </Tooltip>
      );

      const kbd = document.querySelector('kbd');
      expect(kbd).toBeInTheDocument();
      expect(kbd).toHaveTextContent('R');
    });
  });

  describe('ARIA Attributes', () => {
    it('should have proper ARIA attributes structure', () => {
      render(
        <Tooltip content="Test tooltip">
          <button>Hover me</button>
        </Tooltip>
      );

      const tooltip = document.querySelector('[role="tooltip"]');
      expect(tooltip).toHaveAttribute('role', 'tooltip');
      expect(tooltip).toHaveAttribute('id');
    });

    it('should link trigger to tooltip with aria-describedby when visible', () => {
      render(
        <Tooltip content="Test tooltip" delay={100}>
          <button>Hover me</button>
        </Tooltip>
      );

      const button = screen.getByRole('button');
      const wrapper = button.parentElement!;
      const tooltip = document.querySelector('[role="tooltip"]') as HTMLElement;
      const tooltipId = tooltip.getAttribute('id');

      // Initially not linked
      expect(wrapper).not.toHaveAttribute('aria-describedby');

      // Trigger hover
      fireEvent.mouseEnter(wrapper);
      act(() => {
        vi.advanceTimersByTime(100);
      });

      // Should be linked when visible
      expect(wrapper).toHaveAttribute('aria-describedby', tooltipId);
    });

    it('should remove aria-describedby when tooltip is hidden', () => {
      render(
        <Tooltip content="Test tooltip" delay={100}>
          <button>Hover me</button>
        </Tooltip>
      );

      const button = screen.getByRole('button');
      const wrapper = button.parentElement!;

      // Show tooltip
      fireEvent.mouseEnter(wrapper);
      act(() => {
        vi.advanceTimersByTime(100);
      });
      expect(wrapper).toHaveAttribute('aria-describedby');

      // Hide tooltip
      fireEvent.mouseLeave(wrapper);
      expect(wrapper).not.toHaveAttribute('aria-describedby');
    });

    it('should have unique tooltip id for each instance', () => {
      const { unmount: unmount1 } = render(
        <Tooltip content="Tooltip 1">
          <button>Button 1</button>
        </Tooltip>
      );

      const tooltip1 = document.querySelector('[role="tooltip"]');
      const id1 = tooltip1?.getAttribute('id');
      expect(id1).toBeTruthy();

      unmount1();

      const { unmount: unmount2 } = render(
        <Tooltip content="Tooltip 2">
          <button>Button 2</button>
        </Tooltip>
      );

      const tooltip2 = document.querySelector('[role="tooltip"]');
      const id2 = tooltip2?.getAttribute('id');

      expect(id2).toBeTruthy();
      expect(id1).not.toBe(id2);

      unmount2();
    });

    it('should show the tooltip when a child receives keyboard focus', () => {
      render(
        <Tooltip content="Test tooltip">
          <button>Focus me</button>
        </Tooltip>
      );

      const button = screen.getByRole('button', { name: 'Focus me' });
      const wrapper = button.parentElement!;
      const tooltip = document.querySelector('[role="tooltip"]') as HTMLElement;
      const tooltipId = tooltip.getAttribute('id');

      fireEvent.focus(button);

      expect(wrapper).toHaveAttribute('aria-describedby', tooltipId);
      expect(tooltip).toHaveStyle({ visibility: 'visible' });
    });
  });

  describe('Hover Delay Timing', () => {
    it('should not show tooltip immediately on hover', () => {
      render(
        <Tooltip content="Test tooltip" delay={500}>
          <button>Hover me</button>
        </Tooltip>
      );

      const button = screen.getByRole('button');
      const tooltip = document.querySelector('[role="tooltip"]') as HTMLElement;

      fireEvent.mouseEnter(button);

      // Should not be visible immediately
      expect(tooltip.style.visibility).toBe('hidden');
      expect(tooltip.style.opacity).toBe('0');
    });

    it('should show tooltip after default delay of 500ms', () => {
      render(
        <Tooltip content="Test tooltip">
          <button>Hover me</button>
        </Tooltip>
      );

      const button = screen.getByRole('button');
      const tooltip = document.querySelector('[role="tooltip"]') as HTMLElement;

      fireEvent.mouseEnter(button);
      act(() => {
        vi.advanceTimersByTime(500);
      });

      expect(tooltip.style.visibility).toBe('visible');
      expect(tooltip.style.opacity).toBe('1');
    });

    it('should show tooltip after custom delay', () => {
      render(
        <Tooltip content="Test tooltip" delay={1000}>
          <button>Hover me</button>
        </Tooltip>
      );

      const button = screen.getByRole('button');
      const tooltip = document.querySelector('[role="tooltip"]') as HTMLElement;

      fireEvent.mouseEnter(button);

      // Not visible before delay
      act(() => {
        vi.advanceTimersByTime(999);
      });
      expect(tooltip.style.visibility).toBe('hidden');

      // Visible after delay
      act(() => {
        vi.advanceTimersByTime(1);
      });
      expect(tooltip.style.visibility).toBe('visible');
    });

    it('should cancel tooltip display if mouse leaves before delay', () => {
      render(
        <Tooltip content="Test tooltip" delay={500}>
          <button>Hover me</button>
        </Tooltip>
      );

      const button = screen.getByRole('button');
      const tooltip = document.querySelector('[role="tooltip"]') as HTMLElement;

      fireEvent.mouseEnter(button);
      vi.advanceTimersByTime(300);
      fireEvent.mouseLeave(button);
      vi.advanceTimersByTime(200);

      expect(tooltip.style.visibility).toBe('hidden');
    });

    it('should hide tooltip immediately on mouse leave', () => {
      render(
        <Tooltip content="Test tooltip" delay={100}>
          <button>Hover me</button>
        </Tooltip>
      );

      const button = screen.getByRole('button');
      const tooltip = document.querySelector('[role="tooltip"]') as HTMLElement;

      // Show tooltip
      fireEvent.mouseEnter(button);
      act(() => {
        vi.advanceTimersByTime(100);
      });
      expect(tooltip.style.visibility).toBe('visible');

      // Hide on leave
      fireEvent.mouseLeave(button);
      expect(tooltip.style.visibility).toBe('hidden');
    });
  });

  describe('Positioning Logic at Viewport Edges', () => {
    beforeEach(() => {
      // Mock window dimensions
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1024,
      });
      Object.defineProperty(window, 'innerHeight', {
        writable: true,
        configurable: true,
        value: 768,
      });
    });

    it('should clamp tooltip position to stay within viewport horizontally', () => {
      render(
        <Tooltip content="Test tooltip" placement="top" delay={100}>
          <button>Hover me</button>
        </Tooltip>
      );

      const button = screen.getByRole('button');
      const tooltip = document.querySelector('[role="tooltip"]') as HTMLElement;

      // Mock trigger element at far right edge
      vi.spyOn(button.parentElement!, 'getBoundingClientRect').mockReturnValue({
        left: 1000,
        top: 100,
        right: 1020,
        bottom: 130,
        width: 20,
        height: 30,
        x: 1000,
        y: 100,
        toJSON: () => ({}),
      });

      // Mock tooltip dimensions
      vi.spyOn(tooltip, 'getBoundingClientRect').mockReturnValue({
        left: 0,
        top: 0,
        right: 200,
        bottom: 60,
        width: 200,
        height: 60,
        x: 0,
        y: 0,
        toJSON: () => ({}),
      });

      fireEvent.mouseEnter(button);
      act(() => {
        vi.advanceTimersByTime(100);
      });

      const leftPosition = parseInt(tooltip.style.left);
      // Should be clamped to viewport with padding
      expect(leftPosition).toBeLessThanOrEqual(1024 - 200 - 8);
    });

    it('should clamp tooltip position to stay within viewport vertically', () => {
      render(
        <Tooltip content="Test tooltip" placement="top" delay={100}>
          <button>Hover me</button>
        </Tooltip>
      );

      const button = screen.getByRole('button');
      const tooltip = document.querySelector('[role="tooltip"]') as HTMLElement;

      // Mock trigger element at top edge
      vi.spyOn(button.parentElement!, 'getBoundingClientRect').mockReturnValue({
        left: 100,
        top: 5,
        right: 120,
        bottom: 35,
        width: 20,
        height: 30,
        x: 100,
        y: 5,
        toJSON: () => ({}),
      });

      // Mock tooltip dimensions
      vi.spyOn(tooltip, 'getBoundingClientRect').mockReturnValue({
        left: 0,
        top: 0,
        right: 200,
        bottom: 60,
        width: 200,
        height: 60,
        x: 0,
        y: 0,
        toJSON: () => ({}),
      });

      fireEvent.mouseEnter(button);
      act(() => {
        vi.advanceTimersByTime(100);
      });

      const topPosition = parseInt(tooltip.style.top);
      // Should be clamped to minimum padding
      expect(topPosition).toBeGreaterThanOrEqual(8);
    });

    it('should position tooltip above trigger for top placement', () => {
      render(
        <Tooltip content="Test tooltip" placement="top" delay={100}>
          <button>Hover me</button>
        </Tooltip>
      );

      const button = screen.getByRole('button');
      const tooltip = document.querySelector('[role="tooltip"]') as HTMLElement;

      // Mock trigger element in center
      vi.spyOn(button.parentElement!, 'getBoundingClientRect').mockReturnValue({
        left: 400,
        top: 400,
        right: 420,
        bottom: 430,
        width: 20,
        height: 30,
        x: 400,
        y: 400,
        toJSON: () => ({}),
      });

      // Mock tooltip dimensions
      vi.spyOn(tooltip, 'getBoundingClientRect').mockReturnValue({
        left: 0,
        top: 0,
        right: 200,
        bottom: 60,
        width: 200,
        height: 60,
        x: 0,
        y: 0,
        toJSON: () => ({}),
      });

      fireEvent.mouseEnter(button);
      act(() => {
        vi.advanceTimersByTime(100);
      });

      const topPosition = parseInt(tooltip.style.top);
      // Should be above trigger (400 - 60 - 8 = 332)
      expect(topPosition).toBeLessThan(400);
    });

    it('should position tooltip below trigger for bottom placement', () => {
      render(
        <Tooltip content="Test tooltip" placement="bottom" delay={100}>
          <button>Hover me</button>
        </Tooltip>
      );

      const button = screen.getByRole('button');
      const tooltip = document.querySelector('[role="tooltip"]') as HTMLElement;

      // Mock trigger element in center
      vi.spyOn(button.parentElement!, 'getBoundingClientRect').mockReturnValue({
        left: 400,
        top: 400,
        right: 420,
        bottom: 430,
        width: 20,
        height: 30,
        x: 400,
        y: 400,
        toJSON: () => ({}),
      });

      // Mock tooltip dimensions
      vi.spyOn(tooltip, 'getBoundingClientRect').mockReturnValue({
        left: 0,
        top: 0,
        right: 200,
        bottom: 60,
        width: 200,
        height: 60,
        x: 0,
        y: 0,
        toJSON: () => ({}),
      });

      fireEvent.mouseEnter(button);
      act(() => {
        vi.advanceTimersByTime(100);
      });

      const topPosition = parseInt(tooltip.style.top);
      // Should be below trigger (430 + 8 = 438)
      expect(topPosition).toBeGreaterThan(430);
    });

    it('should position tooltip to left of trigger for left placement', () => {
      render(
        <Tooltip content="Test tooltip" placement="left" delay={100}>
          <button>Hover me</button>
        </Tooltip>
      );

      const button = screen.getByRole('button');
      const tooltip = document.querySelector('[role="tooltip"]') as HTMLElement;

      // Mock trigger element in center
      vi.spyOn(button.parentElement!, 'getBoundingClientRect').mockReturnValue({
        left: 400,
        top: 400,
        right: 420,
        bottom: 430,
        width: 20,
        height: 30,
        x: 400,
        y: 400,
        toJSON: () => ({}),
      });

      // Mock tooltip dimensions
      vi.spyOn(tooltip, 'getBoundingClientRect').mockReturnValue({
        left: 0,
        top: 0,
        right: 200,
        bottom: 60,
        width: 200,
        height: 60,
        x: 0,
        y: 0,
        toJSON: () => ({}),
      });

      fireEvent.mouseEnter(button);
      act(() => {
        vi.advanceTimersByTime(100);
      });

      const leftPosition = parseInt(tooltip.style.left);
      // Should be to left of trigger (400 - 200 - 8 = 192)
      expect(leftPosition).toBeLessThan(400);
    });

    it('should position tooltip to right of trigger for right placement', () => {
      render(
        <Tooltip content="Test tooltip" placement="right" delay={100}>
          <button>Hover me</button>
        </Tooltip>
      );

      const button = screen.getByRole('button');
      const tooltip = document.querySelector('[role="tooltip"]') as HTMLElement;

      // Mock trigger element in center
      vi.spyOn(button.parentElement!, 'getBoundingClientRect').mockReturnValue({
        left: 400,
        top: 400,
        right: 420,
        bottom: 430,
        width: 20,
        height: 30,
        x: 400,
        y: 400,
        toJSON: () => ({}),
      });

      // Mock tooltip dimensions
      vi.spyOn(tooltip, 'getBoundingClientRect').mockReturnValue({
        left: 0,
        top: 0,
        right: 200,
        bottom: 60,
        width: 200,
        height: 60,
        x: 0,
        y: 0,
        toJSON: () => ({}),
      });

      fireEvent.mouseEnter(button);
      act(() => {
        vi.advanceTimersByTime(100);
      });

      const leftPosition = parseInt(tooltip.style.left);
      // Should be to right of trigger (420 + 8 = 428)
      expect(leftPosition).toBeGreaterThan(420);
    });
  });

  it('should apply fixed positioning styles', () => {
    render(
      <Tooltip content="Test tooltip">
        <button>Hover me</button>
      </Tooltip>
    );

    const tooltip = document.querySelector('[role="tooltip"]') as HTMLElement;
    expect(tooltip.style.position).toBe('fixed');
    expect(tooltip.style.zIndex).toBe('1000');
  });

  it('should clean up timeout on unmount', () => {
    const { unmount } = render(
      <Tooltip content="Test tooltip" delay={500}>
        <button>Hover me</button>
      </Tooltip>
    );

    unmount();
    
    // Should not throw
    vi.advanceTimersByTime(500);
    expect(true).toBe(true);
  });

  it('should remove resize and scroll listeners on unmount after becoming visible', () => {
    const addEventListenerSpy = vi.spyOn(window, 'addEventListener');
    const removeEventListenerSpy = vi.spyOn(window, 'removeEventListener');

    const { unmount } = render(
      <Tooltip content="Test tooltip" delay={100}>
        <button>Hover me</button>
      </Tooltip>
    );

    const button = screen.getByRole('button', { name: 'Hover me' });
    const wrapper = button.parentElement!;

    fireEvent.mouseEnter(wrapper);
    act(() => {
      vi.advanceTimersByTime(100);
    });

    unmount();

    expect(addEventListenerSpy).toHaveBeenCalledWith('resize', expect.any(Function));
    expect(addEventListenerSpy).toHaveBeenCalledWith('scroll', expect.any(Function), true);
    expect(removeEventListenerSpy).toHaveBeenCalledWith('resize', expect.any(Function));
    expect(removeEventListenerSpy).toHaveBeenCalledWith('scroll', expect.any(Function), true);
  });

  it('should support different placement props', () => {
    const placements = ['top', 'bottom', 'left', 'right'] as const;
    
    placements.forEach((placement) => {
      const { unmount } = render(
        <Tooltip content="Test tooltip" placement={placement}>
          <button>Hover me</button>
        </Tooltip>
      );
      
      const tooltip = document.querySelector('[role="tooltip"]');
      expect(tooltip).toBeInTheDocument();
      
      unmount();
    });
  });

  it('should accept custom delay prop', () => {
    render(
      <Tooltip content="Test tooltip" delay={1000}>
        <button>Hover me</button>
      </Tooltip>
    );

    const tooltip = document.querySelector('[role="tooltip"]');
    expect(tooltip).toBeInTheDocument();
  });
});
