import { useState, useRef, useCallback, useEffect } from 'react';
import { createPortal } from 'react-dom';

interface TooltipProps {
  content: string | React.ReactNode;
  children: React.ReactNode;
  placement?: 'top' | 'bottom' | 'left' | 'right';
  delay?: number;
  shortcut?: string;
  block?: boolean;
}

interface Position {
  x: number;
  y: number;
}

function calculatePosition(
  triggerRect: DOMRect,
  placement: 'top' | 'bottom' | 'left' | 'right',
  tooltipWidth: number,
  tooltipHeight: number
): Position {
  const gap = 8;
  let x = 0;
  let y = 0;

  switch (placement) {
    case 'top':
      x = triggerRect.left + triggerRect.width / 2 - tooltipWidth / 2;
      y = triggerRect.top - tooltipHeight - gap;
      break;
    case 'bottom':
      x = triggerRect.left + triggerRect.width / 2 - tooltipWidth / 2;
      y = triggerRect.bottom + gap;
      break;
    case 'left':
      x = triggerRect.left - tooltipWidth - gap;
      y = triggerRect.top + triggerRect.height / 2 - tooltipHeight / 2;
      break;
    case 'right':
      x = triggerRect.right + gap;
      y = triggerRect.top + triggerRect.height / 2 - tooltipHeight / 2;
      break;
  }

  // Clamp to viewport with padding
  const padding = 8;
  x = Math.max(padding, Math.min(x, window.innerWidth - tooltipWidth - padding));
  y = Math.max(padding, Math.min(y, window.innerHeight - tooltipHeight - padding));

  return { x, y };
}

export function Tooltip({
  content,
  children,
  placement = 'top',
  delay = 500,
  shortcut,
  block = false,
}: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState<Position>({ x: 0, y: 0 });
  const timeoutRef = useRef<number | undefined>(undefined);
  const triggerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const tooltipId = useRef(`tooltip-${Math.random().toString(36).substr(2, 9)}`);
  const measurementIdRef = useRef<string | null>(null);

  const updateTooltip = useCallback(() => {
    if (triggerRef.current && tooltipRef.current) {
      const triggerRect = triggerRef.current.getBoundingClientRect();
      const tooltipRect = tooltipRef.current.getBoundingClientRect();

      const pos = calculatePosition(
        triggerRect,
        placement,
        tooltipRect.width,
        tooltipRect.height
      );

      setPosition(pos);
      setIsVisible(true);
    }
  }, [placement]);

  const show = useCallback(() => {
    if (import.meta.env.DEV && typeof performance !== 'undefined') {
      const measurementId = `tooltip-show:${tooltipId.current}:${performance.now()}`;
      measurementIdRef.current = measurementId;
      performance.mark(`${measurementId}:start`);
    }

    timeoutRef.current = window.setTimeout(() => {
      updateTooltip();

      if (import.meta.env.DEV && typeof performance !== 'undefined' && measurementIdRef.current) {
        const measurementId = measurementIdRef.current;
        performance.mark(`${measurementId}:end`);
        performance.measure(measurementId, `${measurementId}:start`, `${measurementId}:end`);
        performance.clearMarks(`${measurementId}:start`);
        performance.clearMarks(`${measurementId}:end`);
        measurementIdRef.current = null;
      }
    }, delay);
  }, [delay, updateTooltip]);

  const showImmediately = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    updateTooltip();
  }, [updateTooltip]);

  const hide = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsVisible(false);
  }, []);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!isVisible) {
      return;
    }

    const reposition = () => {
      updateTooltip();
    };

    window.addEventListener('resize', reposition);
    window.addEventListener('scroll', reposition, true);

    return () => {
      window.removeEventListener('resize', reposition);
      window.removeEventListener('scroll', reposition, true);
    };
  }, [isVisible, updateTooltip]);

  return (
    <>
      <div
        ref={triggerRef}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocusCapture={showImmediately}
        onBlurCapture={hide}
        onKeyDownCapture={(event) => {
          if (event.key === 'Escape') {
            hide();
          }
        }}
        aria-describedby={isVisible ? tooltipId.current : undefined}
        style={{ display: block ? 'block' : 'inline-block' }}
      >
        {children}
      </div>
      {createPortal(
        <div
          ref={tooltipRef}
          id={tooltipId.current}
          className="tooltip"
          role="tooltip"
          style={{
            position: 'fixed',
            left: `${position.x}px`,
            top: `${position.y}px`,
            opacity: isVisible ? 1 : 0,
            pointerEvents: 'none',
            visibility: isVisible ? 'visible' : 'hidden',
            transition: isVisible
              ? 'opacity 150ms ease-out'
              : 'opacity 100ms ease-out, visibility 0s 100ms',
            zIndex: 1000,
          }}
        >
          <div className="tooltip__content">{content}</div>
          {shortcut && (
            <div className="tooltip__shortcut">
              <kbd>{shortcut}</kbd>
            </div>
          )}
        </div>,
        document.body
      )}
    </>
  );
}
