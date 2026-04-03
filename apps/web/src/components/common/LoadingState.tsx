import { createPortal } from 'react-dom';

interface LoadingStateProps {
  isLoading: boolean;
  message?: string;
  inline?: boolean;
}

export function LoadingState({ isLoading, message, inline = false }: LoadingStateProps) {
  if (!isLoading) {
    return null;
  }

  const indicator = (
    <div
      className={`loading-state ${inline ? 'loading-state--inline' : 'loading-state--overlay'}`}
      role="status"
      aria-live="polite"
    >
      <div
        className={`loading-spinner ${inline ? '' : 'loading-spinner--large'}`.trim()}
        aria-hidden="true"
      />
      {message && <span className="loading-state__message">{message}</span>}
    </div>
  );

  if (inline) {
    return indicator;
  }

  return createPortal(<div className="loading-overlay">{indicator}</div>, document.body);
}
