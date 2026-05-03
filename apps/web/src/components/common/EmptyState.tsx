interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  titleId?: string;
  description?: string;
  descriptionId?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export function EmptyState({ icon, title, titleId, description, descriptionId, action }: EmptyStateProps) {
  return (
    <div className="empty-state">
      {icon && <div className="empty-state__icon" aria-hidden="true">{icon}</div>}
      <h3 className="empty-state__title" id={titleId}>{title}</h3>
      {description && <p className="empty-state__description" id={descriptionId}>{description}</p>}
      {action && (
        <button type="button" className="btn btn--primary" onClick={action.onClick}>
          {action.label}
        </button>
      )}
    </div>
  );
}
