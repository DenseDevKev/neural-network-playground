import { createPortal } from 'react-dom';
import { useId, useRef, useState, type KeyboardEvent } from 'react';
import {
    getConceptById,
    type ConceptId,
    type ConceptUiTarget,
} from '../../concepts/conceptCatalog.ts';
import type { GuidanceLevel } from '../../productShell/audienceProfiles.ts';

const TARGET_LABELS: Readonly<Record<ConceptUiTarget, string>> = Object.freeze({
    presets: 'Presets',
    data: 'Data',
    features: 'Features',
    network: 'Network',
    hyperparams: 'Hyperparameters',
    config: 'Configuration',
    boundary: 'Boundary',
    loss: 'Loss',
    confusion: 'Confusion',
    inspection: 'Inspect',
    code: 'Code',
    history: 'History',
});

export interface ConceptHelpProps {
    readonly conceptId: ConceptId;
    readonly guidanceLevel: GuidanceLevel;
    readonly onNavigateToTarget?: (target: ConceptUiTarget) => void;
    readonly className?: string;
}

// A stable ref callback focuses once on mount, not on live-evidence rerenders.
function focusViewportHelp(node: HTMLSpanElement | null) { node?.focus(); }

export function ConceptHelp({
    conceptId,
    guidanceLevel,
    onNavigateToTarget,
    className,
}: ConceptHelpProps) {
    const [isOpen, setIsOpen] = useState(false);
    const triggerRef = useRef<HTMLButtonElement>(null);
    const contentId = useId();
    const titleId = useId();
    const entry = getConceptById(conceptId);

    if (!entry) {
        return null;
    }

    const showExtended = guidanceLevel !== 'compact' && Boolean(entry.extendedExplanation);
    const showExamples = guidanceLevel === 'high' && Boolean(entry.examples?.length);
    const uiTarget = entry.uiTarget;
    const relatedEntries = entry.related.flatMap((relatedId) => {
        const relatedEntry = getConceptById(relatedId);
        return relatedEntry ? [relatedEntry] : [];
    });

    const handleKeyDown = (event: KeyboardEvent<HTMLSpanElement>) => {
        if (event.key !== 'Escape' || !isOpen) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();
        setIsOpen(false);
        triggerRef.current?.focus();
    };

    // Paint containment establishes a containing block for fixed descendants.
    // Mount Precision Lab help beside the workspace, retaining theme and React
    // event ancestry without clipping the panel or changing scientific tracks.
    const viewportHost = isOpen
        ? triggerRef.current?.closest('.precision-layout, [data-precision-workspace]')
            ?.closest('.forge-shell')
        : null;
    const content = isOpen ? (
        <span
            id={contentId}
            ref={viewportHost ? focusViewportHelp : undefined}
            tabIndex={viewportHost ? -1 : undefined}
            className={`concept-help__content${viewportHost ? ' concept-help__content--viewport' : ''}`}
            role="region"
            aria-labelledby={titleId}
        >
            <strong id={titleId} className="concept-help__title">
                {entry.canonicalTerm}
            </strong>
            <span className="concept-help__difficulty">
                Difficulty: {entry.difficulty[0].toUpperCase() + entry.difficulty.slice(1)}
            </span>
            <span className="concept-help__definition">{entry.plainDefinition}</span>
            {showExtended ? (
                <span className="concept-help__extended">{entry.extendedExplanation}</span>
            ) : null}
            {showExamples ? (
                <span className="concept-help__examples">
                    <strong>Examples</strong>
                    <span role="list">
                        {entry.examples?.map((example) => (
                            <span key={example} role="listitem">{example}</span>
                        ))}
                    </span>
                </span>
            ) : null}
            {relatedEntries.length > 0 ? (
                <span className="concept-help__related">
                    <strong>Related concepts</strong>
                    <span role="list">
                        {relatedEntries.map((relatedEntry) => (
                            <span key={relatedEntry.id} role="listitem">
                                {relatedEntry.canonicalTerm}
                            </span>
                        ))}
                    </span>
                </span>
            ) : null}
            {uiTarget && onNavigateToTarget ? (
                <button
                    className="concept-help__target"
                    type="button"
                    onClick={() => onNavigateToTarget(uiTarget)}
                >
                    Open {TARGET_LABELS[uiTarget]}
                </button>
            ) : null}
            {entry.documentationUrl ? (
                <a className="concept-help__documentation" href={entry.documentationUrl}>
                    Read more about {entry.canonicalTerm}
                </a>
            ) : null}
        </span>
    ) : null;

    return (
        <span
            className={`concept-help${className ? ` ${className}` : ''}`}
            onKeyDown={handleKeyDown}
        >
            <button
                ref={triggerRef}
                className="concept-help__trigger"
                type="button"
                aria-label={`Learn about ${entry.canonicalTerm}`}
                aria-expanded={isOpen}
                aria-controls={contentId}
                onClick={() => setIsOpen((open) => !open)}
            >
                <span aria-hidden="true">?</span>
            </button>
            {content && viewportHost ? createPortal(content, viewportHost) : content}
        </span>
    );
}
