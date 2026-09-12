import { memo, type ReactNode } from 'react';
import { edgeFilterOptions, type EdgeFilter, type GraphViewMode } from './networkGraphPainter.ts';

interface NetworkGraphFrameProps {
    children: ReactNode;
    story: string;
    capacity: string;
    datasetHint?: string | null;
    healthHint?: string | null;
    lesson?: string;
    zoom: number;
    viewMode: GraphViewMode;
    edgeFilter: EdgeFilter;
    onZoomOut: () => void;
    onZoomIn: () => void;
    onFit: () => void;
    onViewMode: (mode: GraphViewMode) => void;
    onEdgeFilter: (filter: EdgeFilter) => void;
}

/** Shared chrome is outside the measured plot: it cannot cover neuron hit targets. */
export const NetworkGraphFrame = memo(function NetworkGraphFrame({
    children, story, capacity, datasetHint, healthHint, lesson,
    zoom, viewMode, edgeFilter, onZoomOut, onZoomIn, onFit, onViewMode, onEdgeFilter,
}: NetworkGraphFrameProps) {
    return <div className="network-graph-frame">
        <div className="network-graph-toolbar" role="toolbar" aria-label="Network graph toolbar">
            <div className="network-graph-controls" aria-label="Graph view controls">
                <button type="button" aria-label="Zoom out graph" title="Zoom out" onClick={onZoomOut}>-</button>
                <span className="network-graph-controls__zoom">{Math.round(zoom * 100)}%</span>
                <button type="button" aria-label="Zoom in graph" title="Zoom in" onClick={onZoomIn}>+</button>
                <button type="button" aria-label="Fit graph to view" title="Fit graph" onClick={onFit}>Fit</button>
            </div>
            <div className="network-graph-mode-toggle" role="group" aria-label="Topology view mode">
                {(['weights', 'activations'] as const).map((mode) => <button key={mode} type="button"
                    className={`network-graph-mode-toggle__button${viewMode === mode ? ' network-graph-mode-toggle__button--active' : ''}`}
                    aria-pressed={viewMode === mode} onClick={() => onViewMode(mode)}>{mode === 'weights' ? 'Weights' : 'Activations'}</button>)}
            </div>
        </div>
        <div className="network-graph-summary" role="region" aria-label="Architecture summary" tabIndex={0}>
            <div className="network-graph-summary__row"><span className="network-graph-summary__story">{story}</span><span className="network-graph-summary__badge">{capacity}</span></div>
            {datasetHint && <div className="network-graph-summary__hint">{datasetHint}</div>}
            {healthHint && <div className="network-graph-summary__hint network-graph-summary__hint--stats">{healthHint}</div>}
            {lesson && <div role="note"><span className="network-graph-lesson-callout__label">Lesson</span><span>{lesson}</span></div>}
        </div>
        {children}
        <div className="network-graph-legend" aria-label="Edge weight legend">
            <div className="network-graph-legend__scale">
                <span><i className="network-graph-legend__swatch network-graph-legend__swatch--positive" /> Positive</span>
                <span><i className="network-graph-legend__swatch network-graph-legend__swatch--negative" /> Negative</span>
                <span className="network-graph-legend__hint">width = |weight|; selected negative paths are dashed</span>
            </div>
            <div className="network-graph-legend__filters">{edgeFilterOptions.map((option) => <button key={option.id} type="button"
                aria-label={option.id === 'strong' ? 'Show only strong edges' : `Show ${option.label.toLowerCase()} edges`}
                aria-pressed={edgeFilter === option.id}
                className={`network-graph-legend__filter${edgeFilter === option.id ? ' network-graph-legend__filter--active' : ''}`}
                onClick={() => onEdgeFilter(option.id)}>{option.label}</button>)}</div>
        </div>
    </div>;
});
