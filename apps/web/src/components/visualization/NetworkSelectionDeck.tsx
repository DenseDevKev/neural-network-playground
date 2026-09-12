import { HeatmapTile } from './NetworkGraphCanvas.tsx';
import './networkAtelier.css';
import { memo } from 'react';
import type { NetworkInfluence, NetworkSelectionDisplayModel } from './networkSelectionModel.ts';

function finite(value: number | null | undefined): string {
    return value !== null && value !== undefined && Number.isFinite(value) ? value.toFixed(3) : 'Not available';
}
function InfluenceList({ label, items }: { label: string; items: readonly NetworkInfluence[] }) {
    return <div className="precision-influences"><h3>{label}</h3>{items.length === 0 ? <p>No available connections</p> :
        <ul>{items.map((item) => <li key={item.edgeKey} data-sign={item.weight < 0 ? 'negative' : 'positive'}><span>{item.peerLabel}</span>{' '}<span>{item.sign} {item.weight > 0 ? '+' : ''}{finite(item.weight)}</span></li>)}</ul>}</div>;
}

/** Display values are derived from accepted artifacts, never from another worker request. */
export const NetworkSelectionDeck = memo(function NetworkSelectionDeck({ model, onClear }: {
    readonly model: NetworkSelectionDisplayModel;
    readonly onClear: () => void;
}) {
    if (model.kind === 'empty') return <p className="precision-selection-empty">Select a neuron to inspect its activation grid, bias, and strongest connections.</p>;
    return <section className="precision-selection-details" aria-label="Selected neuron details">
        <header><h2>{model.nodeLabel}</h2><button type="button" onClick={onClear}>Clear selection</button></header>
        <p className="precision-selection-basis">Weights at step {model.parameterStep ?? 'not available'}; activation grid at step {model.activationStep ?? 'not available'}. Grid summary, not a full-split evaluation.</p>
        <div className="atelier-neuron-map" role="img" aria-label={`${model.nodeLabel} activation map at step ${model.activationStep ?? 'not available'}`}>
            {model.grid && model.gridSize > 0 ? <HeatmapTile grid={model.grid} gridSize={model.gridSize} displaySize={256} /> : <span>Activation grid not available</span>}
        </div>
        <dl>
            <div><dt>Bias</dt><dd>{finite(model.bias)}</dd></div>
            <div><dt>Mean activation</dt><dd>{finite(model.activation?.mean)}</dd></div>
            <div><dt>Minimum</dt><dd>{finite(model.activation?.minimum)}</dd></div>
            <div><dt>Maximum</dt><dd>{finite(model.activation?.maximum)}</dd></div>
            <div><dt>Standard deviation</dt><dd>{finite(model.activation?.standardDeviation)}</dd></div>
        </dl>
        
        <InfluenceList label="Strongest inputs" items={model.incoming} />
        <InfluenceList label="Strongest outputs" items={model.outgoing} />
    </section>;
});
