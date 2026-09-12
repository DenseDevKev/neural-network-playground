import { useMemo, type CSSProperties } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { NetworkGraph } from '../visualization/NetworkGraph.tsx';
import { DecisionBoundaryCanvas } from '../visualization/DecisionBoundaryCanvas.tsx';
import { NetworkSelectionDeck } from '../visualization/NetworkSelectionDeck.tsx';
import type { DecisionBoundaryController } from '../visualization/useDecisionBoundaryController.ts';
import type { NetworkSelectionController } from '../visualization/useNetworkSelectionController.ts';
import type { SetupTab } from '../../productShell/atelierTypes.ts';
import { Icon } from './ui.tsx';
import { getDatasetContract } from '@nn-playground/engine';
import { useThemeStore, readPlotPalette } from '../../store/theme.ts';
import { CLASS_COLORS, fieldColor, hexRgb } from '../visualization/plotColors.ts';

function DataPlot() {
    const points = useTrainingStore((s) => s.trainPoints);
    const test = useTrainingStore((s) => s.testPoints);
    const task = usePlaygroundStore((s) => s.access.status === 'ready' ? s.access.prepared.document.recipe.task : null);
    const noise = usePlaygroundStore((s) => s.access.status === 'ready' ? s.access.prepared.document.recipe.data.noise : 0);
    const theme = useThemeStore((s) => s.resolved);
    const background = useMemo(() => { void theme; return hexRgb(readPlotPalette().background); }, [theme]);
    const regression = task?.kind === 'regression';
    const range = useMemo(() => {
        const domain = task && getDatasetContract(task.dataset).targetDomain;
        return domain?.kind === 'continuous' ? domain.boundsForNoise(noise) : [0,1];
    }, [task,noise]);
    return <>
        <svg className="atelier-data-plot" viewBox="0 0 256 256" role="img" aria-label={`${points.length} training samples plotted by x and y; ${regression ? 'color shows continuous target' : 'color shows target class'}`}>
            {[64,128,192].map((p) => <g key={p} stroke="var(--border-color)" strokeWidth=".5"><path d={`M${p} 0V256 M0 ${p}H256`} /></g>)}
            {points.map((point, index) => <circle key={index} cx={16+(point.x+1)*112} cy={16+(1-point.y)*112} r={2.3} fill={regression ? `rgb(${fieldColor((point.label-range[0])/(range[1]-range[0] || 1),background).join(',')})` : CLASS_COLORS[point.label] ?? CLASS_COLORS[0]} stroke="var(--bg-primary)" strokeWidth=".5" />)}
            <text x="240" y="250" fill="var(--text-secondary)" fontSize="10">x</text><text x="6" y="12" fill="var(--text-secondary)" fontSize="10">y</text>
        </svg>
        <p className="atelier-plot-caption">{points.length} training · {test.length} test samples<br />Active experiment · deterministic data split</p>
        {regression ? <p className="atelier-plot-caption">Target range {Number.isFinite(range[0]) ? range[0].toFixed(2) : '—'} to {Number.isFinite(range[1]) ? range[1].toFixed(2) : '—'}</p> : <div className="atelier-class-legend">{CLASS_COLORS.slice(0,task?.kind === 'multiclass-classification' ? 3 : 2).map((color,index) => <span key={index}><i style={{ '--class-color':color } as CSSProperties} />Class {index}</span>)}</div>}
    </>;
}

export function AtelierNetwork({ boundary, selection, onSetup, onResults }: { boundary: DecisionBoundaryController; selection: NetworkSelectionController; onSetup(tab: SetupTab): void; onResults(): void }) {
    return <>
        <div className="atelier-network">
            <section><div className="atelier-section-title"><h2>Data</h2><button type="button" onClick={() => onSetup('dataset')} aria-label="Edit dataset"><Icon name="settings" /></button></div><DataPlot /></section>
            <section><div className="atelier-section-title"><h2>Network</h2><button type="button" onClick={() => onSetup('network')}>Edit layers <Icon name="next" size={14} /></button></div><div className="atelier-network-graph"><NetworkGraph selectionController={selection} /></div></section>
            <section><div className="atelier-section-title"><h2>Prediction</h2><button type="button" onClick={onResults} aria-label="Expand prediction results"><Icon name="expand" /></button></div><DecisionBoundaryCanvas model={boundary.model} /><p className="atelier-plot-caption">{boundary.model.kind !== 'empty' && boundary.model.kind !== 'unavailable' && boundary.model.provenance && <>Grid at step {boundary.model.provenance.model.step}. </>}The model’s output across the input space. Select Results for evaluation and overlays.</p></section>
        </div>
        {selection.model.kind === 'selected' && <div className="atelier-neuron-detail"><NetworkSelectionDeck model={selection.model} onClear={selection.commands.clearSelection} /></div>}
    </>;
}
