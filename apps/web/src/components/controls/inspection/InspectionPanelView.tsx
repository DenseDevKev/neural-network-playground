import { memo } from 'react';
import {
    type InspectionPanelCommands,
    type InspectionPanelDisplayModel,
    type InspectionTraceSource,
} from './inspectionPanelModel.ts';
import { Tabs } from '../../atelier/ui.tsx';
import type { InspectTab } from '../../../productShell/atelierTypes.ts';
import './inspection.css';
import { ConceptHelp } from '../../common/ConceptHelp.tsx';
import type { GuidanceLevel } from '../../../productShell/audienceProfiles.ts';

export interface InspectionPanelViewProps {
    readonly tab?: InspectTab;
    readonly onTabChange?: (tab: InspectTab) => void;
    readonly running?: boolean;
    readonly onPause?: () => void;
    readonly model: InspectionPanelDisplayModel;
    readonly commands: InspectionPanelCommands;
    readonly guidanceLevel: GuidanceLevel;
}

export const InspectionPanelView = memo(function InspectionPanelView({
    model,
    commands,
    guidanceLevel,
    tab, onTabChange, running = false, onPause,
}: InspectionPanelViewProps) {
    return (
        <div className="inspection-panel atelier-inspection">
            {tab && onTabChange && <Tabs panelPrefix="inspection-panel" label="Inspection views" items={[{ id: 'trace', label: 'Trace' }, { id: 'activations', label: 'Activations' }, { id: 'gradients', label: 'Gradients' }]} value={tab} onChange={onTabChange} />}
            <header className="inspection__intro"><h1>{tab === 'gradients' ? 'See how the model would change' : tab === 'activations' ? 'Look inside each layer' : 'Follow one prediction'}</h1><p>{tab === 'gradients' ? 'Preview gradients and explore a local parameter grid.' : tab === 'activations' ? 'Activation distributions and per-layer statistics from sampled training examples.' : 'Inspect the activations and outputs for a single data point.'}</p></header>
            {running && <div className="inspection__pause" role="status">Pause training to trace a sample, preview backpropagation, or run a probe.{onPause && <button className="btn" type="button" onClick={onPause}>Pause training</button>}</div>}
            {(tab === undefined || tab === 'activations') && <section id={tab ? `inspection-panel-${tab}` : undefined} aria-labelledby={tab && onTabChange ? `inspection-panel-tab-${tab}` : undefined} role={tab ? 'tabpanel' : undefined} aria-label={tab ? 'Activations' : 'Layer activations'}>

            {model.activationStatus && <p role="status">{model.activationStatus}</p>}
            {model.activationBasis && (
                <p className="inspection__basis">
                    <span>{model.activationBasis.label}</span>
                    {model.activationBasis.suffix}
                </p>
            )}
            {model.layers.length === 0 ? (
                <div className="inspection__empty">Train the model to see stats</div>
            ) : (
                <div className="inspection__layers">
                    {model.layers.map((layer) => (
                        <div key={layer.key} className="inspection__layer">
                            <div className="inspection__layer-name">{layer.name}</div>

                            <div className="inspection__stat-row">
                                <span className="inspection__stat-label">|∇w|</span>
                                <div className="inspection__bar-track">
                                    <div
                                        className="inspection__bar-fill inspection__bar-fill--grad"
                                        style={{ width: layer.gradientWidth }}
                                    />
                                </div>
                                <span className="inspection__stat-value">
                                    {layer.gradientValue}
                                </span>
                            </div>

                            <div className="inspection__stat-row">
                                <span className="inspection__stat-label">|w|</span>
                                <div className="inspection__bar-track">
                                    <div
                                        className="inspection__bar-fill inspection__bar-fill--weight"
                                        style={{ width: layer.weightWidth }}
                                    />
                                </div>
                                <span className="inspection__stat-value">
                                    {layer.weightValue}
                                </span>
                            </div>

                            <div className="inspection__stat-row">
                                <span className="inspection__stat-label">μ(a)</span>
                                <span className="inspection__stat-value" style={{ marginLeft: 'auto' }}>
                                    {layer.meanActivation}
                                </span>
                                <span className="inspection__stat-label" style={{ marginLeft: 8 }}>σ</span>
                                <span className="inspection__stat-value">
                                    {layer.activationStd}
                                </span>
                            </div>
                        </div>
                    ))}
                </div>
            )}
            <section className="inspection__layer" aria-label="Activation histogram explorer">
                <div className="inspection__layer-name">Activation Histogram</div>
                {model.histogramBasis && <p className="inspection__basis" role="status">{model.histogramBasis}</p>}
                {!model.histogram ? (
                    <div className="inspection__empty">Open inspection while training to sample layer activations.</div>
                ) : (
                    <>
                        <div className="control-row">
                            <label htmlFor="activation-histogram-layer">Histogram layer</label>
                            <select
                                id="activation-histogram-layer"
                                className="select"
                                value={model.histogram.selectedLayerIndex}
                                onChange={(event) => {
                                    const next = Number(event.currentTarget.value);
                                    commands.selectHistogramLayer(Number.isFinite(next) ? next : 0);
                                }}
                            >
                                {model.histogram.options.map((option) => (
                                    <option key={option.key} value={option.value}>
                                        {option.label}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div
                            className="inspection__histogram"
                            role="img"
                            aria-label={model.histogram.summary}
                        >
                            {model.histogram.bins.map((bin) => (
                                <div
                                    key={bin.key}
                                    className="inspection__histogram-bin"
                                    style={{ height: bin.height }}
                                    aria-hidden="true"
                                />
                            ))}
                        </div>
                        <div className="inspection__histogram-summary">
                            <strong>{model.histogram.title}</strong>
                            <span>{model.histogram.nearZeroText}</span>
                            <span>{model.histogram.saturatedText}</span>
                            <span>{model.histogram.rangeText}</span>
                        </div>
                        <p className="sr-only">{model.histogram.summary}</p>
                    </>
                )}
            </section>
            </section>}
            {(tab === undefined || tab === 'trace') && <div id={tab ? `inspection-panel-${tab}` : undefined} aria-labelledby={tab && onTabChange ? `inspection-panel-tab-${tab}` : undefined} role={tab ? 'tabpanel' : undefined} className="inspection__layer" aria-label="Prediction trace">
                <div className="inspection__layer-name">Prediction Trace</div>
                <div className="control-row">
                    <label htmlFor="trace-source">Sample</label>
                    <select
                        id="trace-source"
                        className="select"
                        value={model.trace.source}
                        onChange={(event) => {
                            commands.selectTraceSource(
                                event.currentTarget.value as InspectionTraceSource,
                            );
                        }}
                    >
                        <option value="train">Training</option>
                        <option value="test">Test</option>
                    </select>
                </div>
                <div className="control-row">
                    <label htmlFor="trace-index">Index</label>
                    <input
                        id="trace-index"
                        className="input"
                        type="number"
                        min={0}
                        max={model.trace.maxSampleIndex}
                        value={model.trace.sampleIndex}
                        onChange={(event) => {
                            commands.selectSampleIndex(event.currentTarget.valueAsNumber);
                        }}
                    />
                </div>

                {model.trace.emptyMessage ? (
                    <div className="inspection__empty">
                        {model.trace.emptyMessage}
                    </div>
                ) : null}
                <button
                    type="button"
                    className="btn"
                    onClick={() => { void commands.requestTrace(); }}
                    disabled={running || model.trace.buttonDisabled}
                >
                    {model.trace.buttonLabel}
                </button>
                <div aria-live="polite">
                    {model.trace.errorMessage ? (
                        <div className="inspection__empty">
                            {model.trace.errorMessage}
                        </div>
                    ) : null}
                    {model.trace.result ? (
                        <div className="inspection__layers inspection__trace-result">
                            <p className="inspection__basis">
                                {model.trace.result.provenance}
                            </p>
                            {model.trace.result.sample && <p className="inspection__sample">Selected data point · x₁ {model.trace.result.sample.x} · x₂ {model.trace.result.sample.y} · Target {model.trace.result.sample.target}</p>}
                            <p className="inspection__direction">Forward pass: input → hidden layers → output</p>
                            <div className="inspection__stat-row">
                                <span className="inspection__stat-label">Output</span>
                                <span className="inspection__stat-value">
                                    {model.trace.result.output}
                                </span>
                                <span className="inspection__stat-label" style={{ marginLeft: 8 }}>sample data loss</span>
                                <span className="inspection__stat-value">
                                    {model.trace.result.sampleDataLoss}
                                </span>
                                <span className="inspection__stat-label" style={{ marginLeft: 8 }}>model penalty</span>
                                <span className="inspection__stat-value">
                                    {model.trace.result.regularizationPenalty}
                                </span>
                            </div>
                            <div className="inspection__trace-flow" aria-label="Forward activation flow">
                                {model.trace.result.sample && <section className="inspection__trace-stage"><h3>Input coordinates</h3><div className="inspection__activation-values"><span title={model.trace.result.sample.x}>{Number(model.trace.result.sample.x).toFixed(4)}</span><span title={model.trace.result.sample.y}>{Number(model.trace.result.sample.y).toFixed(4)}</span></div></section>}
                                {model.trace.result.layers.map((layer) => (
                                    <section key={layer.key} className="inspection__trace-stage"><h3>{layer.label}</h3>
                                        <div className="inspection__activation-values">{layer.activations.split(', ').map((value, index) => <span key={index} data-sign={Number(value) > 0 ? 'positive' : Number(value) < 0 ? 'negative' : 'zero'}>{value}</span>)}</div>
                                    </section>
                                ))}
                                <section className="inspection__trace-stage"><h3>Output</h3><div className="inspection__activation-values">{model.trace.result.output.split(', ').map((value, index) => <span key={index} data-sign={Number(value) > 0 ? 'positive' : Number(value) < 0 ? 'negative' : 'zero'}>{value}</span>)}</div></section>
                            </div>
                        </div>
                    ) : null}
                </div>
            </div>
            }
            {(tab === undefined || tab === 'gradients') && <div id={tab ? `inspection-panel-${tab}` : undefined} aria-labelledby={tab && onTabChange ? `inspection-panel-tab-${tab}` : undefined} role={tab ? 'tabpanel' : undefined} aria-label={tab ? 'Gradients' : undefined} className="inspection__gradient-grid"><section className="inspection__layer" aria-label="Slow-motion backprop preview">
                <div className="inspection__layer-name inspection__layer-name--concept">
                    <span>Slow-Motion Backprop</span>
                    <ConceptHelp conceptId="gradient" guidanceLevel={guidanceLevel} />
                </div>
                <button
                    type="button"
                    className="btn"
                    onClick={() => { void commands.requestBackprop(); }}
                    disabled={running || model.backprop.buttonDisabled}
                >
                    {model.backprop.buttonLabel}
                </button>
                <div
                    role="status"
                    aria-live="polite"
                    aria-label="Backprop preview status"
                    className="inspection__empty"
                >
                    {model.backprop.statusText}
                </div>
                {model.backprop.result ? (
                    <div className="inspection__layers">
                        <div className="inspection__stat-row">
                            <span className="inspection__stat-label">
                                {model.backprop.result.provenance}
                            </span>
                        </div>
                        <div className="inspection__stat-row">
                            {model.backprop.result.objective.map((value, index) => (
                                <span
                                    key={value}
                                    className={index === 0
                                        ? 'inspection__stat-label'
                                        : 'inspection__stat-value'}
                                >
                                    {value}
                                </span>
                            ))}
                        </div>
                        <div className="inspection__stat-row">
                            <span className="inspection__stat-label">
                                {model.backprop.result.gradient[0]}
                            </span>
                            <span className="inspection__stat-value">
                                {model.backprop.result.gradient[1]}
                            </span>
                        </div>
                        <p className="inspection__direction">Backward pass: output → hidden layers → input</p>
                        <ul className="inspection__backprop-list" aria-label="Backprop layer summaries">
                            {model.backprop.result.layers.map((layer) => (
                                <li key={layer.key} className="inspection__backprop-item">
                                    <div className="inspection__backprop-heading">
                                        <span className="inspection__stat-label">
                                            {layer.name}
                                        </span>
                                        <span className="inspection__stat-value">
                                            {layer.status}
                                        </span>
                                    </div>
                                    <div className="inspection__backprop-metrics">
                                        {layer.metrics.map((metric) => (
                                            <span key={metric} className="inspection__stat-value"><span>{metric.slice(0, metric.lastIndexOf(' '))}</span><strong>{metric.slice(metric.lastIndexOf(' ') + 1)}</strong></span>
                                        ))}
                                    </div>
                                    <div className="inspection__backprop-note">
                                        {layer.note}
                                    </div>
                                </li>
                            ))}
                        </ul>
                    </div>
                ) : null}
            </section>
            <section className="inspection__layer" aria-label="Loss landscape probe">
                <div className="inspection__layer-name">Loss Landscape Probe</div>
                <button
                    type="button"
                    className="btn"
                    onClick={() => { void commands.requestLandscape(); }}
                    disabled={running || model.landscape.buttonDisabled}
                >
                    {model.landscape.buttonLabel}
                </button>
                <div
                    role="status"
                    aria-live="polite"
                    aria-label="Loss landscape probe status"
                    className="inspection__empty"
                >
                    {model.landscape.statusText}
                </div>
                {model.landscape.result ? (
                    <div className="inspection__layers">
                        <div className="inspection__stat-row">
                            <span className="inspection__stat-label">
                                {model.landscape.result.provenance}
                            </span>
                        </div>
                        {model.landscape.result.axes && <p className="inspection__axes">Horizontal: {model.landscape.result.axes[0]} · Vertical: {model.landscape.result.axes[1]}</p>}
                        <div
                            className="inspection__loss-heatmap"
                            role="img"
                            aria-label={model.landscape.result.summary}
                            style={{
                                gridTemplateColumns: model.landscape.result.gridTemplateColumns,
                            }}
                        >
                            {model.landscape.result.cells.map((cell) => (
                                <span
                                    key={cell.key}
                                    className="inspection__loss-cell"
                                    style={{ opacity: cell.opacity }}
                                    aria-hidden="true"
                                />
                            ))}
                        </div>
                        <div className="inspection__probe-legend"><span>Higher objective</span><i aria-hidden="true" /><span>Lower objective</span></div>
                        <div className="inspection__histogram-summary">
                            <strong>{model.landscape.result.title}</strong>
                            {model.landscape.result.values.map((value) => (
                                <span key={value}>{value}</span>
                            ))}
                            <span>{model.landscape.result.basis}</span>
                            <span>{model.landscape.result.bestDirection}</span>
                        </div>
                    </div>
                ) : null}
            </section></div>}
            <footer className="inspection__footer">Single-sample traces and local diagnostic views complement full-split evaluations in Results. Previews and probes do not update model weights.</footer>
        </div>
    );
});
