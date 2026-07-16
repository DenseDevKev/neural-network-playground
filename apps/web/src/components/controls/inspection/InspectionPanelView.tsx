import { memo } from 'react';
import {
    type InspectionPanelCommands,
    type InspectionPanelDisplayModel,
    type InspectionTraceSource,
} from './inspectionPanelModel.ts';
import { ConceptHelp } from '../../common/ConceptHelp.tsx';
import type { GuidanceLevel } from '../../../productShell/audienceProfiles.ts';

export interface InspectionPanelViewProps {
    readonly model: InspectionPanelDisplayModel;
    readonly commands: InspectionPanelCommands;
    readonly guidanceLevel: GuidanceLevel;
}

export const InspectionPanelView = memo(function InspectionPanelView({
    model,
    commands,
    guidanceLevel,
}: InspectionPanelViewProps) {
    return (
        <div className="inspection-panel">
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
            <div className="inspection__layer" aria-label="Prediction trace">
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
                    disabled={model.trace.buttonDisabled}
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
                        <div className="inspection__layers">
                            <p className="inspection__basis">
                                {model.trace.result.provenance}
                            </p>
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
                            {model.trace.result.layers.map((layer) => (
                                <div key={layer.key} className="inspection__stat-row" style={{ alignItems: 'flex-start', flexDirection: 'column', gap: 2, marginBottom: 6 }}>
                                    <span className="inspection__stat-label">{layer.label}</span>
                                    <div style={{
                                        width: '100%',
                                        background: 'rgba(0,0,0,0.2)',
                                        padding: '4px 6px',
                                        borderRadius: 4,
                                        fontFamily: 'var(--font-mono)',
                                        fontSize: 9,
                                        color: 'var(--text-secondary)',
                                        overflowX: 'auto',
                                        whiteSpace: 'nowrap',
                                        border: '1px solid rgba(255,255,255,0.05)',
                                    }}>
                                        {layer.activations}
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : null}
                </div>
            </div>
            <section className="inspection__layer" aria-label="Slow-motion backprop preview">
                <div className="inspection__layer-name inspection__layer-name--concept">
                    <span>Slow-Motion Backprop</span>
                    <ConceptHelp conceptId="gradient" guidanceLevel={guidanceLevel} />
                </div>
                <button
                    type="button"
                    className="btn"
                    onClick={() => { void commands.requestBackprop(); }}
                    disabled={model.backprop.buttonDisabled}
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
                                            <span key={metric} className="inspection__stat-value">
                                                {metric}
                                            </span>
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
                    disabled={model.landscape.buttonDisabled}
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
            </section>
        </div>
    );
});
