import { useEffect, useRef, useState } from 'react';
import { StoredLearningChart } from './StoredLearningChart.tsx';
import type { ExperimentRunRecordV2 } from '@nn-playground/shared';

function primaryRows(record: ExperimentRunRecordV2): Record<string, string> {
    const r = record.recipe;
    const e = record.snapshot.evaluation;
    const describe = (value: unknown): string => typeof value === 'object' && value !== null
        ? Object.entries(value).map(([key, child]) => key === 'kind' ? String(child) : `${key.replace(/([a-z])([A-Z])/g, '$1 $2')} ${String(child)}`).join(' · ')
        : String(value);
    return {
        'Dataset': `${r.task.dataset} · ${r.task.kind}`,
        'Data seed / noise / split': `${r.data.seed} / ${r.data.noise}% / ${r.data.trainFraction * 100}% training`,
        'Evaluated populations': `${e.train.basis.sampleCount} training · ${e.test.basis.sampleCount} test`,
        'Architecture': `${r.inputs.featureIds.length} → ${r.model.hiddenLayers.length ? r.model.hiddenLayers.join(' → ') + ' → ' : ''}${r.task.kind === 'multiclass-classification' ? 3 : 1}`,
        'Input features': r.inputs.featureIds.join(', '),
        'Activation / initialization': `${r.model.hiddenActivation} / ${r.model.initialization}`,
        'Model seed': String(r.model.seed),
        'Optimizer': describe(r.training.optimizer),
        'Learning rate / batch size': `${r.training.learningRate} / ${r.training.batchSize}`,
        'Schedule': describe(r.training.schedule),
        'Data loss': describe(r.objective.dataLoss),
        'Regularization': describe(r.objective.penalty),
        'Gradient clipping': describe(r.training.gradientClipping),
        'Evaluated step': String(e.model.step),
        'Train data loss': String(e.train.values.dataLoss),
        'Test data loss': String(e.test.values.dataLoss),
        'Training objective': String(e.objective.trainTotalObjective),
        'Test accuracy': e.test.values.accuracy === undefined ? 'Not applicable' : `${e.test.values.accuracy * 100}%`,
    };
}

function displayValue(key: string, value: string) {
    if (['Train data loss', 'Test data loss', 'Training objective'].includes(key)) return Number(value).toFixed(4);
    if (key === 'Test accuracy' && value.endsWith('%')) return `${parseFloat(value).toFixed(1)}%`;
    return value;
}

export function SavedRunComparison({ current, baseline, currentLabel, baselineLabel, onBack }: {
    current: ExperimentRunRecordV2; baseline: ExperimentRunRecordV2;
    currentLabel: string; baselineLabel: string; onBack(): void;
}) {
    const heading = useRef<HTMLHeadingElement>(null);
    useEffect(() => { heading.current?.focus(); }, []);
    const [differences, setDifferences] = useState(false);
    const left = current.snapshot.evaluation;
    const right = baseline.snapshot.evaluation;
    const comparable = left.dataset.datasetKey === right.dataset.datasetKey && left.objectiveKey === right.objectiveKey;
    const delta = left.test.values.dataLoss - right.test.values.dataLoss;
    const summary = !comparable ? 'Not directly comparable' : delta === 0 ? `Equal test data loss at ${left.test.values.dataLoss.toFixed(4)}` : `${delta < 0 ? currentLabel : baselineLabel} has lower test data loss by ${Math.abs(delta).toFixed(4)}`;
    const x = primaryRows(current); const y = primaryRows(baseline);
    const keys = Object.keys(x).filter((key) => !differences || x[key] !== y[key]);
    const records = [current,baseline];
    const maximumStep = Math.max(1,...records.flatMap((r) => [...r.snapshot.trendHistory.map((p) => p.model.step),...r.snapshot.evaluationHistory.map((p) => p.model.step)]));
    const maximumLoss = Math.max(.000001,...records.flatMap((r) => [...r.snapshot.trendHistory.map((p) => p.dataLoss),...r.snapshot.evaluationHistory.flatMap((p) => [p.train.values.dataLoss,p.test.values.dataLoss,p.objective.trainTotalObjective])]))*1.05;
    return <section role="group" aria-label={`Saved run comparison: ${currentLabel} and ${baselineLabel}`} className="saved-comparison">
        <button type="button" onClick={onBack}>← Saved runs</button>
        <h2 ref={heading} tabIndex={-1}>Compare experiments</h2>
        <p>{summary}</p>
        {!comparable && <p>Dataset and objective identities must both match before losses can be ranked.</p>}
        <h3>Stored learning history</h3><div className="saved-learning-pair"><StoredLearningChart record={current} label={currentLabel} maximumStep={maximumStep} maximumLoss={maximumLoss} /><StoredLearningChart record={baseline} label={baselineLabel} maximumStep={maximumStep} maximumLoss={maximumLoss} /></div>
        <label className="saved-differences"><input type="checkbox" checked={differences} onChange={(event) => setDifferences(event.target.checked)} />Only show differences</label>
        <div className="saved-table-scroll"><table className="saved-comparison-table"><thead><tr><th scope="col">Configuration and evidence</th><th scope="col">{currentLabel}</th><th scope="col">{baselineLabel}</th></tr></thead><tbody>
            {keys.map((key) => <tr key={key} className={x[key] !== y[key] ? 'saved-different' : undefined}><th scope="row">{key}</th><td title={x[key]}>{displayValue(key, x[key])}</td><td title={y[key]}>{displayValue(key, y[key])}</td></tr>)}
            {!keys.length && <tr><td colSpan={3}>No differences in these configuration and evidence rows.</td></tr>}
        </tbody></table></div>
        <details className="saved-evidence-details"><summary>Exact identities and complete scientific evidence</summary>
            <p>All recorded values, recipe identities, model revisions and stored histories.</p>
            {records.map((record, index) => <section key={record.id}><h3>{index === 0 ? currentLabel : baselineLabel}</h3><pre>{JSON.stringify(record, null, 2)}</pre></section>)}
        </details>
        <p>Saved evidence contains no trained parameters. Applying a recipe initializes a fresh model.</p>
    </section>;
}