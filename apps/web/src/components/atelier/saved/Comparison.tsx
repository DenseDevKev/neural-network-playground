import { useEffect, useRef, useState } from 'react';
import type { ExperimentRunRecordV2 } from '@nn-playground/shared';

function flatten(value: unknown, prefix = ''): Record<string, string> {
    if (value !== null && typeof value === 'object') {
        const entries = Object.entries(value);
        return entries.length ? Object.assign({}, ...entries.map(([key, child]) => flatten(child, prefix ? `${prefix} / ${key}` : key))) : { [prefix]: 'None' };
    }
    return { [prefix]: String(value) };
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
    const sections: [string, unknown, unknown][] = [
        ['Full evaluation', left, right],
        ['Recipe', current.recipe, baseline.recipe],
        ['Recipe identity', current.recipeFingerprint, baseline.recipeFingerprint],
        ['Model identity', current.snapshot.model, baseline.snapshot.model],
        ['Stored learning history', current.snapshot.trendHistory, baseline.snapshot.trendHistory],
        ['Stored evaluation history', current.snapshot.evaluationHistory, baseline.snapshot.evaluationHistory],
    ];
    return <section role="group" aria-label={`Saved run comparison: ${currentLabel} and ${baselineLabel}`} className="saved-comparison">
        <button type="button" onClick={onBack}>← Saved runs</button>
        <h2 ref={heading} tabIndex={-1}>Compare experiments</h2>
        <p>{summary}</p>
        {!comparable && <p>Dataset and objective identities must both match before losses can be ranked.</p>}
        <label className="saved-differences"><input type="checkbox" checked={differences} onChange={(event) => setDifferences(event.target.checked)} />Only show differences</label>
        <div className="saved-table-scroll"><table className="saved-comparison-table"><thead><tr><th scope="col">Configuration and evidence</th><th scope="col">{currentLabel}</th><th scope="col">{baselineLabel}</th></tr></thead><tbody>
            {sections.map(([title, a, b]) => {
                const x = flatten(a); const y = flatten(b);
                const keys = [...new Set([...Object.keys(x), ...Object.keys(y)])].filter((key) => !differences || x[key] !== y[key]);
                return keys.length ? <SectionRows key={title} title={title} keys={keys} left={x} right={y} /> : null;
            })}
        </tbody></table></div>
        <p>Saved evidence contains no trained parameters. Applying a recipe initializes a fresh model.</p>
    </section>;
}
function SectionRows({ title, keys, left, right }: { title: string; keys: string[]; left: Record<string, string>; right: Record<string, string> }) {
    return <><tr className="saved-section-row"><th colSpan={3}>{title}</th></tr>{keys.map((key) => <tr key={key} className={left[key] !== right[key] ? 'saved-different' : undefined}><th scope="row">{key ? key.replace(/([a-z])([A-Z])/g, '$1 $2') : 'Value'}</th><td>{left[key] ?? 'Not recorded'}</td><td>{right[key] ?? 'Not recorded'}</td></tr>)}</>;
}
