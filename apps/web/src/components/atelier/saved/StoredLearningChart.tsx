import type { ExperimentRunRecordV2 } from '@nn-playground/shared';

export function StoredLearningChart({ record, label, maximumStep, maximumLoss }: {
    record: ExperimentRunRecordV2; label: string; maximumStep: number; maximumLoss: number;
}) {
    const { trendHistory, evaluationHistory } = record.snapshot;
    const series = [
        { label: 'Batch EMA', color: 'var(--text-secondary)', dash: '2 5', points: trendHistory.map((p) => [p.model.step, p.dataLoss]) },
        { label: 'Train loss', color: '#3984BD', dash: undefined, points: evaluationHistory.map((p) => [p.model.step, p.train.values.dataLoss]) },
        { label: 'Test loss', color: '#DF633B', dash: '8 4', points: evaluationHistory.map((p) => [p.model.step, p.test.values.dataLoss]) },
        { label: 'Objective', color: '#4D9568', dash: '10 3 2 3', points: evaluationHistory.map((p) => [p.model.step, p.objective.trainTotalObjective]) },
    ];
    return <figure className="saved-learning-chart"><figcaption>{label}</figcaption>
        {trendHistory.length + evaluationHistory.length === 0 ? <p>No learning history was stored with this record.</p> : <>
            <svg viewBox="0 0 500 250" role="img" aria-label={`${label}: stored batch EMA, full train and test losses, and objective through step ${record.snapshot.model.step}`}>
                {[0,1,2,3,4].map((i) => <g key={i}><line x1="56" x2="480" y1={20+i*48} y2={20+i*48} stroke="var(--border-color)" /><text x="48" y={25+i*48} textAnchor="end">{(maximumLoss*(1-i/4)).toPrecision(2)}</text></g>)}
                {series.map((s) => <g key={s.label} fill="none" stroke={s.color} strokeWidth="2" strokeDasharray={s.dash}><polyline points={s.points.map(([step,value]) => `${56+step/maximumStep*424},${212-value/maximumLoss*192}`).join(' ')} />{s.points.length === 1 && <circle cx={56+s.points[0][0]/maximumStep*424} cy={212-s.points[0][1]/maximumLoss*192} r="3" fill={s.color} />}</g>)}
                <text x="56" y="242">0</text><text x="260" y="242" textAnchor="middle">Model step</text><text x="480" y="242" textAnchor="end">{maximumStep.toLocaleString()}</text>
            </svg>
            <div className="saved-chart-legend">{series.map((s) => <span key={s.label}><svg viewBox="0 0 26 8" aria-hidden="true"><path d="M0 4H26" stroke={s.color} strokeWidth="2" strokeDasharray={s.dash} /></svg>{s.label}</span>)}</div>
            <p>{trendHistory.length} batch observations · {evaluationHistory.length} full evaluations. Stored steps share the same axes across both records.</p>
        </>}
    </figure>;
}
