// ── Confusion Matrix Component ──
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';

export function ConfusionMatrix() {
    const problemType = usePlaygroundStore((s) => s.data.problemType);
    const cm = usePlaygroundStore((s) => s.snapshot?.testMetrics.confusionMatrix);

    if (problemType !== 'classification') return null;
    if (!cm) return null;

    const total = cm.tp + cm.tn + cm.fp + cm.fn;
    if (total === 0) return null;

    // Helper to render a cell with a background intensity
    const Cell = ({ value, label, rgb }: { value: number; label: string; rgb: string }) => {
        const pct = value / total;
        // alpha scales with percentage, min 0.05 so it's not totally black, max 0.4
        const alpha = Math.max(0.05, pct * 0.4).toFixed(2);
        return (
            <div className={`cm-cell cm-${label.toLowerCase()}`} style={{ backgroundColor: `rgba(${rgb}, ${alpha})` }}>
                <div className="cm-value">{value}</div>
                <div className="cm-label">{label}</div>
            </div>
        );
    };

    return (
        <div className="panel confusion-matrix">
            <div className="panel__title">Confusion Matrix (Test Set)</div>
            <div className="cm-grid-container">
                <div className="cm-grid">
                    <div className="cm-header empty"></div>
                    <div className="cm-header">Pred 0</div>
                    <div className="cm-header">Pred 1</div>

                    <div className="cm-header side">Actual 0</div>
                    <Cell value={cm.tn} label="TN" rgb="59, 130, 246" /> {/* Blue for Class 0 */}
                    <Cell value={cm.fp} label="FP" rgb="239, 68, 68" />  {/* Red for Error */}

                    <div className="cm-header side">Actual 1</div>
                    <Cell value={cm.fn} label="FN" rgb="239, 68, 68" />  {/* Red for Error */}
                    <Cell value={cm.tp} label="TP" rgb="249, 115, 22" /> {/* Orange for Class 1 */}
                </div>
            </div>
        </div>
    );
}
