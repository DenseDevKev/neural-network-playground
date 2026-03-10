// ── Header Component ──
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';

export function Header() {
    const snapshot = usePlaygroundStore((s) => s.snapshot);

    const epoch = snapshot?.epoch ?? 0;
    const trainLoss = snapshot?.trainLoss ?? 0;
    const testLoss = snapshot?.testLoss ?? 0;
    const accuracy = snapshot?.trainMetrics?.accuracy;

    return (
        <header className="header">
            <div className="header__brand">
                <div className="header__logo">NN</div>
                <h1 className="header__title">
                    <span>Neural Network</span> Playground
                </h1>
            </div>
            <div className="header__metrics">
                <div className="header__metric">
                    <span>Epoch</span>
                    <span className="header__metric-value header__metric-value--epoch">
                        {String(epoch).padStart(4, '0')}
                    </span>
                </div>
                <div className="header__metric">
                    <span>Train Loss</span>
                    <span className="header__metric-value header__metric-value--train">
                        {trainLoss.toFixed(4)}
                    </span>
                </div>
                <div className="header__metric">
                    <span>Test Loss</span>
                    <span className="header__metric-value header__metric-value--test">
                        {testLoss.toFixed(4)}
                    </span>
                </div>
                {accuracy != null && (
                    <div className="header__metric">
                        <span>Accuracy</span>
                        <span className="header__metric-value header__metric-value--train">
                            {(accuracy * 100).toFixed(1)}%
                        </span>
                    </div>
                )}
            </div>
        </header>
    );
}
