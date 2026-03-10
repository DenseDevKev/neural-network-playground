// ── Advanced Inspection Panel ──
// Displays per-layer gradient magnitudes, activation stats, and weight distributions.

import { useMemo, useState } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';

export function InspectionPanel() {
    const snapshot = usePlaygroundStore((s) => s.snapshot);
    const hiddenLayers = usePlaygroundStore((s) => s.network.hiddenLayers);
    const [expanded, setExpanded] = useState(false);

    const layerStats = snapshot?.layerStats;

    const layerNames = useMemo(() => {
        const names: string[] = [];
        for (let i = 0; i < hiddenLayers.length; i++) {
            names.push(`Hidden ${i + 1}`);
        }
        names.push('Output');
        return names;
    }, [hiddenLayers]);

    if (!expanded) {
        return (
            <div className="panel">
                <div
                    className="panel__title"
                    style={{ cursor: 'pointer', userSelect: 'none' }}
                    onClick={() => setExpanded(true)}
                >
                    Inspection ▸
                </div>
            </div>
        );
    }

    return (
        <div className="panel inspection-panel">
            <div
                className="panel__title"
                style={{ cursor: 'pointer', userSelect: 'none' }}
                onClick={() => setExpanded(false)}
            >
                Inspection ▾
            </div>

            {!layerStats || layerStats.length === 0 ? (
                <div className="inspection__empty">Train the model to see stats</div>
            ) : (
                <div className="inspection__layers">
                    {layerStats.map((stats, idx) => {
                        const maxGrad = Math.max(...layerStats.map((s) => s.meanAbsGradient), 0.001);
                        const maxWeight = Math.max(...layerStats.map((s) => s.meanAbsWeight), 0.001);
                        const gradPct = (stats.meanAbsGradient / maxGrad) * 100;
                        const weightPct = (stats.meanAbsWeight / maxWeight) * 100;

                        return (
                            <div key={idx} className="inspection__layer">
                                <div className="inspection__layer-name">{layerNames[idx]}</div>

                                <div className="inspection__stat-row">
                                    <span className="inspection__stat-label">|∇w|</span>
                                    <div className="inspection__bar-track">
                                        <div
                                            className="inspection__bar-fill inspection__bar-fill--grad"
                                            style={{ width: `${Math.max(2, gradPct)}%` }}
                                        />
                                    </div>
                                    <span className="inspection__stat-value">
                                        {stats.meanAbsGradient < 0.0001
                                            ? stats.meanAbsGradient.toExponential(1)
                                            : stats.meanAbsGradient.toFixed(4)}
                                    </span>
                                </div>

                                <div className="inspection__stat-row">
                                    <span className="inspection__stat-label">|w|</span>
                                    <div className="inspection__bar-track">
                                        <div
                                            className="inspection__bar-fill inspection__bar-fill--weight"
                                            style={{ width: `${Math.max(2, weightPct)}%` }}
                                        />
                                    </div>
                                    <span className="inspection__stat-value">
                                        {stats.meanAbsWeight.toFixed(4)}
                                    </span>
                                </div>

                                <div className="inspection__stat-row">
                                    <span className="inspection__stat-label">μ(a)</span>
                                    <span className="inspection__stat-value" style={{ marginLeft: 'auto' }}>
                                        {stats.meanActivation.toFixed(4)}
                                    </span>
                                    <span className="inspection__stat-label" style={{ marginLeft: 8 }}>σ</span>
                                    <span className="inspection__stat-value">
                                        {stats.activationStd.toFixed(4)}
                                    </span>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
