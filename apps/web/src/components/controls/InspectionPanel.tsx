// ── Advanced Inspection Panel ──
// Displays per-layer gradient magnitudes, activation stats, and weight distributions.

import { useEffect, useMemo, memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { getFrameBuffer } from '../../worker/frameBuffer.ts';

export const InspectionPanel = memo(function InspectionPanel() {
    const snapshot = useTrainingStore((s) => s.snapshot);
    const frameVersion = useTrainingStore((s) => s.frameVersion);
    const hiddenLayers = usePlaygroundStore((s) => s.network.hiddenLayers);

    useEffect(() => {
        const enableLayerStats = (needLayerStats: boolean) => {
            const { demand, setDemand } = usePlaygroundStore.getState();
            if (demand.needLayerStats === needLayerStats) return;
            setDemand({ ...demand, needLayerStats });
        };

        enableLayerStats(true);
        return () => enableLayerStats(false);
    }, []);

    const layerStats = useMemo(
        () => {
            void frameVersion;
            return getFrameBuffer().layerStats ?? snapshot?.layerStats;
        },
        [frameVersion, snapshot?.layerStats],
    );

    const layerNames = useMemo(() => {
        const names: string[] = [];
        for (let i = 0; i < hiddenLayers.length; i++) {
            names.push(`Hidden ${i + 1}`);
        }
        names.push('Output');
        return names;
    }, [hiddenLayers]);

    const layerStatMaxima = useMemo(() => {
        if (!layerStats || layerStats.length === 0) {
            return { maxGrad: 0.001, maxWeight: 0.001 };
        }

        let maxGrad = 0.001;
        let maxWeight = 0.001;
        for (const stats of layerStats) {
            maxGrad = Math.max(maxGrad, stats.meanAbsGradient);
            maxWeight = Math.max(maxWeight, stats.meanAbsWeight);
        }
        return { maxGrad, maxWeight };
    }, [layerStats]);

    return (
        <div className="inspection-panel">
            {!layerStats || layerStats.length === 0 ? (
                <div className="inspection__empty">Train the model to see stats</div>
            ) : (
                <div className="inspection__layers">
                    {layerStats.map((stats, idx) => {
                        const gradPct = (stats.meanAbsGradient / layerStatMaxima.maxGrad) * 100;
                        const weightPct = (stats.meanAbsWeight / layerStatMaxima.maxWeight) * 100;

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
});
