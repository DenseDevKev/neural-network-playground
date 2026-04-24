// ── Code Export Panel ──
// Tabbed panel that generates pseudocode, NumPy, and TF.js code from the current network.

import { useMemo, useCallback, memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { useLayoutStore, type CodeExportTab } from '../../store/useLayoutStore.ts';
import { generatePseudocode, generateNumPy, generateTFJS } from '@nn-playground/shared';
import { Tooltip } from '../common/Tooltip.tsx';
import { getFrameBuffer, unflattenBiases, unflattenWeights } from '../../worker/frameBuffer.ts';
import { useTimedState } from '../../hooks/useTimedState.ts';

const TABS: { id: CodeExportTab; label: string }[] = [
    { id: 'pseudocode', label: 'Pseudocode' },
    { id: 'numpy', label: 'NumPy' },
    { id: 'tfjs', label: 'TF.js' },
];

export const CodeExportPanel = memo(function CodeExportPanel() {
    const activeTab = useLayoutStore((s) => s.codeExportTab);
    const setActiveTab = useLayoutStore((s) => s.setCodeExportTab);
    const [copied, setCopied] = useTimedState(false, 2000);

    const network = usePlaygroundStore((s) => s.network);
    const training = usePlaygroundStore((s) => s.training);
    const features = usePlaygroundStore((s) => s.features);
    const snapshot = useTrainingStore((s) => s.snapshot);
    const frameVersion = useTrainingStore((s) => s.frameVersion);

    const exportSnapshot = useMemo(() => {
        if (!snapshot) return null;

        const frameBuffer = getFrameBuffer();
        if (frameBuffer.weights && frameBuffer.biases && frameBuffer.weightLayout) {
            return {
                ...snapshot,
                weights: unflattenWeights(frameBuffer.weights, frameBuffer.weightLayout.layerSizes),
                biases: unflattenBiases(frameBuffer.biases, frameBuffer.weightLayout.layerSizes),
            };
        }

        return snapshot;
    }, [snapshot, frameVersion]);

    const code = useMemo(() => {
        const config = {
            ...network,
            inputSize: Object.values(features).filter(Boolean).length,
            outputSize: 1,
        };
        switch (activeTab) {
            case 'pseudocode':
                return generatePseudocode(config, training, features, exportSnapshot);
            case 'numpy':
                return generateNumPy(config, training, features, exportSnapshot);
            case 'tfjs':
                return generateTFJS(config, training, features, exportSnapshot);
        }
    }, [activeTab, network, training, features, exportSnapshot]);

    const handleCopy = useCallback(() => {
        navigator.clipboard.writeText(code).then(() => {
            setCopied(true);
        });
    }, [code, setCopied]);

    return (
        <div className="code-export-panel">
            <div className="code-export__tabs">
                {TABS.map((tab) => (
                    <button
                        key={tab.id}
                        className={`chip ${activeTab === tab.id ? 'active' : ''}`}
                        onClick={() => setActiveTab(tab.id)}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>

            <div className="code-export__code-container">
                <pre className="code-export__code">{code}</pre>
            </div>

            <Tooltip content="Copy the generated code to your clipboard" block>
                <button
                    className="btn btn--ghost btn--sm"
                    style={{ marginTop: 6, width: '100%' }}
                    onClick={handleCopy}
                >
                    {copied ? '✓ Copied!' : '📋 Copy Code'}
                </button>
            </Tooltip>
        </div>
    );
});
