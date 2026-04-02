// ── Code Export Panel ──
// Tabbed panel that generates pseudocode, NumPy, and TF.js code from the current network.

import { useState, useMemo, useCallback, memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { generatePseudocode, generateNumPy, generateTFJS } from '@nn-playground/shared';

type CodeTab = 'pseudocode' | 'numpy' | 'tfjs';

const TABS: { id: CodeTab; label: string }[] = [
    { id: 'pseudocode', label: 'Pseudocode' },
    { id: 'numpy', label: 'NumPy' },
    { id: 'tfjs', label: 'TF.js' },
];

export const CodeExportPanel = memo(function CodeExportPanel() {
    const [activeTab, setActiveTab] = useState<CodeTab>('pseudocode');
    const [copied, setCopied] = useState(false);
    const [expanded, setExpanded] = useState(false);

    const network = usePlaygroundStore((s) => s.network);
    const training = usePlaygroundStore((s) => s.training);
    const features = usePlaygroundStore((s) => s.features);
    const snapshot = useTrainingStore((s) => s.snapshot);

    const code = useMemo(() => {
        const config = {
            ...network,
            inputSize: Object.values(features).filter(Boolean).length,
            outputSize: 1,
        };
        switch (activeTab) {
            case 'pseudocode':
                return generatePseudocode(config, training, features, snapshot);
            case 'numpy':
                return generateNumPy(config, training, features, snapshot);
            case 'tfjs':
                return generateTFJS(config, training, features, snapshot);
        }
    }, [activeTab, network, training, features, snapshot]);

    const handleCopy = useCallback(() => {
        navigator.clipboard.writeText(code).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        });
    }, [code]);

    if (!expanded) {
        return (
            <div className="panel">
                <div
                    className="panel__title"
                    style={{ cursor: 'pointer', userSelect: 'none' }}
                    onClick={() => setExpanded(true)}
                >
                    Code Export ▸
                </div>
            </div>
        );
    }

    return (
        <div className="panel code-export-panel">
            <div
                className="panel__title"
                style={{ cursor: 'pointer', userSelect: 'none' }}
                onClick={() => setExpanded(false)}
            >
                Code Export ▾
            </div>

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

            <button
                className="btn btn--ghost btn--sm"
                style={{ marginTop: 6, width: '100%' }}
                onClick={handleCopy}
            >
                {copied ? '✓ Copied!' : '📋 Copy Code'}
            </button>
        </div>
    );
});
