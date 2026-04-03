// ── Code Export Panel ──
// Tabbed panel that generates pseudocode, NumPy, and TF.js code from the current network.

import { useState, useMemo, useCallback, memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { generatePseudocode, generateNumPy, generateTFJS } from '@nn-playground/shared';
import { Tooltip } from '../common/Tooltip.tsx';

type CodeTab = 'pseudocode' | 'numpy' | 'tfjs';

const TABS: { id: CodeTab; label: string }[] = [
    { id: 'pseudocode', label: 'Pseudocode' },
    { id: 'numpy', label: 'NumPy' },
    { id: 'tfjs', label: 'TF.js' },
];

export const CodeExportPanel = memo(function CodeExportPanel() {
    const [activeTab, setActiveTab] = useState<CodeTab>('pseudocode');
    const [copied, setCopied] = useState(false);

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
