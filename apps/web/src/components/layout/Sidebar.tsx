// ── Sidebar Component ──
import { lazy, memo, useState } from 'react';
import { PresetPanel } from '../controls/PresetPanel.tsx';
import { DataPanel } from '../controls/DataPanel.tsx';
import { FeaturesPanel } from '../controls/FeaturesPanel.tsx';
import { NetworkConfigPanel } from '../controls/NetworkConfigPanel.tsx';
import { CollapsiblePanel } from '../common/CollapsiblePanel.tsx';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { LoadingState } from '../common/LoadingState.tsx';

interface SidebarProps {
    onReset: () => void;
}

const HyperparamPanel = lazy(() =>
    import('../controls/HyperparamPanel.tsx').then((module) => ({ default: module.HyperparamPanel })),
);

const ConfigPanel = lazy(() =>
    import('../controls/ConfigPanel.tsx').then((module) => ({ default: module.ConfigPanel })),
);

function InlinePanelFallback({ message }: { message: string }) {
    return <LoadingState isLoading inline message={message} />;
}

export const Sidebar = memo(function Sidebar({ onReset }: SidebarProps) {
    const hiddenLayers = usePlaygroundStore((s) => s.network.hiddenLayers);
    const [isSidebarExpanded, setIsSidebarExpanded] = useState(true);

    return (
        <aside className={`sidebar ${isSidebarExpanded ? '' : 'sidebar--collapsed'}`} role="complementary" aria-label="Configuration">
            <div className="sidebar__toggle-container">
                <button
                    className="sidebar__toggle btn btn--icon"
                    onClick={() => setIsSidebarExpanded(!isSidebarExpanded)}
                    aria-label={isSidebarExpanded ? 'Collapse sidebar' : 'Expand sidebar'}
                >
                    {isSidebarExpanded ? '←' : '→'}
                </button>
            </div>
            <div className="sidebar__content" style={{ display: isSidebarExpanded ? 'block' : 'none' }}>

            <CollapsiblePanel title="Presets" className="preset-panel">
                <PresetPanel onReset={onReset} />
            </CollapsiblePanel>
            <CollapsiblePanel title="Data">
                <DataPanel onReset={onReset} />
            </CollapsiblePanel>
            <CollapsiblePanel title="Features">
                <FeaturesPanel />
            </CollapsiblePanel>
            <CollapsiblePanel title="Network" badge={hiddenLayers.length}>
                <NetworkConfigPanel />
            </CollapsiblePanel>
            <CollapsiblePanel
                title="Hyperparameters"
                defaultExpanded={false}
                lazyMount
                fallback={<InlinePanelFallback message="Loading hyperparameters..." />}
            >
                <HyperparamPanel />
            </CollapsiblePanel>
            <CollapsiblePanel
                title="Config"
                defaultExpanded={false}
                lazyMount
                fallback={<InlinePanelFallback message="Loading config tools..." />}
            >
                <ConfigPanel onReset={onReset} />
            </CollapsiblePanel>

            </div>
        </aside>
    );
});
