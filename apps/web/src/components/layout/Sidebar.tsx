// ── Sidebar Component ──
import { memo, useState } from 'react';
import { PresetPanel } from '../controls/PresetPanel.tsx';
import { DataPanel } from '../controls/DataPanel.tsx';
import { FeaturesPanel } from '../controls/FeaturesPanel.tsx';
import { NetworkConfigPanel } from '../controls/NetworkConfigPanel.tsx';
import { HyperparamPanel } from '../controls/HyperparamPanel.tsx';
import { ConfigPanel } from '../controls/ConfigPanel.tsx';
import { CollapsiblePanel } from '../common/CollapsiblePanel.tsx';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';

interface SidebarProps {
    onReset: () => void;
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
            <CollapsiblePanel title="Hyperparameters" defaultExpanded={false}>
                <HyperparamPanel />
            </CollapsiblePanel>
            <CollapsiblePanel title="Config" defaultExpanded={false}>
                <ConfigPanel onReset={onReset} />
            </CollapsiblePanel>

            </div>
        </aside>
    );
});
