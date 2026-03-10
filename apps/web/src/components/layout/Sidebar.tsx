// ── Sidebar Component ──
import { PresetPanel } from '../controls/PresetPanel.tsx';
import { DataPanel } from '../controls/DataPanel.tsx';
import { FeaturesPanel } from '../controls/FeaturesPanel.tsx';
import { NetworkConfigPanel } from '../controls/NetworkConfigPanel.tsx';
import { HyperparamPanel } from '../controls/HyperparamPanel.tsx';
import { ConfigPanel } from '../controls/ConfigPanel.tsx';

interface SidebarProps {
    onReset: () => void;
}

export function Sidebar({ onReset }: SidebarProps) {
    return (
        <aside className="sidebar" role="complementary" aria-label="Configuration">
            <PresetPanel onReset={onReset} />
            <DataPanel onReset={onReset} />
            <FeaturesPanel />
            <NetworkConfigPanel />
            <HyperparamPanel />
            <ConfigPanel onReset={onReset} />
        </aside>
    );
}

