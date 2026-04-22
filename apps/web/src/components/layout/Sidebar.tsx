// ── Sidebar ── left-column wrapper using Panel primitives
// Used in legacy/fallback layout paths. In Dock layout, RegionShell
// renders the left tab pane directly without this wrapper.

import { lazy, memo, Suspense } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { PresetPanel } from '../controls/PresetPanel.tsx';
import { DataPanel } from '../controls/DataPanel.tsx';
import { FeaturesPanel } from '../controls/FeaturesPanel.tsx';
import { NetworkConfigPanel } from '../controls/NetworkConfigPanel.tsx';
import { Panel } from '../common/Panel.tsx';
import { LoadingState } from '../common/LoadingState.tsx';

interface SidebarProps {
    onReset: () => void;
}

const HyperparamPanel = lazy(() =>
    import('../controls/HyperparamPanel.tsx').then((m) => ({ default: m.HyperparamPanel })),
);
const ConfigPanel = lazy(() =>
    import('../controls/ConfigPanel.tsx').then((m) => ({ default: m.ConfigPanel })),
);

function Fallback({ msg }: { msg: string }) {
    return <LoadingState isLoading inline message={msg} />;
}

export const Sidebar = memo(function Sidebar({ onReset }: SidebarProps) {
    const hiddenLayers = usePlaygroundStore((s) => s.network.hiddenLayers);

    return (
        <aside
            className="sidebar"
            role="complementary"
            aria-label="Configuration"
            style={{ display: 'flex', flexDirection: 'column', gap: 6, padding: 8, overflowY: 'auto' }}
        >
            <Panel title="Presets" phase="build">
                <PresetPanel onReset={onReset} />
            </Panel>

            <Panel title="Data" phase="build">
                <DataPanel onReset={onReset} />
            </Panel>

            <Panel title="Features" phase="build">
                <FeaturesPanel />
            </Panel>

            <Panel title={`Network${hiddenLayers.length ? ` (${hiddenLayers.length})` : ''}`} phase="build">
                <NetworkConfigPanel />
            </Panel>

            <Suspense fallback={<Fallback msg="Loading hyperparameters…" />}>
                <Panel title="Hyperparameters" phase="both">
                    <HyperparamPanel />
                </Panel>
            </Suspense>

            <Suspense fallback={<Fallback msg="Loading config…" />}>
                <Panel title="Config" phase="both">
                    <ConfigPanel onReset={onReset} />
                </Panel>
            </Suspense>
        </aside>
    );
});
