// ── RegionShell ── three layout variants driven by useLayoutStore
// Dock: icon-rail + left-tabs + canvas + right-tabs + bottom-transport
// Grid: 12-col × 8-row freeform tiles with named regions
// Split: build ↔ run mode columns

import { memo, type ReactNode } from 'react';
import { useLayoutStore } from '../../store/useLayoutStore.ts';

// ─── Tab definitions ───────────────────────────────────────────────────────
interface TabDef {
    id: string;
    label: string;
}

export const LEFT_TABS: TabDef[] = [
    { id: 'presets',     label: 'Presets' },
    { id: 'data',        label: 'Data' },
    { id: 'features',    label: 'Features' },
    { id: 'network',     label: 'Network' },
    { id: 'hyperparams', label: 'Hyperparams' },
    { id: 'config',      label: 'Config' },
];

export const RIGHT_TABS: TabDef[] = [
    { id: 'boundary',   label: 'Boundary' },
    { id: 'loss',       label: 'Loss' },
    { id: 'confusion',  label: 'Confusion' },
    { id: 'inspection', label: 'Inspection' },
    { id: 'code',       label: 'Code' },
];

// ─── Tab strip ────────────────────────────────────────────────────────────
interface TabStripProps {
    tabs: TabDef[];
    active: string;
    onSelect: (id: string) => void;
}

function TabStrip({ tabs, active, onSelect }: TabStripProps) {
    return (
        <div className="forge-tabs" role="tablist">
            {tabs.map((t) => (
                <button
                    key={t.id}
                    role="tab"
                    aria-selected={active === t.id}
                    className={`forge-tab ${active === t.id ? 'forge-tab--active' : ''}`}
                    onClick={() => onSelect(t.id)}
                >
                    <span className="forge-tab__dot" aria-hidden />
                    {t.label}
                </button>
            ))}
        </div>
    );
}

// ─── DOCK ─────────────────────────────────────────────────────────────────
interface DockProps {
    leftTabContent: Record<string, ReactNode>;
    rightTabContent: Record<string, ReactNode>;
    canvasContent: ReactNode;
    transportContent: ReactNode;
}

const RAIL_ICONS = [
    { id: 'data',        icon: '◉', label: 'Data' },
    { id: 'network',     icon: '⬢', label: 'Network' },
    { id: 'hyperparams', icon: 'λ', label: 'Hyperparams' },
    { id: 'config',      icon: '⚙', label: 'Config' },
] as const;

export const DockShell = memo(function DockShell({
    leftTabContent, rightTabContent, canvasContent, transportContent,
}: DockProps) {
    const activeTabLeft  = useLayoutStore((s) => s.activeTabLeft);
    const activeTabRight = useLayoutStore((s) => s.activeTabRight);
    const setTabLeft  = useLayoutStore((s) => s.setActiveTabLeft);
    const setTabRight = useLayoutStore((s) => s.setActiveTabRight);

    return (
        <div className="forge-dock">
            {/* Icon rail */}
            <nav className="forge-rail" aria-label="Navigation rail">
                {RAIL_ICONS.map((r) => (
                    <button
                        key={r.id}
                        className={`forge-rail__btn ${activeTabLeft === r.id ? 'forge-rail__btn--active' : ''}`}
                        title={r.label}
                        aria-label={r.label}
                        onClick={() => setTabLeft(r.id)}
                    >
                        {r.icon}
                    </button>
                ))}
                <span className="forge-rail__spacer" />
                <button
                    className={`forge-rail__btn ${activeTabLeft === 'presets' ? 'forge-rail__btn--active' : ''}`}
                    title="Presets"
                    aria-label="Presets"
                    onClick={() => setTabLeft('presets')}
                >
                    ★
                </button>
            </nav>

            {/* Left panel */}
            <div className="forge-dock__left">
                <TabStrip tabs={LEFT_TABS} active={activeTabLeft} onSelect={(id) => setTabLeft(id as any)} />
                <div className="forge-tabs__tray">
                    <div className="forge-tabs__content">
                        {leftTabContent[activeTabLeft] ?? null}
                    </div>
                </div>
            </div>

            {/* Canvas */}
            <div className="forge-dock__center">
                {canvasContent}
            </div>

            {/* Right panel */}
            <div className="forge-dock__right">
                <TabStrip tabs={RIGHT_TABS} active={activeTabRight} onSelect={(id) => setTabRight(id as any)} />
                <div className="forge-tabs__tray">
                    <div className="forge-tabs__content">
                        {rightTabContent[activeTabRight] ?? null}
                    </div>
                </div>
            </div>

            {/* Bottom transport */}
            <div className="forge-dock__bottom">
                {transportContent}
            </div>
        </div>
    );
});

// ─── GRID ─────────────────────────────────────────────────────────────────
interface GridProps {
    topologyContent: ReactNode;
    boundaryContent: ReactNode;
    configContent: ReactNode;
    lossContent: ReactNode;
    confusionContent: ReactNode;
    inspectContent: ReactNode;
    transportContent: ReactNode;
}

export const GridShell = memo(function GridShell({
    topologyContent, boundaryContent, configContent, lossContent,
    confusionContent, inspectContent, transportContent,
}: GridProps) {
    return (
        <div className="forge-grid">
            <div className="forge-grid__topology">{topologyContent}</div>
            <div className="forge-grid__boundary">{boundaryContent}</div>
            <div className="forge-grid__config">{configContent}</div>
            <div className="forge-grid__loss">{lossContent}</div>
            <div className="forge-grid__confusion">{confusionContent}</div>
            <div className="forge-grid__inspect">{inspectContent}</div>
            <div className="forge-grid__transport">
                <div className="forge-panel" style={{ flexDirection: 'row', height: '100%' }}>
                    <div style={{ padding: '0 16px', display: 'flex', alignItems: 'center', gap: 8, width: '100%' }}>
                        {transportContent}
                    </div>
                </div>
            </div>
        </div>
    );
});

// ─── SPLIT ────────────────────────────────────────────────────────────────
interface SplitProps {
    buildLeft: ReactNode;
    buildCenter: ReactNode;
    buildRight: ReactNode;
    runLeft: ReactNode;
    runCenter: ReactNode;
    runRight: ReactNode;
    transportContent: ReactNode;
}

export const SplitShell = memo(function SplitShell({
    buildLeft, buildCenter, buildRight,
    runLeft, runCenter, runRight,
    transportContent,
}: SplitProps) {
    const phase    = useLayoutStore((s) => s.phase);
    const isBuild  = phase === 'build';
    return (
        <div className="forge-split">
            <div className="forge-split__modebar">
                <span className={`forge-split__indicator forge-split__indicator--${phase}`}>
                    {isBuild ? '◉ Build phase — design your network' : '▶ Run phase — observe training'}
                </span>
                <span className="forge-split__spacer" />
                <span className="forge-split__hint">
                    {isBuild ? 'Press ▶ to start training →' : '← Switch to Build to edit architecture'}
                </span>
            </div>
            <div className="forge-split__cols">
                <div className="forge-split__col">{isBuild ? buildLeft  : runLeft}</div>
                <div className="forge-split__col">{isBuild ? buildCenter : runCenter}</div>
                <div className="forge-split__col">{isBuild ? buildRight : runRight}</div>
            </div>
            <div className="forge-split__transport">
                {transportContent}
            </div>
        </div>
    );
});
