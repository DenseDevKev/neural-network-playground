import { memo, type ReactNode } from 'react';
import type { EvidenceViewId, WorkspaceView } from '../../store/useLayoutStore.ts';

type SurfaceId = 'presets' | 'lessons' | 'history' | 'more';

const EVIDENCE_TABS: readonly { id: EvidenceViewId; label: string }[] = [
    { id: 'boundary', label: 'Boundary' },
    { id: 'loss', label: 'Loss' },
    { id: 'confusion', label: 'Confusion' },
    { id: 'inspection', label: 'Inspect' },
    { id: 'code', label: 'Code' },
];

const SURFACE_LABELS: Record<SurfaceId, string> = {
    presets: 'Presets',
    lessons: 'Lessons',
    history: 'History',
    more: 'More / Commands',
};

interface BuildRunShellProps {
    view: WorkspaceView;
    activeEvidenceView: EvidenceViewId;
    onSelectEvidence: (view: EvidenceViewId) => void;
    openSurface: SurfaceId | null;
    onCloseSurface: () => void;

    recipeContent: ReactNode;
    runContent: ReactNode;
    dataContent: ReactNode;
    networkContent: ReactNode;
    featuresContent: ReactNode;
    hyperparamContent: ReactNode;
    topologyContent: ReactNode;
    transportContent: ReactNode;
    evidenceContent: Record<EvidenceViewId, ReactNode>;
    presetContent: ReactNode;
    lessonContent: ReactNode;
    historyContent: ReactNode;
    moreContent: ReactNode;
}

function InstrumentModule({
    title,
    phase,
    targets,
    fill = false,
    children,
}: {
    title: string;
    phase?: WorkspaceView | 'both';
    targets?: string;
    fill?: boolean;
    children: ReactNode;
}) {
    return (
        <section
            className={`forge-instrument-module ${fill ? 'forge-instrument-module--fill' : ''}`}
            data-forge-panel-targets={targets}
            tabIndex={-1}
            aria-label={title}
        >
            <div className="forge-instrument-module__head">
                <span className="forge-instrument-module__grip" aria-hidden />
                <span className="forge-instrument-module__title">{title}</span>
                {phase && phase !== 'both' && (
                    <span className={`forge-instrument-module__tag forge-instrument-module__tag--${phase}`}>
                        {phase}
                    </span>
                )}
            </div>
            <div className="forge-instrument-module__body">
                {children}
            </div>
        </section>
    );
}

function DrawerSurface({
    surface,
    onClose,
    children,
}: {
    surface: SurfaceId | null;
    onClose: () => void;
    children: ReactNode;
}) {
    if (!surface) return null;
    return (
        <aside
            className={`forge-instrument-drawer forge-instrument-drawer--${surface}`}
            role="dialog"
            aria-modal="false"
            aria-label={SURFACE_LABELS[surface]}
        >
            <div className="forge-instrument-drawer__head">
                <span className="forge-instrument-drawer__title">{SURFACE_LABELS[surface]}</span>
                <button
                    type="button"
                    className="forge-instrument-drawer__close"
                    onClick={onClose}
                    aria-label={`Close ${SURFACE_LABELS[surface]}`}
                >
                    ×
                </button>
            </div>
            <div className="forge-instrument-drawer__body">
                {children}
            </div>
        </aside>
    );
}

export const BuildRunShell = memo(function BuildRunShell({
    view,
    activeEvidenceView,
    onSelectEvidence,
    openSurface,
    onCloseSurface,
    recipeContent,
    runContent,
    dataContent,
    networkContent,
    featuresContent,
    hyperparamContent,
    topologyContent,
    transportContent,
    evidenceContent,
    presetContent,
    lessonContent,
    historyContent,
    moreContent,
}: BuildRunShellProps) {
    const visibleEvidenceView = activeEvidenceView === 'history' ? 'boundary' : activeEvidenceView;
    const activeDrawerContent = openSurface === 'presets'
        ? presetContent
        : openSurface === 'lessons'
            ? lessonContent
            : openSurface === 'history'
                ? historyContent
                : openSurface === 'more'
                    ? moreContent
                    : null;

    return (
        <div className={`forge-buildrun forge-buildrun--${view}`}>
            {view === 'build' ? (
                <div className="forge-buildrun__grid forge-buildrun__grid--build">
                    <div className="forge-buildrun__left">
                        <InstrumentModule title="Current Recipe" phase="build" targets="experiment">
                            {recipeContent}
                        </InstrumentModule>
                        <InstrumentModule title="Data" phase="build" targets="data">
                            {dataContent}
                        </InstrumentModule>
                    </div>

                    <div className="forge-buildrun__center">
                        <InstrumentModule title="Network Topology" phase="build" targets="topology" fill>
                            {topologyContent}
                        </InstrumentModule>
                        <InstrumentModule title="Network" phase="build" targets="network">
                            {networkContent}
                        </InstrumentModule>
                    </div>

                    <div className="forge-buildrun__right">
                        <InstrumentModule title="Features" phase="build" targets="features">
                            {featuresContent}
                        </InstrumentModule>
                        <InstrumentModule title="Hyperparameters" phase="build" targets="hyperparams">
                            {hyperparamContent}
                        </InstrumentModule>
                    </div>
                </div>
            ) : (
                <div className="forge-buildrun__grid forge-buildrun__grid--run">
                    <div className="forge-buildrun__left">
                        <InstrumentModule title="Current Run" phase="run" targets="run">
                            {runContent}
                        </InstrumentModule>
                        <InstrumentModule title="Recipe" phase="both" targets="experiment">
                            {recipeContent}
                        </InstrumentModule>
                    </div>

                    <div className="forge-buildrun__center">
                        <InstrumentModule title="Topology + Training State" phase="run" targets="topology" fill>
                            {topologyContent}
                        </InstrumentModule>
                    </div>

                    <div className="forge-buildrun__evidence">
                        <div className="forge-evidence-tabs" role="tablist" aria-label="Evidence views">
                            {EVIDENCE_TABS.map((tab) => (
                                <button
                                    key={tab.id}
                                    type="button"
                                    role="tab"
                                    id={`forge-right-tab-${tab.id}`}
                                    aria-selected={visibleEvidenceView === tab.id}
                                    aria-controls={`forge-right-panel-${tab.id}`}
                                    className={`forge-evidence-tab ${visibleEvidenceView === tab.id ? 'forge-evidence-tab--active' : ''}`}
                                    onClick={() => onSelectEvidence(tab.id)}
                                >
                                    {tab.label}
                                </button>
                            ))}
                        </div>
                        <div
                            className="forge-buildrun__evidence-body"
                            role="tabpanel"
                            id={`forge-right-panel-${visibleEvidenceView}`}
                            aria-labelledby={`forge-right-tab-${visibleEvidenceView}`}
                            data-forge-panel-targets={visibleEvidenceView}
                            tabIndex={-1}
                        >
                            {evidenceContent[visibleEvidenceView]}
                        </div>
                    </div>
                </div>
            )}

            <div className="forge-buildrun__transport">
                {transportContent}
            </div>

            <DrawerSurface surface={openSurface} onClose={onCloseSurface}>
                {activeDrawerContent}
            </DrawerSurface>
        </div>
    );
});
