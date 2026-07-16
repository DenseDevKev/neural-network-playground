import { memo, useEffect, useRef, type KeyboardEvent, type ReactNode } from 'react';
import type { EvidenceViewId, WorkspaceView } from '../../store/useLayoutStore.ts';
import type { AudienceMode, ShellEvidenceViewId } from '../../productShell/audienceProfiles.ts';
import {
    ADVANCED_TOOLS_REGION_ID,
    type DrawerSurfaceId,
} from '../../productShell/shellTypes.ts';
import {
    getVisibleBuildModules,
    getVisibleEvidenceViews,
    resolveVisibleEvidenceView,
} from '../../productShell/visibleShell.ts';
import type { TrainingStatus } from '@nn-playground/shared';

const EVIDENCE_TABS: readonly { id: ShellEvidenceViewId; label: string }[] = [
    { id: 'boundary', label: 'Boundary' },
    { id: 'loss', label: 'Loss' },
    { id: 'confusion', label: 'Confusion' },
    { id: 'inspection', label: 'Inspect' },
    { id: 'code', label: 'Code' },
];

const SURFACE_LABELS: Record<DrawerSurfaceId, string> = {
    presets: 'Presets',
    lessons: 'Lessons',
    history: 'History',
};

interface BuildRunShellProps {
    view: WorkspaceView;
    status: TrainingStatus;
    activeEvidenceView: EvidenceViewId;
    audienceMode: AudienceMode;
    advancedToolsOpen: boolean;
    onSelectEvidence: (view: EvidenceViewId) => void;
    openSurface: DrawerSurfaceId | null;
    onCloseSurface: () => void;

    recipeContent: ReactNode;
    runContent: ReactNode;
    dataContent: ReactNode;
    networkContent: ReactNode;
    featuresContent: ReactNode;
    hyperparamContent: ReactNode;
    configurationContent: ReactNode;
    topologyContent: ReactNode;
    transportContent: ReactNode;
    evidenceContent: Record<EvidenceViewId, ReactNode>;
    presetContent: ReactNode;
    lessonContent: ReactNode;
    historyContent: ReactNode;
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
    surface: DrawerSurfaceId | null;
    onClose: () => void;
    children: ReactNode;
}) {
    const closeButtonRef = useRef<HTMLButtonElement>(null);

    useEffect(() => {
        if (surface) closeButtonRef.current?.focus();
    }, [surface]);

    if (!surface) return null;
    return (
        <aside
            className={`forge-instrument-drawer forge-instrument-drawer--${surface}`}
            id={`forge-surface-${surface}`}
            role="dialog"
            aria-modal="false"
            aria-label={SURFACE_LABELS[surface]}
        >
            <div className="forge-instrument-drawer__head">
                <span className="forge-instrument-drawer__title">{SURFACE_LABELS[surface]}</span>
                <button
                    type="button"
                    className="forge-instrument-drawer__close"
                    ref={closeButtonRef}
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
    status,
    activeEvidenceView,
    audienceMode,
    advancedToolsOpen,
    onSelectEvidence,
    openSurface,
    onCloseSurface,
    recipeContent,
    runContent,
    dataContent,
    networkContent,
    featuresContent,
    hyperparamContent,
    configurationContent,
    topologyContent,
    transportContent,
    evidenceContent,
    presetContent,
    lessonContent,
    historyContent,
}: BuildRunShellProps) {
    const visibleEvidenceView = resolveVisibleEvidenceView(
        audienceMode,
        advancedToolsOpen,
        activeEvidenceView,
    );
    const visibleBuildModules = getVisibleBuildModules(audienceMode, advancedToolsOpen);
    const visibleEvidenceViews = getVisibleEvidenceViews(audienceMode, advancedToolsOpen);
    const visibleEvidenceTabs = EVIDENCE_TABS.filter((tab) => visibleEvidenceViews.includes(tab.id));
    const hasRightBuildModules = visibleBuildModules.includes('features')
        || visibleBuildModules.includes('hyperparams')
        || visibleBuildModules.includes('config');
    const activeDrawerContent = openSurface === 'presets'
        ? presetContent
        : openSurface === 'lessons'
            ? lessonContent
            : openSurface === 'history'
                ? historyContent
                : null;

    const selectAndFocusEvidence = (view: ShellEvidenceViewId) => {
        onSelectEvidence(view);
        document.getElementById(`forge-right-tab-${view}`)?.focus();
    };

    const handleEvidenceKeyDown = (
        event: KeyboardEvent<HTMLButtonElement>,
        currentIndex: number,
    ) => {
        let nextIndex: number | null = null;
        if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
            nextIndex = (currentIndex + 1) % visibleEvidenceTabs.length;
        } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
            nextIndex = (currentIndex - 1 + visibleEvidenceTabs.length) % visibleEvidenceTabs.length;
        } else if (event.key === 'Home') {
            nextIndex = 0;
        } else if (event.key === 'End') {
            nextIndex = visibleEvidenceTabs.length - 1;
        }

        if (nextIndex === null) return;
        event.preventDefault();
        selectAndFocusEvidence(visibleEvidenceTabs[nextIndex].id);
    };

    return (
        <div className={`forge-buildrun forge-buildrun--${view}`}>
            <section
                className={`forge-buildrun__workspace-tools ${advancedToolsOpen ? 'forge-buildrun__workspace-tools--advanced' : ''}`}
                id={ADVANCED_TOOLS_REGION_ID}
                aria-label="Workspace tools"
            >
                {advancedToolsOpen && (
                    <p className="forge-buildrun__advanced-note" role="note">
                        Advanced Tools are visible. Configuration and diagnostic views are available without changing the experiment.
                    </p>
                )}

                {view === 'build' ? (
                    <div className={`forge-buildrun__grid forge-buildrun__grid--build ${hasRightBuildModules ? '' : 'forge-buildrun__grid--build-core'}`}>
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

                        {hasRightBuildModules && (
                            <div className="forge-buildrun__right">
                                {visibleBuildModules.includes('features') && (
                                    <InstrumentModule title="Features" phase="build" targets="features">
                                        {featuresContent}
                                    </InstrumentModule>
                                )}
                                {visibleBuildModules.includes('hyperparams') && (
                                    <InstrumentModule title="Hyperparameters" phase="build" targets="hyperparams">
                                        {hyperparamContent}
                                    </InstrumentModule>
                                )}
                                {visibleBuildModules.includes('config') && (
                                    <InstrumentModule title="Configuration" phase="build" targets="config">
                                        {configurationContent}
                                    </InstrumentModule>
                                )}
                            </div>
                        )}
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
                                {visibleEvidenceTabs.map((tab, index) => (
                                    <button
                                        key={tab.id}
                                        type="button"
                                        role="tab"
                                        id={`forge-right-tab-${tab.id}`}
                                        aria-selected={visibleEvidenceView === tab.id}
                                        aria-controls={`forge-right-panel-${tab.id}`}
                                        tabIndex={visibleEvidenceView === tab.id ? 0 : -1}
                                        className={`forge-evidence-tab ${visibleEvidenceView === tab.id ? 'forge-evidence-tab--active' : ''}`}
                                        onClick={() => onSelectEvidence(tab.id)}
                                        onKeyDown={(event) => handleEvidenceKeyDown(event, index)}
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
            </section>

            <div className="forge-buildrun__transport" data-status={status}>
                {transportContent}
            </div>

            <DrawerSurface surface={openSurface} onClose={onCloseSurface}>
                {activeDrawerContent}
            </DrawerSurface>
        </div>
    );
});
