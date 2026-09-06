import {
    memo,
    useCallback,
    type KeyboardEvent,
    type ReactNode,
} from 'react';
import type { TrainingStatus } from '@nn-playground/shared';
import type {
    EvidenceViewId,
    RecipeSectionId,
    WorkspaceView,
} from '../../../store/useLayoutStore.ts';
import type {
    AudienceMode,
    BuildModuleId,
    ShellEvidenceViewId,
} from '../../../productShell/audienceProfiles.ts';
import type { DrawerSurfaceId } from '../../../productShell/shellTypes.ts';
import {
    getVisibleBuildModules,
    getVisibleEvidenceViews,
    resolveVisibleEvidenceView,
} from '../../../productShell/visibleShell.ts';

const BUILD_LABELS: Readonly<Record<BuildModuleId, string>> = Object.freeze({
    data: 'Data',
    network: 'Network',
    features: 'Features',
    hyperparams: 'Hyperparameters',
    config: 'Configuration',
});

const EVIDENCE_LABELS: Readonly<Record<ShellEvidenceViewId, string>> = Object.freeze({
    boundary: 'Boundary',
    loss: 'Loss',
    confusion: 'Confusion',
    inspection: 'Inspect',
    code: 'Code',
});

const SURFACE_LABELS: Readonly<Record<DrawerSurfaceId, string>> = Object.freeze({
    presets: 'Presets',
    lessons: 'Lessons',
    history: 'History',
});

export interface PrecisionLabShellProps {
    readonly view: WorkspaceView;
    readonly status: TrainingStatus;
    readonly activeRecipeSection: RecipeSectionId;
    readonly activeEvidenceView: EvidenceViewId;
    readonly audienceMode: AudienceMode;
    readonly advancedToolsOpen: boolean;
    readonly buildContextOpen: boolean;
    readonly openSurface: DrawerSurfaceId | null;
    readonly onSelectRecipeSection: (section: RecipeSectionId) => void;
    readonly onCloseRecipeSection: () => void;
    readonly onSelectEvidence: (view: EvidenceViewId) => void;
    readonly onCloseSurface: () => void;
    readonly recipeStripContent: ReactNode;
    readonly runSummaryContent: ReactNode;
    readonly buildContent: Readonly<Record<BuildModuleId, ReactNode>>;
    readonly topologyContent: ReactNode;
    readonly boundaryRailContent: ReactNode;
    readonly selectionContent: ReactNode;
    readonly evidenceContent: Readonly<Record<ShellEvidenceViewId, ReactNode>>;
    readonly transportContent: ReactNode;
    readonly presetContent: ReactNode;
    readonly lessonContent: ReactNode;
    readonly historyContent: ReactNode;
}

function normalizedBuildSection(section: RecipeSectionId): BuildModuleId {
    return section === 'presets' ? 'data' : section;
}

export const PrecisionLabShell = memo(function PrecisionLabShell({
    view,
    status,
    activeRecipeSection,
    activeEvidenceView,
    audienceMode,
    advancedToolsOpen,
    buildContextOpen,
    openSurface,
    onSelectRecipeSection,
    onCloseRecipeSection,
    onSelectEvidence,
    onCloseSurface,
    recipeStripContent,
    runSummaryContent,
    buildContent,
    topologyContent,
    boundaryRailContent,
    selectionContent,
    evidenceContent,
    transportContent,
    presetContent,
    lessonContent,
    historyContent,
}: PrecisionLabShellProps) {
    const visibleBuildModules = getVisibleBuildModules(audienceMode, advancedToolsOpen);
    const visibleEvidenceViews = getVisibleEvidenceViews(audienceMode, advancedToolsOpen);
    const visibleEvidenceView = resolveVisibleEvidenceView(
        audienceMode,
        advancedToolsOpen,
        activeEvidenceView,
    );
    const selectedBuildSection = normalizedBuildSection(activeRecipeSection);

    const focusRailTrigger = useCallback((id: string) => {
        requestAnimationFrame(() => {
            document.getElementById(`precision-rail-${id}`)?.focus();
        });
    }, []);

    const closeBuildContext = useCallback(() => {
        const selected = selectedBuildSection;
        onCloseRecipeSection();
        focusRailTrigger(selected);
    }, [focusRailTrigger, onCloseRecipeSection, selectedBuildSection]);

    const selectAndFocusEvidence = useCallback((nextView: ShellEvidenceViewId) => {
        onSelectEvidence(nextView);
        document.getElementById(`precision-evidence-tab-${nextView}`)?.focus();
    }, [onSelectEvidence]);

    const handleEvidenceKeyDown = useCallback((
        event: KeyboardEvent<HTMLButtonElement>,
        currentIndex: number,
    ) => {
        if (visibleEvidenceViews.length === 0) return;
        let nextIndex: number | null = null;
        if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
            nextIndex = (currentIndex + 1) % visibleEvidenceViews.length;
        } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
            nextIndex = (currentIndex - 1 + visibleEvidenceViews.length) % visibleEvidenceViews.length;
        } else if (event.key === 'Home') {
            nextIndex = 0;
        } else if (event.key === 'End') {
            nextIndex = visibleEvidenceViews.length - 1;
        }
        if (nextIndex === null) return;
        event.preventDefault();
        selectAndFocusEvidence(visibleEvidenceViews[nextIndex]);
    }, [selectAndFocusEvidence, visibleEvidenceViews]);

    const railContent = view === 'build'
        ? visibleBuildModules.map((moduleId) => (
            <button
                key={moduleId}
                id={`precision-rail-${moduleId}`}
                type="button"
                aria-pressed={buildContextOpen && selectedBuildSection === moduleId}
                onClick={() => onSelectRecipeSection(moduleId)}
            >
                {BUILD_LABELS[moduleId]}
            </button>
        ))
        : visibleEvidenceViews.map((evidenceView) => (
            <button
                key={evidenceView}
                id={`precision-rail-${evidenceView}`}
                type="button"
                aria-pressed={visibleEvidenceView === evidenceView}
                onClick={() => onSelectEvidence(evidenceView)}
            >
                {EVIDENCE_LABELS[evidenceView]}
            </button>
        ));

    const evidenceTablist = (
        <div className="precision-evidence__tabs" role="tablist" aria-label="Evidence views">
            {visibleEvidenceViews.map((evidenceView, index) => (
                <button
                    key={evidenceView}
                    type="button"
                    role="tab"
                    id={`precision-evidence-tab-${evidenceView}`}
                    aria-selected={visibleEvidenceView === evidenceView}
                    aria-controls="precision-evidence-panel"
                    tabIndex={visibleEvidenceView === evidenceView ? 0 : -1}
                    onClick={() => onSelectEvidence(evidenceView)}
                    onKeyDown={(event) => handleEvidenceKeyDown(event, index)}
                >
                    {EVIDENCE_LABELS[evidenceView]}
                </button>
            ))}
        </div>
    );

    const buildContext = view === 'build' && buildContextOpen ? (
        <aside
            className="precision-context"
            role="region"
            aria-label={`${BUILD_LABELS[selectedBuildSection]} context`}
            tabIndex={-1}
            onKeyDown={(event) => {
                if (event.key !== 'Escape') return;
                event.preventDefault();
                event.stopPropagation();
                closeBuildContext();
            }}
        >
            <div className="precision-context__head">
                <span>{BUILD_LABELS[selectedBuildSection]}</span>
                <button
                    type="button"
                    aria-label={`Close ${BUILD_LABELS[selectedBuildSection]} context`}
                    onClick={closeBuildContext}
                >
                    ×
                </button>
            </div>
            <div className="precision-context__body">
                {buildContent[selectedBuildSection]}
            </div>
        </aside>
    ) : null;

    const activeDrawerContent = openSurface === 'presets'
        ? presetContent
        : openSurface === 'lessons'
            ? lessonContent
            : openSurface === 'history'
                ? historyContent
                : null;
    const drawer = openSurface === null ? null : (
        <aside
            className={`precision-drawer precision-drawer--${openSurface}`}
            id={`forge-surface-${openSurface}`}
            role="dialog"
            aria-modal="false"
            aria-label={SURFACE_LABELS[openSurface]}
        >
            <div className="precision-drawer__head">
                <span>{SURFACE_LABELS[openSurface]}</span>
                <button
                    type="button"
                    aria-label={`Close ${SURFACE_LABELS[openSurface]}`}
                    onClick={onCloseSurface}
                >
                    ×
                </button>
            </div>
            <div className="precision-drawer__body">{activeDrawerContent}</div>
        </aside>
    );

    return (
        <div
            className={`precision-shell precision-shell--${view}`}
            data-precision-workspace
        >
            <nav className="precision-rail" aria-label={view === 'build' ? 'Build tools' : 'Run tools'}>
                {railContent}
            </nav>
            {recipeStripContent}
            {view === 'run' && (
                <aside className="precision-run-summary" aria-label="Current run">
                    {runSummaryContent}
                </aside>
            )}
            <section
                className="precision-topology"
                aria-label="Neural network"
                data-precision-region="topology"
            >
                {topologyContent}
            </section>
            <aside
                className="precision-boundary"
                aria-label="Pinned decision boundary"
                data-precision-region="boundary"
            >
                {boundaryRailContent}
            </aside>
            <section className="precision-selection" aria-label="Neuron selection">
                {selectionContent}
            </section>
            <section
                className="precision-evidence"
                aria-label="Evidence"
                data-precision-region="evidence"
            >
                {evidenceTablist}
                <div
                    id="precision-evidence-panel"
                    role="tabpanel"
                    aria-labelledby={`precision-evidence-tab-${visibleEvidenceView}`}
                    tabIndex={-1}
                >
                    {evidenceContent[visibleEvidenceView]}
                </div>
            </section>
            <footer
                className="precision-transport"
                data-status={status}
                data-precision-region="transport"
            >
                {transportContent}
            </footer>
            {buildContext}
            {drawer}
        </div>
    );
});
