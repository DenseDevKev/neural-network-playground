// ── Advanced Inspection Panel ──
// Displays per-layer gradient magnitudes, activation stats, and weight distributions.

import { memo } from 'react';
import { InspectionPanelView } from './inspection/InspectionPanelView.tsx';
import { useInspectionPanelController } from './inspection/useInspectionPanelController.ts';
import { useAudienceGuidanceLevel } from '../../hooks/useAudienceGuidanceLevel.ts';

import { useLayoutStore } from '../../store/useLayoutStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';

export const InspectionPanel = memo(function InspectionPanel({ onPause }: { onPause?: () => void }) {
    const tab = useLayoutStore((state) => state.inspectTab);
    const onTabChange = useLayoutStore((state) => state.setInspectTab);
    const running = useTrainingStore((state) => state.status === 'running');
    const controller = useInspectionPanelController();
    const guidanceLevel = useAudienceGuidanceLevel();
    return <InspectionPanelView {...controller} guidanceLevel={guidanceLevel} tab={tab} onTabChange={onTabChange} running={running} onPause={onPause} />;
});
