// ── Advanced Inspection Panel ──
// Displays per-layer gradient magnitudes, activation stats, and weight distributions.

import { memo } from 'react';
import { InspectionPanelView } from './inspection/InspectionPanelView.tsx';
import { useInspectionPanelController } from './inspection/useInspectionPanelController.ts';
import { useAudienceGuidanceLevel } from '../../hooks/useAudienceGuidanceLevel.ts';

export const InspectionPanel = memo(function InspectionPanel() {
    const controller = useInspectionPanelController();
    const guidanceLevel = useAudienceGuidanceLevel();
    return <InspectionPanelView {...controller} guidanceLevel={guidanceLevel} />;
});
