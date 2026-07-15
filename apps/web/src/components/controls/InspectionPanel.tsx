// ── Advanced Inspection Panel ──
// Displays per-layer gradient magnitudes, activation stats, and weight distributions.

import { memo } from 'react';
import { InspectionPanelView } from './inspection/InspectionPanelView.tsx';
import { useInspectionPanelController } from './inspection/useInspectionPanelController.ts';

export const InspectionPanel = memo(function InspectionPanel() {
    const controller = useInspectionPanelController();
    return <InspectionPanelView {...controller} />;
});
