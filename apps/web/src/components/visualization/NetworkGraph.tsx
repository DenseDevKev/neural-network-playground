// ── NetworkGraph (renderer switcher, AS-5) ─────────────────────────────────
// Public entry point used by the rest of the app. Picks between the
// performance-optimized canvas renderer and the legacy SVG renderer based
// on the `featuresUI.canvasNetworkGraph` flag in the playground store.
//
// Rationale: keeps the SVG implementation around as a documented fallback
// for one release, so a long-tail bug in the canvas path (e.g. a hover-
// region miscompute on a niche browser) can be flipped off in the UI
// without redeploying.

import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { NetworkGraphCanvas } from './NetworkGraphCanvas.tsx';
import { NetworkGraphSVG } from './NetworkGraphSVG.tsx';

export function NetworkGraph() {
    const useCanvas = usePlaygroundStore((s) => s.featuresUI.canvasNetworkGraph);
    return useCanvas ? <NetworkGraphCanvas /> : <NetworkGraphSVG />;
}
