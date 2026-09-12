import { NetworkGraphCanvasView } from './NetworkGraphCanvas.tsx';
import { useNetworkSelectionController, type NetworkSelectionController } from './useNetworkSelectionController.ts';

/** Same layout, viewport and accepted data as Canvas; SVG paints the topology. */
export function NetworkGraphSVG({ controller }: { readonly controller?: NetworkSelectionController }) {
    return controller ? <NetworkGraphCanvasView controller={controller} renderer="svg" /> : <LocalSVG />;
}
function LocalSVG() {
    const controller = useNetworkSelectionController();
    return <NetworkGraphCanvasView controller={controller} renderer="svg" />;
}
