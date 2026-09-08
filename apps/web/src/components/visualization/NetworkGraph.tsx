import { memo } from 'react';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { NetworkGraphCanvas } from './NetworkGraphCanvas.tsx';
import { NetworkGraphSVG } from './NetworkGraphSVG.tsx';
import { useNetworkSelectionController, type NetworkSelectionController } from './useNetworkSelectionController.ts';

export interface NetworkGraphProps { readonly selectionController?: NetworkSelectionController }
export interface NetworkGraphRendererProps { readonly controller: NetworkSelectionController }

function NetworkGraphRenderer({ controller }: NetworkGraphRendererProps) {
    const useCanvas = usePlaygroundStore((s) => s.featuresUI.canvasNetworkGraph);
    return useCanvas ? <NetworkGraphCanvas controller={controller} /> : <NetworkGraphSVG controller={controller} />;
}
function NetworkGraphWithLocalController() {
    const controller = useNetworkSelectionController();
    return <NetworkGraphRenderer controller={controller} />;
}
export const NetworkGraph = memo(function NetworkGraph({ selectionController }: NetworkGraphProps) {
    return selectionController ? <NetworkGraphRenderer controller={selectionController} /> : <NetworkGraphWithLocalController />;
});
