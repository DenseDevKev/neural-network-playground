import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { render } from '@testing-library/react';
import { InspectionPanel } from './InspectionPanel.tsx';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import {
    DEFAULT_DATA,
    DEFAULT_DEMAND,
    DEFAULT_FEATURES,
    DEFAULT_NETWORK,
    DEFAULT_TRAINING,
} from '@nn-playground/shared';

describe('InspectionPanel demand', () => {
    beforeEach(() => {
        usePlaygroundStore.setState({
            data: { ...DEFAULT_DATA },
            network: { ...DEFAULT_NETWORK, inputSize: 2, outputSize: 1, seed: DEFAULT_DATA.seed },
            features: { ...DEFAULT_FEATURES },
            training: { ...DEFAULT_TRAINING },
            ui: { showTestData: false, discretizeOutput: false },
            demand: { ...DEFAULT_DEMAND, needLayerStats: false },
        });
        useTrainingStore.setState({
            snapshot: null,
            frameVersion: 0,
        });
    });

    afterEach(() => {
        usePlaygroundStore.setState({
            demand: { ...DEFAULT_DEMAND, needLayerStats: false },
        });
    });

    it('requests layer stats only while inspection is mounted', () => {
        const { unmount } = render(<InspectionPanel />);

        expect(usePlaygroundStore.getState().demand.needLayerStats).toBe(true);

        unmount();

        expect(usePlaygroundStore.getState().demand.needLayerStats).toBe(false);
    });
});
