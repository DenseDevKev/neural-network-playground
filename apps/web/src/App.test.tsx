import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { axe } from 'jest-axe';
import App from './App';
import { useTrainingStore } from './store/useTrainingStore.ts';
import { useLayoutStore } from './store/useLayoutStore.ts';

const trainingMock = {
    play: vi.fn(),
    pause: vi.fn(),
    step: vi.fn(),
    reset: vi.fn(),
};

vi.mock('./hooks/useTraining.ts', () => ({
    useTraining: () => trainingMock,
}));

vi.mock('./components/layout/Header.tsx', () => ({
    Header: () => <header>Header</header>,
}));

vi.mock('./components/layout/RegionShell.tsx', () => ({
    DockShell:  () => <section aria-label="Dock workspace">Workspace</section>,
    FocusShell: () => <section aria-label="Focus workspace">Workspace</section>,
    GridShell:  () => <section aria-label="Grid workspace">Workspace</section>,
    SplitShell: () => <section aria-label="Split workspace">Workspace</section>,
}));

vi.mock('./components/layout/MainArea.tsx', () => ({
    CanvasContent:   () => <div>Canvas</div>,
    BoundaryContent: () => <div>Boundary</div>,
    LossContent:     () => <div>Loss</div>,
    ConfusionContent:() => <div>Confusion</div>,
    InspectContent:  () => <div>Inspect</div>,
    CodeContent:     () => <div>Code</div>,
}));

vi.mock('./components/controls/TrainingControls.tsx', () => ({ TrainingControls: () => <div>Controls</div> }));
vi.mock('./components/visualization/NetworkGraph.tsx', () => ({ NetworkGraph: () => <div>Graph</div> }));
vi.mock('./components/controls/PresetPanel.tsx',       () => ({ PresetPanel: () => <div>Presets</div> }));
vi.mock('./components/controls/DataPanel.tsx',         () => ({ DataPanel: () => <div>Data</div> }));
vi.mock('./components/controls/FeaturesPanel.tsx',     () => ({ FeaturesPanel: () => <div>Features</div> }));
vi.mock('./components/controls/NetworkConfigPanel.tsx',() => ({ NetworkConfigPanel: () => <div>Network</div> }));
vi.mock('./components/controls/HyperparamPanel.tsx',   () => ({ HyperparamPanel: () => <div>Hyperparams</div> }));
vi.mock('./components/controls/ConfigPanel.tsx',       () => ({ ConfigPanel: () => <div>Config</div> }));
vi.mock('./components/controls/InspectionPanel.tsx',   () => ({ InspectionPanel: () => <div>Inspection</div> }));
vi.mock('./components/controls/CodeExportPanel.tsx',   () => ({ CodeExportPanel: () => <div>CodeExport</div> }));

describe('App accessibility shell', () => {
    beforeEach(() => {
        window.localStorage.clear();
        trainingMock.play.mockReset();
        trainingMock.pause.mockReset();
        trainingMock.step.mockReset();
        trainingMock.reset.mockReset();

        useTrainingStore.setState({
            status: 'idle',
            dataConfigLoading: false,
            networkConfigLoading: false,
            configError: null,
            configErrorSource: null,
            workerError: null,
            snapshot: null,
        });

        useLayoutStore.setState({
            layout: 'dock',
            phase: 'build',
            activeTabLeft: 'data',
            activeTabRight: 'boundary',
        });
    });

    it('renders a skip link to the main content', () => {
        render(<App />);
        expect(screen.getByRole('link', { name: 'Skip to main content' })).toHaveAttribute('href', '#main-content');
    });

    it('exposes the redesigned workspace as the main landmark', () => {
        render(<App />);
        expect(screen.getByRole('main', { name: 'Neural network playground workspace' })).toHaveAttribute('id', 'main-content');
    });

    it('has no obvious accessibility violations in the shell', async () => {
        const { container } = render(<App />);
        const results = await axe(container);
        expect(results.violations).toHaveLength(0);
    });

    it('shows a recovery overlay when the worker connection fails', () => {
        useTrainingStore.setState({ workerError: 'Worker channel closed unexpectedly.' });

        render(<App />);

        expect(screen.getByText('Worker connection lost')).toBeInTheDocument();
        expect(screen.getByText('Worker channel closed unexpectedly. Refresh the page to restart the playground.')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Refresh page' })).toBeInTheDocument();
    });

    it('handles global keyboard shortcuts for training controls', () => {
        render(<App />);

        fireEvent.keyDown(window, { code: 'Space' });
        fireEvent.keyDown(window, { code: 'ArrowRight' });
        fireEvent.keyDown(window, { code: 'KeyR' });

        expect(trainingMock.play).toHaveBeenCalledTimes(1);
        expect(trainingMock.step).toHaveBeenCalledTimes(1);
        expect(trainingMock.reset).toHaveBeenCalledTimes(1);
    });

    it('pauses training with Space when already running and ignores shortcuts in inputs', () => {
        useTrainingStore.setState({ status: 'running' });
        render(
            <>
                <App />
                <input aria-label="Typing field" />
            </>,
        );

        fireEvent.keyDown(window, { code: 'Space' });
        expect(trainingMock.pause).toHaveBeenCalledTimes(1);

        const input = screen.getByRole('textbox', { name: 'Typing field' });
        fireEvent.keyDown(input, { code: 'KeyR' });
        expect(trainingMock.reset).not.toHaveBeenCalled();
    });

    it('renders forge-shell with status bar', () => {
        const { container } = render(<App />);
        expect(container.querySelector('.forge-shell')).toBeTruthy();
        expect(screen.getByRole('status', { name: 'Status bar' })).toBeInTheDocument();
    });

    it('switches layout variants through the store', () => {
        render(<App />);

        act(() => {
            useLayoutStore.getState().setLayout('grid');
        });
        expect(useLayoutStore.getState().layout).toBe('grid');

        act(() => {
            useLayoutStore.getState().setLayout('focus');
        });
        expect(useLayoutStore.getState().layout).toBe('focus');

        act(() => {
            useLayoutStore.getState().setLayout('split');
        });
        expect(useLayoutStore.getState().layout).toBe('split');
    });
});
