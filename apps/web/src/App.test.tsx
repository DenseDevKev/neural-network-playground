import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { axe } from 'jest-axe';
import App from './App';
import { useTrainingStore } from './store/useTrainingStore.ts';

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

vi.mock('./components/layout/Sidebar.tsx', () => ({
    Sidebar: () => <aside>Sidebar</aside>,
}));

vi.mock('./components/layout/MainArea.tsx', () => ({
    MainArea: () => <main id="main-content" tabIndex={-1}>Main content</main>,
}));

describe('App accessibility shell', () => {
    beforeEach(() => {
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
        });

        Object.defineProperty(window, 'innerWidth', {
            writable: true,
            configurable: true,
            value: 1200,
        });
    });

    it('renders a skip link to the main content', () => {
        render(<App />);

        expect(screen.getByRole('link', { name: 'Skip to main content' })).toHaveAttribute('href', '#main-content');
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

    it('tracks responsive viewport transitions across breakpoints', () => {
        const { container } = render(<App />);

        const shell = container.querySelector('.app-shell');
        expect(shell).toHaveAttribute('data-viewport', 'wide');

        act(() => {
            window.innerWidth = 1000;
            window.dispatchEvent(new Event('resize'));
        });
        expect(shell).toHaveAttribute('data-viewport', 'desktop');

        act(() => {
            window.innerWidth = 700;
            window.dispatchEvent(new Event('resize'));
        });
        expect(shell).toHaveAttribute('data-viewport', 'tablet');

        act(() => {
            window.innerWidth = 320;
            window.dispatchEvent(new Event('resize'));
        });
        expect(shell).toHaveAttribute('data-viewport', 'mobile');
    });
});
