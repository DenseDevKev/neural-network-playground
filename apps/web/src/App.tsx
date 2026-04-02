// ── Root App Component ──
import { useEffect, useRef, useCallback } from 'react';
import { Header } from './components/layout/Header.tsx';
import { Sidebar } from './components/layout/Sidebar.tsx';
import { MainArea } from './components/layout/MainArea.tsx';
import { useTraining } from './hooks/useTraining.ts';
import { useTrainingStore } from './store/useTrainingStore.ts';

export default function App() {
    const training = useTraining();
    const status = useTrainingStore((s) => s.status);

    // Stable ref so the keydown handler never goes stale between renders.
    const trainingRef = useRef(training);
    const statusRef = useRef(status);
    useEffect(() => { trainingRef.current = training; }, [training]);
    useEffect(() => { statusRef.current = status; }, [status]);

    // Global keyboard shortcuts: Space=play/pause, →=step, R=reset
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            // Don't intercept when user is typing in a form element
            if (
                e.target instanceof HTMLInputElement ||
                e.target instanceof HTMLSelectElement ||
                e.target instanceof HTMLTextAreaElement
            ) return;

            if (e.code === 'Space') {
                e.preventDefault();
                if (statusRef.current === 'running') {
                    trainingRef.current.pause();
                } else {
                    trainingRef.current.play();
                }
            } else if (e.code === 'ArrowRight') {
                e.preventDefault();
                trainingRef.current.step();
            } else if (e.code === 'KeyR') {
                e.preventDefault();
                trainingRef.current.reset();
            }
        };

        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, []); // empty deps — handler reads always-current refs
    // Stable callback for Sidebar — delegates to the always-current ref
    // so React.memo on Sidebar actually prevents re-renders.
    const stableReset = useCallback(() => trainingRef.current.reset(), []);

    return (
        <div className="app-shell">
            <Header />
            <div className="main-layout">
                <Sidebar onReset={stableReset} />
                <MainArea training={training} />
            </div>
            <footer className="footer">
                Inspired by{' '}
                <a
                    href="https://playground.tensorflow.org"
                    target="_blank"
                    rel="noopener noreferrer"
                >
                    TensorFlow Playground
                </a>{' '}
                by Daniel Smilkov &amp; Shan Carter
            </footer>
        </div>
    );
}
