import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import { E2E_FAULTS_ENABLED } from './testing/e2eFaults.ts';
import './styles/fonts.css';
import './styles/index.css';
import './styles/forge.css';
import './styles/precisionLab.css';

// E2E specs probe this marker to skip fault-injection scenarios when the
// served bundle was built without VITE_E2E_FAULTS=1.
if (E2E_FAULTS_ENABLED) {
    (window as typeof window & { __nnpE2EFaultsEnabled?: boolean })
        .__nnpE2EFaultsEnabled = true;
}

ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <App />
    </React.StrictMode>,
);
