import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { NetworkGraphFrame } from './NetworkGraphFrame.tsx';

describe('NetworkGraphFrame scrolling summary', () => {
    it('is a named keyboard focus stop without hiding architecture evidence', () => {
        const command = vi.fn();
        render(<NetworkGraphFrame story="Two hidden layers" capacity="17 parameters"
            datasetHint="XOR" healthHint="Gradients available" lesson="Inspect the neurons"
            zoom={1} viewMode="weights" edgeFilter="all"
            onZoomOut={command} onZoomIn={command} onFit={command}
            onViewMode={command} onEdgeFilter={command}>
            <div>Plot</div>
        </NetworkGraphFrame>);
        const summary = screen.getByRole('region', { name: 'Architecture summary' });
        expect(summary).toHaveAttribute('tabindex', '0');
        for (const copy of ['Two hidden layers', '17 parameters', 'XOR', 'Gradients available', 'Inspect the neurons']) {
            expect(summary).toHaveTextContent(copy);
        }
        summary.focus();
        expect(summary).toHaveFocus();
        expect(command).not.toHaveBeenCalled();
    });
});
