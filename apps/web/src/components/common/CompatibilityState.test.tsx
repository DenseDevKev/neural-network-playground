import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'jest-axe';
import { CompatibilityState } from './CompatibilityState.tsx';

describe('CompatibilityState', () => {
    it('keeps the exact URL source and structured issues visible until Start fresh', async () => {
        const onStartFresh = vi.fn().mockResolvedValue(undefined);
        render(<CompatibilityState access={{
            status: 'incompatible',
            prepared: null,
            source: { kind: 'url', rawHash: '#d=xor&n=0.2' },
            issues: [{
                code: 'legacy-state',
                path: 'schemaVersion',
                message: 'unversioned experiment documents are incompatible',
            }],
        }} onStartFresh={onStartFresh} />);

        expect(screen.getByRole('heading', { name: 'Experiment is incompatible' }))
            .toBeInTheDocument();
        expect(screen.getByText('#d=xor&n=0.2')).toBeInTheDocument();
        expect(screen.getByRole('list', { name: 'Compatibility issues' }))
            .toHaveTextContent('schemaVersion');

        await userEvent.click(screen.getByRole('button', { name: 'Start fresh' }));
        expect(onStartFresh).toHaveBeenCalledTimes(1);
    });

    it('identifies the exact rejected file and has no obvious accessibility violations', async () => {
        const file = new File(['legacy'], 'legacy-experiment.json', {
            type: 'application/json',
        });
        const { container } = render(<CompatibilityState access={{
            status: 'incompatible',
            prepared: null,
            source: { kind: 'file', file },
            issues: [{ code: 'unsupported-version', path: 'schemaVersion', message: 'unsupported' }],
        }} onStartFresh={vi.fn()} />);

        expect(screen.getByText('legacy-experiment.json')).toBeInTheDocument();
        expect(screen.getByText(`${file.size} bytes`)).toBeInTheDocument();
        expect((await axe(container)).violations).toHaveLength(0);
    });
});
