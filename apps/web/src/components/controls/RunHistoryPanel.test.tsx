import { makeSavedRunRecord } from '../../test/savedRunFixtures.ts';
import { StrictMode } from 'react';
import { useSaveCurrentRun } from '../../hooks/useSaveCurrentRun.ts';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
    EXPERIMENT_MEMORY_ENVELOPE_KIND,
    EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS,
    PREPARED_PRESETS,
} from '@nn-playground/shared';
import type {
    ExperimentRunRecordV2,
} from '@nn-playground/shared';
import { RunHistoryPanel } from './RunHistoryPanel.tsx';
import {
    EXPERIMENT_MEMORY_STORAGE_KEY,
    LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY,
    useExperimentMemoryStore,
} from '../../store/experimentMemoryStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';
import { currentPreparedForTest } from '../../test/playgroundStoreTestUtils.ts';
import { STATE_EFFECTS } from '../../copy/stateEffects.ts';

const workerApi = vi.hoisted(() => ({
    captureRunArtifact: vi.fn(),
}));

vi.mock('../../worker/workerBridge.ts', () => ({
    getWorkerApi: async () => workerApi,
}));

const IDS = [
    '00000000-0000-0000-0000-000000000001',
    '00000000-0000-0000-0000-000000000002',
    '00000000-0000-0000-0000-000000000003',
    '00000000-0000-0000-0000-000000000004',
] as const;

function preset(id: (typeof PREPARED_PRESETS)[number]['id']) {
    const entry = PREPARED_PRESETS.find((candidate) => candidate.id === id);
    if (!entry) throw new Error(`Missing preset ${id}`);
    return entry.prepared;
}


async function hydrateSingleton(): Promise<void> {
    await act(async () => {
        await useExperimentMemoryStore.getState().hydrate();
    });
}

async function openActions() {
    for (const summary of document.querySelectorAll('summary')) {
        if (!summary.parentElement?.hasAttribute('open')) await userEvent.click(summary);
    }
}

describe('RunHistoryPanel V2 evidence memory', () => {
    beforeEach(async () => {
        window.localStorage.clear();
        workerApi.captureRunArtifact.mockReset();
        vi.restoreAllMocks();
        await act(async () => {
            const result = await usePlaygroundStore.getState()
                .replaceDocument(DEFAULT_EXPERIMENT_DOCUMENT);
            if (!result.ok) throw new Error('Could not reset the playground document for the test.');
        });
        await hydrateSingleton();
    });

    it('delegates capture to the injected App controller', async () => {
        const commands = { save: vi.fn(async () => true), retry: vi.fn(async () => true),
            discard: vi.fn(async () => {}), downloadPending: vi.fn(async () => true), dismiss: vi.fn() };
        render(<RunHistoryPanel saveController={{ busy: false, error: null, pending: false,
            disabledReason: null, commands }} />);
        await userEvent.type(screen.getByRole('textbox', { name: 'Run name' }), ' Shared title ');
        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));
        expect(commands.save).toHaveBeenCalledWith('Shared title');
        expect(workerApi.captureRunArtifact).not.toHaveBeenCalled();
    });

    it('retains the shared capture guard when the History drawer unmounts and reopens', async () => {
        let finish!: (record: ExperimentRunRecordV2) => void;
        workerApi.captureRunArtifact.mockImplementation(() => new Promise<ExperimentRunRecordV2>((resolve) => { finish = resolve; }));
        function Owner({ historyOpen }: { historyOpen: boolean }) {
            const save = useSaveCurrentRun();
            return <>
                <button onClick={() => { void save.commands.save(); }}>Transport save</button>
                <span data-testid="capture-busy">{String(save.busy)}</span>
                {historyOpen && <RunHistoryPanel saveController={save} />}
            </>;
        }
        const view = render(<Owner historyOpen />);
        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));
        await waitFor(() => expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1));
        view.rerender(<Owner historyOpen={false} />);
        await userEvent.click(screen.getByRole('button', { name: 'Transport save' }));
        expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1);
        expect(screen.getByTestId('capture-busy')).toHaveTextContent('true');
        const metadata = workerApi.captureRunArtifact.mock.calls[0][0];
        await act(async () => finish(makeSavedRunRecord(currentPreparedForTest()!, metadata.id, 'Closed drawer capture')));
        await waitFor(() => expect(screen.getByTestId('capture-busy')).toHaveTextContent('false'));
        view.rerender(<Owner historyOpen />);
        expect(screen.getByRole('article', { name: 'Closed drawer capture' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Save current run' })).toBeEnabled();
    });

    it('asks the worker to author the record from metadata only and persists its artifact', async () => {
        const prepared = currentPreparedForTest()!;
        workerApi.captureRunArtifact.mockImplementation(async (metadata: {
            id: string;
            createdAt: string;
            updatedAt: string;
        }) => makeSavedRunRecord(prepared, metadata.id, 'Captured run'));

        render(<RunHistoryPanel />);
        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));

        await waitFor(() => expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1));
        const request = workerApi.captureRunArtifact.mock.calls[0][0];
        expect(request).toEqual({
            id: expect.stringMatching(/^[0-9a-f-]{36}$/u),
            createdAt: expect.stringMatching(/^\d{4}-\d{2}-\d{2}T/u),
            updatedAt: expect.stringMatching(/^\d{4}-\d{2}-\d{2}T/u),
        });
        expect(Object.keys(request).sort()).toEqual(['createdAt', 'id', 'updatedAt']);
        await waitFor(() => expect(screen.getByText('Captured run')).toBeInTheDocument());
        const persisted = window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY) ?? '';
        expect(persisted).toContain('"kind":"nn-playground-run"');
        expect(persisted).not.toMatch(/weights|biases|parameters/i);
    });

    it('re-enables saving after a successful capture under StrictMode', async () => {
        const prepared = currentPreparedForTest()!;
        workerApi.captureRunArtifact.mockImplementation(async (metadata: { id: string }) => (
            makeSavedRunRecord(prepared, metadata.id, 'Strict capture')
        ));

        render(
            <StrictMode>
                <RunHistoryPanel />
            </StrictMode>,
        );
        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));

        expect(await screen.findByRole('article', { name: 'Strict capture' })).toBeInTheDocument();
        const saveButton = await screen.findByRole('button', { name: 'Save current run' });
        expect(saveButton).toBeEnabled();
        expect(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY))
            .toContain('"title":"Strict capture"');
    });

    it('re-enables saving and announces a rejected capture under StrictMode', async () => {
        workerApi.captureRunArtifact.mockRejectedValue(new Error('Strict capture rejected'));

        render(
            <StrictMode>
                <RunHistoryPanel />
            </StrictMode>,
        );
        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));

        expect(await screen.findByRole('alert')).toHaveTextContent('Strict capture rejected');
        expect(screen.getByRole('button', { name: 'Save current run' })).toBeEnabled();
    });

    it('trims a custom run name, gives it to the worker as metadata, and renders the saved title', async () => {
        const prepared = currentPreparedForTest()!;
        workerApi.captureRunArtifact.mockImplementation(async (metadata: {
            id: string;
            title: string;
        }) => makeSavedRunRecord(prepared, metadata.id, metadata.title));

        const view = render(<RunHistoryPanel />);
        await userEvent.type(screen.getByRole('textbox', { name: 'Run name' }), '  XOR baseline  ');
        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));

        await waitFor(() => expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1));
        expect(workerApi.captureRunArtifact.mock.calls[0][0]).toEqual(expect.objectContaining({
            title: 'XOR baseline',
        }));
        expect(await screen.findByRole('article', { name: 'XOR baseline' })).toBeInTheDocument();
        expect(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY))
            .toContain('"title":"XOR baseline"');

        view.unmount();
        useExperimentMemoryStore.setState({ records: [] });
        await hydrateSingleton();
        render(<RunHistoryPanel />);
        expect(await screen.findByRole('article', { name: 'XOR baseline' })).toBeInTheDocument();
    });

    it('uses the worker-authored snapshot step in the deterministic default when the run name is blank', async () => {
        const prepared = currentPreparedForTest()!;
        workerApi.captureRunArtifact.mockImplementation(async (metadata: { id: string }) => {
            const { title: _title, ...record } = makeSavedRunRecord(prepared, metadata.id);
            return record;
        });

        render(<RunHistoryPanel />);
        fireEvent.change(screen.getByRole('textbox', { name: 'Run name' }), {
            target: { value: '   ' },
        });
        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));

        await waitFor(() => expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1));
        expect(workerApi.captureRunArtifact.mock.calls[0][0]).not.toHaveProperty('title');
        expect(await screen.findByRole('article', { name: 'Circle · 2-4-4-1 · step 4' }))
            .toBeInTheDocument();
    });

    it('accepts exactly 120 Unicode code points, including astral characters', async () => {
        const prepared = currentPreparedForTest()!;
        const title = '🧠'.repeat(EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS);
        workerApi.captureRunArtifact.mockImplementation(async (metadata: {
            id: string;
            title: string;
        }) => makeSavedRunRecord(prepared, metadata.id, metadata.title));

        render(<RunHistoryPanel />);
        fireEvent.change(screen.getByRole('textbox', { name: 'Run name' }), {
            target: { value: title },
        });
        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));

        await waitFor(() => expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1));
        expect(workerApi.captureRunArtifact.mock.calls[0][0]).toEqual(expect.objectContaining({ title }));
    });

    it('rejects a custom run name beyond the schema Unicode code-point limit', async () => {
        render(<RunHistoryPanel />);
        const input = screen.getByRole('textbox', { name: 'Run name' });
        fireEvent.change(input, {
            target: { value: '🧠'.repeat(EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS + 1) },
        });

        expect(input).toHaveAttribute('aria-invalid', 'true');
        expect(screen.getByRole('alert')).toHaveTextContent(
            `Run name must contain at most ${EXPERIMENT_MEMORY_MAX_TITLE_CODE_POINTS} Unicode code points.`,
        );
        expect(screen.getByRole('button', { name: 'Save current run' })).toBeDisabled();
        expect(workerApi.captureRunArtifact).not.toHaveBeenCalled();
    });

    it('finishes the exact clicked-title save after unmount without reading later input', async () => {
        const prepared = currentPreparedForTest()!;
        let resolveCapture!: () => void;
        workerApi.captureRunArtifact.mockImplementation((metadata: { id: string; title: string }) => (
            new Promise<ExperimentRunRecordV2>((resolve) => {
                resolveCapture = () => resolve(makeSavedRunRecord(prepared, metadata.id, metadata.title));
            })
        ));

        const view = render(<RunHistoryPanel />);
        const input = screen.getByRole('textbox', { name: 'Run name' });
        await userEvent.type(input, 'First title');
        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));
        fireEvent.change(input, { target: { value: 'Later title' } });
        view.unmount();

        await act(async () => {
            resolveCapture();
            await Promise.resolve();
        });

        expect(workerApi.captureRunArtifact.mock.calls[0][0]).toEqual(expect.objectContaining({
            title: 'First title',
        }));
        await waitFor(() => expect(useExperimentMemoryStore.getState().records[0]?.title)
            .toBe('First title'));
    });

    it('retries persistence with the captured record instead of recapturing scientific state', async () => {
        const prepared = currentPreparedForTest()!;
        workerApi.captureRunArtifact.mockImplementation(async (metadata: { id: string }) => (
            makeSavedRunRecord(prepared, metadata.id, 'Retry me')
        ));
        const prototype = Object.getPrototypeOf(window.localStorage) as Storage;
        const setItem = vi.spyOn(prototype, 'setItem').mockImplementation(() => {
            throw new DOMException('Quota exceeded', 'QuotaExceededError');
        });

        render(<RunHistoryPanel />);
        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));
        await screen.findByText(/quota/i);
        expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1);

        setItem.mockRestore();
        await userEvent.click(screen.getByRole('button', { name: 'Retry saving' }));

        await waitFor(() => expect(screen.getByText('Retry me')).toBeInTheDocument());
        expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1);
        expect(useExperimentMemoryStore.getState().persistenceError).toBeNull();
    });

    it('blocks recapture and keeps retry or discard available after error dismissal', async () => {
        const prepared = currentPreparedForTest()!;
        workerApi.captureRunArtifact.mockImplementation(async (metadata: { id: string }) => (
            makeSavedRunRecord(prepared, metadata.id, 'Pending exact artifact')
        ));
        const prototype = Object.getPrototypeOf(window.localStorage) as Storage;
        const setItem = vi.spyOn(prototype, 'setItem').mockImplementation(() => {
            throw new DOMException('Quota exceeded', 'QuotaExceededError');
        });

        render(<RunHistoryPanel />);
        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));
        await screen.findByText(/quota/i);
        expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1);
        expect(screen.getByRole('button', { name: 'Save current run' })).toBeDisabled();

        await userEvent.click(screen.getByRole('button', { name: 'Dismiss error' }));
        expect(screen.queryByText(/quota/i)).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Retry saving' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Discard pending save' })).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));
        expect(workerApi.captureRunArtifact).toHaveBeenCalledTimes(1);

        setItem.mockRestore();
        await userEvent.click(screen.getByRole('button', { name: 'Discard pending save' }));
        await userEvent.click(screen.getByRole('button', { name: 'Confirm' }));
        expect(screen.queryByRole('button', { name: 'Retry saving' })).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Save current run' })).toBeEnabled();
    });

    it('applies a saved recipe through a fresh version-2 document', async () => {
        const prepared = preset('xor-hidden');
        await act(async () => {
            await useExperimentMemoryStore.getState().saveRecord(makeSavedRunRecord(prepared));
        });
        const replaceDocument = vi.spyOn(usePlaygroundStore.getState(), 'replaceDocument');

        render(<RunHistoryPanel />);
        await openActions();
        await userEvent.click(screen.getByRole('button', { name: 'Apply saved recipe' }));
        expect(replaceDocument).not.toHaveBeenCalled();
        await userEvent.click(screen.getByRole('button', { name: 'Confirm' }));

        await waitFor(() => expect(replaceDocument).toHaveBeenCalledWith(expect.objectContaining({
            kind: 'nn-playground-experiment',
            schemaVersion: 2,
            recipe: prepared.document.recipe,
        })));
        expect(screen.queryByRole('button', { name: /restore/i })).not.toBeInTheDocument();
    });

    it('owns unique run-name and saved-recipe descriptions across mounted panels', async () => {
        const prepared = preset('circle-one-layer');
        await act(async () => {
            await useExperimentMemoryStore.getState().saveRecord(
                makeSavedRunRecord(prepared, IDS[0], 'First'),
            );
            await useExperimentMemoryStore.getState().saveRecord(
                makeSavedRunRecord(prepared, IDS[1], 'Second'),
            );
        });

        const { container } = render(
            <>
                <RunHistoryPanel />
                <RunHistoryPanel />
            </>,
        );
        await openActions();
        const panels = [...container.querySelectorAll<HTMLElement>('.run-history-panel')];
        const allOwnedIds: string[] = [];

        expect(panels).toHaveLength(2);
        for (const panel of panels) {
            const runName = within(panel).getByRole('textbox', { name: 'Run name' });
            const runNameIds = (runName.getAttribute('aria-describedby') ?? '')
                .split(/\s+/u)
                .filter(Boolean);
            expect(runNameIds).toHaveLength(1);
            expect(document.getElementById(runNameIds[0])).toHaveTextContent(
                'Optional. Leave blank to use dataset, architecture, and saved step.',
            );

            const effectsNote = within(panel).getByText(
                STATE_EFFECTS['saved-recipe-apply'],
                { exact: true },
            );
            expect(effectsNote).toBeVisible();
            expect(effectsNote.closest('[aria-live], [role="status"], [role="alert"]'))
                .toBeNull();
            const applyButtons = within(panel).getAllByRole('button', {
                name: 'Apply saved recipe',
            });
            const applyIds = applyButtons.map((button) => {
                expect(button).toHaveAccessibleDescription(STATE_EFFECTS['saved-recipe-apply']);
                const ids = (button.getAttribute('aria-describedby') ?? '')
                    .split(/\s+/u)
                    .filter(Boolean);
                expect(ids).toHaveLength(1);
                expect(document.getElementById(ids[0])).toBe(effectsNote);
                return ids[0];
            });
            expect(new Set(applyIds).size).toBe(1);
            allOwnedIds.push(runNameIds[0], applyIds[0]);
        }

        expect(new Set(allOwnedIds).size).toBe(allOwnedIds.length);
    });

    it('shows no numeric winner when dataset or objective identity differs', async () => {
        await act(async () => {
            await useExperimentMemoryStore.getState().saveRecord(makeSavedRunRecord(
                preset('circle-one-layer'),
                IDS[0],
                'Circle',
                0.4,
            ));
            await useExperimentMemoryStore.getState().saveRecord(makeSavedRunRecord(
                preset('regression-plane'),
                IDS[1],
                'Regression',
                0.2,
            ));
        });

        render(<RunHistoryPanel />);

        await userEvent.click(screen.getByRole('button', { name: 'Compare selected' }));
        const comparison = screen.getByRole('group', { name: /Saved run comparison/ });
        expect(comparison).toHaveTextContent('Not directly comparable');
        expect(comparison).not.toHaveTextContent(/winner|lower by|better/i);
    });

    it('computes a numeric winner only for equal dataset and objective identities', async () => {
        const prepared = preset('circle-one-layer');
        await act(async () => {
            await useExperimentMemoryStore.getState().saveRecord(makeSavedRunRecord(prepared, IDS[0], 'Baseline', 0.6));
            await useExperimentMemoryStore.getState().saveRecord(makeSavedRunRecord(prepared, IDS[1], 'Tuned', 0.4));
        });

        render(<RunHistoryPanel />);

        await userEvent.click(screen.getByRole('button', { name: 'Compare selected' }));
        expect(screen.getByRole('group', { name: /Saved run comparison/ }))
            .toHaveTextContent('Tuned has lower test data loss by 0.2000');
    });

    it('compares two nonadjacent choices, prunes a deleted choice, and preserves order across updates', async () => {
        const prepared = preset('circle-one-layer');
        const first = makeSavedRunRecord(prepared, IDS[0], 'First', 0.8);
        const middle = makeSavedRunRecord(prepared, IDS[1], 'Middle', 0.2);
        const latest = makeSavedRunRecord(prepared, IDS[2], 'Latest', 0.5);
        await act(async () => {
            await useExperimentMemoryStore.getState().saveRecord(first);
            await useExperimentMemoryStore.getState().saveRecord(middle);
            await useExperimentMemoryStore.getState().saveRecord(latest);
        });

        const view = render(<RunHistoryPanel />);
        const latestChoice = screen.getByRole('checkbox', { name: 'Compare Latest' });
        const middleChoice = screen.getByRole('checkbox', { name: 'Compare Middle' });
        const firstChoice = screen.getByRole('checkbox', { name: 'Compare First' });
        expect(latestChoice).toBeChecked();
        expect(middleChoice).toBeChecked();
        expect(firstChoice).not.toBeChecked();

        await act(async () => {
            await useExperimentMemoryStore.getState().saveRecord(
                makeSavedRunRecord(prepared, IDS[3], 'Newly saved', 0.1),
            );
        });
        expect(screen.getByRole('checkbox', { name: 'Compare Newly saved' })).not.toBeChecked();
        expect(latestChoice).toBeChecked();
        expect(middleChoice).toBeChecked();

        await userEvent.click(middleChoice);
        await userEvent.click(firstChoice);

        await userEvent.click(screen.getByRole('button', { name: 'Compare selected' }));
        let comparison = screen.getByRole('group', {
            name: 'Saved run comparison: Latest and First',
        });
        expect(comparison).toHaveTextContent('Latest has lower test data loss by 0.3000');
        expect(comparison).not.toHaveTextContent('Middle');

        act(() => {
            useExperimentMemoryStore.setState({
                records: [
                    first,
                    middle,
                    latest,
                    makeSavedRunRecord(prepared, IDS[3], 'Newly saved', 0.1),
                ],
            });
        });
        comparison = screen.getByRole('group', {
            name: 'Saved run comparison: Latest and First',
        });
        expect(comparison).toBeInTheDocument();

        await userEvent.click(screen.getByRole('button', { name: '← Saved runs' }));
        await openActions();
        await userEvent.click(screen.getByRole('button', { name: 'Delete Latest' }));
        await userEvent.click(screen.getByRole('button', { name: 'Confirm' }));
        await waitFor(() => expect(screen.queryByRole('article', { name: 'Latest' }))
            .not.toBeInTheDocument());
        expect(screen.queryByRole('group', { name: /Saved run comparison/ })).not.toBeInTheDocument();
        expect(screen.getByRole('checkbox', { name: 'Compare First' })).toBeChecked();
        expect(screen.getByRole('checkbox', { name: 'Compare Middle' })).not.toBeChecked();

        act(() => {
            useExperimentMemoryStore.setState({
                records: [
                    first,
                    middle,
                    latest,
                    makeSavedRunRecord(prepared, IDS[3], 'Newly saved', 0.1),
                ],
            });
        });
        expect(screen.getByRole('checkbox', { name: 'Compare Latest' })).not.toBeChecked();
        expect(screen.getByRole('checkbox', { name: 'Compare First' })).toBeChecked();

        view.unmount();
        render(<RunHistoryPanel />);
        expect(screen.getByRole('checkbox', { name: 'Compare First' })).toBeChecked();
        expect(screen.getByRole('checkbox', { name: 'Compare Middle' })).toBeChecked();
    });

    it('uses distinct human labels for duplicate long titles in controls and the heading', async () => {
        const prepared = preset('circle-one-layer');
        const title = `Duplicate ${'🧠'.repeat(50)}`;
        await act(async () => {
            await useExperimentMemoryStore.getState().saveRecord(
                makeSavedRunRecord(prepared, IDS[0], title, 0.6),
            );
            await useExperimentMemoryStore.getState().saveRecord(
                makeSavedRunRecord(prepared, IDS[1], title, 0.4),
            );
        });

        render(<RunHistoryPanel />);

        expect(screen.getByRole('checkbox', { name: `Compare ${title} (${IDS[0]})` }))
            .toBeChecked();
        expect(screen.getByRole('checkbox', { name: `Compare ${title} (${IDS[1]})` }))
            .toBeChecked();
        await userEvent.click(screen.getByRole('button', { name: 'Compare selected' }));
        expect(screen.getByRole('group', {
            name: `Saved run comparison: ${title} (${IDS[1]}) and ${title} (${IDS[0]})`,
        })).toHaveTextContent(`${title} (${IDS[1]}) has lower test data loss by 0.2000`);
    });

    it('keeps the warning local when only objective identity differs', async () => {
        const prepared = preset('circle-one-layer');
        const baseline = makeSavedRunRecord(prepared, IDS[0], 'Baseline objective', 0.6);
        const changed = makeSavedRunRecord(prepared, IDS[1], 'Changed objective', 0.4);
        const incompatible = {
            ...changed,
            snapshot: {
                ...changed.snapshot,
                evaluation: {
                    ...changed.snapshot.evaluation,
                    objectiveKey: `${changed.snapshot.evaluation.objectiveKey}:different`,
                },
            },
        } as ExperimentRunRecordV2;
        act(() => {
            useExperimentMemoryStore.setState({ records: [incompatible, baseline] });
        });

        render(<RunHistoryPanel />);

        await userEvent.click(screen.getByRole('button', { name: 'Compare selected' }));
        const comparison = screen.getByRole('group', { name: /Saved run comparison/ });
        expect(comparison).toHaveTextContent('Not directly comparable');
        expect(comparison).toHaveTextContent(
            'Dataset and objective identities must both match before losses can be ranked.',
        );
        expect(screen.queryByRole('alert')).toBeNull();
    });

    it('requires explicit comparison, caps two choices, filters differences and returns to retained selection', async () => {
        const prepared = preset('circle-one-layer');
        await act(async () => {
            for (let index = 0; index < 3; index++) await useExperimentMemoryStore.getState().saveRecord(makeSavedRunRecord(prepared, IDS[index], `Run ${index}`, 0.3 + index / 10));
        });
        render(<RunHistoryPanel />);
        expect(screen.queryByRole('group', { name: /Saved run comparison/ })).toBeNull();
        expect(screen.getByRole('checkbox', { name: 'Compare Run 0' })).toBeDisabled();
        await userEvent.click(screen.getByRole('button', { name: 'Compare selected' }));
        expect(screen.getByText('Stored learning history')).toBeVisible();
        expect(screen.getByText('Recipe identity')).toBeVisible();
        await userEvent.click(screen.getByRole('checkbox', { name: 'Only show differences' }));
        expect(screen.queryByText('Recipe identity')).toBeNull();
        expect(screen.getByText('Full evaluation')).toBeVisible();
        await userEvent.click(screen.getByRole('button', { name: '← Saved runs' }));
        expect(screen.getByRole('checkbox', { name: 'Compare Run 2' })).toBeChecked();
        expect(screen.getByRole('checkbox', { name: 'Compare Run 1' })).toBeChecked();
        await userEvent.click(screen.getByRole('button', { name: 'Clear' }));
        expect(screen.getByRole('button', { name: 'Compare selected' })).toBeDisabled();
    });

    it('renames within the code-point limit and persists the title without changing evidence', async () => {
        const record = makeSavedRunRecord(preset('circle-one-layer'), IDS[0], 'Before');
        await act(async () => { await useExperimentMemoryStore.getState().saveRecord(record); });
        render(<RunHistoryPanel />); await openActions();
        await userEvent.click(screen.getByRole('button', { name: 'Rename Before' }));
        const dialog = screen.getByRole('dialog', { name: 'Rename run' });
        const name = within(dialog).getByRole('textbox', { name: 'Run name' });
        fireEvent.change(name, { target: { value: '🧠'.repeat(121) } });
        expect(within(dialog).getByRole('button', { name: 'Save name' })).toBeDisabled();
        fireEvent.change(name, { target: { value: '🧠'.repeat(120) } });
        await userEvent.click(within(dialog).getByRole('button', { name: 'Save name' }));
        await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
        expect(useExperimentMemoryStore.getState().records[0].title).toBe('🧠'.repeat(120));
        expect(useExperimentMemoryStore.getState().records[0].snapshot).toEqual(record.snapshot);
    });

    it('cancels deletion and keeps an action-specific failed deletion error available', async () => {
        await act(async () => { await useExperimentMemoryStore.getState().saveRecord(makeSavedRunRecord(preset('circle-one-layer'), IDS[0], 'Keep me')); });
        render(<RunHistoryPanel />); await openActions();
        await userEvent.click(screen.getByRole('button', { name: 'Delete Keep me' }));
        await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
        expect(useExperimentMemoryStore.getState().records).toHaveLength(1);
        vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('quota'); });
        await userEvent.click(screen.getByRole('button', { name: 'Delete Keep me' }));
        await userEvent.click(screen.getByRole('button', { name: 'Confirm' }));
        expect(await within(screen.getByRole('alertdialog')).findByRole('alert')).toHaveTextContent('Delete saved run failed');
        expect(useExperimentMemoryStore.getState().records).toHaveLength(1);
    });

    it('preserves legacy bytes through dismissal and deletes only explicitly', async () => {
        const raw = '{"schemaVersion":1,"records":[{"id":"old"}]}';
        window.localStorage.setItem(LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY, raw);
        await hydrateSingleton();

        render(<RunHistoryPanel />);
        expect(screen.getByRole('note', { name: 'Earlier saved runs' })).toHaveTextContent(/incompatible/i);
        expect(screen.getByRole('button', { name: 'Download earlier runs' })).toBeInTheDocument();

        await userEvent.click(screen.getByRole('button', { name: 'Dismiss earlier runs notice' }));
        expect(screen.queryByRole('note', { name: 'Earlier saved runs' })).not.toBeInTheDocument();
        expect(window.localStorage.getItem(LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY)).toBe(raw);

        await act(async () => {
            await useExperimentMemoryStore.getState().deleteLegacyStorage();
        });
        expect(window.localStorage.getItem(LEGACY_EXPERIMENT_MEMORY_STORAGE_KEY)).toBeNull();
    });

    it('keeps an incompatible whole envelope recoverable while blocking unrelated saves', async () => {
        const raw = '{ "kind": "nn-playground-experiment-memory", "schemaVersion": 3, "records": [] }';
        const prepared = currentPreparedForTest()!;
        window.localStorage.setItem(EXPERIMENT_MEMORY_STORAGE_KEY, raw);
        workerApi.captureRunArtifact.mockImplementation(async (metadata: { id: string }) => (
            makeSavedRunRecord(prepared, metadata.id, 'Saved after recovery')
        ));
        await hydrateSingleton();

        render(<RunHistoryPanel />);
        expect(screen.getByRole('note', { name: 'Incompatible saved-run file' }))
            .toHaveTextContent(/bytes.*not.*changed/i);
        expect(screen.getByRole('button', { name: 'Download incompatible saved-run file' }))
            .toBeInTheDocument();

        await userEvent.click(screen.getByRole('button', { name: 'Save current run' }));
        await screen.findByText(/incompatible.*delete/i);
        expect(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY)).toBe(raw);
        expect(useExperimentMemoryStore.getState().pendingSave?.title).toBe('Saved after recovery');

        await userEvent.click(screen.getByRole('button', { name: 'Delete incompatible saved-run file' }));
        await userEvent.click(screen.getByRole('button', { name: 'Confirm' }));
        await userEvent.click(screen.getByRole('button', { name: 'Retry saving' }));
        await screen.findByText('Saved after recovery');
        expect(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY)).not.toBe(raw);
    });

    it('offers raw download and explicit deletion for a rejected record', async () => {
        const valid = makeSavedRunRecord(preset('circle-one-layer'));
        const rejected = { schemaVersion: 1, id: 'rejected' };
        window.localStorage.setItem(EXPERIMENT_MEMORY_STORAGE_KEY, JSON.stringify({
            kind: EXPERIMENT_MEMORY_ENVELOPE_KIND,
            schemaVersion: 2,
            records: [valid, rejected],
        }));
        await hydrateSingleton();

        render(<RunHistoryPanel />);
        expect(screen.getByText('Rejected saved record')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Download rejected record' })).toBeInTheDocument();

        await userEvent.click(screen.getByRole('button', { name: 'Delete rejected record' }));
        await userEvent.click(screen.getByRole('button', { name: 'Confirm' }));
        await waitFor(() => expect(screen.queryByText('Rejected saved record')).not.toBeInTheDocument());
        expect(window.localStorage.getItem(EXPERIMENT_MEMORY_STORAGE_KEY)).not.toContain('"schemaVersion":1');
    });
});
