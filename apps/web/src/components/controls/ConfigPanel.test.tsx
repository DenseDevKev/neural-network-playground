import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
    MAX_EXPERIMENT_JSON_BYTES,
    decodeExperimentJson,
    encodeExperimentJson,
    encodeExperimentUrl,
    type ExperimentDocumentV2,
    type PreparedExperimentDocumentV2,
} from '@nn-playground/shared';
import { ConfigPanel } from './ConfigPanel';
import { useTrainingStore } from '../../store/useTrainingStore.ts';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';

function currentPrepared(): PreparedExperimentDocumentV2 | null {
    const { access } = usePlaygroundStore.getState();
    return access.status === 'ready' ? access.prepared : null;
}

function deferred<T>() {
    let resolve!: (value: T) => void;
    let reject!: (reason?: unknown) => void;
    const promise = new Promise<T>((next, fail) => {
        resolve = next;
        reject = fail;
    });
    return { promise, resolve, reject };
}

function documentWithNoise(
    prepared: PreparedExperimentDocumentV2,
    noise: number,
): ExperimentDocumentV2 {
    return {
        ...prepared.document,
        recipe: {
            ...prepared.document.recipe,
            data: { ...prepared.document.recipe.data, noise },
        },
    };
}

async function selectFile(input: HTMLInputElement, file: File): Promise<void> {
    await act(async () => {
        fireEvent.change(input, { target: { files: [file] } });
    });
}

describe('ConfigPanel strict V2 transport', () => {
    beforeEach(async () => {
        vi.restoreAllMocks();
        vi.useRealTimers();
        useTrainingStore.setState({ pendingConfigSource: null, configError: null, workerError: null, trainedRecipeFingerprint: null, trainedRecipeSource: null });
        window.history.replaceState(null, '', '/');
        const restored = await usePlaygroundStore.getState()
            .replaceDocument(DEFAULT_EXPERIMENT_DOCUMENT);
        expect(restored.ok).toBe(true);

        Object.defineProperty(navigator, 'clipboard', {
            value: { writeText: vi.fn().mockResolvedValue(undefined) },
            configurable: true,
        });
        Object.defineProperty(URL, 'createObjectURL', {
            value: vi.fn(() => 'blob:nn-playground-experiment-v2'),
            configurable: true,
        });
        Object.defineProperty(URL, 'revokeObjectURL', {
            value: vi.fn(),
            configurable: true,
        });
        vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined);
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('syncs V2 state before copying the resulting current absolute URL', async () => {
        const prepared = currentPrepared()!;
        window.history.replaceState(null, '', '/#stale-legacy-hash');
        const syncToUrl = vi.spyOn(usePlaygroundStore.getState(), 'syncToUrl');

        render(<ConfigPanel onReset={vi.fn()} />);
        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /copy setup link/i }));
        });

        expect(syncToUrl).toHaveBeenCalledTimes(1);
        expect(window.location.hash).toBe(encodeExperimentUrl(prepared.document));
        expect(navigator.clipboard.writeText).toHaveBeenCalledWith(window.location.href);
        expect(navigator.clipboard.writeText).not.toHaveBeenCalledWith(
            expect.stringContaining('stale-legacy-hash'),
        );
        expect(syncToUrl.mock.invocationCallOrder[0]).toBeLessThan(
            (navigator.clipboard.writeText as ReturnType<typeof vi.fn>)
                .mock.invocationCallOrder[0],
        );
        expect(screen.getByRole('status')).toHaveTextContent('URL copied');
    });

    it('announces a structured persistent sync failure and does not touch the clipboard', async () => {
        vi.useFakeTimers();
        vi.spyOn(usePlaygroundStore.getState(), 'syncToUrl').mockReturnValue({
            ok: false,
            issues: [{
                code: 'invalid-field',
                path: '$',
                message: 'No compatible version-2 experiment is active',
            }],
        });
        render(<ConfigPanel onReset={vi.fn()} />);

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /copy setup link/i }));
        });

        expect(screen.getByRole('alert')).toHaveTextContent(
            '$: No compatible version-2 experiment is active',
        );
        expect(navigator.clipboard.writeText).not.toHaveBeenCalled();
        act(() => vi.advanceTimersByTime(10_000));
        expect(screen.getByRole('alert')).toHaveTextContent(
            '$: No compatible version-2 experiment is active',
        );
    });

    it('announces clipboard rejection after synchronizing', async () => {
        (navigator.clipboard.writeText as ReturnType<typeof vi.fn>)
            .mockRejectedValueOnce(new Error('permission denied'));
        const syncToUrl = vi.spyOn(usePlaygroundStore.getState(), 'syncToUrl');
        render(<ConfigPanel onReset={vi.fn()} />);

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /copy setup link/i }));
        });

        expect(syncToUrl).toHaveBeenCalledTimes(1);
        expect(screen.getByRole('alert')).toHaveTextContent(/could not copy setup link/i);
    });

    it('replaces a failed URL copy alert with URL copied feedback on a later copy', async () => {
        (navigator.clipboard.writeText as ReturnType<typeof vi.fn>)
            .mockRejectedValueOnce(new Error('permission denied'));
        render(<ConfigPanel onReset={vi.fn()} />);

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /copy setup link/i }));
        });
        expect(screen.getByRole('alert')).toHaveTextContent(/could not copy setup link/i);

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /copy setup link/i }));
        });

        expect(screen.queryByRole('alert')).not.toBeInTheDocument();
        expect(screen.getByRole('status')).toHaveTextContent('URL copied');
    });

    it('announces unavailable clipboard support after synchronizing', async () => {
        Object.defineProperty(navigator, 'clipboard', {
            value: undefined,
            configurable: true,
        });
        const syncToUrl = vi.spyOn(usePlaygroundStore.getState(), 'syncToUrl');
        render(<ConfigPanel onReset={vi.fn()} />);

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /copy setup link/i }));
        });

        expect(syncToUrl).toHaveBeenCalledTimes(1);
        expect(screen.getByRole('alert')).toHaveTextContent(/could not copy setup link/i);
    });

    it('deduplicates rapid Copy URL requests until the current clipboard write settles', async () => {
        const copyResult = deferred<void>();
        (navigator.clipboard.writeText as ReturnType<typeof vi.fn>)
            .mockReturnValue(copyResult.promise);
        const syncToUrl = vi.spyOn(usePlaygroundStore.getState(), 'syncToUrl');
        render(<ConfigPanel onReset={vi.fn()} />);
        const copyButton = screen.getByRole('button', { name: /copy setup link/i });

        fireEvent.click(copyButton);
        await waitFor(() => {
            expect(navigator.clipboard.writeText).toHaveBeenCalledTimes(1);
        });
        const disabledWhilePending = copyButton.hasAttribute('disabled');
        fireEvent.click(copyButton);
        const callsAfterSecondClick = (navigator.clipboard.writeText as ReturnType<typeof vi.fn>)
            .mock.calls.length;

        copyResult.resolve();
        expect(await screen.findByRole('status')).toHaveTextContent('URL copied');

        expect(disabledWhilePending).toBe(true);
        expect(callsAfterSecondClick).toBe(1);
        expect(syncToUrl).toHaveBeenCalledTimes(1);
        expect(copyButton).toBeEnabled();
    });

    it('keeps a newer JSON export error when an older URL copy succeeds', async () => {
        const copyResult = deferred<void>();
        (navigator.clipboard.writeText as ReturnType<typeof vi.fn>)
            .mockReturnValue(copyResult.promise);
        (URL.createObjectURL as ReturnType<typeof vi.fn>)
            .mockImplementationOnce(() => {
                throw new Error('URL creation failed');
            });
        render(<ConfigPanel onReset={vi.fn()} />);

        fireEvent.click(screen.getByRole('button', { name: /copy setup link/i }));
        await waitFor(() => {
            expect(navigator.clipboard.writeText).toHaveBeenCalledTimes(1);
        });

        fireEvent.click(screen.getByRole('button', { name: /export json/i }));
        expect(screen.getByRole('alert')).toHaveTextContent(/could not export experiment/i);

        await act(async () => {
            copyResult.resolve();
            await copyResult.promise;
        });

        expect(screen.getByRole('status')).toHaveTextContent('URL copied');
        expect(screen.getByRole('alert')).toHaveTextContent(/could not export experiment/i);
    });

    it('exports the exact prepared V2 document with a V2-specific filename', async () => {
        const prepared = currentPrepared()!;
        render(<ConfigPanel onReset={vi.fn()} />);

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /export json/i }));
        });

        expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
        const blob = (URL.createObjectURL as ReturnType<typeof vi.fn>)
            .mock.calls[0][0] as Blob;
        const json = await blob.text();
        expect(json).toBe(encodeExperimentJson(prepared.document));
        expect(decodeExperimentJson(json)).toEqual({
            ok: true,
            value: prepared.document,
        });
        const clickedAnchor = (HTMLAnchorElement.prototype.click as ReturnType<typeof vi.fn>)
            .mock.instances[0] as HTMLAnchorElement;
        expect(clickedAnchor.download).toBe('nn-playground-experiment-v2.json');
        expect(URL.revokeObjectURL).toHaveBeenCalledWith(
            'blob:nn-playground-experiment-v2',
        );
        expect(screen.getByRole('status')).toHaveTextContent('Exported');
    });

    it('replaces a failed JSON export alert with Exported feedback on a later export', async () => {
        (URL.createObjectURL as ReturnType<typeof vi.fn>)
            .mockImplementationOnce(() => {
                throw new Error('URL creation failed');
            });
        render(<ConfigPanel onReset={vi.fn()} />);

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /export json/i }));
        });
        expect(screen.getByRole('alert')).toHaveTextContent(/could not export experiment/i);

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /export json/i }));
        });

        expect(screen.queryByRole('alert')).not.toBeInTheDocument();
        expect(screen.getByRole('status')).toHaveTextContent('Exported');
    });

    it('shows a structured persistent error and creates no blob without an active V2 document', async () => {
        vi.useFakeTimers();
        usePlaygroundStore.setState({
            access: {
                status: 'incompatible',
                prepared: null,
                source: { kind: 'url', rawHash: '#invalid' },
                issues: [{
                    code: 'invalid-field',
                    path: '$',
                    message: 'No compatible version-2 experiment is active',
                }],
            },
        });
        render(<ConfigPanel onReset={vi.fn()} />);

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /export json/i }));
        });

        expect(URL.createObjectURL).not.toHaveBeenCalled();
        expect(HTMLAnchorElement.prototype.click).not.toHaveBeenCalled();
        expect(screen.getByRole('alert')).toHaveTextContent(
            '$: No compatible version-2 experiment is active',
        );
        act(() => vi.advanceTimersByTime(10_000));
        expect(screen.getByRole('alert')).toHaveTextContent(
            '$: No compatible version-2 experiment is active',
        );
    });

    async function stageSetup(noise = 17) {
        const before = currentPrepared()!;
        const document = documentWithNoise(before, noise);
        const file = new File([JSON.stringify(document)], 'setup.json', { type: 'application/json' });
        const input = screen.getByLabelText('Import setup JSON file') as HTMLInputElement;
        await selectFile(input, file);
        await screen.findByRole('region', { name: 'Imported setup review' });
        return { before, document, input, file };
    }

    function acknowledge() {
        const prepared = currentPrepared()!;
        act(() => useTrainingStore.setState({ pendingConfigSource: null, configError: null,
            trainedRecipeSource: 'config-sync', trainedRecipeFingerprint: prepared.identities.recipeFingerprint }));
    }

    it('stages without publication, applies once, and only succeeds after exact worker acknowledgement', async () => {
        const onReset = vi.fn();
        const replace = vi.spyOn(usePlaygroundStore.getState(), 'replaceDocument');
        render(<ConfigPanel onReset={onReset} />);
        const { before, document } = await stageSetup();
        expect(currentPrepared()).toBe(before);
        expect(replace).not.toHaveBeenCalled();
        const apply = screen.getByRole('button', { name: 'Apply imported setup' });
        fireEvent.click(apply); fireEvent.click(apply);
        await waitFor(() => expect(currentPrepared()?.document).toEqual(document));
        expect(replace).toHaveBeenCalledTimes(1);
        expect(screen.queryByText(/Imported setup applied/)).not.toBeInTheDocument();
        expect(onReset).not.toHaveBeenCalled();
        acknowledge();
        expect(await screen.findByText(/Imported setup applied/)).toBeInTheDocument();
    });

    it('cancel leaves the active recipe untouched', async () => {
        render(<ConfigPanel onReset={vi.fn()} />);
        const { before } = await stageSetup();
        fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
        expect(currentPrepared()).toBe(before);
        expect(screen.queryByRole('region', { name: 'Imported setup review' })).not.toBeInTheDocument();
    });

    it.each([
        ['malformed JSON', '{'],
        ['old schema', JSON.stringify({ ...DEFAULT_EXPERIMENT_DOCUMENT, schemaVersion: 1 })],
        ['future schema', JSON.stringify({ ...DEFAULT_EXPERIMENT_DOCUMENT, schemaVersion: 3 })],
        ['duplicate key', encodeExperimentJson(DEFAULT_EXPERIMENT_DOCUMENT).replace('"schemaVersion": 2', '"schemaVersion": 1, "schemaVersion": 2')],
    ])('rejects %s without marking the active experiment incompatible; same-file retry stays available', async (_name, json) => {
        const before = currentPrepared();
        render(<ConfigPanel onReset={vi.fn()} />);
        const input = screen.getByLabelText('Import setup JSON file') as HTMLInputElement;
        const file = new File([json], 'invalid.json');
        await selectFile(input, file);
        expect(await screen.findByRole('alert')).toBeInTheDocument();
        expect(currentPrepared()).toBe(before);
        expect(input.value).toBe('');
        await selectFile(input, file);
        expect(await screen.findByRole('alert')).toBeInTheDocument();
        expect(currentPrepared()).toBe(before);
    });

    it('rejects oversized bytes before reading and preserves unsupported initial URL access', async () => {
        const source = { kind: 'url' as const, rawHash: '#unsupported' };
        usePlaygroundStore.getState().markIncompatible(source, [{ code: 'invalid-field', path: '$', message: 'Unsupported URL' }]);
        const access = usePlaygroundStore.getState().access;
        const read = vi.spyOn(FileReader.prototype, 'readAsText');
        render(<ConfigPanel onReset={vi.fn()} />);
        await selectFile(screen.getByLabelText('Import setup JSON file') as HTMLInputElement,
            new File([new Uint8Array(MAX_EXPERIMENT_JSON_BYTES + 1)], 'big.json'));
        expect(screen.getByRole('alert')).toHaveTextContent('exceeds');
        expect(read).not.toHaveBeenCalled();
        expect(usePlaygroundStore.getState().access).toBe(access);
    });

    it('allows exactly the resource byte limit to reach strict decoding', async () => {
        const read = vi.spyOn(FileReader.prototype, 'readAsText');
        render(<ConfigPanel onReset={vi.fn()} />);
        await selectFile(screen.getByLabelText('Import setup JSON file') as HTMLInputElement,
            new File([new Uint8Array(MAX_EXPERIMENT_JSON_BYTES)], 'limit.json'));
        expect(await screen.findByRole('alert')).toHaveTextContent('invalid experiment JSON');
        expect(read).toHaveBeenCalledTimes(1);
    });

    it('preserves the stage and active access on preparation failure and allows retry', async () => {
        render(<ConfigPanel onReset={vi.fn()} />);
        const { before } = await stageSetup();
        vi.spyOn(usePlaygroundStore.getState(), 'replaceDocument').mockResolvedValueOnce({ ok: false,
            issues: [{ code: 'invalid-field', path: 'recipe', message: 'Preparation failed' }] });
        fireEvent.click(screen.getByRole('button', { name: 'Apply imported setup' }));
        expect(await screen.findByRole('alert')).toHaveTextContent('Preparation failed');
        expect(currentPrepared()).toBe(before);
        expect(screen.getByRole('region', { name: 'Imported setup review' })).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Apply imported setup' }));
        await waitFor(() => expect(currentPrepared()).not.toBe(before));
        acknowledge();
        expect(await screen.findByText(/Imported setup applied/)).toBeInTheDocument();
    });

    it('offers synchronization retry without publishing or resetting the setup twice', async () => {
        render(<ConfigPanel onReset={vi.fn()} />);
        await stageSetup();
        const replace = vi.spyOn(usePlaygroundStore.getState(), 'replaceDocument');
        fireEvent.click(screen.getByRole('button', { name: 'Apply imported setup' }));
        await waitFor(() => expect(currentPrepared()?.document.recipe.data.noise).toBe(17));
        act(() => useTrainingStore.getState().failConfigChange('Worker refused setup'));
        expect(await screen.findByRole('alert')).toHaveTextContent('Worker refused setup');
        fireEvent.click(screen.getByRole('button', { name: 'Retry synchronization' }));
        expect(useTrainingStore.getState().pendingConfigSource).toBe('setup');
        acknowledge();
        expect(await screen.findByText(/Imported setup applied/)).toBeInTheDocument();
        expect(replace).toHaveBeenCalledTimes(1);
    });

    it('invalidates a staged recipe when the active setup changes', async () => {
        render(<ConfigPanel onReset={vi.fn()} />);
        const { before } = await stageSetup();
        await act(async () => { await usePlaygroundStore.getState().replaceDocument(documentWithNoise(before, 31)); });
        fireEvent.click(screen.getByRole('button', { name: 'Apply imported setup' }));
        expect(screen.getByRole('alert')).toHaveTextContent('active setup changed');
        expect(currentPrepared()?.document.recipe.data.noise).toBe(31);
    });

    it('never reports success for an imported publication superseded before acknowledgement', async () => {
        render(<ConfigPanel onReset={vi.fn()} />);
        const { before } = await stageSetup();
        fireEvent.click(screen.getByRole('button', { name: 'Apply imported setup' }));
        await waitFor(() => expect(currentPrepared()?.document.recipe.data.noise).toBe(17));
        await act(async () => { await usePlaygroundStore.getState().replaceDocument(documentWithNoise(before, 31)); });
        acknowledge();
        expect(screen.queryByText(/Imported setup applied/)).not.toBeInTheDocument();
        expect(screen.getByRole('alert')).toHaveTextContent('active setup changed');
    });

    it('keeps a selectable setup link and downloadable JSON when clipboard is unavailable', async () => {
        Object.defineProperty(navigator, 'clipboard', { value: undefined, configurable: true });
        render(<ConfigPanel onReset={vi.fn()} />);
        fireEvent.click(screen.getByRole('button', { name: /copy setup link/i }));
        expect(await screen.findByRole('alert')).toHaveTextContent('Clipboard API');
        expect(screen.getByRole('textbox')).toHaveValue(window.location.href);
        expect(screen.getByRole('button', { name: /Export JSON/ })).toBeEnabled();
    });
    it.each(['error', 'abort'] as const)('preserves active setup after FileReader %s and permits the same file retry', async (event) => {
        const before = currentPrepared();
        vi.spyOn(FileReader.prototype, 'readAsText').mockImplementation(function (this: FileReader) {
            this.dispatchEvent(new ProgressEvent(event));
        });
        render(<ConfigPanel onReset={vi.fn()} />);
        const input = screen.getByLabelText('Import setup JSON file') as HTMLInputElement;
        const file = new File(['{}'], 'unreadable.json');
        await selectFile(input, file);
        expect(await screen.findByRole('alert')).toHaveTextContent(/Select the file again/);
        expect(currentPrepared()).toBe(before);
        expect(input.value).toBe('');
    });

    it('ignores an old read after a newer file is selected', async () => {
        const readers: FileReader[] = [];
        vi.spyOn(FileReader.prototype, 'readAsText').mockImplementation(function (this: FileReader) { readers.push(this); });
        render(<ConfigPanel onReset={vi.fn()} />);
        const input = screen.getByLabelText('Import setup JSON file') as HTMLInputElement;
        await selectFile(input, new File(['{}'], 'older.json'));
        await selectFile(input, new File(['{}'], 'newer.json'));
        const complete = async (index: number, noise: number) => act(async () => {
            readers[index].onload?.call(readers[index], { target: { result: JSON.stringify(documentWithNoise(currentPrepared()!, noise)) } } as unknown as ProgressEvent<FileReader>);
        });
        await complete(1, 11); await complete(0, 22);
        expect(screen.getByText('newer.json')).toBeInTheDocument();
        expect(screen.queryByText('older.json')).not.toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Apply imported setup' }));
        await waitFor(() => expect(currentPrepared()?.document.recipe.data.noise).toBe(11));
    });

    it('ignores a stale successful preparation after another recipe is published', async () => {
        render(<ConfigPanel onReset={vi.fn()} />);
        const { before } = await stageSetup();
        const original = usePlaygroundStore.getState().replaceDocument;
        const result = deferred<Awaited<ReturnType<typeof original>>>();
        vi.spyOn(usePlaygroundStore.getState(), 'replaceDocument').mockReturnValueOnce(result.promise);
        fireEvent.click(screen.getByRole('button', { name: 'Apply imported setup' }));
        const newer = await original(documentWithNoise(before, 33));
        await act(async () => result.resolve(newer));
        expect(currentPrepared()?.document.recipe.data.noise).toBe(33);
        expect(screen.queryByText(/Imported setup applied/)).not.toBeInTheDocument();
    });

    it.each(['result', 'rejection'] as const)('releases a failed import after unmount for a deferred %s', async (outcome) => {
        const { unmount } = render(<ConfigPanel onReset={vi.fn()} />);
        const { before } = await stageSetup();
        const pending = deferred<Awaited<ReturnType<ReturnType<typeof usePlaygroundStore.getState>['replaceDocument']>>>();
        vi.spyOn(usePlaygroundStore.getState(), 'replaceDocument').mockReturnValueOnce(pending.promise);
        fireEvent.click(screen.getByRole('button', { name: 'Apply imported setup' }));
        expect(useTrainingStore.getState().pendingConfigSource).toBe('setup');
        unmount();
        await act(async () => {
            if (outcome === 'result') pending.resolve({ ok: false, issues: [{ code: 'invalid-field', path: 'recipe', message: 'Preparation failed' }] });
            else pending.reject(new Error('Preparation failed'));
            await pending.promise.catch(() => undefined);
        });
        expect(currentPrepared()).toBe(before);
        expect(useTrainingStore.getState().pendingConfigSource).toBeNull();
        expect(useTrainingStore.getState().configError).toMatch(/Preparation failed.*Reopen Export/);
        expect(useTrainingStore.getState().configErrorSource).toBe('setup');
    });

    it.each(['result', 'rejection'] as const)('does not release a newer transaction after an unmounted stale %s', async (outcome) => {
        const { unmount } = render(<ConfigPanel onReset={vi.fn()} />);
        const { before } = await stageSetup();
        const original = usePlaygroundStore.getState().replaceDocument;
        const pending = deferred<Awaited<ReturnType<typeof original>>>();
        vi.spyOn(usePlaygroundStore.getState(), 'replaceDocument').mockReturnValueOnce(pending.promise);
        fireEvent.click(screen.getByRole('button', { name: 'Apply imported setup' }));
        unmount();
        useTrainingStore.getState().beginConfigChange('network');
        await original(documentWithNoise(before, 44));
        const newer = currentPrepared();
        await act(async () => {
            if (outcome === 'result') pending.resolve({ ok: false, issues: [{ code: 'invalid-field', path: 'recipe', message: 'Obsolete failure' }] });
            else pending.reject(new Error('Obsolete failure'));
            await pending.promise.catch(() => undefined);
        });
        expect(currentPrepared()).toBe(newer);
        expect(useTrainingStore.getState().pendingConfigSource).toBe('network');
        expect(useTrainingStore.getState().configError).toBeNull();
    });

    it('bounds synchronization errors shown after import publication', async () => {
        render(<ConfigPanel onReset={vi.fn()} />);
        await stageSetup();
        fireEvent.click(screen.getByRole('button', { name: 'Apply imported setup' }));
        await waitFor(() => expect(currentPrepared()?.document.recipe.data.noise).toBe(17));
        act(() => useTrainingStore.getState().failConfigChange('Worker refused: ' + 'x'.repeat(10_000)));
        const alert = await screen.findByRole('alert');
        expect(alert.textContent!.length).toBeLessThanOrEqual(1800);
        expect(screen.getByRole('button', { name: 'Retry synchronization' })).toBeEnabled();
    });

});
