import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { StrictMode } from 'react';
import {
    DEFAULT_EXPERIMENT_DOCUMENT,
    MAX_EXPERIMENT_JSON_BYTES,
    decodeExperimentJson,
    encodeExperimentJson,
    encodeExperimentUrl,
    prepareExperimentDocument,
    type ExperimentDocumentV2,
    type PreparedExperimentDocumentV2,
    type SchemaResult,
} from '@nn-playground/shared';
import { ConfigPanel } from './ConfigPanel';
import { usePlaygroundStore } from '../../store/usePlaygroundStore.ts';

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

async function requirePrepared(
    document: ExperimentDocumentV2,
): Promise<PreparedExperimentDocumentV2> {
    const result = await prepareExperimentDocument(document);
    expect(result.ok).toBe(true);
    if (!result.ok) throw new Error(JSON.stringify(result.issues));
    return result.value;
}

function fileInput(container: HTMLElement): HTMLInputElement {
    const input = container.querySelector<HTMLInputElement>('input[type="file"]');
    if (!input) throw new Error('Config file input was not rendered');
    return input;
}

async function selectFile(input: HTMLInputElement, file: File): Promise<void> {
    await act(async () => {
        fireEvent.change(input, { target: { files: [file] } });
    });
}

describe('ConfigPanel strict V2 transport', () => {
    beforeEach(async () => {
        vi.restoreAllMocks();
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
        const prepared = usePlaygroundStore.getState().prepared!;
        window.history.replaceState(null, '', '/#stale-legacy-hash');
        const syncToUrl = vi.spyOn(usePlaygroundStore.getState(), 'syncToUrl');

        render(<ConfigPanel onReset={vi.fn()} />);
        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /copy url/i }));
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
            fireEvent.click(screen.getByRole('button', { name: /copy url/i }));
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
            fireEvent.click(screen.getByRole('button', { name: /copy url/i }));
        });

        expect(syncToUrl).toHaveBeenCalledTimes(1);
        expect(screen.getByRole('alert')).toHaveTextContent(/could not copy url/i);
    });

    it('announces unavailable clipboard support after synchronizing', async () => {
        Object.defineProperty(navigator, 'clipboard', {
            value: undefined,
            configurable: true,
        });
        const syncToUrl = vi.spyOn(usePlaygroundStore.getState(), 'syncToUrl');
        render(<ConfigPanel onReset={vi.fn()} />);

        await act(async () => {
            fireEvent.click(screen.getByRole('button', { name: /copy url/i }));
        });

        expect(syncToUrl).toHaveBeenCalledTimes(1);
        expect(screen.getByRole('alert')).toHaveTextContent(/could not copy url/i);
    });

    it('deduplicates rapid Copy URL requests until the current clipboard write settles', async () => {
        const copyResult = deferred<void>();
        (navigator.clipboard.writeText as ReturnType<typeof vi.fn>)
            .mockReturnValue(copyResult.promise);
        const syncToUrl = vi.spyOn(usePlaygroundStore.getState(), 'syncToUrl');
        render(<ConfigPanel onReset={vi.fn()} />);
        const copyButton = screen.getByRole('button', { name: /copy url/i });

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

    it('exports the exact prepared V2 document with a V2-specific filename', async () => {
        const prepared = usePlaygroundStore.getState().prepared!;
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

    it('shows a structured persistent error and creates no blob without an active V2 document', async () => {
        vi.useFakeTimers();
        usePlaygroundStore.setState({ prepared: null });
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

    it('awaits V2 preparation and resets only after the exact import is published', async () => {
        const onReset = vi.fn();
        const before = usePlaygroundStore.getState().prepared!;
        const target = await requirePrepared(documentWithNoise(before, 17));
        const gate = deferred<void>();
        const originalReplace = usePlaygroundStore.getState().replaceDocument;
        vi.spyOn(usePlaygroundStore.getState(), 'replaceImportedDocument')
            .mockImplementation(async (value) => {
                const result = await originalReplace(value);
                await gate.promise;
                return result;
            });
        const { container } = render(<ConfigPanel onReset={onReset} />);
        const input = fileInput(container);

        await selectFile(input, new File(
            [encodeExperimentJson(target.document)],
            'experiment-v2.json',
            { type: 'application/json' },
        ));
        await waitFor(() => {
            expect(usePlaygroundStore.getState().replaceImportedDocument).toHaveBeenCalledTimes(1);
        });
        expect(onReset).not.toHaveBeenCalled();
        expect(screen.queryByText('Imported!')).not.toBeInTheDocument();
        expect(input.value).toBe('');

        gate.resolve();
        expect(await screen.findByRole('status')).toHaveTextContent('Imported');
        expect(onReset).toHaveBeenCalledTimes(1);
        expect(usePlaygroundStore.getState().prepared?.document).toEqual(target.document);
        expect(usePlaygroundStore.getState().prepared?.identities).toEqual(target.identities);
    });

    it('keeps import publication callbacks active through the StrictMode effect probe', async () => {
        const onReset = vi.fn();
        const before = usePlaygroundStore.getState().prepared!;
        const target = await requirePrepared(documentWithNoise(before, 18));
        const { container } = render(
            <StrictMode>
                <ConfigPanel onReset={onReset} />
            </StrictMode>,
        );

        await selectFile(fileInput(container), new File(
            [encodeExperimentJson(target.document)],
            'strict-mode-v2.json',
            { type: 'application/json' },
        ));

        expect(await screen.findByRole('status')).toHaveTextContent('Imported');
        expect(onReset).toHaveBeenCalledTimes(1);
    });

    it.each([
        ['malformed JSON', '{', '$: invalid experiment JSON'],
        [
            'unversioned V1 runtime config',
            JSON.stringify(usePlaygroundStore.getState().getConfig()),
            'schemaVersion: unversioned experiment documents are incompatible',
        ],
        [
            'schema version 1',
            JSON.stringify({ ...DEFAULT_EXPERIMENT_DOCUMENT, schemaVersion: 1 }),
            'schemaVersion: schema version 1 experiment documents are incompatible',
        ],
        [
            'future schema version',
            JSON.stringify({ ...DEFAULT_EXPERIMENT_DOCUMENT, schemaVersion: 3 }),
            'schemaVersion: schemaVersion 3 is unsupported',
        ],
        [
            'duplicate object key',
            encodeExperimentJson(DEFAULT_EXPERIMENT_DOCUMENT).replace(
                '"schemaVersion": 2',
                '"schemaVersion": 1,\n  "schemaVersion": 2',
            ),
            '$: invalid experiment JSON: duplicate JSON object member "schemaVersion"',
        ],
    ])('rejects %s into an exact source-preserving incompatible state', async (_label, json, message) => {
        const onReset = vi.fn();
        const replaceDocument = vi.spyOn(usePlaygroundStore.getState(), 'replaceImportedDocument');
        const { container } = render(<ConfigPanel onReset={onReset} />);
        const input = fileInput(container);
        const file = new File([json], 'incompatible.json', {
            type: 'application/json',
        });

        await selectFile(input, file);

        expect(await screen.findByRole('alert')).toHaveTextContent(message);
        expect(replaceDocument).not.toHaveBeenCalled();
        expect(usePlaygroundStore.getState().access).toMatchObject({
            status: 'incompatible',
            prepared: null,
            source: { kind: 'file', file },
        });
        expect(usePlaygroundStore.getState().prepared).toBeNull();
        expect(onReset).not.toHaveBeenCalled();
        expect(input.value).toBe('');
    });

    it('rejects files over the exact UTF-8 byte limit before reading', async () => {
        const onReset = vi.fn();
        const readAsText = vi.spyOn(FileReader.prototype, 'readAsText');
        const replaceDocument = vi.spyOn(usePlaygroundStore.getState(), 'replaceImportedDocument');
        const { container } = render(<ConfigPanel onReset={onReset} />);
        const input = fileInput(container);

        const file = new File(
            [new Uint8Array(MAX_EXPERIMENT_JSON_BYTES + 1)],
            'oversized.json',
            { type: 'application/json' },
        );
        await selectFile(input, file);

        expect(screen.getByRole('alert')).toHaveTextContent(
            `$: experiment JSON exceeds ${MAX_EXPERIMENT_JSON_BYTES} UTF-8 bytes`,
        );
        expect(readAsText).not.toHaveBeenCalled();
        expect(replaceDocument).not.toHaveBeenCalled();
        expect(usePlaygroundStore.getState().access).toMatchObject({
            status: 'incompatible',
            prepared: null,
            source: { kind: 'file', file },
        });
        expect(onReset).not.toHaveBeenCalled();
        expect(input.value).toBe('');
    });

    it('allows a file exactly at the byte limit to reach strict decoding', async () => {
        const originalRead = FileReader.prototype.readAsText;
        const readAsText = vi.spyOn(FileReader.prototype, 'readAsText')
            .mockImplementation(function (blob, encoding) {
                return originalRead.call(this, blob, encoding);
            });
        const { container } = render(<ConfigPanel onReset={vi.fn()} />);

        await selectFile(fileInput(container), new File(
            [new Uint8Array(MAX_EXPERIMENT_JSON_BYTES)],
            'exact-limit.json',
            { type: 'application/json' },
        ));

        const alert = await screen.findByRole('alert');
        expect(readAsText).toHaveBeenCalledTimes(1);
        expect(alert).toHaveTextContent('$: invalid experiment JSON');
        expect(alert).not.toHaveTextContent('exceeds');
    });

    it('announces FileReader failures and resets the file control', async () => {
        vi.spyOn(FileReader.prototype, 'readAsText')
            .mockImplementation(function () {
                this.onerror?.call(this, new ProgressEvent('error'));
            });
        const onReset = vi.fn();
        const before = usePlaygroundStore.getState().prepared!;
        const { container } = render(<ConfigPanel onReset={onReset} />);
        const input = fileInput(container);
        const file = new File(['{}'], 'unreadable.json', {
            type: 'application/json',
        });

        await selectFile(input, file);

        expect(screen.getByRole('alert')).toHaveTextContent('$: Could not read config file');
        expect(usePlaygroundStore.getState().access).toMatchObject({
            status: 'incompatible',
            prepared: null,
            source: { kind: 'file', file },
            issues: [{ path: '$', message: 'Could not read config file' }],
        });
        expect(usePlaygroundStore.getState().access.source.kind === 'file'
            ? usePlaygroundStore.getState().access.source.file
            : null).toBe(file);
        expect(onReset).not.toHaveBeenCalled();
        expect(input.value).toBe('');
    });

    it('releases the import transaction when FileReader aborts', async () => {
        vi.spyOn(FileReader.prototype, 'readAsText')
            .mockImplementation(function () {
                this.onabort?.call(this, new ProgressEvent('abort'));
            });
        const { container } = render(<ConfigPanel onReset={vi.fn()} />);
        const input = fileInput(container);
        const file = new File(['{}'], 'aborted.json', {
            type: 'application/json',
        });

        await selectFile(input, file);

        expect(screen.getByRole('alert')).toHaveTextContent('$: Config file read was canceled');
        expect(screen.getByRole('button', { name: /import json/i })).toBeEnabled();
        expect(usePlaygroundStore.getState().access).toMatchObject({
            status: 'incompatible',
            prepared: null,
            source: { kind: 'file', file },
        });
        expect(input.value).toBe('');
    });

    it('retains the exact file and reports preparation issues', async () => {
        const onReset = vi.fn();
        const before = usePlaygroundStore.getState().prepared!;
        const target = await requirePrepared(documentWithNoise(before, 19));
        vi.spyOn(usePlaygroundStore.getState(), 'replaceImportedDocument').mockResolvedValue({
            ok: false,
            issues: [{
                code: 'incompatible-task',
                path: 'recipe.objective.dataLoss',
                message: 'objective cannot be compiled for this task',
            }],
        });
        const { container } = render(<ConfigPanel onReset={onReset} />);
        const file = new File(
            [encodeExperimentJson(target.document)],
            'unpreparable-v2.json',
            { type: 'application/json' },
        );

        await selectFile(fileInput(container), file);

        expect(await screen.findByRole('alert')).toHaveTextContent(
            'recipe.objective.dataLoss: objective cannot be compiled for this task',
        );
        expect(usePlaygroundStore.getState().access).toMatchObject({
            status: 'incompatible',
            prepared: null,
            source: { kind: 'file', file },
        });
        expect(usePlaygroundStore.getState().prepared).toBeNull();
        expect(onReset).not.toHaveBeenCalled();
    });

    it('does not reset or claim success for a stale successful import result', async () => {
        const onReset = vi.fn();
        const before = usePlaygroundStore.getState().prepared!;
        const imported = await requirePrepared(documentWithNoise(before, 21));
        const newer = documentWithNoise(before, 23);
        const importResult = deferred<SchemaResult<PreparedExperimentDocumentV2>>();
        const originalReplace = usePlaygroundStore.getState().replaceDocument;
        vi.spyOn(usePlaygroundStore.getState(), 'replaceImportedDocument')
            .mockImplementationOnce(() => importResult.promise);
        const { container } = render(<ConfigPanel onReset={onReset} />);

        await selectFile(fileInput(container), new File(
            [encodeExperimentJson(imported.document)],
            'stale-import-v2.json',
            { type: 'application/json' },
        ));
        await waitFor(() => {
            expect(usePlaygroundStore.getState().replaceImportedDocument).toHaveBeenCalledTimes(1);
        });
        const newerResult = await originalReplace(newer);
        expect(newerResult.ok).toBe(true);
        const published = usePlaygroundStore.getState().prepared;

        importResult.resolve({ ok: true, value: imported });
        await act(async () => {
            await importResult.promise;
        });

        expect(usePlaygroundStore.getState().prepared).toBe(published);
        expect(usePlaygroundStore.getState().prepared?.document.recipe.data.noise).toBe(23);
        expect(onReset).not.toHaveBeenCalled();
        expect(screen.queryByText('Imported!')).not.toBeInTheDocument();
    });

    it('deduplicates a second file selection while the first import is pending', async () => {
        const onReset = vi.fn();
        const before = usePlaygroundStore.getState().prepared!;
        const first = await requirePrepared(documentWithNoise(before, 25));
        const second = await requirePrepared(documentWithNoise(before, 27));
        const gate = deferred<void>();
        const originalReplace = usePlaygroundStore.getState().replaceDocument;
        const replaceDocument = vi.spyOn(usePlaygroundStore.getState(), 'replaceImportedDocument')
            .mockImplementation(async (value) => {
                const result = await originalReplace(value);
                await gate.promise;
                return result;
            });
        const { container } = render(<ConfigPanel onReset={onReset} />);
        const input = fileInput(container);

        await selectFile(input, new File(
            [encodeExperimentJson(first.document)],
            'first-v2.json',
            { type: 'application/json' },
        ));
        await waitFor(() => expect(replaceDocument).toHaveBeenCalledTimes(1));
        const disabledWhilePending = screen.getByRole('button', { name: /import json/i })
            .hasAttribute('disabled');

        await selectFile(input, new File(
            [encodeExperimentJson(second.document)],
            'second-v2.json',
            { type: 'application/json' },
        ));
        const callsAfterSecondSelection = replaceDocument.mock.calls.length;

        gate.resolve();
        expect(await screen.findByRole('status')).toHaveTextContent('Imported');

        expect(disabledWhilePending).toBe(true);
        expect(callsAfterSecondSelection).toBe(1);
        expect(usePlaygroundStore.getState().prepared?.document.recipe.data.noise).toBe(25);
        expect(onReset).toHaveBeenCalledTimes(1);
        expect(screen.getByRole('button', { name: /import json/i })).toBeEnabled();
    });

    it('does not apply a file whose read finishes after a newer store edit', async () => {
        const onReset = vi.fn();
        const before = usePlaygroundStore.getState().prepared!;
        const imported = await requirePrepared(documentWithNoise(before, 29));
        const newer = documentWithNoise(before, 31);
        const readers: FileReader[] = [];
        vi.spyOn(FileReader.prototype, 'readAsText').mockImplementation(function () {
            readers.push(this);
        });
        const { container } = render(<ConfigPanel onReset={onReset} />);

        await selectFile(fileInput(container), new File(
            [encodeExperimentJson(imported.document)],
            'slow-read-v2.json',
            { type: 'application/json' },
        ));
        const newerResult = await usePlaygroundStore.getState().replaceDocument(newer);
        expect(newerResult.ok).toBe(true);
        const published = usePlaygroundStore.getState().prepared;

        const reader = readers[0];
        const onload = reader.onload;
        expect(onload).not.toBeNull();
        await act(async () => {
            await onload?.call(reader, {
                target: { result: encodeExperimentJson(imported.document) },
            } as unknown as ProgressEvent<FileReader>);
        });

        expect(usePlaygroundStore.getState().prepared).toBe(published);
        expect(usePlaygroundStore.getState().prepared?.document.recipe.data.noise).toBe(31);
        expect(onReset).not.toHaveBeenCalled();
        expect(screen.queryByText('Imported!')).not.toBeInTheDocument();
    });

    it('does not report an older preparation failure over a newer store edit', async () => {
        const onReset = vi.fn();
        const before = usePlaygroundStore.getState().prepared!;
        const imported = await requirePrepared(documentWithNoise(before, 33));
        const newer = documentWithNoise(before, 35);
        const importResult = deferred<SchemaResult<PreparedExperimentDocumentV2>>();
        const originalReplace = usePlaygroundStore.getState().replaceDocument;
        vi.spyOn(usePlaygroundStore.getState(), 'replaceImportedDocument')
            .mockImplementationOnce(() => importResult.promise);
        const { container } = render(<ConfigPanel onReset={onReset} />);

        await selectFile(fileInput(container), new File(
            [encodeExperimentJson(imported.document)],
            'stale-failure-v2.json',
            { type: 'application/json' },
        ));
        await waitFor(() => {
            expect(usePlaygroundStore.getState().replaceImportedDocument).toHaveBeenCalledTimes(1);
        });
        const newerResult = await originalReplace(newer);
        expect(newerResult.ok).toBe(true);
        const published = usePlaygroundStore.getState().prepared;

        importResult.resolve({
            ok: false,
            issues: [{ code: 'invalid-field', path: 'recipe', message: 'Obsolete failure' }],
        });
        await act(async () => {
            await importResult.promise;
        });

        expect(usePlaygroundStore.getState().prepared).toBe(published);
        expect(screen.queryByRole('alert')).not.toBeInTheDocument();
        expect(onReset).not.toHaveBeenCalled();
    });

    it('does not report an older preparation rejection over a newer store edit', async () => {
        const onReset = vi.fn();
        const before = usePlaygroundStore.getState().prepared!;
        const imported = await requirePrepared(documentWithNoise(before, 37));
        const newer = documentWithNoise(before, 39);
        const importResult = deferred<SchemaResult<PreparedExperimentDocumentV2>>();
        const originalReplace = usePlaygroundStore.getState().replaceDocument;
        vi.spyOn(usePlaygroundStore.getState(), 'replaceImportedDocument')
            .mockImplementationOnce(() => importResult.promise);
        const { container } = render(<ConfigPanel onReset={onReset} />);

        await selectFile(fileInput(container), new File(
            [encodeExperimentJson(imported.document)],
            'stale-rejection-v2.json',
            { type: 'application/json' },
        ));
        await waitFor(() => {
            expect(usePlaygroundStore.getState().replaceImportedDocument).toHaveBeenCalledTimes(1);
        });
        const newerResult = await originalReplace(newer);
        expect(newerResult.ok).toBe(true);
        const published = usePlaygroundStore.getState().prepared;
        const observedRejection = importResult.promise.catch(() => undefined);

        importResult.reject(new Error('Obsolete rejection'));
        await act(async () => {
            await observedRejection;
        });

        expect(usePlaygroundStore.getState().prepared).toBe(published);
        expect(screen.queryByRole('alert')).not.toBeInTheDocument();
        expect(onReset).not.toHaveBeenCalled();
    });
});
