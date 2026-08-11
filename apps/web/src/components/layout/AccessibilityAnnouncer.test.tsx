import { act, render, screen } from '@testing-library/react';
import { PREPARED_PRESETS } from '@nn-playground/shared';
import type { ComponentProps } from 'react';
import { describe, expect, it } from 'vitest';
import { AccessibilityAnnouncer } from './AccessibilityAnnouncer';

type AnnouncerProps = ComponentProps<typeof AccessibilityAnnouncer>;

const recipe = PREPARED_PRESETS[0].prepared.document.recipe;
const initialRecipePublication = structuredClone(recipe);

function props(overrides: Partial<AnnouncerProps> = {}): AnnouncerProps {
    return {
        status: 'idle',
        pauseReason: null,
        workerError: null,
        pendingConfigSource: null,
        configError: null,
        configErrorSource: null,
        evidenceGenerationId: 1,
        trainedRecipe: initialRecipePublication,
        trainedRecipeSource: 'initialize',
        ...overrides,
    };
}

async function flushMutations() {
    await act(async () => Promise.resolve());
}

describe('AccessibilityAnnouncer', () => {
    it('owns one named polite atomic region that starts empty', () => {
        render(<AccessibilityAnnouncer {...props()} />);

        const region = screen.getByRole('status', {
            name: 'Training and configuration announcements',
        });
        expect(region).toHaveAttribute('aria-live', 'polite');
        expect(region).toHaveAttribute('aria-atomic', 'true');
        expect(region).toBeEmptyDOMElement();
        expect(screen.getAllByRole('status')).toHaveLength(1);
    });

    it.each([
        { status: 'running' as const },
        { pendingConfigSource: 'data' as const },
        { configError: 'already failed', configErrorSource: 'network' as const },
    ])('does not announce state that already existed at mount: $status$pendingConfigSource$configError', (initial) => {
        render(<AccessibilityAnnouncer {...props(initial)} />);

        expect(screen.getByRole('status', {
            name: 'Training and configuration announcements',
        })).toBeEmptyDOMElement();
    });

    it.each([
        ['data', 'Generating data', 'Data update complete', 'Data error: failed'],
        ['network', 'Initializing network', 'Network update complete', 'Network error: failed'],
        ['features', 'Updating features', 'Features update complete', 'Features error: failed'],
        ['training', 'Updating training', 'Training update complete', 'Training error: failed'],
        ['preset', 'Applying preset', 'Preset update complete', 'Preset error: failed'],
    ] as const)(
        'announces one owned %s interval across start, completion, failure, coalescing, and retry',
        async (source, startMessage, completionMessage, errorMessage) => {
            const { rerender } = render(<AccessibilityAnnouncer {...props()} />);
            const region = screen.getByRole('status', {
                name: 'Training and configuration announcements',
            });
            const writes: MutationRecord[] = [];
            const observer = new MutationObserver((records) => writes.push(...records));
            observer.observe(region, { childList: true, characterData: true, subtree: true });
            await flushMutations();
            expect(writes).toHaveLength(0);

            rerender(<AccessibilityAnnouncer {...props({ pendingConfigSource: source })} />);
            expect(region).toHaveTextContent(startMessage);
            await flushMutations();
            const startWrites = writes.length;
            expect(startWrites).toBe(1);

            rerender(<AccessibilityAnnouncer {...props({ pendingConfigSource: source })} />);
            await flushMutations();
            expect(writes).toHaveLength(startWrites);

            rerender(<AccessibilityAnnouncer {...props()} />);
            expect(region).toHaveTextContent(completionMessage);
            await flushMutations();
            expect(writes).toHaveLength(startWrites + 1);

            rerender(<AccessibilityAnnouncer {...props({ pendingConfigSource: source })} />);
            expect(region).toHaveTextContent(startMessage);
            await flushMutations();
            expect(writes).toHaveLength(startWrites + 2);
            rerender(<AccessibilityAnnouncer {...props({
                configError: 'failed',
                configErrorSource: source,
            })} />);
            expect(region).toHaveTextContent(errorMessage);
            expect(region).not.toHaveTextContent(completionMessage);
            await flushMutations();
            expect(writes).toHaveLength(startWrites + 3);

            const errorWrites = writes.length;
            rerender(<AccessibilityAnnouncer {...props({
                configError: 'failed',
                configErrorSource: source,
            })} />);
            await flushMutations();
            expect(writes).toHaveLength(errorWrites);

            rerender(<AccessibilityAnnouncer {...props({ pendingConfigSource: source })} />);
            expect(region).toHaveTextContent(startMessage);
            await flushMutations();
            expect(writes).toHaveLength(errorWrites + 1);
            rerender(<AccessibilityAnnouncer {...props({
                configError: 'failed',
                configErrorSource: source,
            })} />);
            expect(region).toHaveTextContent(errorMessage);
            await flushMutations();
            expect(writes).toHaveLength(errorWrites + 2);

            observer.disconnect();
        },
    );

    it('announces a superseding scope once without completing or replaying the superseded interval', async () => {
        const { rerender } = render(<AccessibilityAnnouncer {...props()} />);
        const region = screen.getByRole('status', {
            name: 'Training and configuration announcements',
        });
        const writes: MutationRecord[] = [];
        const observer = new MutationObserver((records) => writes.push(...records));
        observer.observe(region, { childList: true, characterData: true, subtree: true });
        await flushMutations();
        expect(writes).toHaveLength(0);

        rerender(<AccessibilityAnnouncer {...props({ pendingConfigSource: 'data' })} />);
        expect(region).toHaveTextContent('Generating data');
        await flushMutations();
        expect(writes).toHaveLength(1);
        rerender(<AccessibilityAnnouncer {...props({ pendingConfigSource: 'network' })} />);
        expect(region).toHaveTextContent('Initializing network');
        expect(region).not.toHaveTextContent('Data update complete');
        await flushMutations();
        expect(writes).toHaveLength(2);

        rerender(<AccessibilityAnnouncer {...props({ pendingConfigSource: 'network' })} />);
        await flushMutations();
        expect(writes).toHaveLength(2);
        observer.disconnect();
    });

    it('treats the error source and message as identity', async () => {
        const { rerender } = render(<AccessibilityAnnouncer {...props()} />);
        const region = screen.getByRole('status', {
            name: 'Training and configuration announcements',
        });
        const writes: MutationRecord[] = [];
        const observer = new MutationObserver((records) => writes.push(...records));
        observer.observe(region, { childList: true, characterData: true, subtree: true });
        await flushMutations();
        expect(writes).toHaveLength(0);

        rerender(<AccessibilityAnnouncer {...props({
            configError: 'same failure',
            configErrorSource: 'data',
        })} />);
        await flushMutations();
        expect(region).toHaveTextContent('Data error: same failure');
        expect(writes).toHaveLength(1);

        rerender(<AccessibilityAnnouncer {...props({
            configError: 'same failure',
            configErrorSource: 'network',
        })} />);
        await flushMutations();
        expect(region).toHaveTextContent('Network error: same failure');
        expect(writes).toHaveLength(2);

        rerender(<AccessibilityAnnouncer {...props({ pendingConfigSource: 'network' })} />);
        rerender(<AccessibilityAnnouncer {...props({
            configError: 'same failure',
            configErrorSource: 'network',
        })} />);
        await flushMutations();
        expect(region).toHaveTextContent('Network error: same failure');
        expect(writes).toHaveLength(4);
        observer.disconnect();
    });

    it('uses deterministic priority and suppresses internal and duplicate pause announcements', () => {
        const { rerender } = render(<AccessibilityAnnouncer {...props()} />);
        const region = screen.getByRole('status', {
            name: 'Training and configuration announcements',
        });

        rerender(<AccessibilityAnnouncer {...props({ status: 'running' })} />);
        expect(region).toHaveTextContent('Training started');

        rerender(<AccessibilityAnnouncer {...props({ status: 'paused', pauseReason: 'manual' })} />);
        expect(region).toHaveTextContent('Training paused');

        rerender(<AccessibilityAnnouncer {...props({ status: 'running' })} />);
        rerender(<AccessibilityAnnouncer {...props({ status: 'paused', pauseReason: 'target-loss-reached' })} />);
        expect(region).toHaveTextContent('Training paused');

        rerender(<AccessibilityAnnouncer {...props({ status: 'running' })} />);
        rerender(<AccessibilityAnnouncer {...props({ status: 'paused', pauseReason: null, pendingConfigSource: 'data' })} />);
        expect(region).toHaveTextContent('Generating data');
        rerender(<AccessibilityAnnouncer {...props({ status: 'idle', pauseReason: null })} />);
        expect(region).toHaveTextContent('Data update complete');

        rerender(<AccessibilityAnnouncer {...props({ status: 'running' })} />);
        rerender(<AccessibilityAnnouncer {...props({
            status: 'paused',
            pauseReason: 'error',
            workerError: 'worker failed',
        })} />);
        expect(region).toHaveTextContent('Training started');

        rerender(<AccessibilityAnnouncer {...props({
            status: 'running',
            pendingConfigSource: 'network',
            configError: 'failed',
            configErrorSource: 'data',
        })} />);
        expect(region).toHaveTextContent('Data error: failed');
    });

    it('announces split reset publications exactly once and ignores stale reset sources', async () => {
        const { rerender } = render(<AccessibilityAnnouncer {...props()} />);
        const region = screen.getByRole('status', {
            name: 'Training and configuration announcements',
        });
        const writes: MutationRecord[] = [];
        const observer = new MutationObserver((records) => writes.push(...records));
        observer.observe(region, { childList: true, characterData: true, subtree: true });
        await flushMutations();
        expect(writes).toHaveLength(0);

        rerender(<AccessibilityAnnouncer {...props({ evidenceGenerationId: 2 })} />);
        await flushMutations();
        expect(writes).toHaveLength(0);

        const firstResetPublication = structuredClone(recipe);
        rerender(<AccessibilityAnnouncer {...props({
            evidenceGenerationId: 2,
            trainedRecipe: firstResetPublication,
            trainedRecipeSource: 'reset',
        })} />);
        expect(region).toHaveTextContent('Training reset');
        await flushMutations();
        const firstResetWrites = writes.length;
        expect(firstResetWrites).toBe(1);

        const sameGenerationResetPublication = structuredClone(recipe);
        rerender(<AccessibilityAnnouncer {...props({
            evidenceGenerationId: 2,
            trainedRecipe: sameGenerationResetPublication,
            trainedRecipeSource: 'reset',
        })} />);
        await flushMutations();
        expect(writes).toHaveLength(firstResetWrites);

        rerender(<AccessibilityAnnouncer {...props({
            evidenceGenerationId: 3,
            trainedRecipe: sameGenerationResetPublication,
            trainedRecipeSource: 'reset',
        })} />);
        const configSyncPublication = structuredClone(recipe);
        rerender(<AccessibilityAnnouncer {...props({
            evidenceGenerationId: 3,
            trainedRecipe: configSyncPublication,
            trainedRecipeSource: 'config-sync',
        })} />);
        await flushMutations();
        expect(writes).toHaveLength(firstResetWrites);

        rerender(<AccessibilityAnnouncer {...props({
            evidenceGenerationId: 4,
            trainedRecipe: configSyncPublication,
            trainedRecipeSource: 'config-sync',
        })} />);
        const secondResetPublication = structuredClone(recipe);
        rerender(<AccessibilityAnnouncer {...props({
            evidenceGenerationId: 4,
            trainedRecipe: secondResetPublication,
            trainedRecipeSource: 'reset',
        })} />);
        await flushMutations();
        expect(writes).toHaveLength(firstResetWrites + 1);
        const secondResetWrites = writes.length;

        rerender(<AccessibilityAnnouncer {...props({
            evidenceGenerationId: 4,
            trainedRecipe: secondResetPublication,
            trainedRecipeSource: 'reset',
        })} />);
        await flushMutations();
        expect(writes).toHaveLength(secondResetWrites);

        observer.disconnect();
    });
});
