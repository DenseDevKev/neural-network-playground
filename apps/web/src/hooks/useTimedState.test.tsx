import { StrictMode, type ReactNode } from 'react';
import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, expectTypeOf, it, vi } from 'vitest';
import { useTimedState } from './useTimedState';

function StrictWrapper({ children }: { children: ReactNode }) {
    return <StrictMode>{children}</StrictMode>;
}

describe('useTimedState', () => {
    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.clearAllTimers();
        vi.restoreAllMocks();
        vi.useRealTimers();
    });

    it('preserves its tuple contract and setter identity while resetting to the latest default', () => {
        const { result, rerender } = renderHook(
            ({ defaultValue, duration }: { defaultValue: string; duration: number }) =>
                useTimedState(defaultValue, duration),
            {
                initialProps: { defaultValue: 'idle', duration: 100 },
                wrapper: StrictWrapper,
            },
        );
        const firstSetter = result.current[1];

        expectTypeOf(result.current).toEqualTypeOf<[string, (value: string) => void]>();
        expect(result.current[0]).toBe('idle');

        act(() => firstSetter('saved'));
        rerender({ defaultValue: 'ready', duration: 50 });

        expect(result.current[0]).toBe('saved');
        expect(result.current[1]).toBe(firstSetter);

        act(() => vi.advanceTimersByTime(100));

        expect(result.current[0]).toBe('ready');
    });

    it('keeps an active deadline while a retained setter uses the latest duration later', () => {
        const { result, rerender } = renderHook(
            ({ duration }: { duration: number }) => useTimedState('idle', duration),
            {
                initialProps: { duration: 100 },
                wrapper: StrictWrapper,
            },
        );
        const retainedSetter = result.current[1];

        act(() => retainedSetter('first'));
        act(() => vi.advanceTimersByTime(40));
        rerender({ duration: 10 });

        act(() => vi.advanceTimersByTime(59));
        expect(result.current[0]).toBe('first');

        act(() => vi.advanceTimersByTime(1));
        expect(result.current[0]).toBe('idle');

        act(() => retainedSetter('second'));
        act(() => vi.advanceTimersByTime(9));
        expect(result.current[0]).toBe('second');

        act(() => vi.advanceTimersByTime(1));
        expect(result.current[0]).toBe('idle');
    });

    it('clears replacement and unmount timers once and makes a retained setter inert', () => {
        const { result, unmount } = renderHook(() => useTimedState('idle', 100), {
            wrapper: StrictWrapper,
        });
        const retainedSetter = result.current[1];
        const clearTimeoutSpy = vi.spyOn(globalThis, 'clearTimeout');

        act(() => retainedSetter('first'));
        expect(vi.getTimerCount()).toBe(1);

        act(() => retainedSetter('replacement'));
        expect(clearTimeoutSpy).toHaveBeenCalledTimes(1);
        expect(vi.getTimerCount()).toBe(1);

        unmount();
        expect(clearTimeoutSpy).toHaveBeenCalledTimes(2);
        expect(vi.getTimerCount()).toBe(0);

        act(() => retainedSetter('late'));
        expect(clearTimeoutSpy).toHaveBeenCalledTimes(2);
        expect(vi.getTimerCount()).toBe(0);
    });

    it('rejects a captured stale callback after a replacement takes timer ownership', () => {
        const { result } = renderHook(() => useTimedState('idle', 100), {
            wrapper: StrictWrapper,
        });
        const retainedSetter = result.current[1];
        const fakeSetTimeout = globalThis.setTimeout;
        const setTimeoutSpy = vi
            .spyOn(globalThis, 'setTimeout')
            .mockImplementation(fakeSetTimeout);

        act(() => retainedSetter('first'));
        const staleCallback = setTimeoutSpy.mock.calls.find((call) => call[1] === 100)?.[0];
        setTimeoutSpy.mockRestore();

        expect(typeof staleCallback).toBe('function');

        act(() => retainedSetter('replacement'));
        act(() => {
            if (typeof staleCallback === 'function') staleCallback();
        });

        expect(result.current[0]).toBe('replacement');
        expect(vi.getTimerCount()).toBe(1);

        act(() => vi.advanceTimersByTime(100));

        expect(result.current[0]).toBe('idle');
        expect(vi.getTimerCount()).toBe(0);
    });
});
