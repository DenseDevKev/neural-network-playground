// ── useTimedState Hook ──
// Manages state that auto-resets to its default value after a duration.
// Automatically cleans up timeouts on unmount to prevent memory leaks.

import { useCallback, useEffect, useRef, useState } from 'react';

export function useTimedState<T>(defaultValue: T, duration: number): [T, (value: T) => void] {
    const [value, setValue] = useState<T>(defaultValue);
    const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const defaultValueRef = useRef(defaultValue);
    const durationRef = useRef(duration);
    const mountedRef = useRef(true);

    defaultValueRef.current = defaultValue;
    durationRef.current = duration;

    const clearPendingTimeout = useCallback(() => {
        if (timeoutRef.current === null) return;
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
    }, []);

    useEffect(() => {
        mountedRef.current = true;
        return () => {
            mountedRef.current = false;
            clearPendingTimeout();
        };
    }, [clearPendingTimeout]);

    const setTimed = useCallback((next: T) => {
        if (!mountedRef.current) return;
        clearPendingTimeout();
        setValue(next);
        const timeoutId = setTimeout(() => {
            if (timeoutRef.current !== timeoutId) return;
            timeoutRef.current = null;
            if (mountedRef.current) setValue(defaultValueRef.current);
        }, durationRef.current);
        timeoutRef.current = timeoutId;
    }, [clearPendingTimeout]);

    return [value, setTimed];
}
