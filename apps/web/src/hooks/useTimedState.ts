// ── useTimedState Hook ──
// Manages state that auto-resets to its default value after a duration.
// Automatically cleans up timeouts on unmount to prevent memory leaks.

import { useState, useRef, useEffect } from 'react';

export function useTimedState<T>(defaultValue: T, duration: number): [T, (value: T) => void] {
    const [value, setValue] = useState<T>(defaultValue);
    const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    useEffect(() => {
        // Clean up any pending timeout on unmount
        return () => {
            if (timeoutRef.current !== null) {
                clearTimeout(timeoutRef.current);
            }
        };
    }, []);

    const setTimed = (newValue: T) => {
        // Clear any pending timeout before setting new value
        if (timeoutRef.current !== null) {
            clearTimeout(timeoutRef.current);
        }

        setValue(newValue);

        // Schedule reset to default value
        timeoutRef.current = setTimeout(() => {
            setValue(defaultValue);
            timeoutRef.current = null;
        }, duration);
    };

    return [value, setTimed];
}
