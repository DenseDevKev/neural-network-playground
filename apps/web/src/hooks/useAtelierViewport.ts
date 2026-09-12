import { useLayoutEffect, useState } from 'react';

function readViewport() {
    const zoom = Number.parseFloat(getComputedStyle(document.documentElement).zoom) || 1;
    const width = window.innerWidth / zoom;
    const height = (window.visualViewport?.height ?? window.innerHeight) / zoom;
    return { width, height, compact: width < 760, short: height <= 600 };
}

/** CSS document zoom and the virtual keyboard also reduce the usable viewport. */
export function useAtelierViewport() {
    const [viewport, setViewport] = useState(readViewport);
    useLayoutEffect(() => {
        const root = document.documentElement;
        let frame: number | null = null;
        let dimensions: { width: number; height: number } | null = null;
        const sync = () => {
            frame = null;
            const next = readViewport();
            if (dimensions?.width === next.width && dimensions.height === next.height) return;
            dimensions = next;
            root.style.setProperty('--atelier-viewport-width', `${next.width}px`);
            root.style.setProperty('--atelier-viewport-height', `${next.height}px`);
            setViewport((current) => current.width === next.width && current.height === next.height ? current : next);
        };
        // ResizeObserver callbacks must not write dimensions during delivery.
        // Coalesce all resize sources before the next layout, and only publish changes.
        const schedule = () => {
            if (frame === null) frame = requestAnimationFrame(sync);
        };
        sync();
        window.addEventListener('resize', schedule);
        window.visualViewport?.addEventListener('resize', schedule);
        // A document zoom change affects layout without dispatching a window resize.
        const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(schedule);
        observer?.observe(root);
        return () => {
            observer?.disconnect();
            if (frame !== null) cancelAnimationFrame(frame);
            window.removeEventListener('resize', schedule);
            window.visualViewport?.removeEventListener('resize', schedule);
            root.style.removeProperty('--atelier-viewport-width');
            root.style.removeProperty('--atelier-viewport-height');
        };
    }, []);
    return viewport;
}
