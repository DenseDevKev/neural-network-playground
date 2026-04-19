// ── WebGPU capability detection (AS-4) ──────────────────────────────────────
// Pure async function returning a `GPUDevice` when WebGPU is available in
// this global, or `null` otherwise. Designed to never throw — every failure
// mode (missing API, adapter denial, device denial) collapses to a
// `null` return so callers can branch once and forget.
//
// Cached per process: repeated calls return the same device. WebGPU
// devices are designed to be long-lived and cheap to share across
// pipelines, so caching here matches real-world usage.

import type { GPUDevice, GPUNavigator } from './types.js';

let _cached: GPUDevice | null | undefined; // undefined = not yet attempted

/**
 * Best-effort WebGPU device acquisition. Returns the cached device on
 * subsequent calls. The first call may fail for several reasons:
 *   - `navigator.gpu` doesn't exist (Node, older browsers, Firefox without flag)
 *   - `requestAdapter()` returns null (no compatible adapter)
 *   - `requestDevice()` rejects (driver / sandbox denied)
 * In every failure mode we cache `null` so we don't keep re-trying on the
 * hot path; callers fall back to the CPU predictor without cost.
 */
export async function detectWebGPU(): Promise<GPUDevice | null> {
    if (_cached !== undefined) return _cached;

    const nav = (globalThis as unknown as { navigator?: { gpu?: GPUNavigator } }).navigator;
    const gpu = nav?.gpu;
    if (!gpu) {
        _cached = null;
        return null;
    }

    try {
        // low-power: this is a visualization, not a training accelerator —
        // we don't want to spin up the user's discrete GPU.
        const adapter = await gpu.requestAdapter({ powerPreference: 'low-power' });
        if (!adapter) {
            _cached = null;
            return null;
        }
        // The grid-predictor shader uses 9 storage buffers which exceeds
        // the default per-stage limit of 8. Request a higher limit from
        // the adapter when available.
        const adapterLimit = (adapter as any).limits?.maxStorageBuffersPerShaderStage ?? 8;
        const requiredStorageBuffers = Math.min(adapterLimit, 10);
        const device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBuffersPerShaderStage: requiredStorageBuffers,
            },
        });
        _cached = device;
        return device;
    } catch {
        _cached = null;
        return null;
    }
}

/** Test-only: forget the cached device so a subsequent detect re-runs. */
export function resetWebGPUDetectionCache(): void {
    _cached = undefined;
}
