// ── Seeded pseudo-random number generator ──
// Uses mulberry32 — fast, deterministic, 32-bit state.

const UINT32_MAX = 0xffff_ffff;
const GAUSSIAN_STANDARD_DEVIATION_LIMIT = 3;

export function normalizeUint32Seed(seed: number): number {
    if (!Number.isInteger(seed) || seed < 0 || seed > UINT32_MAX) {
        throw new RangeError('Seed must be an unsigned 32-bit integer from 0 through 4294967295');
    }
    return seed;
}

export class PRNG {
    private state: number;

    constructor(seed: number) {
        this.state = normalizeUint32Seed(seed);
    }

    /** Returns a float in [0, 1). */
    next(): number {
        this.state = (this.state + 0x6d2b79f5) >>> 0;
        let t = this.state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }

    /** Returns float in [min, max). */
    range(min: number, max: number): number {
        return min + this.next() * (max - min);
    }

    /** Gaussian via Box-Muller with deterministic rejection outside mean ± 3σ. */
    gaussian(mean = 0, std = 1): number {
        if (!Number.isFinite(mean) || !Number.isFinite(std) || std < 0) {
            throw new RangeError('Gaussian mean must be finite and standard deviation must be finite and non-negative');
        }

        const minimum = mean - GAUSSIAN_STANDARD_DEVIATION_LIMIT * std;
        const maximum = mean + GAUSSIAN_STANDARD_DEVIATION_LIMIT * std;
        while (true) {
            const u1 = 1 - this.next();
            const u2 = this.next();
            const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            const value = mean + z * std;
            if (value >= minimum && value <= maximum) return value;
        }
    }

    /** Shuffle array in place (Fisher-Yates). */
    shuffle<T>(arr: T[]): T[] {
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(this.next() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
        return arr;
    }

    /** Fork a new PRNG with a derived seed. */
    fork(): PRNG {
        return new PRNG(Math.floor(this.next() * 4294967296));
    }
}
