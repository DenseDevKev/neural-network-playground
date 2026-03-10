// ── Seeded pseudo-random number generator ──
// Uses mulberry32 — fast, deterministic, 32-bit state.

export class PRNG {
    private state: number;

    constructor(seed: number) {
        this.state = seed | 0;
    }

    /** Returns a float in [0, 1). */
    next(): number {
        let t = (this.state += 0x6d2b79f5);
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }

    /** Returns float in [min, max). */
    range(min: number, max: number): number {
        return min + this.next() * (max - min);
    }

    /** Standard normal via Box-Muller. */
    gaussian(mean = 0, std = 1): number {
        const u1 = this.next();
        const u2 = this.next();
        const z = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
        return mean + z * std;
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
        return new PRNG(Math.floor(this.next() * 2147483647));
    }
}
