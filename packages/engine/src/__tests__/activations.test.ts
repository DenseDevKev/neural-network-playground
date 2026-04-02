import { describe, it, expect } from 'vitest';
import { getActivation } from '../activations.js';
import type { ActivationType } from '../types.js';

describe('getActivation', () => {
    it('returns a function pair for all known types', () => {
        const types: ActivationType[] = [
            'relu', 'tanh', 'sigmoid', 'linear',
            'leakyRelu', 'elu', 'swish', 'softplus',
        ];
        for (const t of types) {
            const act = getActivation(t);
            expect(act).toBeDefined();
            expect(typeof act.f).toBe('function');
            expect(typeof act.df).toBe('function');
        }
    });
});

describe('ReLU', () => {
    const relu = getActivation('relu');

    it('passes positive values through', () => {
        expect(relu.f(2)).toBe(2);
        expect(relu.f(0.5)).toBe(0.5);
    });

    it('clamps negative values to 0', () => {
        expect(relu.f(-1)).toBe(0);
        expect(relu.f(-100)).toBe(0);
    });

    it('returns 0 for x=0', () => {
        expect(relu.f(0)).toBe(0);
    });

    it('derivative is 1 for positive output', () => {
        expect(relu.df(1, 1)).toBe(1);
    });

    it('derivative is 0 for zero/negative output', () => {
        expect(relu.df(-1, 0)).toBe(0);
    });
});

describe('Tanh', () => {
    const tanh = getActivation('tanh');

    it('maps 0 to 0', () => {
        expect(tanh.f(0)).toBeCloseTo(0, 10);
    });

    it('output is in (-1, 1)', () => {
        expect(tanh.f(10)).toBeCloseTo(1, 4);
        expect(tanh.f(-10)).toBeCloseTo(-1, 4);
    });

    it('derivative at output=0 is 1', () => {
        expect(tanh.df(0, 0)).toBeCloseTo(1, 10);
    });

    it('derivative satisfies 1 - output²', () => {
        const out = tanh.f(0.5);
        expect(tanh.df(0.5, out)).toBeCloseTo(1 - out * out, 8);
    });
});

describe('Sigmoid', () => {
    const sigmoid = getActivation('sigmoid');

    it('maps 0 to 0.5', () => {
        expect(sigmoid.f(0)).toBeCloseTo(0.5, 10);
    });

    it('output approaches 1 for large positive input', () => {
        expect(sigmoid.f(20)).toBeCloseTo(1, 4);
    });

    it('output approaches 0 for large negative input', () => {
        expect(sigmoid.f(-20)).toBeCloseTo(0, 4);
    });

    it('derivative satisfies output * (1 - output)', () => {
        const out = sigmoid.f(1);
        expect(sigmoid.df(1, out)).toBeCloseTo(out * (1 - out), 8);
    });
});

describe('Linear', () => {
    const linear = getActivation('linear');

    it('passes through unchanged', () => {
        expect(linear.f(42)).toBe(42);
        expect(linear.f(-3.14)).toBe(-3.14);
    });

    it('derivative is always 1', () => {
        expect(linear.df(0, 0)).toBe(1);
        expect(linear.df(100, 100)).toBe(1);
    });
});

describe('Leaky ReLU', () => {
    const lrelu = getActivation('leakyRelu');

    it('passes positive values through', () => {
        expect(lrelu.f(5)).toBe(5);
    });

    it('scales negative values by 0.01', () => {
        expect(lrelu.f(-10)).toBeCloseTo(-0.1, 8);
    });

    it('derivative is 1 for positive, 0.01 for negative', () => {
        expect(lrelu.df(1, 1)).toBe(1);
        expect(lrelu.df(-1, -0.01)).toBeCloseTo(0.01, 8);
    });
});

describe('ELU', () => {
    const elu = getActivation('elu');

    it('passes positive values through', () => {
        expect(elu.f(3)).toBe(3);
    });

    it('maps negative values via exp(x) - 1', () => {
        expect(elu.f(-1)).toBeCloseTo(Math.exp(-1) - 1, 8);
    });

    it('derivative is 1 for positive, output + 1 for negative', () => {
        expect(elu.df(1, 1)).toBe(1);
        const out = elu.f(-0.5);
        expect(elu.df(-0.5, out)).toBeCloseTo(out + 1, 8);
    });
});

describe('Softplus', () => {
    const sp = getActivation('softplus');

    it('approximates ReLU smoothly', () => {
        expect(sp.f(10)).toBeCloseTo(10, 2);
        expect(sp.f(-10)).toBeCloseTo(0, 2);
    });

    it('f(0) = ln(2)', () => {
        expect(sp.f(0)).toBeCloseTo(Math.log(2), 8);
    });

    it('derivative is sigmoid', () => {
        const sigmoid = 1 / (1 + Math.exp(-2));
        expect(sp.df(2, sp.f(2))).toBeCloseTo(sigmoid, 8);
    });
});

describe('Numerical gradient check (all activations)', () => {
    const h = 1e-5;
    const types: ActivationType[] = [
        'relu', 'tanh', 'sigmoid', 'linear',
        'leakyRelu', 'elu', 'swish', 'softplus',
    ];

    // Test at a few x values (skip x=0 for relu-family discontinuities)
    const testPoints = [0.5, -0.5, 1, -1, 2];

    for (const t of types) {
        it(`${t}: analytical derivative matches numerical approximation`, () => {
            const act = getActivation(t);
            for (const x of testPoints) {
                // Skip discontinuity at 0 for relu/leakyRelu
                if (x === 0 && (t === 'relu' || t === 'leakyRelu')) continue;

                const numGrad = (act.f(x + h) - act.f(x - h)) / (2 * h);
                const output = act.f(x);
                const analyticalGrad = act.df(x, output);
                expect(analyticalGrad).toBeCloseTo(numGrad, 3);
            }
        });
    }
});
