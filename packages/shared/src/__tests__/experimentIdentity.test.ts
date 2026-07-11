import { describe, expect, it } from 'vitest';
import { canonicalizeJson } from '../canonicalJson.js';

describe('canonicalizeJson', () => {
    it('pins RFC 8785 scalar, nested, escaping, and number serialization', () => {
        const value = {
            numbers: [Number('333333333.33333329'), 1e30, 4.5, 2e-3, 1e-27, -0],
            string: '€$\u000f\nA\'B"\\/',
            literals: [null, true, false],
        };

        expect(canonicalizeJson(value)).toBe(
            '{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27,0],"string":"€$\\u000f\\nA\'B\\"\\\\/"}',
        );
    });

    it('sorts object keys by UTF-16 code units at every nesting level', () => {
        const value = {
            '😀': 'Emoji: Grinning Face',
            '€': 'Euro Sign',
            '\r': 'Carriage Return',
            'ö': 'Latin Small Letter O With Diaeresis',
            '1': 'One',
            '': 'Control',
            nested: { z: 1, a: 2 },
        };

        expect(canonicalizeJson(value)).toBe(
            '{"\\r":"Carriage Return","1":"One","nested":{"a":2,"z":1},"":"Control","ö":"Latin Small Letter O With Diaeresis","€":"Euro Sign","😀":"Emoji: Grinning Face"}',
        );
    });

    it('is invariant to insertion order and supports null-prototype records', () => {
        const first = { z: 3, a: { d: 4, b: 2 } };
        const second = { a: { b: 2, d: 4 }, z: 3 };
        const nullPrototype = Object.assign(Object.create(null) as Record<string, unknown>, {
            z: 3,
            a: { d: 4, b: 2 },
        });

        expect(canonicalizeJson(first)).toBe('{"a":{"b":2,"d":4},"z":3}');
        expect(canonicalizeJson(second)).toBe(canonicalizeJson(first));
        expect(canonicalizeJson(nullPrototype)).toBe(canonicalizeJson(first));
    });

    it.each([
        Number.NaN,
        Number.POSITIVE_INFINITY,
        Number.NEGATIVE_INFINITY,
        undefined,
        () => undefined,
        Symbol('unsupported'),
        1n,
        new Date(0),
        new Number(1),
        new String('value'),
    ])('rejects non-I-JSON value %#', (value) => {
        expect(() => canonicalizeJson(value)).toThrow();
    });

    it.each(['\ud800', '\udc00', `prefix\ud800suffix`])(
        'rejects lone surrogate string %#',
        (value) => {
            expect(() => canonicalizeJson(value)).toThrow(/surrogate|Unicode/iu);
        },
    );

    it('rejects sparse arrays, inherited indexes, and extra properties', () => {
        const sparse = new Array<unknown>(2);
        sparse[1] = true;

        const inherited = new Array<unknown>(1);
        const inheritedPrototype = Object.create(Array.prototype) as unknown[];
        inheritedPrototype[0] = 'inherited';
        Object.setPrototypeOf(inherited, inheritedPrototype);

        const extra = [1, 2] as unknown[] & { note?: string };
        extra.note = 'hidden from JSON arrays';

        expect(() => canonicalizeJson(sparse)).toThrow(/dense|index/iu);
        expect(() => canonicalizeJson(inherited)).toThrow(/array|prototype|index/iu);
        expect(() => canonicalizeJson(extra)).toThrow(/property|array/iu);
    });

    it('rejects hidden, symbol, accessor, and inherited object state', () => {
        const hidden = { visible: true };
        Object.defineProperty(hidden, 'secret', { value: 1, enumerable: false });

        const symbol = { visible: true } as Record<PropertyKey, unknown>;
        symbol[Symbol('secret')] = 1;

        let getterCalls = 0;
        const accessor = {} as Record<string, unknown>;
        Object.defineProperty(accessor, 'value', {
            enumerable: true,
            get: () => {
                getterCalls += 1;
                return 1;
            },
        });

        const inherited = Object.create({ inherited: true }) as Record<string, unknown>;
        inherited.visible = true;

        expect(() => canonicalizeJson(hidden)).toThrow(/enumerable|property/iu);
        expect(() => canonicalizeJson(symbol)).toThrow(/symbol/iu);
        expect(() => canonicalizeJson(accessor)).toThrow(/accessor|data property/iu);
        expect(getterCalls).toBe(0);
        expect(() => canonicalizeJson(inherited)).toThrow(/plain|prototype/iu);
    });

    it('rejects cycles but permits repeated acyclic references', () => {
        const cyclic: { self?: unknown } = {};
        cyclic.self = cyclic;
        const shared = { value: 1 };

        expect(() => canonicalizeJson(cyclic)).toThrow(/cyclic/iu);
        expect(canonicalizeJson([shared, shared])).toBe(
            '[{"value":1},{"value":1}]',
        );
    });
});
