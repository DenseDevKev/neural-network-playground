/**
 * Serialize an I-JSON value according to RFC 8785 (JSON Canonicalization Scheme).
 *
 * The function intentionally accepts `unknown`: identities are only trustworthy
 * when unsupported JavaScript state is rejected instead of being silently
 * omitted by `JSON.stringify`.
 */
export class StableJsonSnapshotError extends TypeError {
    constructor(
        readonly path: string,
        message: string,
    ) {
        super(message);
        this.name = 'StableJsonSnapshotError';
    }
}

/**
 * Capture one immutable plain-data view without invoking accessors. Proxy traps
 * may participate in this single capture, but every later consumer receives
 * only the detached snapshot, so validation, compilation, and identity cannot
 * observe different versions of the input.
 */
export function createStableJsonSnapshot<T>(value: T): T {
    return snapshotValue(value, new Set<object>(), '') as T;
}

export function canonicalizeJson(value: unknown): string {
    const snapshot = createStableJsonSnapshot(value);
    return serializeCanonicalValue(snapshot, new Set<object>());
}

function snapshotValue(
    value: unknown,
    ancestors: Set<object>,
    path: string,
): unknown {
    if ((typeof value !== 'object' && typeof value !== 'function') || value === null) {
        return value;
    }
    if (typeof value === 'function') return value;
    if (ancestors.has(value)) {
        throw snapshotError(path, 'plain-data snapshots cannot contain cycles');
    }

    ancestors.add(value);
    try {
        return Array.isArray(value)
            ? snapshotArray(value, ancestors, path)
            : snapshotRecord(value, ancestors, path);
    } finally {
        ancestors.delete(value);
    }
}

function snapshotArray(
    value: unknown[],
    ancestors: Set<object>,
    path: string,
): readonly unknown[] {
    if (Object.getPrototypeOf(value) !== Array.prototype) {
        throw snapshotError(path, 'arrays must use a standard Array prototype');
    }
    const descriptors = new Map<number, PropertyDescriptor>();
    for (const key of Reflect.ownKeys(value)) {
        if (typeof key === 'symbol') {
            throw snapshotError(path, 'arrays cannot contain symbol properties');
        }
        if (key === 'length') continue;
        const index = canonicalArrayIndex(key, value.length);
        if (index === undefined) {
            throw snapshotError(
                propertyPath(path, key),
                'arrays cannot contain non-index properties',
            );
        }
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (!descriptor || !descriptor.enumerable) {
            throw snapshotError(arrayPath(path, index), 'array indexes must be enumerable');
        }
        if (!('value' in descriptor)) {
            throw snapshotError(arrayPath(path, index), 'array accessors are not allowed');
        }
        descriptors.set(index, descriptor);
    }

    const snapshot: unknown[] = [];
    for (let index = 0; index < value.length; index += 1) {
        const descriptor = descriptors.get(index);
        if (!descriptor) {
            throw snapshotError(arrayPath(path, index), 'arrays must contain every own index');
        }
        snapshot.push(snapshotValue(descriptor.value, ancestors, arrayPath(path, index)));
    }
    return Object.freeze(snapshot);
}

function snapshotRecord(
    value: object,
    ancestors: Set<object>,
    path: string,
): Readonly<Record<string, unknown>> {
    if (!isPlainRecordPrototype(Object.getPrototypeOf(value))) {
        throw snapshotError(path, 'objects must be plain records');
    }

    const snapshot: Record<string, unknown> = {};
    for (const key of Reflect.ownKeys(value)) {
        if (typeof key === 'symbol') {
            throw snapshotError(path, 'objects cannot contain symbol properties');
        }
        const child = propertyPath(path, key);
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (!descriptor || !descriptor.enumerable) {
            throw snapshotError(child, 'object properties must be enumerable');
        }
        if (!('value' in descriptor)) {
            throw snapshotError(child, 'object accessors are not allowed');
        }
        Object.defineProperty(snapshot, key, {
            value: snapshotValue(descriptor.value, ancestors, child),
            enumerable: true,
            configurable: false,
            writable: false,
        });
    }
    return Object.freeze(snapshot);
}

function isPlainRecordPrototype(prototype: object | null): boolean {
    return prototype === null || prototype === Object.prototype;
}

function propertyPath(path: string, key: string): string {
    return path ? `${path}.${key}` : key;
}

function arrayPath(path: string, index: number): string {
    return `${path}[${index}]`;
}

function snapshotError(path: string, message: string): StableJsonSnapshotError {
    const displayPath = path || '$';
    return new StableJsonSnapshotError(displayPath, `${displayPath}: ${message}`);
}

function serializeCanonicalValue(value: unknown, ancestors: Set<object>): string {
    if (value === null) return 'null';

    switch (typeof value) {
        case 'boolean':
            return value ? 'true' : 'false';
        case 'number':
            if (!Number.isFinite(value)) {
                throw new TypeError('Canonical JSON numbers must be finite');
            }
            // JSON.stringify uses the ECMAScript number serialization required
            // by RFC 8785, including canonicalizing negative zero to zero.
            return JSON.stringify(value);
        case 'string':
            assertUnicodeScalarString(value);
            return JSON.stringify(value);
        case 'object':
            return serializeCanonicalObject(value, ancestors);
        case 'undefined':
        case 'function':
        case 'symbol':
        case 'bigint':
            throw new TypeError(`Canonical JSON does not support ${typeof value}`);
        default:
            throw new TypeError(`Unsupported canonical JSON value: ${String(value)}`);
    }
}

function serializeCanonicalObject(value: object, ancestors: Set<object>): string {
    if (ancestors.has(value)) {
        throw new TypeError('Canonical JSON cannot serialize cyclic values');
    }

    ancestors.add(value);
    try {
        return Array.isArray(value)
            ? serializeCanonicalArray(value, ancestors)
            : serializeCanonicalRecord(value, ancestors);
    } finally {
        ancestors.delete(value);
    }
}

function serializeCanonicalArray(value: unknown[], ancestors: Set<object>): string {
    if (Object.getPrototypeOf(value) !== Array.prototype) {
        throw new TypeError('Canonical JSON arrays must use the standard Array prototype');
    }

    const keys = Reflect.ownKeys(value);
    const indexDescriptors = new Map<number, PropertyDescriptor>();

    for (const key of keys) {
        if (typeof key === 'symbol') {
            throw new TypeError('Canonical JSON arrays cannot contain symbol properties');
        }
        if (key === 'length') continue;

        const index = canonicalArrayIndex(key, value.length);
        if (index === undefined) {
            throw new TypeError(`Canonical JSON arrays cannot contain extra property ${key}`);
        }
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (!descriptor || !descriptor.enumerable || !('value' in descriptor)) {
            throw new TypeError(
                `Canonical JSON array index ${key} must be an enumerable data property`,
            );
        }
        indexDescriptors.set(index, descriptor);
    }

    if (indexDescriptors.size !== value.length) {
        throw new TypeError('Canonical JSON arrays must contain every own dense index');
    }

    const serialized = new Array<string>(value.length);
    for (let index = 0; index < value.length; index += 1) {
        const descriptor = indexDescriptors.get(index);
        if (!descriptor) {
            throw new TypeError(`Canonical JSON array is missing own index ${index}`);
        }
        serialized[index] = serializeCanonicalValue(descriptor.value, ancestors);
    }
    return `[${serialized.join(',')}]`;
}

function canonicalArrayIndex(key: string, length: number): number | undefined {
    if (key === '' || key === '-0') return undefined;
    const index = Number(key);
    if (
        !Number.isInteger(index) ||
        index < 0 ||
        index >= length ||
        String(index) !== key
    ) {
        return undefined;
    }
    return index;
}

function serializeCanonicalRecord(value: object, ancestors: Set<object>): string {
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
        throw new TypeError('Canonical JSON objects must be plain records');
    }

    const descriptors = new Map<string, PropertyDescriptor>();
    for (const key of Reflect.ownKeys(value)) {
        if (typeof key === 'symbol') {
            throw new TypeError('Canonical JSON objects cannot contain symbol properties');
        }
        assertUnicodeScalarString(key);
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        if (!descriptor || !descriptor.enumerable) {
            throw new TypeError(
                `Canonical JSON object property ${JSON.stringify(key)} must be enumerable`,
            );
        }
        if (!('value' in descriptor)) {
            throw new TypeError(
                `Canonical JSON object property ${JSON.stringify(key)} must be a data property, not an accessor`,
            );
        }
        descriptors.set(key, descriptor);
    }

    const members: string[] = [];
    for (const key of [...descriptors.keys()].sort()) {
        const descriptor = descriptors.get(key);
        if (!descriptor) {
            throw new TypeError('Canonical JSON descriptor invariant failed');
        }
        members.push(
            `${JSON.stringify(key)}:${serializeCanonicalValue(descriptor.value, ancestors)}`,
        );
    }
    return `{${members.join(',')}}`;
}

function assertUnicodeScalarString(value: string): void {
    for (let index = 0; index < value.length; index += 1) {
        const codeUnit = value.charCodeAt(index);
        if (codeUnit >= 0xd800 && codeUnit <= 0xdbff) {
            const next = value.charCodeAt(index + 1);
            if (!Number.isInteger(next) || next < 0xdc00 || next > 0xdfff) {
                throw new TypeError('Canonical JSON strings cannot contain lone Unicode surrogates');
            }
            index += 1;
        } else if (codeUnit >= 0xdc00 && codeUnit <= 0xdfff) {
            throw new TypeError('Canonical JSON strings cannot contain lone Unicode surrogates');
        }
    }
}
