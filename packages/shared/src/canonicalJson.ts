/**
 * Serialize an I-JSON value according to RFC 8785 (JSON Canonicalization Scheme).
 *
 * The function intentionally accepts `unknown`: identities are only trustworthy
 * when unsupported JavaScript state is rejected instead of being silently
 * omitted by `JSON.stringify`.
 */
export function canonicalizeJson(value: unknown): string {
    return serializeCanonicalValue(value, new Set<object>());
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
