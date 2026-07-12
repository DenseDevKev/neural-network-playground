/// <reference lib="dom" />

import { canonicalizeJson } from './canonicalJson.js';
import { DEFAULT_EXPERIMENT_DOCUMENT } from './constants.js';
import { validateExperimentDocument } from './experimentSchema.js';
import type {
    ExperimentSchemaIssueCode,
    SchemaResult,
    ValidatedExperimentDocumentV2,
} from './types.js';

export const MAX_EXPERIMENT_JSON_BYTES = 4 * 1024 * 1024;

const MAX_BASE64URL_PAYLOAD_CHARACTERS = Math.ceil(
    MAX_EXPERIMENT_JSON_BYTES * 4 / 3,
);
const BASE64URL_ALPHABET =
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';
const BASE64URL_INDEX = new Map<string, number>(
    [...BASE64URL_ALPHABET].map((character, index) => [character, index]),
);

function schemaFailure(
    code: ExperimentSchemaIssueCode,
    path: string,
    message: string,
): SchemaResult<never> {
    return { ok: false, issues: [{ code, path, message }] };
}

function encodeBase64UrlBytes(bytes: Uint8Array): string {
    let encoded = '';
    for (let offset = 0; offset < bytes.length; offset += 3) {
        const remaining = bytes.length - offset;
        const first = bytes[offset];
        const second = remaining > 1 ? bytes[offset + 1] : 0;
        const third = remaining > 2 ? bytes[offset + 2] : 0;
        const chunk = (first << 16) | (second << 8) | third;
        encoded += BASE64URL_ALPHABET[(chunk >>> 18) & 0x3f];
        encoded += BASE64URL_ALPHABET[(chunk >>> 12) & 0x3f];
        if (remaining > 1) encoded += BASE64URL_ALPHABET[(chunk >>> 6) & 0x3f];
        if (remaining > 2) encoded += BASE64URL_ALPHABET[chunk & 0x3f];
    }
    return encoded;
}

function decodeBase64UrlBytes(value: string): Uint8Array {
    if (value.length === 0 || value.length % 4 === 1) {
        throw new TypeError('base64url payload has an invalid length');
    }

    const bytes = new Uint8Array(Math.floor(value.length * 6 / 8));
    let byteIndex = 0;
    let accumulator = 0;
    let bitCount = 0;
    for (const character of value) {
        const digit = BASE64URL_INDEX.get(character);
        if (digit === undefined) {
            throw new TypeError('base64url payload contains an invalid character');
        }
        accumulator = (accumulator << 6) | digit;
        bitCount += 6;
        if (bitCount >= 8) {
            bitCount -= 8;
            bytes[byteIndex] = (accumulator >>> bitCount) & 0xff;
            byteIndex += 1;
            accumulator &= bitCount === 0 ? 0 : (1 << bitCount) - 1;
        }
    }
    if (accumulator !== 0) {
        throw new TypeError('base64url payload has non-canonical trailing bits');
    }
    if (byteIndex !== bytes.length || encodeBase64UrlBytes(bytes) !== value) {
        throw new TypeError('base64url payload is not canonical');
    }
    return bytes;
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function validateDecodedDocument(
    value: unknown,
): SchemaResult<ValidatedExperimentDocumentV2> {
    const looksUnversioned = isRecord(value) && (
        value.kind === 'nn-playground-experiment'
        || Object.hasOwn(value, 'recipe')
        || ['network', 'training', 'data', 'features', 'ui'].every((key) =>
            Object.hasOwn(value, key))
    );
    if (looksUnversioned && !Object.hasOwn(value, 'schemaVersion')) {
        return schemaFailure(
            'legacy-state',
            'schemaVersion',
            'unversioned experiment documents are incompatible with schema version 2',
        );
    }
    if (isRecord(value) && value.schemaVersion === 1) {
        return schemaFailure(
            'legacy-state',
            'schemaVersion',
            'schema version 1 experiment documents are incompatible with schema version 2',
        );
    }
    return validateExperimentDocument(value);
}

/** Encode the exact validated V2 document into its canonical share hash. */
export function encodeExperimentUrl(document: ValidatedExperimentDocumentV2): string {
    const canonicalDocument = canonicalizeJson(document);
    const payload = encodeBase64UrlBytes(new TextEncoder().encode(canonicalDocument));
    return `#v=2&r=${payload}`;
}

/** Decode a V2 share hash without repair, clamping, or fallback. */
export function decodeExperimentUrl(
    hash: string,
): SchemaResult<ValidatedExperimentDocumentV2> {
    if (hash === '') return { ok: true, value: DEFAULT_EXPERIMENT_DOCUMENT };
    if (!hash.startsWith('#')) {
        return schemaFailure('invalid-field', '$', 'V2 URL state must begin with #');
    }

    const source = hash.slice(1);
    if (source.length > 6 + MAX_BASE64URL_PAYLOAD_CHARACTERS) {
        return schemaFailure(
            'resource-limit',
            'r',
            `URL document exceeds ${MAX_EXPERIMENT_JSON_BYTES} UTF-8 bytes`,
        );
    }
    const parameters = new URLSearchParams(source);
    const versions = parameters.getAll('v');
    if (versions.length === 0) {
        return schemaFailure(
            'legacy-state',
            '$',
            'unversioned URL state is incompatible with schema version 2',
        );
    }
    if (versions.length !== 1) {
        return schemaFailure('invalid-field', 'v', 'URL must contain exactly one version');
    }
    const version = versions[0];
    if (version === '0' || version === '1') {
        return schemaFailure(
            'legacy-state',
            'v',
            `URL version ${version} is incompatible with schema version 2`,
        );
    }
    if (version !== '2') {
        if (/^[0-9]+$/u.test(version) && Number(version) > 2) {
            return schemaFailure(
                'unsupported-version',
                'v',
                `URL version ${version} is not supported`,
            );
        }
        return schemaFailure('invalid-field', 'v', 'URL version must be the integer 2');
    }

    let unknownKey: string | null = null;
    parameters.forEach((_value, key) => {
        if (unknownKey === null && key !== 'v' && key !== 'r') unknownKey = key;
    });
    if (unknownKey !== null) {
        return schemaFailure(
            'unknown-field',
            unknownKey,
            `URL field ${unknownKey} is not allowed`,
        );
    }
    const payloads = parameters.getAll('r');
    if (payloads.length === 0) {
        return schemaFailure('missing-field', 'r', 'URL recipe payload is required');
    }
    if (payloads.length !== 1) {
        return schemaFailure('invalid-field', 'r', 'URL must contain exactly one recipe payload');
    }
    if (source !== `v=2&r=${payloads[0]}`) {
        return schemaFailure(
            'invalid-field',
            '$',
            'V2 URL must use the exact #v=2&r=<canonical-base64url> format',
        );
    }

    try {
        const bytes = decodeBase64UrlBytes(payloads[0]);
        if (bytes.byteLength > MAX_EXPERIMENT_JSON_BYTES) {
            return schemaFailure(
                'resource-limit',
                'r',
                `URL document exceeds ${MAX_EXPERIMENT_JSON_BYTES} UTF-8 bytes`,
            );
        }
        const json = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
        const parsed = JSON.parse(json) as unknown;
        if (json !== canonicalizeJson(parsed)) {
            return schemaFailure(
                'invalid-field',
                'r',
                'V2 URL payload must contain canonical JSON',
            );
        }
        return validateDecodedDocument(parsed);
    } catch (error) {
        const detail = error instanceof Error ? error.message : 'malformed URL payload';
        return schemaFailure('invalid-field', 'r', `invalid V2 URL payload: ${detail}`);
    }
}

function assertNoDuplicateJsonObjectKeys(json: string): void {
    let index = 0;
    const skipWhitespace = (): void => {
        while (/\s/u.test(json[index] ?? '')) index += 1;
    };
    const parseString = (): string => {
        const start = index;
        index += 1;
        while (index < json.length) {
            const character = json[index];
            index += 1;
            if (character === '"') return JSON.parse(json.slice(start, index)) as string;
            if (character === '\\') {
                const escape = json[index];
                index += 1;
                if (escape === 'u') index += 4;
            }
        }
        throw new SyntaxError('unterminated JSON string');
    };
    const parseValue = (): void => {
        skipWhitespace();
        const character = json[index];
        if (character === '{') {
            index += 1;
            skipWhitespace();
            const keys = new Set<string>();
            if (json[index] === '}') {
                index += 1;
                return;
            }
            while (index < json.length) {
                skipWhitespace();
                const key = parseString();
                if (keys.has(key)) {
                    throw new SyntaxError(`duplicate JSON object member ${JSON.stringify(key)}`);
                }
                keys.add(key);
                skipWhitespace();
                index += 1;
                parseValue();
                skipWhitespace();
                const separator = json[index];
                index += 1;
                if (separator === '}') return;
            }
            return;
        }
        if (character === '[') {
            index += 1;
            skipWhitespace();
            if (json[index] === ']') {
                index += 1;
                return;
            }
            while (index < json.length) {
                parseValue();
                skipWhitespace();
                const separator = json[index];
                index += 1;
                if (separator === ']') return;
            }
            return;
        }
        if (character === '"') {
            parseString();
            return;
        }
        while (index < json.length && !/[\s,}\]]/u.test(json[index])) index += 1;
    };
    parseValue();
}

/** Encode a validated V2 document as deterministic, pretty JSON. */
export function encodeExperimentJson(document: ValidatedExperimentDocumentV2): string {
    const canonicalDocument = canonicalizeJson(document);
    return JSON.stringify(JSON.parse(canonicalDocument) as unknown, null, 2);
}

/** Parse and strictly validate one exact V2 document envelope. */
export function decodeExperimentJson(
    json: string,
): SchemaResult<ValidatedExperimentDocumentV2> {
    if (new TextEncoder().encode(json).byteLength > MAX_EXPERIMENT_JSON_BYTES) {
        return schemaFailure(
            'resource-limit',
            '$',
            `experiment JSON exceeds ${MAX_EXPERIMENT_JSON_BYTES} UTF-8 bytes`,
        );
    }
    try {
        const parsed = JSON.parse(json) as unknown;
        assertNoDuplicateJsonObjectKeys(json);
        return validateDecodedDocument(parsed);
    } catch (error) {
        const detail = error instanceof Error ? error.message : 'malformed JSON';
        return schemaFailure('invalid-field', '$', `invalid experiment JSON: ${detail}`);
    }
}
