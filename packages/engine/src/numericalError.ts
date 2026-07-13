/**
 * Structured failure for a numerical value that became NaN or infinite.
 *
 * Extending RangeError preserves the engine's existing public validation
 * contract while allowing runtime boundaries to classify divergence without
 * parsing an error message.
 */
export class NonFiniteNumericalError extends RangeError {
    readonly path: string;
    readonly value: number;

    constructor(path: string, value: number) {
        super(`${path} must be finite`);
        this.name = 'NonFiniteNumericalError';
        this.path = path;
        this.value = value;
    }
}
