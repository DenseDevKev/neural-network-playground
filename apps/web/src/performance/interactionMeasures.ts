export const APP_PERFORMANCE_MEASURE_PREFIX = 'nn-playground:';
export const SLOW_INTERACTION_THRESHOLD_MS = 16;

export function createAppMeasureName(label: string): string {
    return `${APP_PERFORMANCE_MEASURE_PREFIX}${label}`;
}

export function shouldReportSlowInteraction(
    entry: Pick<PerformanceEntry, 'name' | 'duration'>,
): boolean {
    return entry.name.startsWith(APP_PERFORMANCE_MEASURE_PREFIX)
        && entry.duration > SLOW_INTERACTION_THRESHOLD_MS;
}
