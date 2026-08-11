interface ComparisonRecordIdentity {
    readonly id: string;
}

export function reconcileComparisonSelection(
    selectedIds: readonly string[],
    records: readonly ComparisonRecordIdentity[],
    toggledId: string | null,
): readonly string[] {
    const existingIds = new Set(records.map((record) => record.id));
    const seen = new Set<string>();
    const reconciled = selectedIds.filter((id) => {
        if (!existingIds.has(id) || seen.has(id)) return false;
        seen.add(id);
        return true;
    }).slice(-2);

    if (toggledId === null || !existingIds.has(toggledId)) return reconciled;
    if (reconciled.includes(toggledId)) {
        return reconciled.filter((id) => id !== toggledId);
    }
    return [...reconciled, toggledId].slice(-2);
}
