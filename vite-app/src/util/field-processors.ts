import { state } from '../App';
import { createFilterFunction as createFilterFunctionUtil } from '../util/filter-utils';
import { type FilterGroup } from '../types/filters';

/**
 * Utility functions for processing field configurations and creating handlers
 * Centralizes common field manipulation logic
 */

/**
 * Creates a field change handler for a specific index
 */
export function createFieldHandler(
	updater: (index: number, value: string) => void
) {
	return (index: number, value: string) => {
		updater(index, value);
	};
}

/**
 * Creates an add field handler that respects the maximum limit
 */
export function createAddHandler(
	fields: string[],
	updater: (fields: string[]) => void,
	maxFields: number = 3
) {
	return () => {
		if (fields.length < maxFields) {
			updater([...fields, '']);
		}
	};
}

/**
 * Creates a remove field handler
 */
export function createRemoveHandler(
	fields: string[],
	updater: (fields: string[]) => void
) {
	return (index: number) => {
		updater(fields.filter((_, i) => i !== index));
	};
}

/**
 * Creates a complete field handler set for a field array
 */
export function createFieldHandlerSet(
	fields: string[],
	updater: (fields: string[]) => void,
	maxFields: number = 3
) {
	return {
		onFieldChange: createFieldHandler((index: number, value: string) => {
			const newFields = [...fields];
			newFields[index] = value;
			updater(newFields);
		}),
		onAddField: createAddHandler(fields, updater, maxFields),
		onRemoveField: createRemoveHandler(fields, updater),
	};
}

/**
 * Gets available keys from the current dataset state
 */
export function getAvailableKeys(): string[] {
	return state.flattenedDatasetKeys;
}

/**
 * Processes pivot configuration from state
 */
export function getPivotConfig() {
	return state.pivotConfig;
}

/**
 * Updates pivot configuration
 */
export function updatePivotConfig(updates: Partial<typeof state.pivotConfig>) {
	state.updatePivotConfig(updates);
}

/**
 * Resets pivot configuration to defaults
 */
export function resetPivotConfig() {
	state.resetPivotConfig();
}

/**
 * Updates filter configuration
 */
export function updateFilterConfig(filters: FilterGroup[]) {
	state.updateFilterConfig(filters);
}

/**
 * Gets the flattened dataset from state
 */
export function getFlattenedDataset() {
	return state.flattenedDataset;
}

/**
 * Creates a filter function using the current filter config
 */
export function createFilterFunction() {
	return createFilterFunctionUtil(state.filterConfig);
}

/**
 * Gets the current filter configuration
 */
export function getFilterConfig() {
	return state.filterConfig;
}
