import { useMemo } from 'react';
import { computePivot } from '../util/pivot';
import { createFilterFunction } from '../util/filter-utils';
import { state } from '../App';

export interface PivotDataConfig {
	rowFields: string[];
	columnFields: string[];
	valueField?: string;
	aggregator: 'count' | 'sum' | 'avg' | 'min' | 'max';
	showRowTotals?: boolean;
	showColumnTotals?: boolean;
}

export interface ProcessedPivotData {
	rowFields: string[];
	columnFields: string[];
	valueField?: string;
	aggregator: 'count' | 'sum' | 'avg' | 'min' | 'max';
	pivotResult: ReturnType<typeof computePivot<any>>;
	hasValidConfiguration: boolean;
}

/**
 * Custom hook that processes pivot configuration and computes pivot data
 * Centralizes all pivot-related logic to avoid duplication
 */
export function usePivotData(
	config: PivotDataConfig
): ProcessedPivotData {
	const { rowFields, columnFields, valueField, aggregator, showRowTotals = true, showColumnTotals = true } = config;

	// Filter out empty fields and cast to proper types
	const processedRowFields = useMemo(
		() => rowFields.filter((field) => field !== '') as string[],
		[rowFields]
	);

	const processedColumnFields = useMemo(
		() => columnFields.filter((field) => field !== '') as string[],
		[columnFields]
	);

	const processedValueField = useMemo(
		() => (valueField && valueField !== '' ? valueField : undefined) as string | undefined,
		[valueField]
	);

	// Check if we have a valid configuration for pivot computation
	const hasValidConfiguration = useMemo(
		() => processedRowFields.length > 0 && processedColumnFields.length > 0,
		[processedRowFields, processedColumnFields]
	);

	// Compute pivot data only when configuration is valid
	const pivotResult = useMemo(() => {
		if (!hasValidConfiguration) {
			// Return empty pivot result structure
			return {
				rowKeyTuples: [],
				colKeyTuples: [],
				cells: {},
				rowTotals: {},
				colTotals: {},
				grandTotal: 0,
			} as ReturnType<typeof computePivot<any>>;
		}

		return computePivot<any>({
			data: state.flattenedDataset,
			rowFields: processedRowFields,
			columnFields: processedColumnFields,
			valueField: processedValueField,
			aggregator,
			filter: createFilterFunction(state.filterConfig),
		});
	}, [hasValidConfiguration, processedRowFields, processedColumnFields, processedValueField, aggregator]);

	return {
		rowFields: processedRowFields,
		columnFields: processedColumnFields,
		valueField: processedValueField,
		aggregator,
		pivotResult,
		hasValidConfiguration,
	};
}
