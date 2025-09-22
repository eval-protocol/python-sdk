/**
 * Module for handling the reactivity, update, and querying of query params based on GlobalState
 */

import { autorun } from "mobx";
import { useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import type { GlobalConfig } from "../types/configs";
import { DEFAULT_GLOBAL_CONFIG, GlobalState } from "../GlobalState";

export class QueryParamsWatcher {
  queryParams: Record<string, string>;
  private updateUrlCallback:
    | ((queryParams: Record<string, string>) => void)
    | null = null;
  private state: GlobalState;

  constructor(state: GlobalState) {
    this.state = state;
    this.queryParams = nonDefaultValues(this.state.globalConfig);
    this.init();
  }

  init() {
    autorun(() => {
      const globalConfig = this.state.globalConfig;
      const diff = nonDefaultValues(globalConfig);
      const previousUrlEncodedQueryParams =
        this.generateUrlEncodedQueryParams();
      this.queryParams = diff;
      const newUrlEncodedQueryParams = this.generateUrlEncodedQueryParams();
      if (previousUrlEncodedQueryParams !== newUrlEncodedQueryParams) {
        console.log(
          `Query params changed from ${previousUrlEncodedQueryParams} to ${newUrlEncodedQueryParams}`
        );
        this.updateUrl();
      }
    });
  }

  setUpdateUrlCallback(
    callback: (queryParams: Record<string, string>) => void
  ) {
    this.updateUrlCallback = callback;
  }

  stableQueryParams(): [string, string][] {
    /**
     * Returns a stable query params object that is idempotent based on this.queryParams. First sorts by key and then by value.
     */
    return Object.entries(this.queryParams).sort((a, b) => {
      const keyCompare = a[0].localeCompare(b[0]);
      if (keyCompare !== 0) return keyCompare;
      return a[1].localeCompare(b[1]);
    });
  }

  generateUrlEncodedQueryParams(): string {
    return this.stableQueryParams()
      .map(
        ([key, value]) =>
          `${encodeURIComponent(key)}=${encodeURIComponent(value)}`
      )
      .join("&");
  }

  private updateUrl() {
    /**
     * Update the browser URL with current query params using React Router callback
     */
    if (this.updateUrlCallback) {
      this.updateUrlCallback(this.queryParams);
    }
  }
}

export function nonDefaultValues(
  globalConfig: GlobalConfig,
  defaultConfig: GlobalConfig = DEFAULT_GLOBAL_CONFIG
): Record<string, string> {
  /**
   * Return a collection of non-default values based on an instance of GlobalConfig
   *
   * This is particularly useful for computing query params since we want to
   * keep the links as minimal as possible so they are easy to understand and
   * log to console.
   *
   * Return
   * - The key is a JSON path to the field
   * - The value is the JSON serialized value of the field
   */
  return calculateDifferentValues(globalConfig, defaultConfig);
}

function calculateDifferentValues(
  globalConfig: GlobalConfig,
  defaultConfig: GlobalConfig
): Record<string, string> {
  const differences: Record<string, string> = {};

  function compareObjects(obj1: any, obj2: any, path: string = ""): void {
    // Handle null/undefined cases
    if (obj1 === null || obj1 === undefined) {
      if (obj2 !== null && obj2 !== undefined) {
        differences[path] = JSON.stringify(obj1);
      }
      return;
    }

    if (obj2 === null || obj2 === undefined) {
      if (obj1 !== null && obj1 !== undefined) {
        differences[path] = JSON.stringify(obj1);
      }
      return;
    }

    // Handle primitive types
    if (typeof obj1 !== "object" || typeof obj2 !== "object") {
      if (obj1 !== obj2) {
        differences[path] = JSON.stringify(obj1);
      }
      return;
    }

    // Handle arrays
    if (Array.isArray(obj1) && Array.isArray(obj2)) {
      if (JSON.stringify(obj1) !== JSON.stringify(obj2)) {
        differences[path] = JSON.stringify(obj1);
      }
      return;
    }

    // Handle objects
    if (Array.isArray(obj1) || Array.isArray(obj2)) {
      if (JSON.stringify(obj1) !== JSON.stringify(obj2)) {
        differences[path] = JSON.stringify(obj1);
      }
      return;
    }

    // Get all unique keys from both objects
    const allKeys = new Set([...Object.keys(obj1), ...Object.keys(obj2)]);

    for (const key of allKeys) {
      const currentPath = path ? `${path}.${key}` : key;

      if (!(key in obj1)) {
        // Key exists in obj2 but not obj1
        differences[currentPath] = JSON.stringify(undefined);
      } else if (!(key in obj2)) {
        // Key exists in obj1 but not obj2
        differences[currentPath] = JSON.stringify(obj1[key]);
      } else {
        // Key exists in both objects, compare recursively
        compareObjects(obj1[key], obj2[key], currentPath);
      }
    }
  }

  compareObjects(globalConfig, defaultConfig);
  return differences;
}

/**
 * Converts serialized query params back into a partial GlobalConfig
 * This is the inverse operation of nonDefaultValues()
 *
 * @param queryParams - Record where keys are JSON paths and values are JSON-serialized values
 * @returns Partial<GlobalConfig> that can be applied to update GlobalConfig
 */
export function queryParamsToPartialConfig(
  queryParams: Record<string, string>
): Partial<GlobalConfig> {
  const result: any = {};

  for (const [path, serializedValue] of Object.entries(queryParams)) {
    try {
      const value = JSON.parse(serializedValue);
      setNestedValue(result, path, value);
    } catch (error) {
      console.warn(
        `Failed to parse query param value for path "${path}":`,
        error
      );
    }
  }

  return result;
}

/**
 * Helper function to set a nested value in an object using dot notation path
 * @param obj - The object to modify
 * @param path - Dot notation path (e.g., "pivotConfig.selectedRowFields")
 * @param value - The value to set
 */
function setNestedValue(obj: any, path: string, value: any): void {
  const keys = path.split(".");
  let current = obj;

  // Navigate to the parent of the target key
  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i];
    if (!(key in current)) {
      current[key] = {};
    }
    current = current[key];
  }

  // Set the final value
  const finalKey = keys[keys.length - 1];
  current[finalKey] = value;
}

/**
 * Custom hook that integrates QueryParamsWatcher with React Router's useSearchParams
 * This hook should be used in components that need to sync global state with URL query params
 */
export function useQueryParamsSync(queryParamsWatcher: QueryParamsWatcher) {
  const [, setSearchParams] = useSearchParams();

  useEffect(() => {
    // Set up the callback for the QueryParamsWatcher to update URL
    const updateUrl = (queryParams: Record<string, string>) => {
      const newSearchParams = new URLSearchParams();

      // Add all query params to URLSearchParams
      Object.entries(queryParams).forEach(([key, value]) => {
        newSearchParams.set(key, value);
      });

      // Update the URL using React Router
      setSearchParams(newSearchParams, { replace: true });
    };

    // Set the callback on the global queryParamsWatcher
    queryParamsWatcher.setUpdateUrlCallback(updateUrl);

    // Cleanup: remove callback when component unmounts
    return () => {
      queryParamsWatcher.setUpdateUrlCallback(() => {});
    };
  }, [setSearchParams]);
}
