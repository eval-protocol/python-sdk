/**
 * Module for handling the reactivity, update, and querying of query params based on GlobalState
 */

import { autorun } from "mobx";
import { state } from "../App";
import type { GlobalConfig } from "../types/configs";
import { DEFAULT_GLOBAL_CONFIG } from "../GlobalState";

export function initWatcher() {
  autorun(() => {
    const globalConfig = state.globalConfig;
    console.log(JSON.stringify(globalConfig));
  });
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
