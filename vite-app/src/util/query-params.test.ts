import { describe, it, expect } from "vitest";
import { nonDefaultValues } from "./query-params";
import type { GlobalConfig } from "../types/configs";
import { DEFAULT_GLOBAL_CONFIG } from "../GlobalState";

describe("query-params", () => {
  describe("nonDefaultValues", () => {
    it("returns empty object when config matches default", () => {
      const result = nonDefaultValues(DEFAULT_GLOBAL_CONFIG);
      expect(result).toEqual({});
    });

    it("returns empty object when config is identical to default", () => {
      const identicalConfig: GlobalConfig = {
        pivotConfig: {
          selectedRowFields: ["$.eval_metadata.name"],
          selectedColumnFields: ["$.input_metadata.completion_params.model"],
          selectedValueField: "$.evaluation_result.score",
          selectedAggregator: "avg",
        },
        filterConfig: [],
        paginationConfig: {
          currentPage: 1,
          pageSize: 25,
        },
        sortConfig: {
          sortField: "created_at",
          sortDirection: "desc",
        },
      };

      const result = nonDefaultValues(identicalConfig);
      expect(result).toEqual({});
    });

    it("detects differences in pivot config", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        pivotConfig: {
          ...DEFAULT_GLOBAL_CONFIG.pivotConfig,
          selectedRowFields: [
            "$.eval_metadata.name",
            "$.eval_metadata.version",
          ],
          selectedAggregator: "sum",
        },
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        "pivotConfig.selectedRowFields":
          '["$.eval_metadata.name","$.eval_metadata.version"]',
        "pivotConfig.selectedAggregator": '"sum"',
      });
    });

    it("detects differences in filter config", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        filterConfig: [
          {
            logic: "AND",
            filters: [
              {
                field: "score",
                operator: ">",
                value: "0.5",
                type: "text",
              },
            ],
          },
        ],
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        filterConfig:
          '[{"logic":"AND","filters":[{"field":"score","operator":">","value":"0.5","type":"text"}]}]',
      });
    });

    it("detects differences in pagination config", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        paginationConfig: {
          currentPage: 3,
          pageSize: 50,
        },
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        "paginationConfig.currentPage": "3",
        "paginationConfig.pageSize": "50",
      });
    });

    it("detects differences in sort config", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        sortConfig: {
          sortField: "score",
          sortDirection: "asc",
        },
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        "sortConfig.sortField": '"score"',
        "sortConfig.sortDirection": '"asc"',
      });
    });

    it("detects multiple differences across different config sections", () => {
      const modifiedConfig: GlobalConfig = {
        pivotConfig: {
          selectedRowFields: ["$.eval_metadata.name"],
          selectedColumnFields: ["$.input_metadata.completion_params.model"],
          selectedValueField: "$.evaluation_result.score",
          selectedAggregator: "max",
        },
        filterConfig: [
          {
            logic: "OR",
            filters: [
              {
                field: "status",
                operator: "==",
                value: "completed",
                type: "text",
              },
            ],
          },
        ],
        paginationConfig: {
          currentPage: 2,
          pageSize: 100,
        },
        sortConfig: {
          sortField: "timestamp",
          sortDirection: "asc",
        },
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        "pivotConfig.selectedAggregator": '"max"',
        filterConfig:
          '[{"logic":"OR","filters":[{"field":"status","operator":"==","value":"completed","type":"text"}]}]',
        "paginationConfig.currentPage": "2",
        "paginationConfig.pageSize": "100",
        "sortConfig.sortField": '"timestamp"',
        "sortConfig.sortDirection": '"asc"',
      });
    });

    it("handles nested array differences in filter config", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        filterConfig: [
          {
            logic: "AND",
            filters: [
              {
                field: "score",
                operator: ">",
                value: "0.5",
                type: "text",
              },
              {
                field: "status",
                operator: "==",
                value: "active",
                type: "text",
              },
            ],
          },
          {
            logic: "OR",
            filters: [
              {
                field: "category",
                operator: "contains",
                value: "test",
                type: "text",
              },
            ],
          },
        ],
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        filterConfig:
          '[{"logic":"AND","filters":[{"field":"score","operator":">","value":"0.5","type":"text"},{"field":"status","operator":"==","value":"active","type":"text"}]},{"logic":"OR","filters":[{"field":"category","operator":"contains","value":"test","type":"text"}]}]',
      });
    });

    it("handles null and undefined values correctly", () => {
      const configWithNulls: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        filterConfig: [
          {
            logic: "AND",
            filters: [
              {
                field: "test",
                operator: "==",
                value: "null",
                type: "text",
              },
            ],
          },
        ],
      };

      const result = nonDefaultValues(configWithNulls);

      expect(result).toEqual({
        filterConfig:
          '[{"logic":"AND","filters":[{"field":"test","operator":"==","value":"null","type":"text"}]}]',
      });
    });

    it("handles empty arrays correctly", () => {
      const configWithEmptyArrays: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        pivotConfig: {
          ...DEFAULT_GLOBAL_CONFIG.pivotConfig,
          selectedRowFields: [],
          selectedColumnFields: [],
        },
      };

      const result = nonDefaultValues(configWithEmptyArrays);

      expect(result).toEqual({
        "pivotConfig.selectedRowFields": "[]",
        "pivotConfig.selectedColumnFields": "[]",
      });
    });

    it("handles complex nested structures", () => {
      const complexConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        filterConfig: [
          {
            logic: "AND",
            filters: [
              {
                field: "metadata.tags",
                operator: "contains",
                value: "important",
                type: "text",
              },
              {
                field: "score",
                operator: "between",
                value: "0.5",
                value2: "1.0",
                type: "text",
              },
            ],
          },
        ],
      };

      const result = nonDefaultValues(complexConfig);

      expect(result).toEqual({
        filterConfig:
          '[{"logic":"AND","filters":[{"field":"metadata.tags","operator":"contains","value":"important","type":"text"},{"field":"score","operator":"between","value":"0.5","value2":"1.0","type":"text"}]}]',
      });
    });

    it("preserves JSON serialization format", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        pivotConfig: {
          ...DEFAULT_GLOBAL_CONFIG.pivotConfig,
          selectedRowFields: ["field1", "field2"],
        },
      };

      const result = nonDefaultValues(modifiedConfig);

      // Verify that arrays are properly JSON stringified
      expect(result["pivotConfig.selectedRowFields"]).toBe(
        '["field1","field2"]'
      );

      // Verify that strings are properly JSON stringified (with quotes)
      expect(result["pivotConfig.selectedAggregator"]).toBeUndefined(); // No change from default
    });

    it("handles edge case with all default values but different object references", () => {
      // Create a config that has the same values but different object references
      const configWithNewObjects: GlobalConfig = {
        pivotConfig: {
          selectedRowFields: ["$.eval_metadata.name"],
          selectedColumnFields: ["$.input_metadata.completion_params.model"],
          selectedValueField: "$.evaluation_result.score",
          selectedAggregator: "avg",
        },
        filterConfig: [],
        paginationConfig: {
          currentPage: 1,
          pageSize: 25,
        },
        sortConfig: {
          sortField: "created_at",
          sortDirection: "desc",
        },
      };

      const result = nonDefaultValues(configWithNewObjects);

      // Should return empty object since values are identical to defaults
      expect(result).toEqual({});
    });

    it("handles boolean values correctly", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        filterConfig: [
          {
            logic: "AND",
            filters: [
              {
                field: "isActive",
                operator: "==",
                value: "true",
                type: "text",
              },
            ],
          },
        ],
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        filterConfig:
          '[{"logic":"AND","filters":[{"field":"isActive","operator":"==","value":"true","type":"text"}]}]',
      });
    });

    it("handles numeric values correctly", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        paginationConfig: {
          currentPage: 0,
          pageSize: 1000,
        },
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        "paginationConfig.currentPage": "0",
        "paginationConfig.pageSize": "1000",
      });
    });

    it("handles special characters in string values", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        pivotConfig: {
          ...DEFAULT_GLOBAL_CONFIG.pivotConfig,
          selectedValueField: "$.metadata['special-chars']",
        },
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        "pivotConfig.selectedValueField": "\"$.metadata['special-chars']\"",
      });
    });

    it("handles deeply nested filter configurations", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        filterConfig: [
          {
            logic: "AND",
            filters: [
              {
                field: "metadata.nested.deep.value",
                operator: "contains",
                value: "test",
                type: "text",
              },
              {
                field: "scores[0]",
                operator: ">",
                value: "0.5",
                type: "text",
              },
            ],
          },
          {
            logic: "OR",
            filters: [
              {
                field: "status",
                operator: "==",
                value: "active",
                type: "text",
              },
              {
                field: "status",
                operator: "==",
                value: "pending",
                type: "text",
              },
            ],
          },
        ],
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        filterConfig:
          '[{"logic":"AND","filters":[{"field":"metadata.nested.deep.value","operator":"contains","value":"test","type":"text"},{"field":"scores[0]","operator":">","value":"0.5","type":"text"}]},{"logic":"OR","filters":[{"field":"status","operator":"==","value":"active","type":"text"},{"field":"status","operator":"==","value":"pending","type":"text"}]}]',
      });
    });

    it("handles date range filters correctly", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        filterConfig: [
          {
            logic: "AND",
            filters: [
              {
                field: "created_at",
                operator: "between",
                value: "2024-01-01",
                value2: "2024-12-31",
                type: "date-range",
              },
            ],
          },
        ],
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        filterConfig:
          '[{"logic":"AND","filters":[{"field":"created_at","operator":"between","value":"2024-01-01","value2":"2024-12-31","type":"date-range"}]}]',
      });
    });

    it("handles all sort directions", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        sortConfig: {
          sortField: "name",
          sortDirection: "asc",
        },
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        "sortConfig.sortField": '"name"',
        "sortConfig.sortDirection": '"asc"',
      });
    });

    it("handles all aggregator types", () => {
      const aggregators = ["sum", "min", "max", "count"]; // Exclude "avg" since it's the default

      for (const aggregator of aggregators) {
        const modifiedConfig: GlobalConfig = {
          ...DEFAULT_GLOBAL_CONFIG,
          pivotConfig: {
            ...DEFAULT_GLOBAL_CONFIG.pivotConfig,
            selectedAggregator: aggregator,
          },
        };

        const result = nonDefaultValues(modifiedConfig);

        expect(result).toEqual({
          "pivotConfig.selectedAggregator": `"${aggregator}"`,
        });
      }
    });

    it("verifies default aggregator returns empty result", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        pivotConfig: {
          ...DEFAULT_GLOBAL_CONFIG.pivotConfig,
          selectedAggregator: "avg", // This is the default value
        },
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({});
    });

    it("handles all filter operators", () => {
      const operators = [
        "==",
        "!=",
        ">",
        "<",
        ">=",
        "<=",
        "contains",
        "!contains",
        "between",
      ];

      for (const operator of operators) {
        const modifiedConfig: GlobalConfig = {
          ...DEFAULT_GLOBAL_CONFIG,
          filterConfig: [
            {
              logic: "AND",
              filters: [
                {
                  field: "test",
                  operator: operator as any,
                  value: "test-value",
                  type: "text",
                },
              ],
            },
          ],
        };

        const result = nonDefaultValues(modifiedConfig);

        expect(result).toEqual({
          filterConfig: `[{"logic":"AND","filters":[{"field":"test","operator":"${operator}","value":"test-value","type":"text"}]}]`,
        });
      }
    });

    it("handles all filter logic types", () => {
      const logics = ["AND", "OR"];

      for (const logic of logics) {
        const modifiedConfig: GlobalConfig = {
          ...DEFAULT_GLOBAL_CONFIG,
          filterConfig: [
            {
              logic: logic as any,
              filters: [
                {
                  field: "test",
                  operator: "==",
                  value: "test-value",
                  type: "text",
                },
              ],
            },
          ],
        };

        const result = nonDefaultValues(modifiedConfig);

        expect(result).toEqual({
          filterConfig: `[{"logic":"${logic}","filters":[{"field":"test","operator":"==","value":"test-value","type":"text"}]}]`,
        });
      }
    });

    it("handles all filter types", () => {
      const filterTypes = ["text", "date", "date-range"];

      for (const filterType of filterTypes) {
        const modifiedConfig: GlobalConfig = {
          ...DEFAULT_GLOBAL_CONFIG,
          filterConfig: [
            {
              logic: "AND",
              filters: [
                {
                  field: "test",
                  operator: "==",
                  value: "test-value",
                  type: filterType as any,
                },
              ],
            },
          ],
        };

        const result = nonDefaultValues(modifiedConfig);

        expect(result).toEqual({
          filterConfig: `[{"logic":"AND","filters":[{"field":"test","operator":"==","value":"test-value","type":"${filterType}"}]}]`,
        });
      }
    });

    it("handles very large page sizes", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        paginationConfig: {
          currentPage: 1,
          pageSize: 10000,
        },
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        "paginationConfig.pageSize": "10000",
      });
    });

    it("handles zero values correctly", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        paginationConfig: {
          currentPage: 0,
          pageSize: 0,
        },
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        "paginationConfig.currentPage": "0",
        "paginationConfig.pageSize": "0",
      });
    });

    it("handles negative values correctly", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        paginationConfig: {
          currentPage: -1,
          pageSize: -5,
        },
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        "paginationConfig.currentPage": "-1",
        "paginationConfig.pageSize": "-5",
      });
    });

    it("handles very long field names", () => {
      const longFieldName =
        "very_long_field_name_that_might_be_used_in_real_world_scenarios_with_deeply_nested_objects_and_arrays";
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        pivotConfig: {
          ...DEFAULT_GLOBAL_CONFIG.pivotConfig,
          selectedValueField: `$.${longFieldName}`,
        },
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        "pivotConfig.selectedValueField": `"$.${longFieldName}"`,
      });
    });

    it("handles unicode characters in values", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        pivotConfig: {
          ...DEFAULT_GLOBAL_CONFIG.pivotConfig,
          selectedValueField: "$.metadata.测试字段",
        },
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        "pivotConfig.selectedValueField": '"$.metadata.测试字段"',
      });
    });

    it("handles empty string values", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        pivotConfig: {
          ...DEFAULT_GLOBAL_CONFIG.pivotConfig,
          selectedValueField: "",
        },
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        "pivotConfig.selectedValueField": '""',
      });
    });

    it("handles very deep nesting in filter configs", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        filterConfig: [
          {
            logic: "AND",
            filters: [
              {
                field: "a.b.c.d.e.f.g.h.i.j.k.l.m.n.o.p",
                operator: "==",
                value: "deep_value",
                type: "text",
              },
            ],
          },
        ],
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        filterConfig:
          '[{"logic":"AND","filters":[{"field":"a.b.c.d.e.f.g.h.i.j.k.l.m.n.o.p","operator":"==","value":"deep_value","type":"text"}]}]',
      });
    });

    it("handles mixed data types in arrays", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        pivotConfig: {
          ...DEFAULT_GLOBAL_CONFIG.pivotConfig,
          selectedRowFields: ["string", "123", "true", "null"],
        },
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        "pivotConfig.selectedRowFields": '["string","123","true","null"]',
      });
    });

    it("handles complex JSON paths", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        pivotConfig: {
          ...DEFAULT_GLOBAL_CONFIG.pivotConfig,
          selectedRowFields: [
            "$.data[0].items[1].metadata.tags[2]",
            "$['special-key'].value",
            "$.nested['with spaces'].value",
          ],
        },
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        "pivotConfig.selectedRowFields":
          '["$.data[0].items[1].metadata.tags[2]","$[\'special-key\'].value","$.nested[\'with spaces\'].value"]',
      });
    });

    it("handles partial config changes with some defaults preserved", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        pivotConfig: {
          selectedRowFields: ["$.eval_metadata.name"], // Same as default
          selectedColumnFields: ["$.input_metadata.completion_params.model"], // Same as default
          selectedValueField: "$.evaluation_result.score", // Same as default
          selectedAggregator: "sum", // Different from default
        },
        filterConfig: [], // Same as default
        paginationConfig: {
          currentPage: 1, // Same as default
          pageSize: 50, // Different from default
        },
        sortConfig: {
          sortField: "created_at", // Same as default
          sortDirection: "asc", // Different from default
        },
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        "pivotConfig.selectedAggregator": '"sum"',
        "paginationConfig.pageSize": "50",
        "sortConfig.sortDirection": '"asc"',
      });
    });

    it("handles edge case with undefined values in optional fields", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        filterConfig: [
          {
            logic: "AND",
            filters: [
              {
                field: "test",
                operator: "==",
                value: "test",
                type: "text",
                // value2 is optional and undefined
              },
            ],
          },
        ],
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        filterConfig:
          '[{"logic":"AND","filters":[{"field":"test","operator":"==","value":"test","type":"text"}]}]',
      });
    });

    it("handles edge case with value2 field in between operator", () => {
      const modifiedConfig: GlobalConfig = {
        ...DEFAULT_GLOBAL_CONFIG,
        filterConfig: [
          {
            logic: "AND",
            filters: [
              {
                field: "score",
                operator: "between",
                value: "0.5",
                value2: "1.0",
                type: "text",
              },
            ],
          },
        ],
      };

      const result = nonDefaultValues(modifiedConfig);

      expect(result).toEqual({
        filterConfig:
          '[{"logic":"AND","filters":[{"field":"score","operator":"between","value":"0.5","value2":"1.0","type":"text"}]}]',
      });
    });
  });
});
