import { state } from "../App";
import type { EvaluationRow } from "../types/eval-protocol";

export interface ToolCall {
  id: string;
  name: string;
  parameters: Record<string, any>;
}

export interface ToolResult {
  id: string;
  success: boolean;
  data?: any;
  error?: string;
  visualizationType?: "table" | "chart" | "text" | "insight";
}

// We'll store tool results separately and associate them with messages by index or timestamp

export class AgentService {
  private messageIdCounter = 0;

  // Available tools for the AI to use
  getAvailableTools() {
    return [
      {
        name: "analyzeData",
        description:
          "Analyze evaluation data with flexible filtering, grouping, and aggregation",
        parameters: {
          type: "object",
          properties: {
            filters: {
              type: "array",
              description: "Array of filter conditions",
              items: {
                type: "object",
                properties: {
                  field: {
                    type: "string",
                    description: 'Field path (e.g., "evaluation_result.score")',
                  },
                  operator: {
                    type: "string",
                    enum: [
                      "=",
                      "!=",
                      ">",
                      ">=",
                      "<",
                      "<=",
                      "contains",
                      "startsWith",
                      "endsWith",
                    ],
                  },
                  value: {
                    type: "string",
                    description: "Value to compare against",
                  },
                },
                required: ["field", "operator", "value"],
              },
            },
            groupBy: {
              type: "array",
              description: "Fields to group by",
              items: { type: "string" },
            },
            aggregations: {
              type: "array",
              description: "Aggregations to perform",
              items: {
                type: "object",
                properties: {
                  field: { type: "string" },
                  operation: {
                    type: "string",
                    enum: ["count", "sum", "avg", "min", "max", "std"],
                  },
                  alias: { type: "string" },
                },
                required: ["field", "operation"],
              },
            },
            limit: { type: "number", description: "Maximum number of results" },
            visualizationType: {
              type: "string",
              enum: ["table", "chart", "text", "insight"],
              description: "Preferred visualization type",
            },
          },
        },
      },
    ];
  }

  // Tool suggestion chips for UI
  getToolSuggestions() {
    return [
      "Show me failed evaluations",
      "Compare model performance",
      "Find score trends over time",
      "Group by evaluation name",
      "Average scores by model",
      "Find common error patterns",
    ];
  }

  // Execute a tool call
  async executeToolCall(toolCall: ToolCall): Promise<ToolResult> {
    try {
      let result;

      switch (toolCall.name) {
        case "analyzeData":
          result = await this.analyzeData(toolCall.parameters);
          break;
        default:
          throw new Error(`Unknown tool: ${toolCall.name}`);
      }

      return {
        id: toolCall.id,
        success: true,
        data: result.data,
        visualizationType: result.visualizationType,
      };
    } catch (error) {
      return {
        id: toolCall.id,
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  // Main analysis tool implementation
  private async analyzeData(params: any) {
    const {
      filters = [],
      groupBy = [],
      aggregations = [],
      limit,
      visualizationType = "table",
    } = params;

    // Get filtered data from GlobalState
    let data = state.filteredOriginalDataset;

    // Apply filters
    if (filters.length > 0) {
      data = this.applyFilters(data, filters);
    }

    // Apply grouping and aggregations
    let result;
    if (groupBy.length > 0 || aggregations.length > 0) {
      result = this.applyGroupingAndAggregation(data, groupBy, aggregations);
    } else {
      result = data;
    }

    // Apply limit
    if (limit && limit > 0) {
      result = result.slice(0, limit);
    }

    // Determine visualization type based on data structure
    const finalVisualizationType = this.determineVisualizationType(
      result,
      visualizationType
    );

    return {
      data: result,
      visualizationType: finalVisualizationType,
      metadata: {
        totalRows: data.length,
        filteredRows: result.length,
        appliedFilters: filters,
        groupBy,
        aggregations,
      },
    };
  }

  private applyFilters(data: EvaluationRow[], filters: any[]) {
    return data.filter((row) => {
      return filters.every((filter) => {
        const value = this.getNestedValue(row, filter.field);
        return this.evaluateFilter(value, filter.operator, filter.value);
      });
    });
  }

  private getNestedValue(obj: any, path: string): any {
    return path.split(".").reduce((current, key) => {
      return current?.[key];
    }, obj);
  }

  private evaluateFilter(
    value: any,
    operator: string,
    filterValue: string
  ): boolean {
    const numValue = typeof value === "string" ? parseFloat(value) : value;
    const numFilterValue = parseFloat(filterValue);

    switch (operator) {
      case "=":
        return value == filterValue;
      case "!=":
        return value != filterValue;
      case ">":
        return numValue > numFilterValue;
      case ">=":
        return numValue >= numFilterValue;
      case "<":
        return numValue < numFilterValue;
      case "<=":
        return numValue <= numFilterValue;
      case "contains":
        return String(value).toLowerCase().includes(filterValue.toLowerCase());
      case "startsWith":
        return String(value)
          .toLowerCase()
          .startsWith(filterValue.toLowerCase());
      case "endsWith":
        return String(value).toLowerCase().endsWith(filterValue.toLowerCase());
      default:
        return false;
    }
  }

  private applyGroupingAndAggregation(
    data: EvaluationRow[],
    groupBy: string[],
    aggregations: any[]
  ) {
    if (groupBy.length === 0 && aggregations.length === 0) {
      return data;
    }

    // Group data
    const groups = new Map<string, EvaluationRow[]>();

    data.forEach((row) => {
      const groupKey = groupBy
        .map((field) => this.getNestedValue(row, field))
        .join("|");
      if (!groups.has(groupKey)) {
        groups.set(groupKey, []);
      }
      groups.get(groupKey)!.push(row);
    });

    // Apply aggregations to each group
    const result = Array.from(groups.entries()).map(([groupKey, rows]) => {
      const groupValues = groupKey.split("|");
      const result: any = {};

      // Add group values
      groupBy.forEach((field, index) => {
        result[field] = groupValues[index];
      });

      // Add aggregations
      aggregations.forEach((agg) => {
        const values = rows
          .map((row) => this.getNestedValue(row, agg.field))
          .filter((v) => v != null);
        const alias = agg.alias || `${agg.operation}_${agg.field}`;

        switch (agg.operation) {
          case "count":
            result[alias] = values.length;
            break;
          case "sum":
            result[alias] = values.reduce(
              (sum, val) => sum + (Number(val) || 0),
              0
            );
            break;
          case "avg":
            result[alias] =
              values.length > 0
                ? values.reduce((sum, val) => sum + (Number(val) || 0), 0) /
                  values.length
                : 0;
            break;
          case "min":
            result[alias] =
              values.length > 0
                ? Math.min(...values.map((v) => Number(v) || 0))
                : 0;
            break;
          case "max":
            result[alias] =
              values.length > 0
                ? Math.max(...values.map((v) => Number(v) || 0))
                : 0;
            break;
          case "std":
            if (values.length > 0) {
              const avg =
                values.reduce((sum, val) => sum + (Number(val) || 0), 0) /
                values.length;
              const variance =
                values.reduce(
                  (sum, val) => sum + Math.pow((Number(val) || 0) - avg, 2),
                  0
                ) / values.length;
              result[alias] = Math.sqrt(variance);
            } else {
              result[alias] = 0;
            }
            break;
        }
      });

      return result;
    });

    return result;
  }

  private determineVisualizationType(
    data: any[],
    preferredType: string
  ): "table" | "chart" | "text" | "insight" {
    // If it's aggregated data with numeric values, prefer chart
    if (preferredType === "chart" && data.length > 0) {
      const firstRow = data[0];
      const hasNumericValues = Object.values(firstRow).some(
        (val) => typeof val === "number"
      );
      if (hasNumericValues) return "chart";
    }

    // If it's a single insight or very small dataset, prefer text
    if (data.length <= 3 && Object.keys(data[0] || {}).length <= 2) {
      return "text";
    }

    // Default to table for most cases
    return "table";
  }

  // Generate a unique message ID
  generateMessageId(): string {
    return `msg_${++this.messageIdCounter}_${Date.now()}`;
  }

  // Generate a unique tool call ID
  generateToolCallId(): string {
    return `tool_${++this.messageIdCounter}_${Date.now()}`;
  }
}
