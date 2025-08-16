import { useRef, useCallback, useState } from "react";
import { Chart as ChartJS, registerables } from "chart.js";
import { Chart } from "react-chartjs-2";
import html2canvas from "html2canvas-oklch";
import Button from "./Button";
import Select from "./Select";

// Register Chart.js components
ChartJS.register(...registerables);

interface ChartExportProps<T extends Record<string, unknown>> {
  /**
   * Pivot table data structure
   */
  pivotData: {
    rowKeyTuples: unknown[][];
    colKeyTuples: unknown[][];
    cells: Record<string, Record<string, { value: number }>>;
    rowTotals: Record<string, number>;
    colTotals: Record<string, number>;
    grandTotal: number;
  };
  /**
   * Row fields configuration
   */
  rowFields: (keyof T)[];
  /**
   * Column fields configuration
   */
  columnFields: (keyof T)[];
  /**
   * Value field configuration
   */
  valueField?: keyof T;
  /**
   * Aggregator type
   */
  aggregator: string;
  /**
   * Chart type to render
   */
  chartType?: "bar" | "line" | "doughnut" | "pie";
  /**
   * Whether to show row totals
   */
  showRowTotals?: boolean;
  /**
   * Whether to show column totals
   */
  showColumnTotals?: boolean;
}

type ChartType = "bar" | "line" | "doughnut" | "pie";

const ChartExport = <T extends Record<string, unknown>>({
  pivotData,
  rowFields,
  columnFields,
  valueField,
  aggregator,
  chartType = "bar",
}: ChartExportProps<T>) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const [selectedChartType, setSelectedChartType] =
    useState<ChartType>(chartType);
  const [isExporting, setIsExporting] = useState(false);

  // Convert pivot data to Chart.js format
  const getChartData = useCallback(() => {
    const { rowKeyTuples, colKeyTuples, cells } = pivotData;

    if (selectedChartType === "bar" || selectedChartType === "line") {
      // For bar/line charts, use row groups as labels and columns as datasets
      const labels = rowKeyTuples.map((tuple) =>
        tuple.map((v) => String(v ?? "")).join(" / ")
      );

      const datasets = colKeyTuples.map((colTuple, colIdx) => {
        const colKey = colTuple.map((v) => String(v ?? "")).join(" / ");
        const colLabel =
          columnFields.length > 0 ? colKey : `Column ${colIdx + 1}`;

        const data = rowKeyTuples.map((rowTuple) => {
          const rowKey = rowTuple.map((v) => String(v ?? "")).join("||");
          const cell = cells[rowKey]?.[colKey];
          return cell ? cell.value : 0;
        });

        // Generate a color for each dataset
        const hue = (colIdx * 137.5) % 360;
        const color = `hsl(${hue}, 70%, 60%)`;

        return {
          label: colLabel,
          data,
          backgroundColor: color,
          borderColor: color,
          borderWidth: 1,
          type: selectedChartType as "bar" | "line",
        };
      });

      return { labels, datasets };
    } else {
      // For pie/doughnut charts, aggregate all data into a single dataset
      const aggregatedData: { [key: string]: number } = {};

      // Sum up all cell values
      Object.values(cells).forEach((colCells) => {
        Object.values(colCells).forEach((cell) => {
          const colKey = Object.keys(colCells).find(
            (key) => colCells[key] === cell
          );
          if (colKey) {
            const label = colKey || "Unknown";
            aggregatedData[label] = (aggregatedData[label] || 0) + cell.value;
          }
        });
      });

      const labels = Object.keys(aggregatedData);
      const data = Object.values(aggregatedData);
      const backgroundColor = labels.map((_, idx) => {
        const hue = (idx * 137.5) % 360;
        return `hsl(${hue}, 60%, 60%)`;
      });

      return {
        labels,
        datasets: [
          {
            data,
            backgroundColor,
            borderColor: backgroundColor.map((color) => color),
            borderWidth: 1,
          },
        ],
      };
    }
  }, [pivotData, rowFields, columnFields, selectedChartType]);

  const chartData = getChartData();

  // Don't render chart if no data
  if (!chartData.labels.length || !chartData.datasets.length) {
    return (
      <div className="mb-6 p-4 border border-gray-200 rounded-lg bg-white">
        <div className="text-center text-gray-500 py-8">
          No data available for chart visualization. Please select row and
          column fields.
        </div>
      </div>
    );
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: true,
        text: `Pivot Table: ${aggregator} of ${String(
          valueField || "records"
        )}`,
        font: {
          size: 16,
          weight: "bold" as const,
        },
      },
      legend: {
        display: true,
        position: "top" as const,
      },
      tooltip: {
        enabled: true,
      },
    },
    scales:
      selectedChartType === "bar" || selectedChartType === "line"
        ? {
            y: {
              type: "linear" as const,
              beginAtZero: true,
              title: {
                display: true,
                text: aggregator === "count" ? "Count" : "Value",
              },
            },
            x: {
              type: "category" as const,
              title: {
                display: true,
                text: rowFields.map((f) => String(f)).join(" / "),
              },
            },
          }
        : undefined,
  };

  const exportChartAsImage = useCallback(async () => {
    if (!chartRef.current) return;

    setIsExporting(true);
    try {
      const canvas = await html2canvas(chartRef.current, {
        backgroundColor: "#ffffff",
        scale: 2, // Higher resolution
        useCORS: true,
        allowTaint: true,
      });

      // Create download link
      const link = document.createElement("a");
      link.download = `pivot-chart-${selectedChartType}-${Date.now()}.png`;
      link.href = canvas.toDataURL("image/png");
      link.click();
    } catch (error) {
      console.error("Error exporting chart:", error);
    } finally {
      setIsExporting(false);
    }
  }, [selectedChartType]);

  const chartTypes: { value: ChartType; label: string }[] = [
    { value: "bar", label: "Bar Chart" },
    { value: "line", label: "Line Chart" },
    { value: "doughnut", label: "Doughnut Chart" },
    { value: "pie", label: "Pie Chart" },
  ];

  return (
    <div className="mb-6 p-4 border border-gray-200 rounded-lg bg-white">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-gray-900">Chart Export</h3>
        <div className="flex items-center space-x-2">
          <Select
            value={selectedChartType}
            onChange={(e) => setSelectedChartType(e.target.value as ChartType)}
            size="sm"
            className="min-w-32"
          >
            {chartTypes.map((type) => (
              <option key={type.value} value={type.value}>
                {type.label}
              </option>
            ))}
          </Select>
          <Button
            onClick={exportChartAsImage}
            disabled={isExporting}
            size="sm"
            variant="secondary"
          >
            {isExporting ? "Exporting..." : "Export as Image"}
          </Button>
        </div>
      </div>

      <div className="text-xs text-gray-600 mb-3">
        Visualize your pivot table data as a chart and export it as a
        high-resolution PNG image. You can adjust your browser window size to
        change the exported image dimensions.
      </div>

      <div
        ref={chartRef}
        className="w-full h-96 bg-white border border-gray-200 rounded p-4"
      >
        <Chart
          type={selectedChartType}
          data={chartData}
          options={chartOptions}
        />
      </div>
    </div>
  );
};

export default ChartExport;
