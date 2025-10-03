import { observer } from "mobx-react";
import { useState, useEffect } from "react";
import {
  LogsResponseSchema,
  type LogEntry,
  type LogsResponse,
} from "../types/eval-protocol";
import { getApiUrl } from "../config";
import Select from "./Select";
import Button from "./Button";

interface LogsSectionProps {
  rolloutId?: string;
}

export const LogsSection = observer(({ rolloutId }: LogsSectionProps) => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedLevel, setSelectedLevel] = useState<string>("");

  const fetchLogs = async (isInitialLoad = false) => {
    if (!rolloutId) return;

    // Only show loading on initial load, not during polling
    if (isInitialLoad) {
      setLoading(true);
    }
    setError(null);

    try {
      const apiUrl = getApiUrl();
      console.log("API URL:", apiUrl);

      const params = new URLSearchParams();
      if (selectedLevel) {
        params.append("level", selectedLevel);
      }
      params.append("limit", "50");

      const fullUrl = `${apiUrl}/api/logs/${rolloutId}?${params}`;
      console.log("Attempting to fetch logs from:", fullUrl);

      let response;
      try {
        response = await fetch(fullUrl);
        console.log("Fetch completed, response:", response);
      } catch (fetchError) {
        console.error("Fetch failed with network error:", fetchError);
        setError(
          `Network error: ${
            fetchError instanceof Error ? fetchError.message : "Unknown error"
          }`
        );
        return;
      }

      if (!response.ok) {
        if (response.status === 503) {
          setError("Elasticsearch is not configured");
          return;
        }

        if (response.status === 404) {
          // Check if we got HTML (server not running) vs JSON (no logs found)
          const contentType = response.headers.get("content-type");
          if (contentType && contentType.includes("text/html")) {
            setError(
              "Logs server not running. Start the logs server to view logs."
            );
            return;
          } else {
            // 404 with JSON content-type means "no logs found" - this is valid
            setLogs([]);
            return;
          }
        }

        // Check if we got HTML instead of JSON (likely a routing issue)
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.includes("text/html")) {
          setError(
            `API endpoint not found. Got HTML response instead of JSON. Status: ${response.status}`
          );
          return;
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data: LogsResponse = LogsResponseSchema.parse(
        await response.json()
      );
      setLogs(data.logs);
    } catch (err) {
      if (err instanceof Error && err.message.includes("Unexpected token")) {
        setError(
          "API returned HTML instead of JSON. Is the logs server running on the correct port?"
        );
      } else {
        setError(err instanceof Error ? err.message : "Failed to fetch logs");
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (rolloutId) {
      fetchLogs(true); // Initial load
      const interval = setInterval(() => fetchLogs(false), 5000); // Poll every 5 seconds without loading state
      return () => clearInterval(interval);
    }
  }, [rolloutId, selectedLevel]);

  if (!rolloutId) {
    return null;
  }

  return (
    <div>
      {/* Content - matching MetadataSection container styling */}
      <div className="border border-gray-200 p-2 w-[1200px] text-xs bg-white">
        {/* Log level filter */}
        <div className="mb-2 flex items-center gap-2">
          <Select
            value={selectedLevel}
            onChange={(e) => setSelectedLevel(e.target.value)}
            size="sm"
          >
            <option value="">All levels</option>
            <option value="DEBUG">DEBUG</option>
            <option value="INFO">INFO</option>
            <option value="WARNING">WARNING</option>
            <option value="ERROR">ERROR</option>
          </Select>
          <Button
            onClick={() => fetchLogs(true)}
            variant="primary"
            size="sm"
            disabled={loading}
          >
            {loading ? "Loading..." : "Refresh Logs"}
          </Button>
        </div>

        {error && (
          <div className="text-red-600 text-xs mb-2 px-3 py-2 bg-red-50 border border-red-200">
            {error}
          </div>
        )}

        {loading && logs.length === 0 && (
          <div className="text-gray-500 text-xs">Loading logs...</div>
        )}

        {logs.length === 0 && !loading && !error && (
          <div className="text-gray-500 text-xs">No logs found</div>
        )}

        {logs.length > 0 && (
          <div className="max-h-80 min-h-4 overflow-auto border border-gray-200">
            <div>
              {logs.map((log, index) => (
                <div
                  key={index}
                  className={`text-xs px-3 py-1 border-b border-gray-200 last:border-b-0 ${
                    index % 2 === 0 ? "bg-white" : "bg-gray-50"
                  }`}
                >
                  <div className="flex items-start gap-2">
                    <span
                      className={`font-medium text-xs flex-shrink-0 ${
                        log.level === "ERROR"
                          ? "text-red-700"
                          : log.level === "WARNING"
                          ? "text-yellow-700"
                          : log.level === "INFO"
                          ? "text-blue-700"
                          : "text-gray-700"
                      }`}
                    >
                      {log.level}
                    </span>
                    <span className="text-gray-500 text-xs flex-shrink-0">
                      {new Date(log["@timestamp"]).toLocaleTimeString()}
                    </span>
                    <span className="text-gray-400 text-xs flex-shrink-0">
                      {log.logger_name}
                    </span>
                    <span className="text-gray-900 break-words min-w-0 flex-1">
                      {log.status_message && (
                        <span className="text-gray-500 font-mono">
                          Status: {log.status_message}{" "}
                        </span>
                      )}
                      {log.message}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
});
