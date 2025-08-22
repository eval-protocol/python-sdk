import { useEffect, useRef } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { observer } from "mobx-react";
import Dashboard from "./components/Dashboard";
import Button from "./components/Button";
import StatusIndicator from "./components/StatusIndicator";
import { EvaluationRowSchema, type EvaluationRow } from "./types/eval-protocol";
import { WebSocketServerMessageSchema } from "./types/websocket";
import { GlobalState } from "./GlobalState";
import logoLight from "./assets/logo-light.png";
import { getWebSocketUrl, discoverServerConfig } from "./config";

export const state = new GlobalState();

const BASE_DELAY = 1000; // 1 second
const MAX_RECONNECT_ATTEMPTS = 5;

const App = observer(() => {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null
  );
  const reconnectAttemptsRef = useRef(0);

  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    const ws = new WebSocket(getWebSocketUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("Connected to file watcher");
      state.setConnected(true);
      state.setLoading(true); // Set loading when connection opens
      reconnectAttemptsRef.current = 0; // Reset reconnect attempts on successful connection
    };

    ws.onmessage = (event) => {
      try {
        const update = WebSocketServerMessageSchema.parse(
          JSON.parse(event.data)
        );
        if (update.type === "initialize_logs") {
          const rows: EvaluationRow[] = update.logs.map((log) => {
            return EvaluationRowSchema.parse(log);
          });
          console.log("initialize_logs", rows);
          state.upsertRows(rows);
        } else if (update.type === "log") {
          state.setLoading(true); // Set loading for individual log updates
          const row: EvaluationRow = EvaluationRowSchema.parse(update.row);
          console.log("log", row);
          state.upsertRows([row]);
        }
      } catch (error) {
        console.error("Failed to parse WebSocket message:", error);
        state.setLoading(false); // Clear loading state on error
      }
    };

    ws.onclose = (event) => {
      console.log("Disconnected from file watcher", event.code, event.reason);
      state.setConnected(false);
      state.setLoading(false); // Clear loading state on disconnect

      // Attempt to reconnect if not a normal closure
      if (
        event.code !== 1000 &&
        reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS
      ) {
        scheduleReconnect();
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      state.setConnected(false);
      state.setLoading(false); // Clear loading state on error
    };
  };

  const scheduleReconnect = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    const delay = BASE_DELAY * Math.pow(2, reconnectAttemptsRef.current); // Exponential backoff
    console.log(
      `Scheduling reconnect attempt ${
        reconnectAttemptsRef.current + 1
      } in ${delay}ms`
    );

    reconnectTimeoutRef.current = setTimeout(() => {
      reconnectAttemptsRef.current++;
      console.log(
        `Attempting to reconnect (attempt ${reconnectAttemptsRef.current}/${MAX_RECONNECT_ATTEMPTS})`
      );
      connectWebSocket();
    }, delay);
  };

  // Manual refresh handler
  const handleManualRefresh = () => {
    state.setLoading(true); // Set loading when manually refreshing
    if (wsRef.current) {
      try {
        wsRef.current.onclose = null; // Prevent triggering reconnect logic
        wsRef.current.close();
      } catch (e) {}
      wsRef.current = null;
    }
    connectWebSocket();
  };

  useEffect(() => {
    // Discover server configuration first, then connect
    const initializeApp = async () => {
      await discoverServerConfig();
      connectWebSocket();
    };

    initializeApp();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-3">
          <div className="flex justify-between items-center h-10">
            <div className="flex items-center space-x-2">
              <a href="https://evalprotocol.io" target="_blank">
                <img
                  src={logoLight}
                  alt="Eval Protocol"
                  className="h-6 w-auto"
                />
              </a>
            </div>
            <div className="flex items-center gap-2">
              <StatusIndicator
                status={
                  state.isConnected
                    ? { code: 0, message: "Connected", details: [] }
                    : { code: 1, message: "Disconnected", details: [] }
                }
              />
              <Button onClick={handleManualRefresh} className="ml-2">
                Refresh
              </Button>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-3 py-4">
        <Routes>
          <Route path="/" element={<Navigate to="/table" replace />} />
          <Route
            path="/table"
            element={<Dashboard onRefresh={handleManualRefresh} />}
          />
          <Route
            path="/pivot"
            element={<Dashboard onRefresh={handleManualRefresh} />}
          />
        </Routes>
      </main>
    </div>
  );
});

export default App;
