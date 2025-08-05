import { useEffect, useRef } from "react";
import { makeAutoObservable } from "mobx";
import { observer } from "mobx-react";
import Dashboard from "./components/Dashboard";
import type { EvaluationRow } from "./types/eval-protocol";
interface FileUpdate {
  type: "file_changed" | "file_created" | "file_deleted";
  path: string;
  timestamp: string;
}

class GlobalState {
  isConnected: boolean = false;
  dataset: EvaluationRow[] = [];
  constructor() {
    makeAutoObservable(this);
  }

  setDataset(dataset: EvaluationRow[]) {
    this.dataset = dataset;
  }
}

const state = new GlobalState();

const BASE_DELAY = 1000; // 1 second
const MAX_RECONNECT_ATTEMPTS = 5;

const App = observer(() => {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef(0);

  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    const ws = new WebSocket("ws://localhost:8000/ws");
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("Connected to file watcher");
      state.isConnected = true;
      reconnectAttemptsRef.current = 0; // Reset reconnect attempts on successful connection
    };

    ws.onmessage = (event) => {
      try {
        const update: FileUpdate = JSON.parse(event.data);
        console.log(update);
      } catch (error) {
        console.error("Failed to parse WebSocket message:", error);
      }
    };

    ws.onclose = (event) => {
      console.log("Disconnected from file watcher", event.code, event.reason);
      state.isConnected = false;

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
      state.isConnected = false;
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

  useEffect(() => {
    connectWebSocket();

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
    <div>
      <nav>
        <div>
          <div>
            <h1>Eval Protocol Logs</h1>
            <div>{state.isConnected ? "Connected" : "Disconnected"}</div>
          </div>
          <div>
            <Dashboard />
          </div>
        </div>
      </nav>

      <main>TODO</main>
    </div>
  );
});

export default App;
