import { useEffect, useState, useRef } from "react";
interface FileUpdate {
  type: "file_changed" | "file_created" | "file_deleted";
  path: string;
  timestamp: string;
}

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;
  const baseDelay = 1000; // 1 second

  const connectWebSocket = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    const ws = new WebSocket("ws://localhost:8000/ws");
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("Connected to file watcher");
      setIsConnected(true);
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
      setIsConnected(false);

      // Attempt to reconnect if not a normal closure
      if (
        event.code !== 1000 &&
        reconnectAttemptsRef.current < maxReconnectAttempts
      ) {
        scheduleReconnect();
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      setIsConnected(false);
    };
  };

  const scheduleReconnect = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    const delay = baseDelay * Math.pow(2, reconnectAttemptsRef.current); // Exponential backoff
    console.log(
      `Scheduling reconnect attempt ${
        reconnectAttemptsRef.current + 1
      } in ${delay}ms`
    );

    reconnectTimeoutRef.current = setTimeout(() => {
      reconnectAttemptsRef.current++;
      console.log(
        `Attempting to reconnect (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`
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
            <div>{isConnected ? "Connected" : "Disconnected"}</div>
          </div>
          <div>TODO</div>
        </div>
      </nav>

      <main>TODO</main>
    </div>
  );
}

export default App;
