import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import "./index.css";

interface FileUpdate {
  type: "file_changed" | "file_created" | "file_deleted";
  path: string;
  timestamp: string;
}

function App() {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Connect to WebSocket for file updates
    const ws = new WebSocket("ws://localhost:4789/ws");

    ws.onopen = () => {
      console.log("Connected to file watcher");
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const update: FileUpdate = JSON.parse(event.data);
        console.log(update);
      } catch (error) {
        console.error("Failed to parse WebSocket message:", error);
      }
    };

    ws.onclose = () => {
      console.log("Disconnected from file watcher");
      setIsConnected(false);
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      setIsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, []);

  return (
    <div className="app">
      <nav className="navbar">
        <div className="nav-container">
          <div className="nav-brand">
            <h1>Eval Protocol Logs</h1>
            <div className="connection-status">
              <span
                className={`status-dot ${
                  isConnected ? "connected" : "disconnected"
                }`}
              ></span>
              {isConnected ? "Connected" : "Disconnected"}
            </div>
          </div>
          <div className="nav-links">
            <Link to="/" className="nav-link">
              Home
            </Link>
            <Link to="/logs" className="nav-link">
              Logs
            </Link>
            <Link to="/about" className="nav-link">
              About
            </Link>
            <Link to="/dashboard" className="nav-link">
              Dashboard
            </Link>
          </div>
        </div>
      </nav>

      <main className="main-content">TODO</main>
    </div>
  );
}

export default App;
