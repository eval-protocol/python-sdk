# Vite App with Runtime Configuration

This Vite app is designed to work with the Python evaluation protocol server and automatically discovers its configuration at runtime.

## Runtime Configuration

The app automatically discovers the server configuration in the following order:

1. **Server-Injected Configuration** (Recommended): The Python server injects configuration directly into the HTML
2. **Location-Based Discovery**: Falls back to discovering configuration from the current URL
3. **Default Values**: Uses localhost:8000 as a last resort

## How It Works

### Server-Side Injection
The Python server (`logs_server.py`) automatically injects configuration into the HTML response:

```html
<script>
window.SERVER_CONFIG = {
    host: "localhost",
    port: "8000",
    protocol: "ws",
    apiProtocol: "http"
};
</script>
```

### Frontend Discovery
The frontend automatically reads this configuration and uses it for WebSocket connections:

```typescript
// First, check if server injected configuration is available
if (window.SERVER_CONFIG) {
    const serverConfig = window.SERVER_CONFIG;
    config.websocket.host = serverConfig.host;
    config.websocket.port = serverConfig.port;
    // ... etc
}
```

## Usage

### Starting the Server
```bash
# Default port 8000
python -m eval_protocol.utils.logs_server

# Custom port
python -m eval_protocol.utils.logs_server --port 9000

# Custom host and port
python -m eval_protocol.utils.logs_server --host 0.0.0.0 --port 9000

# Custom build directory
python -m eval_protocol.utils.logs_server --build-dir /path/to/dist
```

### Building the Frontend
```bash
cd vite-app
pnpm install
pnpm build
```

The built files will be in the `dist/` directory and automatically served by the Python server.

## Benefits

- **No hard-coded ports**: The frontend automatically adapts to whatever port the server runs on
- **Flexible deployment**: Can run on any port without rebuilding the frontend
- **Automatic discovery**: Works whether served from the same origin or different origins
- **Fallback support**: Gracefully handles cases where server injection isn't available
