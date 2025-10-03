import logging
import os
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)


class ViteServer:
    """
    Server for serving Vite-built SPA applications.

    This class creates a FastAPI server that serves static files from a Vite build output
    directory (typically 'dist'). It handles SPA routing by serving index.html for
    non-existent routes.

    Args:
        build_dir: Path to the Vite build output directory (default: "dist")
        host: Host to bind the server to (default: "localhost")
        port: Port to bind the server to (default: 8000)
        index_file: Name of the main index file (default: "index.html")
    """

    def __init__(
        self,
        build_dir: str = "dist",
        host: str = "localhost",
        port: int = 8000,
        index_file: str = "index.html",
        app: Optional[FastAPI] = None,
    ):
        self.build_dir = Path(build_dir)
        self.host = host
        self.port = port
        self.index_file = index_file
        self.app = app if app is not None else FastAPI(title="Vite SPA Server")

        # Validate build directory exists
        if not self.build_dir.exists():
            raise FileNotFoundError(f"Build directory '{self.build_dir}' does not exist")

        if not self.build_dir.is_dir():
            raise NotADirectoryError(f"'{self.build_dir}' is not a directory")

        # Check if index.html exists
        index_path = self.build_dir / self.index_file
        if not index_path.exists():
            raise FileNotFoundError(f"Index file '{index_path}' does not exist")

        logger.info(f"Initialized Vite server for build directory: {self.build_dir}")

        # Setup routes
        self._setup_routes()

    def _inject_config_into_html(self, html_content: str) -> str:
        """Inject server configuration into the HTML content."""
        config_script = f"""
<script>
// Server-injected configuration
window.SERVER_CONFIG = {{
    host: "{self.host}",
    port: "{self.port}",
    protocol: "ws",
    apiProtocol: "http"
}};
</script>
"""

        # Insert the config script before the closing </head> tag
        if "</head>" in html_content:
            return html_content.replace("</head>", f"{config_script}</head>")
        else:
            # If no </head> tag, insert at the beginning
            return f"{config_script}{html_content}"

    def _serve_index_with_config(self) -> HTMLResponse:
        """Serve the index.html file with injected configuration."""
        index_path = self.build_dir / self.index_file
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Inject server configuration
            enhanced_html = self._inject_config_into_html(html_content)
            return HTMLResponse(content=enhanced_html)

        raise HTTPException(status_code=404, detail="Index file not found")

    def _setup_routes(self):
        """Set up the API routes for serving the SPA."""

        # Mount static files
        self.app.mount("/assets", StaticFiles(directory=self.build_dir / "assets"), name="assets")

        @self.app.get("/")
        async def root():
            """Serve the main index.html file with injected configuration."""
            return self._serve_index_with_config()

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "ok", "build_dir": str(self.build_dir)}

        # Serve other static files from build directory - this must be last
        @self.app.get("/{path:path}")
        async def serve_spa(path: str):
            """
            Serve the SPA application.

            For existing files, serve them directly. For non-existent routes,
            serve index.html to enable client-side routing.
            """
            file_path = self.build_dir / path

            # If the file exists, serve it
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)

            # For SPA routing, serve index.html for non-existent routes
            # but exclude API routes and asset requests
            if not path.startswith(("api/", "assets/", "health")):
                return self._serve_index_with_config()

            # If we get here, the file doesn't exist and it's not a SPA route
            raise HTTPException(status_code=404, detail="File not found")

    def run(self):
        """
        Run the Vite server.

        Args:
            reload: Whether to enable auto-reload (default: False)
        """
        logger.info(f"Starting Vite server on {self.host}:{self.port}")
        logger.info(f"Serving files from: {self.build_dir}")

        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")
