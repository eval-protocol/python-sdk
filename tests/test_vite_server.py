import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eval_protocol.utils.vite_server import ViteServer


class TestViteServer:
    """Test ViteServer class."""

    @pytest.fixture
    def temp_build_dir_with_files(self):
        """Create a temporary build directory with index.html and assets/ directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create index.html
            (temp_path / "index.html").write_text("<html><body>Test</body></html>")

            # Create assets directory and some files inside it
            assets_dir = temp_path / "assets"
            assets_dir.mkdir()
            (assets_dir / "app.js").write_text("console.log('test');")
            (assets_dir / "style.css").write_text("body { color: black; }")

            # Optionally, create a nested directory inside assets
            (assets_dir / "nested").mkdir()
            (assets_dir / "nested" / "file.txt").write_text("nested content")

            yield temp_path

    def test_initialization(self, temp_build_dir_with_files):
        """Test ViteServer initialization."""
        vite_server = ViteServer(build_dir=str(temp_build_dir_with_files), host="localhost", port=8000)

        assert vite_server.build_dir == temp_build_dir_with_files
        assert vite_server.host == "localhost"
        assert vite_server.port == 8000
        assert vite_server.index_file == "index.html"
        assert vite_server.app is not None

    def test_initialization_invalid_build_dir(self):
        """Test ViteServer initialization with invalid build directory."""
        with pytest.raises(FileNotFoundError):
            ViteServer(build_dir="nonexistent_dir")

    def test_initialization_invalid_index_file(self, temp_build_dir_with_files):
        """Test ViteServer initialization with invalid index file."""
        # Remove the index.html file
        (temp_build_dir_with_files / "index.html").unlink()

        with pytest.raises(FileNotFoundError):
            ViteServer(build_dir=str(temp_build_dir_with_files))

    def test_html_injection_in_vite_server(self, temp_build_dir_with_files):
        """Test that ViteServer injects server configuration into HTML."""
        # Create a more complex HTML file for testing injection
        index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test App</title>
    <link rel="stylesheet" href="/assets/style.css">
</head>
<body>
    <div id="app">Test Application</div>
    <script src="/assets/app.js"></script>
</body>
</html>"""

        # Write the test HTML
        (temp_build_dir_with_files / "index.html").write_text(index_html)

        # Create ViteServer instance
        vite_server = ViteServer(build_dir=str(temp_build_dir_with_files), host="localhost", port=8000)

        # Test the HTML injection method directly
        injected_html = vite_server._inject_config_into_html(index_html)

        # Verify server configuration is injected
        assert "window.SERVER_CONFIG" in injected_html
        assert 'host: "localhost"' in injected_html
        assert 'port: "8000"' in injected_html
        assert 'protocol: "ws"' in injected_html
        assert 'apiProtocol: "http"' in injected_html

        # Verify injection happens before closing </head> tag
        head_end_index = injected_html.find("</head>")
        config_script_index = injected_html.find("window.SERVER_CONFIG")
        assert config_script_index < head_end_index

        # Verify the original HTML structure is preserved
        assert '<div id="app">Test Application</div>' in injected_html
        assert '<script src="/assets/app.js"></script>' in injected_html

        # Test that the injected config is valid JavaScript
        assert "window.SERVER_CONFIG = {" in injected_html
        assert "};" in injected_html

    def test_html_injection_without_head_tag(self, temp_build_dir_with_files):
        """Test HTML injection when no </head> tag is present."""
        # Create HTML without </head> tag
        simple_html = """<!DOCTYPE html>
<html>
<body>
    <h1>Simple App</h1>
    <p>No head tag</p>
</body>
</html>"""

        (temp_build_dir_with_files / "index.html").write_text(simple_html)

        vite_server = ViteServer(build_dir=str(temp_build_dir_with_files), host="127.0.0.1", port=9000)

        injected_html = vite_server._inject_config_into_html(simple_html)

        # Verify config is injected at the beginnin
        assert injected_html.strip().startswith("<script>")
        assert "window.SERVER_CONFIG" in injected_html
        assert 'host: "127.0.0.1"' in injected_html
        assert 'port: "9000"' in injected_html

        # Verify original content is preserved
        assert "<h1>Simple App</h1>" in injected_html
        assert "<p>No head tag</p>" in injected_html

    def test_vite_server_endpoints_with_injection(self, temp_build_dir_with_files):
        """Test that ViteServer endpoints serve HTML with injected configuration."""
        # Create test HTML
        test_html = """<!DOCTYPE html>
<html>
<head>
    <title>Test</title>
</head>
<body>
    <h1>Hello World</h1>
</body>
</html>"""

        (temp_build_dir_with_files / "index.html").write_text(test_html)

        vite_server = ViteServer(build_dir=str(temp_build_dir_with_files), host="localhost", port=8000)

        client = TestClient(vite_server.app)

        # Test root endpoint returns HTML with injection
        response = client.get("/")
        assert response.status_code == 200
        assert "window.SERVER_CONFIG" in response.text
        assert 'host: "localhost"' in response.text
        assert 'port: "8000"' in response.text

        # Test SPA routing also returns HTML with injection
        response = client.get("/some/route")
        assert response.status_code == 200
        assert "window.SERVER_CONFIG" in response.text
        assert "Hello World" in response.text

        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["build_dir"] == str(temp_build_dir_with_files)

    def test_static_file_serving(self, temp_build_dir_with_files):
        """Test that static files are served correctly."""
        vite_server = ViteServer(build_dir=str(temp_build_dir_with_files))
        client = TestClient(vite_server.app)

        # Test serving static files
        response = client.get("/assets/app.js")
        assert response.status_code == 200
        assert "console.log('test')" in response.text

        response = client.get("/assets/style.css")
        assert response.status_code == 200
        assert "color: black" in response.text

        response = client.get("/assets/nested/file.txt")
        assert response.status_code == 200
        assert "nested content" in response.text

    def test_spa_routing(self, temp_build_dir_with_files):
        """Test SPA routing fallback."""
        vite_server = ViteServer(build_dir=str(temp_build_dir_with_files))
        client = TestClient(vite_server.app)

        # Test that non-existent routes fall back to index.html
        response = client.get("/some/nonexistent/route")
        assert response.status_code == 200
        assert "Test" in response.text

    def test_api_routes_not_affected(self, temp_build_dir_with_files):
        """Test that API routes are not affected by SPA routing."""
        vite_server = ViteServer(build_dir=str(temp_build_dir_with_files))
        client = TestClient(vite_server.app)

        # Test that API routes return 404 (not index.html)
        response = client.get("/api/test")
        assert response.status_code == 404

    def test_assets_routes_not_affected(self, temp_build_dir_with_files):
        """Test that asset routes are not affected by SPA routing."""
        vite_server = ViteServer(build_dir=str(temp_build_dir_with_files))
        client = TestClient(vite_server.app)

        # Test that asset routes return 404 for non-existent files (not index.html)
        response = client.get("/assets/nonexistent.js")
        assert response.status_code == 404

    def test_health_routes_not_affected(self, temp_build_dir_with_files):
        """Test that health routes are not affected by SPA routing."""
        vite_server = ViteServer(build_dir=str(temp_build_dir_with_files))
        client = TestClient(vite_server.app)

        # Test that health routes work correctly
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
