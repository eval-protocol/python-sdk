import http.server
import json
import logging
import os
import socketserver

PORT = int(os.environ.get("MOCK_MCP_PORT", 8080))  # Port the server will listen on
SERVER_ID = os.environ.get("MOCK_SERVER_ID", "mock_server_default_id")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MockMCPHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/mcp":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            try:
                payload = json.loads(post_data.decode("utf-8"))
                tool_name = payload.get("tool_name")
                arguments = payload.get("arguments", {})

                logger.info(f"Mock Server {SERVER_ID} received tool call: {tool_name} with args: {arguments}")

                response_data = {}
                status_code = 200

                if tool_name == "ping":
                    response_data = {"status": "pong", "server_id": SERVER_ID}
                elif tool_name == "echo":
                    response_data = {
                        "echoed_arguments": arguments,
                        "server_id": SERVER_ID,
                    }
                elif tool_name == "get_server_id":
                    response_data = {"server_id": SERVER_ID}
                elif tool_name == "read_file":  # Mocking a filesystem tool
                    file_path = arguments.get("path")
                    if file_path == "test.txt":
                        response_data = {"content": f"Mock content from {SERVER_ID} for {file_path}"}
                    else:
                        response_data = {"error": "File not found", "path": file_path}
                        status_code = 404  # Or a specific MCP error structure
                else:
                    response_data = {"error": f"Tool '{tool_name}' not found on {SERVER_ID}"}
                    status_code = 400  # Bad Request or specific MCP error

                self.send_response(status_code)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))
                logger.info(f"Mock Server {SERVER_ID} responded with: {response_data}")

            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid JSON payload"}).encode("utf-8"))
                logger.error("Invalid JSON payload received.")
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Internal server error: {str(e)}"}).encode("utf-8"))
                logger.error(f"Internal server error: {e}", exc_info=True)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), MockMCPHandler) as httpd:
        logger.info(f"Mock MCP Server '{SERVER_ID}' starting on port {PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Mock MCP Server shutting down.")
        finally:
            httpd.server_close()
