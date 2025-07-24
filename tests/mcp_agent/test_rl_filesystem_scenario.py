import asyncio
import json
import logging
from contextlib import AsyncExitStack

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

INTERMEDIARY_SERVER_URL = "http://localhost:8001/mcp"

# Define expected initial file content
EXPECTED_FILE_CONTENT = "Hello from source"


async def call_tool_on_intermediary(
    mcp_session: ClientSession,
    tool_name: str,  # This is the tool on the *intermediary* server
    rk_session_id: str,
    backend_name_ref: str,
    instance_id: str,
    backend_tool_name: str,  # This is the tool on the *actual backend* MCP server
    backend_tool_args: dict,
) -> dict:
    """Helper to call the 'call_backend_tool' on the intermediary."""
    payload = {
        "args": {
            "rk_session_id": rk_session_id,
            "backend_name_ref": backend_name_ref,
            "instance_id": instance_id,
            "tool_name": backend_tool_name,
            "tool_args": backend_tool_args,
        }
    }
    logger.info(
        f"Calling intermediary tool '{tool_name}' for backend tool '{backend_tool_name}' with payload: {payload}"
    )
    result = await mcp_session.call_tool(tool_name, payload)
    logger.info(f"Raw result from intermediary for '{backend_tool_name}': {result}")

    if result.isError or not result.content or not hasattr(result.content[0], "text"):
        error_message = "Unknown error or non-text content"
        if result.content and hasattr(result.content[0], "text"):
            error_message = result.content[0].text
        elif result.isError:
            error_message = f"Tool call '{tool_name}' (for backend '{backend_tool_name}') returned an error, but no text content part found."
        logger.error(error_message)
        raise ValueError(error_message)

    parsed_result = json.loads(result.content[0].text)
    logger.info(f"Parsed result from intermediary for '{backend_tool_name}': {parsed_result}")

    # Check if the backend tool call itself resulted in an error
    if isinstance(parsed_result, dict) and parsed_result.get("isError"):
        backend_error_message = "Backend tool call failed."
        if (
            parsed_result.get("content")
            and isinstance(parsed_result["content"], list)
            and len(parsed_result["content"]) > 0
        ):
            if isinstance(parsed_result["content"][0], dict) and "text" in parsed_result["content"][0]:
                backend_error_message = parsed_result["content"][0]["text"]
            else:
                backend_error_message = str(parsed_result["content"][0])  # fallback
        raise ValueError(f"Backend tool '{backend_tool_name}' failed: {backend_error_message}")

    return parsed_result


async def main():
    rk_session_id = None
    fs_instance_id = None

    async with AsyncExitStack() as stack:
        try:
            logger.info(f"Connecting to Intermediary MCP server at {INTERMEDIARY_SERVER_URL}")
            transport_tuple = await stack.enter_async_context(streamablehttp_client(INTERMEDIARY_SERVER_URL))
            read_stream, write_stream, _ = transport_tuple

            mcp_client_session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await mcp_client_session.initialize()
            logger.info("ClientSession to Intermediary Server handshake successful.")

            # 1. Initialize session with multiple filesystem_test backends
            num_fs_instances = 2
            init_payload = {
                "args": {
                    "backends": [
                        {
                            "backend_name_ref": "filesystem_test",
                            "num_instances": num_fs_instances,
                        }
                    ]
                }
            }
            logger.info(
                f"Calling 'initialize_session' on Intermediary for {num_fs_instances} instances: {init_payload}"
            )
            init_result_raw = await mcp_client_session.call_tool("initialize_session", init_payload)
            if init_result_raw.isError or not init_result_raw.content:
                raise ValueError(
                    f"initialize_session failed: {getattr(init_result_raw.content[0], 'text', 'Unknown error') if init_result_raw.content else 'No content'}"
                )

            init_result = json.loads(init_result_raw.content[0].text)
            logger.info(f"Parsed 'initialize_session' result: {init_result}")

            rk_session_id = init_result.get("rk_session_id")
            if not rk_session_id:
                raise ValueError("rk_session_id not found in initialize_session response.")

            fs_instances = []
            for backend_res in init_result.get("initialized_backends", []):
                if backend_res.get("backend_name_ref") == "filesystem_test":
                    fs_instances.extend(backend_res.get("instances", []))

            if len(fs_instances) != num_fs_instances:
                raise ValueError(f"Expected {num_fs_instances} filesystem_test instances, got {len(fs_instances)}")

            logger.info(f"Successfully initialized {len(fs_instances)} filesystem instances.")

            for i, instance_info in enumerate(fs_instances):
                fs_instance_id = instance_info.get("instance_id")
                if not fs_instance_id:
                    raise ValueError(f"Instance ID not found for filesystem instance #{i}")

                logger.info(f"\n--- Testing Filesystem Instance #{i+1} (ID: {fs_instance_id}) ---")

                # --- Verify Initial State ---
                logger.info(f"[{fs_instance_id}] Verifying initial state...")
                read_file_args = {"path": "/data/source_dir/file_to_move.txt"}
                file_content_result_wrapper = await call_tool_on_intermediary(
                    mcp_client_session,
                    "call_backend_tool",
                    rk_session_id,
                    "filesystem_test",
                    fs_instance_id,
                    "read_file",
                    read_file_args,
                )
                # The actual file content is nested: result -> 'content' (list) -> first item -> 'text'
                actual_content_list = file_content_result_wrapper.get("content")
                actual_content_text = None
                if actual_content_list and len(actual_content_list) > 0 and isinstance(actual_content_list[0], dict):
                    actual_content_text = actual_content_list[0].get("text")

                if actual_content_text is None:
                    raise ValueError(
                        f"[{fs_instance_id}] Could not extract text content from read_file result: {file_content_result_wrapper}"
                    )

                if actual_content_text.strip() != EXPECTED_FILE_CONTENT:
                    raise ValueError(
                        f"[{fs_instance_id}] Initial file content mismatch. Expected: '{EXPECTED_FILE_CONTENT}', Got: '{actual_content_text.strip()}'"
                    )
                logger.info(f"[{fs_instance_id}] Initial file content verified.")

                # Check target_dir is empty (or just contains .gitkeep if that's copied)
                list_target_args = {"path": "/data/target_dir"}
                target_dir_list = await call_tool_on_intermediary(
                    mcp_client_session,
                    "call_backend_tool",
                    rk_session_id,
                    "filesystem_test",
                    fs_instance_id,
                    "list_directory",
                    list_target_args,
                )
                target_dir_listing_str = target_dir_list.get("content")[0].get("text", "").strip()

                is_target_initially_correct = (
                    target_dir_listing_str == "" or target_dir_listing_str == "[FILE] .gitkeep"
                )
                if not is_target_initially_correct:
                    raise ValueError(
                        f"[{fs_instance_id}] Initial target_dir not as expected. Content string: '{target_dir_listing_str}'"
                    )
                logger.info(
                    f"[{fs_instance_id}] Initial target_dir state verified. Listing: '{target_dir_listing_str}'"
                )

                # --- Simulate Agent Action: Move the file ---
                logger.info(f"[{fs_instance_id}] Simulating agent action: moving file...")
                move_file_args = {
                    "source": "/data/source_dir/file_to_move.txt",
                    "destination": "/data/target_dir/file_to_move.txt",
                }
                await call_tool_on_intermediary(
                    mcp_client_session,
                    "call_backend_tool",
                    rk_session_id,
                    "filesystem_test",
                    fs_instance_id,
                    "move_file",
                    move_file_args,
                )
                logger.info(f"[{fs_instance_id}] Move file action executed.")

                # --- Verify Final State ---
                logger.info(f"[{fs_instance_id}] Verifying final state...")
                read_moved_file_args = {"path": "/data/target_dir/file_to_move.txt"}
                moved_file_content_result_wrapper = await call_tool_on_intermediary(
                    mcp_client_session,
                    "call_backend_tool",
                    rk_session_id,
                    "filesystem_test",
                    fs_instance_id,
                    "read_file",
                    read_moved_file_args,
                )
                moved_actual_content_list = moved_file_content_result_wrapper.get("content")
                moved_actual_content_text = None
                if (
                    moved_actual_content_list
                    and len(moved_actual_content_list) > 0
                    and isinstance(moved_actual_content_list[0], dict)
                ):
                    moved_actual_content_text = moved_actual_content_list[0].get("text")

                if moved_actual_content_text is None:
                    raise ValueError(
                        f"[{fs_instance_id}] Could not extract text content from read_file result after move: {moved_file_content_result_wrapper}"
                    )

                if moved_actual_content_text.strip() != EXPECTED_FILE_CONTENT:
                    raise ValueError(
                        f"[{fs_instance_id}] Moved file content mismatch. Expected: '{EXPECTED_FILE_CONTENT}', Got: '{moved_actual_content_text.strip()}'"
                    )
                logger.info(f"[{fs_instance_id}] File successfully moved to target_dir and content verified.")

                list_source_args = {"path": "/data/source_dir"}
                source_dir_list_result = await call_tool_on_intermediary(
                    mcp_client_session,
                    "call_backend_tool",
                    rk_session_id,
                    "filesystem_test",
                    fs_instance_id,
                    "list_directory",
                    list_source_args,
                )
                source_dir_listing_str = source_dir_list_result.get("content")[0].get("text", "").strip()

                if "file_to_move.txt" in source_dir_listing_str:
                    raise ValueError(
                        f"[{fs_instance_id}] File 'file_to_move.txt' still present in source_dir after move. Listing: '{source_dir_listing_str}'"
                    )
                logger.info(f"[{fs_instance_id}] File successfully removed from source_dir.")
                logger.info(f"--- Test for Instance ID {fs_instance_id} PASSED ---")

            logger.info("All RL Filesystem Scenario Instances Test PASSED!")

        except Exception as e:
            logger.error(
                f"RL Filesystem Scenario Test FAILED (instance {fs_instance_id if fs_instance_id else 'N/A'}): {e}",
                exc_info=True,
            )
            raise  # Re-raise to make test runner fail
        finally:
            # ClientSession does not have a public 'is_closed'.
            # The 'async with mcp_client_session:' handles its closure.
            # For explicit cleanup call, just check if session object exists.
            if rk_session_id and mcp_client_session:
                logger.info(f"Cleaning up session: {rk_session_id}")
                cleanup_payload = {"args": {"rk_session_id": rk_session_id}}
                await mcp_client_session.call_tool("cleanup_session", cleanup_payload)
                logger.info("Cleanup session call completed.")


if __name__ == "__main__":
    asyncio.run(main())
