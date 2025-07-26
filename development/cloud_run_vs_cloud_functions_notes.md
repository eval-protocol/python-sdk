# Cloud Run vs Cloud Functions for MCP Server Deployment

## Decision Context
Date: 2025-06-25
Context: Evaluating deployment options for stateful MCP servers (like FrozenLake game server)

## Current Architecture
- **MCP Server**: FastMCP with global state management
- **State Components**:
  - `GAME_ENV`: Game environment instance
  - `CURRENT_POSITION`: Player position
  - `TOTAL_MOVES`: Move counter
- **Current Deployment**: Google Cloud Run service
- **Transport**: Streamable HTTP with persistent connections

## Analysis

### Cloud Run (Current Choice) ✅
**Advantages:**
- Container stays alive between requests
- Can maintain in-memory state across requests
- Perfect for stateful applications like games
- Supports persistent connections (WebSocket, SSE)
- FastMCP works out-of-the-box
- Only resets state on container restarts/scaling

**Disadvantages:**
- Slightly higher resource usage (always-on containers)
- More complex deployment (container builds)

### Cloud Functions (Considered Alternative) ❌
**Advantages:**
- Potentially faster cold starts
- Pay-per-request model
- Simpler deployment (source-based)

**Disadvantages:**
- **Stateless by design** - each invocation resets memory
- Would require external state storage (Firestore/Redis/etc.)
- Need session management for multi-move games
- Latency overhead for database operations on each move
- Architecture complexity increase
- Cold starts reset all state

## Decision: Stick with Cloud Run

**Reasoning:**
1. **State Requirements**: MCP server inherently needs state across tool calls
2. **Performance**: In-memory state is faster than database operations
3. **Simplicity**: Current FastMCP code works without modification
4. **Architecture Fit**: Cloud Run designed for stateful web services

## Future Considerations

### If Moving to Cloud Functions Later:
- Implement external state storage (Firestore recommended)
- Add session management with unique game IDs
- Modify tool calls to load/save state
- Handle concurrent access to same game session
- Consider state expiration policies

### Cloud Run Optimizations:
- Use minimum instances for faster response times
- Configure appropriate CPU/memory limits
- Implement health checks for reliability
- Consider regional deployment for latency

## Related Files
- `/examples/frozen_lake_mcp/frozen_lake_mcp_server.py` - Current MCP server
- `/eval_protocol/cli_commands/deploy_mcp.py` - Deployment logic
- `/remote_rollout_test/test_remote_north_star.py` - Remote testing
