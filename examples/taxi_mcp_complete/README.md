# Taxi MCP Complete Example

A comprehensive Model Context Protocol (MCP) implementation for the **Taxi-v3** gymnasium environment. This example demonstrates how to create a fully functional MCP server for reinforcement learning environments using the eval-protocol framework, including local development, testing, and deployment patterns.

## 🎯 What is the Taxi Problem?

The **Taxi** environment is a classic reinforcement learning problem where:
- A taxi must navigate a 5x5 grid world with walls and designated locations
- Pick up a passenger from one of 4 locations (Red, Green, Yellow, Blue)
- Drop off the passenger at their destination
- Avoid illegal pickup/dropoff attempts that result in penalties

**Goal**: Successfully complete passenger trips while minimizing steps and avoiding penalties.

## 🏗️ Project Structure

```
taxi_mcp_complete/
├── README.md                           # This comprehensive guide
├── mcp_server/                         # MCP Server Implementation
│   ├── taxi_mcp_server.py              # 🏭 Production server (vanilla, no seeds)
│   ├── simulation_server.py            # 🚀 Simulation server (multi-session, seeds)
│   ├── taxi_adapter.py                 # Taxi environment adapter
│   └── requirements.txt                # Server dependencies
├── tests/                              # Testing
│   ├── test_record_and_replay_e2e.py   # Main test (use this!)
│   └── conftest.py                     # Test configuration
└── shared_data/                        # Shared Data & Configurations
    └── taxi_rollouts.jsonl             # Environment configurations and prompts
```

## 🏭 Server Types Explained

### `taxi_mcp_server.py` - Production Server
- **Purpose**: Single-session production deployment
- **Use Case**: Individual client connections, demos, simple integrations
- **Concurrency**: ❌ NOT suitable for multiple concurrent rollouts
- **Session Management**: Global state (one game per server instance)
- **Seed Handling**: ❌ No seed handling - uses default environment
- **Architecture**: Vanilla MCP server built on `GymProductionServer`

### `simulation_server.py` - Simulation Server
- **Purpose**: Multi-session simulation environment for evaluation
- **Use Case**: ✅ **PREFERRED for concurrent rollouts and testing**
- **Concurrency**: ✅ Handles multiple parallel sessions properly
- **Session Management**: Per-client isolated sessions with proper seeding
- **Seed Handling**: ✅ Supports different seeds per session for reproducible evaluation
- **Architecture**: Built on `SimulationServerBase` framework

**Key Point**: Production servers are intentionally simple and don't handle seeds. For evaluation with different seeds, always use the simulation server.

## 🎮 Game Environment

**Taxi 5x5 Grid World:**
```
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
```

- **R/G/Y/B**: Pickup/dropoff locations (Red, Green, Yellow, Blue)
- **|, +, -**: Walls that block movement
- **:** Empty navigable spaces
- **t**: Empty taxi (needs to pick up passenger)
- **T**: Taxi with passenger (ready for dropoff)
- **r/g/y/b**: Current destination (lowercase)

### State Space & Actions

- **500 discrete states** encoding taxi position, passenger location, and destination
- **6 actions**: SOUTH (0), NORTH (1), EAST (2), WEST (3), PICKUP (4), DROPOFF (5)

### Rewards
- **-1**: Each step (time penalty)
- **+20**: Successful dropoff
- **-10**: Illegal pickup/dropoff attempt

## 🚀 Quick Start

### Prerequisites

Ensure you have the eval-protocol development environment set up:

```bash
# Activate virtual environment
source .venv/bin/activate

# Ensure dependencies are installed
.venv/bin/pip install -e ".[dev]"

# Set up authentication
export FIREWORKS_API_KEY="your_dev_fireworks_api_key"
export FIREWORKS_ACCOUNT_ID="your_account_id"
```

### Proper Testing (Recommended)

```bash
# Run the proper e2e test (tests simulation server with seeds)
python -m pytest tests/test_record_and_replay_e2e.py -v
```

This test validates:
- Multi-session handling with different seeds
- Record/replay functionality
- Environment adapter integration
- MCP protocol compliance

## 🔧 Configuration Options

The taxi environment supports various configuration parameters:

```python
config = {
    "is_raining": False,      # If True, movement success rate is 80%
    "fickle_passenger": False # If True, passenger may change destinations
}
```

## 🔍 Understanding Game States

The taxi adapter provides helpful state decoding utilities:

```python
from mcp_server.taxi_adapter import TaxiAdapter

adapter = TaxiAdapter()

# Decode a state
state = 328
decoded = adapter.decode_state(state)
# Returns: {
#   "taxi_row": 1,
#   "taxi_col": 3,
#   "passenger_location": 2,  # 0-3: locations, 4: in taxi
#   "destination": 0          # 0: Red, 1: Green, 2: Yellow, 3: Blue
# }

# Get human-readable description for the LLM's reasoning
description = adapter.get_state_description(state)
# Returns: "Taxi at T (1, 3), Passenger at Yellow, Destination: r (Red), must pickup passenger"
```

## 📊 Expected Results

### North Star Test Output

**Recording Mode (First Run):**
```
🌟 Testing Simplified North Star Interface - Taxi Environment
📝 === RECORDING MODE ===
🎬 Setting EP_PLAYBACK_FILE=recording_trajectories.jsonl
✅ Policy created in live mode
✅ MCP environments created successfully
✅ Completed 3 trajectories in 45.23s
🚕 Trajectories completed: 3
✅ Successful: 2/3
🏆 Recording phase completed successfully!
```

**Playback Mode (Subsequent Runs):**
```
🌟 Testing Simplified North Star Interface - Taxi Environment
🎬 === PLAYBACK MODE ===
📂 Using existing file: recording_trajectories.jsonl
✅ Policy created in playback mode
✅ MCP environments created successfully
✅ Completed 3 trajectories in 1.45s
⚡ Playback speedup: ~31x faster than recording
🏆 Playback phase completed successfully!
```

### Integration Test Output
```
✅ Adapter created successfully
✅ Environment created with default config
✅ State decoding works correctly
✅ Action parsing works correctly
✅ Basic gameplay sequence successful
✅ All adapter integration tests passed!
```

### Common Issues

**Server Connection Errors:**
```bash
# Check if server is running
curl http://localhost:8000/mcp/

# Check server logs
cd mcp_server && ../../../.venv/bin/python simulation_server.py --verbose
```

**Environment Creation Fails:**
```bash
# Verify gymnasium installation
python -c "import gymnasium as gym; print(gym.make('Taxi-v3'))"

# Install taxi dependencies
pip install gymnasium[toy_text]
```

**State Decoding Issues:**
- Taxi states range from 0-499
- Verify state bounds before decoding
- Check environment reset returns valid initial state

**Import Errors:**
```bash
# Ensure eval-protocol is installed in development mode
pip install -e .

# Check adapter imports
python -c "from mcp_server.taxi_adapter import TaxiAdapter; print('OK')"
```

### Debug Mode

Enable detailed logging:

```bash
# Start server with debug logging
python simulation_server.py --log-level DEBUG

# Run tests with verbose output
python test_north_star.py --verbose
```

## 🔗 Related Examples

- **`frozen_lake_mcp_complete/`**: Similar MCP implementation for FrozenLake
- **`apps_coding_example/`**: Code execution evaluation example
- **`math_example/`**: Mathematical reasoning evaluation

## 📚 Learning Resources

- **[MCP Server Documentation](docs/mcp_server_readme.md)**: Detailed server implementation guide
- **[CONTRIBUTING.md](../../development/CONTRIBUTING.md)**: Development setup and standards

## 🤝 Contributing

When modifying this example:

1. **Follow [CONTRIBUTING.md](../../development/CONTRIBUTING.md)** standards
2. **Test locally first** using the local testing suite
3. **Validate remote deployment** with remote testing
4. **Update documentation** for any structural changes
5. **Run comprehensive tests** before submitting changes

```bash
# Code quality checks
.venv/bin/black examples/taxi_mcp_complete
.venv/bin/flake8 examples/taxi_mcp_complete
.venv/bin/mypy examples/taxi_mcp_complete
```

---

This example demonstrates a production-ready MCP server implementation suitable for cloud deployment and integration with LLM applications requiring taxi navigation capabilities. It showcases the eval-protocol north star API with record-and-playback for efficient development and evaluation workflows.
