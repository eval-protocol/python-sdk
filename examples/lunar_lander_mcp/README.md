# LunarLander MCP Server Example

This example demonstrates a visual Gymnasium environment (LunarLander) integrated with the Model Context Protocol (MCP), featuring image rendering and complex dependencies that test conda isolation.

## Overview

The LunarLander environment is a classic rocket trajectory optimization problem where an agent must learn to land a spacecraft safely on a landing pad. This example serves as a test case for:

- **Visual Environment Support**: Returns rendered frames as base64-encoded images
- **Complex Dependencies**: Requires `swig` and `gymnasium[box2d]` for box2d physics
- **Conda Isolation Testing**: Tests the managed simulation server's ability to handle environments with external dependencies

## Dependencies

This example requires external dependencies that must be properly handled by conda isolation:

```bash
# Required system dependency
pip install swig

# Gymnasium with box2d physics
pip install gymnasium[box2d]

# Rendering support
pip install pygame
```

## Environment Details

- **Action Space**: Discrete(4) - NOTHING, FIRE_LEFT, FIRE_MAIN, FIRE_RIGHT
- **Observation Space**: Box(8) - [x, y, vx, vy, angle, angular_velocity, leg1_contact, leg2_contact]
- **Reward**: Based on distance to landing pad, velocity, angle, fuel consumption, and landing success
- **Episode End**: Landing (success/crash), going out of bounds, or time limit

## Usage

### Direct Server

```bash
cd examples/lunar_lander_mcp/mcp_server
python lunar_lander_mcp_server.py --port 8000 --seed 42
```

### With Managed Simulation Server

```bash
cd examples/frozen_lake_mcp_complete/mcp_server
python managed_simulation_server.py --port 9002 --use-conda-isolation \
  --production-script /path/to/lunar_lander_mcp_server.py \
  --requirements /path/to/requirements.txt
```

### Generate Sample Trajectory Images

```bash
cd examples/lunar_lander_mcp
python generate_sample_images.py
```

This creates a `sample_trajectory/` directory with:
- `step_*.png` - Rendered frames showing the lander in action
- `trajectory_summary.json` - Complete trajectory data with observations and rewards

### Test Conda Isolation

```bash
cd examples/lunar_lander_mcp
python test_lunar_lander_conda.py
```

This verifies that the managed simulation server can properly:
- Create isolated conda environments
- Install complex dependencies (swig, box2d)
- Run lunar lander simulations with visual rendering

## MCP Resources

- `game://initial_state` - Initial environment state with rendered frame
- `game://current_frame` - Current rendered frame as base64 image
- `game://action_space` - Available actions and descriptions
- `game://observation_space` - Observation vector description

## MCP Tools

- `lander_action` - Control the lunar lander with actions: NOTHING, FIRE_LEFT, FIRE_MAIN, FIRE_RIGHT

## Testing Conda Isolation

This example is specifically designed to test conda environment isolation:

1. **Complex Dependencies**: Requires compilation of box2d via swig
2. **Visual Rendering**: Tests pygame integration
3. **Dependency Management**: Verifies requirements.txt installation in isolated environments

To test conda isolation:

```bash
# This should create a fresh conda environment and install all dependencies
python managed_simulation_server.py --port 9002 --use-conda-isolation --verbose
```

## Expected Behavior

- **Successful Landing**: Lander touches down gently with both legs, positive reward
- **Crash**: Lander hits ground too hard or at wrong angle, negative reward
- **Out of Bounds**: Lander leaves the visible area, episode terminates
- **Visual Feedback**: Each action returns a rendered frame showing the current state

## File Structure

```
lunar_lander_mcp/
├── README.md                     # This file
└── mcp_server/
    ├── requirements.txt          # Dependencies for conda isolation
    ├── lunar_lander_adapter.py   # Environment adapter with rendering
    └── lunar_lander_mcp_server.py # MCP server implementation
```

## Troubleshooting

### Swig Installation Issues

If you encounter swig compilation errors:

```bash
# Ubuntu/Debian
sudo apt-get install swig

# macOS
brew install swig

# Windows
# Download from http://www.swig.org/download.html
```

### Box2D Installation Issues

```bash
# Clear pip cache and reinstall
pip cache purge
pip install --no-cache-dir gymnasium[box2d]
```

### Rendering Issues

```bash
# Install pygame if rendering fails
pip install pygame

# For headless environments, you may need virtual display
sudo apt-get install xvfb
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &
```
