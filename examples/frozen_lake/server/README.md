# Frozen Lake Game Server

This is the **server-side implementation** of the Frozen Lake game environment that provides an HTTP API for agent evaluation.

## Overview

This server implements the HTTP Rollout Protocol that allows external evaluation frameworks (like eval-protocol) to interact with the game environment through standardized endpoints.

## API Endpoints

### `POST /start_episode`
Initializes a new game episode.

**Response:**
```json
{
  "episode_id": "uuid-string",
  "observation": {
    "position": [0, 0],
    "current_cell": "S",
    "done": false,
    "won": false,
    "visual": "[S] F  F  F \n F  H  F  H \n F  F  F  H \n H  F  F  G ",
    "message": "You are at position (0, 0) on a S cell. Choose your next move carefully."
  }
}
```

### `POST /step`
Executes an action in the game.

**Request:**
```json
{
  "episode_id": "uuid-string",
  "action": 2
}
```

**Action Values:**
- `0` = Left
- `1` = Down
- `2` = Right
- `3` = Up

**Response:**
```json
{
  "observation": {
    "position": [0, 1],
    "current_cell": "F",
    "done": false,
    "won": false,
    "visual": " S [F] F  F \n F  H  F  H \n F  F  F  H \n H  F  F  G ",
    "message": "You are at position (0, 1) on a F cell. Choose your next move carefully."
  },
  "is_done": false
}
```

### `POST /end_episode`
Cleans up a completed episode.

**Request:**
```json
{
  "episode_id": "uuid-string"
}
```

**Response:**
```json
{
  "message": "Episode ended successfully"
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "game": "frozen_lake"
}
```

## Game Logic

### Board Layout
```
S F F F
F H F H
F F F H
H F F G
```

### Game Rules
- **S**: Starting position (safe)
- **F**: Frozen lake (safe to step on)
- **H**: Hole (game over if stepped on)
- **G**: Goal (win condition)

### Win/Loss Conditions
- **Win**: Reach the goal position (G)
- **Loss**: Step on a hole (H) or exceed maximum steps

## Running the Server

### Prerequisites
- Python 3.8+
- FastAPI
- Uvicorn

### Installation
```bash
pip install fastapi uvicorn
```

### Start Server
```bash
python http_rollout_server.py
```

The server will start on `http://localhost:8080`

### Configuration
Environment variables:
- `PORT`: Server port (default: 8080)
- `HOST`: Server host (default: 0.0.0.0)

## Integration Notes

This server is designed to work with any HTTP rollout-compatible evaluation framework. The client side handles:
- Action translation (string → numeric)
- State interpretation
- Reward calculation
- Episode management

## Customization

### Different Board Layouts
Modify the `FROZEN_LAKE_MAP` constant:
```python
FROZEN_LAKE_MAP = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]
```

### Game Variants
- Slippery surfaces (movement uncertainty)
- Larger boards
- Dynamic obstacles
- Multi-goal scenarios

## Development

### Testing the API
```bash
# Health check
curl http://localhost:8080/health

# Start episode
curl -X POST http://localhost:8080/start_episode

# Take action
curl -X POST http://localhost:8080/step \
  -H "Content-Type: application/json" \
  -d '{"episode_id": "your-episode-id", "action": 2}'
```

### Logging
The server logs all API interactions for debugging and monitoring.

## Protocol Compliance

This implementation follows the HTTP Rollout Protocol specification:
- Stateless episode management
- Structured observation format
- Standardized error handling
- Health monitoring endpoint

Game environment developers can use this as a reference implementation for creating their own HTTP rollout-compatible environments.
