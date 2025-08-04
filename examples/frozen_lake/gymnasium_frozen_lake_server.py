"""
Gymnasium-based Frozen Lake game server implementation.

This implementation wraps the official Gymnasium FrozenLake-v1 environment
and provides the same interface as the hand-rolled implementation for
seamless integration with the HTTP rollout server.
"""

from typing import Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


class GymnasiumFrozenLakeGame:
    """
    Gymnasium-based Frozen Lake game implementation.

    This class wraps the Gymnasium FrozenLake-v1 environment and provides
    a compatible interface with the hand-rolled implementation.

    The game is played on a 4x4 grid where:
    - S: Starting position
    - F: Frozen surface (safe to walk on)
    - H: Hole (game over if you fall in)
    - G: Goal (reach this to win)

    Actions:
    - 0: Left
    - 1: Down
    - 2: Right
    - 3: Up
    """

    def __init__(
        self,
        map_name: str = "4x4",
        is_slippery: bool = False,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the Gymnasium Frozen Lake game.

        Args:
            map_name: Map size ("4x4" or "8x8") - only used if seed is None
            is_slippery: Whether the ice is slippery (stochastic environment)
            render_mode: Rendering mode for Gymnasium environment
            seed: Random seed for reproducible map generation and behavior
            **kwargs: Additional keyword arguments (ignored for compatibility)
        """
        self.map_name = map_name
        self.is_slippery = is_slippery
        self.seed = seed

        # Create the Gymnasium environment
        if seed is not None:
            # Use random map generation with seed for reproducible boards
            size = 4 if map_name == "4x4" else 8
            desc = generate_random_map(size=size, p=0.8, seed=seed)
            self.env = gym.make(
                "FrozenLake-v1",
                desc=desc,
                is_slippery=is_slippery,
                render_mode=render_mode,
            )
        else:
            # Use fixed predefined maps
            self.env = gym.make(
                "FrozenLake-v1",
                map_name=map_name,
                is_slippery=is_slippery,
                render_mode=render_mode,
            )

        # Get environment properties
        self.desc = self.env.unwrapped.desc
        self.nrow, self.ncol = self.desc.shape
        self.nS = self.env.observation_space.n
        self.nA = self.env.action_space.n

        # Find start and goal positions
        self.start_pos = None
        self.goal_pos = None
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.desc[i, j] == b"S":
                    self.start_pos = (i, j)
                elif self.desc[i, j] == b"G":
                    self.goal_pos = (i, j)

        # Initialize state tracking
        self.current_state: Optional[int] = None
        self.current_pos: Optional[Tuple[int, int]] = None
        self.done = False
        self.won = False

        self.reset()

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert state number to (row, col) position."""
        return state // self.ncol, state % self.ncol

    def _pos_to_state(self, row: int, col: int) -> int:
        """Convert (row, col) position to state number."""
        return row * self.ncol + col

    def reset(self) -> Dict:
        """Reset the game to the starting position."""
        if self.seed is not None:
            self.current_state, _ = self.env.reset(seed=self.seed)
        else:
            self.current_state, _ = self.env.reset()
        self.current_pos = self._state_to_pos(self.current_state)
        self.done = False
        self.won = False
        return self._get_observation()

    def step(self, action: Union[int, str]) -> Tuple[Dict, bool]:
        """
        Take a step in the environment.

        Args:
            action: Action to take. Can be:
                - Integer: 0=left, 1=down, 2=right, 3=up
                - String: "left", "down", "right", "up"

        Returns:
            Tuple of (observation, done)
        """
        if self.done:
            return self._get_observation(), True

        # Convert string action to integer if needed
        if isinstance(action, str):
            action_map = {"left": 0, "down": 1, "right": 2, "up": 3}
            if action.lower() not in action_map:
                raise ValueError(f"Invalid action '{action}'. Must be one of: left, down, right, up")
            numeric_action = action_map[action.lower()]
        else:
            numeric_action = action

        if not (0 <= numeric_action < self.nA):
            raise ValueError(f"Invalid action: {numeric_action}. Must be 0-{self.nA-1}")

        # Take the step in the Gymnasium environment
        new_state, reward, terminated, truncated, info = self.env.step(numeric_action)

        # Update our state tracking
        self.current_state = new_state
        self.current_pos = self._state_to_pos(new_state)
        self.done = terminated or truncated
        self.won = reward > 0  # In FrozenLake, reward=1 for reaching goal, 0 otherwise

        return self._get_observation(), self.done

    def _get_observation(self) -> Dict:
        """Get the current observation."""
        if self.current_pos is None:
            raise RuntimeError("Game not initialized")
        row, col = self.current_pos
        cell = self.desc[row, col].decode("utf-8")

        # Create a visual representation
        visual = []
        for i in range(self.nrow):
            row_str = ""
            for j in range(self.ncol):
                if (i, j) == self.current_pos:
                    row_str += "[" + self.desc[i, j].decode("utf-8") + "]"
                else:
                    row_str += " " + self.desc[i, j].decode("utf-8") + " "
            visual.append(row_str)

        obs = {
            "position": self.current_pos,
            "current_cell": cell,
            "done": self.done,
            "won": self.won,
            "visual": "\n".join(visual),
            "message": self._get_message(),
            "state": self.current_state,  # Add the Gymnasium state for compatibility
        }

        return obs

    def _get_message(self) -> str:
        """Get a descriptive message about the current state."""
        if self.done:
            if self.won:
                return "Congratulations! You reached the goal! You win!"
            else:
                return "Oh no! You fell into a hole. Game over."
        else:
            if self.current_pos is None:
                return "Game not initialized"
            row, col = self.current_pos
            cell = self.desc[row, col].decode("utf-8")
            return f"You are at position ({row}, {col}) on a {cell} cell. Choose your next move carefully."

    def close(self):
        """Close the Gymnasium environment."""
        self.env.close()

    def render(self, mode: str = "human"):
        """Render the environment using Gymnasium's rendering."""
        return self.env.render()

    def get_action_meanings(self):
        """Get human-readable action meanings."""
        return ["Left", "Down", "Right", "Up"]

    def get_action_space_info(self):
        """Get information about the action space."""
        return {
            "type": "Discrete",
            "n": int(self.nA),  # Convert numpy int to Python int
            "actions": {0: "left", 1: "down", 2: "right", 3: "up"},
        }

    def get_observation_space_info(self):
        """Get information about the observation space."""
        return {
            "type": "Discrete",
            "n": int(self.nS),  # Convert numpy int to Python int
            "shape": (
                int(self.nrow),
                int(self.ncol),
            ),  # Convert numpy ints to Python ints
            "description": "State number representing position on grid",
        }

    def get_environment_info(self):
        """Get comprehensive environment information."""
        return {
            "name": "FrozenLake-v1",
            "map_name": self.map_name,
            "is_slippery": self.is_slippery,
            "nrow": int(self.nrow),  # Convert numpy int to Python int
            "ncol": int(self.ncol),  # Convert numpy int to Python int
            "action_space": self.get_action_space_info(),
            "observation_space": self.get_observation_space_info(),
            "description": [[cell.decode("utf-8") for cell in row] for row in self.desc],  # Convert to strings
        }


# Backward compatibility: alias the old class name to the new one
FrozenLakeGame = GymnasiumFrozenLakeGame


if __name__ == "__main__":
    """Test the Gymnasium implementation."""
    print("Testing Gymnasium FrozenLake implementation...")

    # Test with deterministic environment
    game = GymnasiumFrozenLakeGame(is_slippery=False)
    print(f"Environment info: {game.get_environment_info()}")

    obs = game.reset()
    print(f"Initial observation: {obs}")

    # Test both string and numeric actions
    test_actions = ["down", "down", "right", "right", "down", "right"]

    for i, action in enumerate(test_actions):
        print(f"\nStep {i+1}: Taking action '{action}'")
        obs, done = game.step(action)
        print(f"Position: {obs['position']}, Done: {done}, Won: {obs['won']}")
        print(f"Message: {obs['message']}")

        if done:
            if obs["won"]:
                print("🎉 Success! Reached the goal!")
            else:
                print("💀 Failed! Fell into a hole!")
            break

    game.close()
    print("\nTest completed!")
