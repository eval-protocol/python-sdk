"""
Grid Rendering Utilities

Utilities for rendering grid-based environments in a human-readable format.
"""

from typing import Any


def render_grid(desc, position: int) -> str:
    """
    Render a grid environment showing the current player position.

    Args:
        desc: Grid description (usually from env.desc)
        position: Current player position as 1D index

    Returns:
        String representation of the grid with player position marked
    """
    if desc is None:
        return f"Position: {position} (no grid available)"

    # Convert numpy array or bytes to string if needed
    if hasattr(desc, "shape"):
        size = desc.shape[0]

        # Convert position to row, col coordinates
        row = position // size
        col = position % size

        # Create grid representation
        grid_lines = []
        for r, desc_row in enumerate(desc):
            line = ""
            for c, cell in enumerate(desc_row):
                # Convert bytes to string if needed
                cell_char = cell.decode("utf-8") if isinstance(cell, bytes) else str(cell)

                if r == row and c == col:
                    # Show player position with 'P', unless it's the goal
                    if cell_char == "G":
                        line += "W"  # Won - player reached goal
                    else:
                        line += "P"
                else:
                    # Show original cell
                    line += cell_char
            grid_lines.append(line)

        return "\n".join(grid_lines)
    else:
        # Fallback for other grid formats
        return f"Position: {position}"
