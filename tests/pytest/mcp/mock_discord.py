"""
Mock MCP for a Discord server
"""

import datetime
from typing import Dict, Optional

from discord.ext import commands
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Initialize FastMCP server
mcp = FastMCP("discord-server")


# Pydantic models for channel data
class ChannelInfo(BaseModel):
    name: str
    type: str
    position: int
    archived: bool = Field(default=False)
    category_id: Optional[str] = None
    topic: Optional[str] = None
    nsfw: Optional[bool] = None
    slowmode_delay: Optional[int] = None
    user_limit: Optional[int] = None
    bitrate: Optional[int] = None
    created_at: Optional[str] = None


class ServerChannelsResponse(BaseModel):
    server_name: str
    server_id: str
    channels: Dict[str, ChannelInfo]
    total_channels: int


# Server Information Tools
@mcp.tool
async def get_server_info(server_id: str) -> str:
    """Get information about a Discord server"""
    return "Server Information:\n" + "mocked"


@mcp.tool
async def get_channels(server_id: str) -> ServerChannelsResponse:
    """Get a list of all channels in a Discord server"""
    return ServerChannelsResponse(
        server_name="mocked",
        server_id="mocked",
        channels={},
        total_channels=0,
    )


@mcp.tool
async def list_members(server_id: str, limit: int = 100) -> str:
    """Get a list of members in a server"""
    return "Server Members (mocked):\n" + "mocked"


@mcp.tool
async def read_messages(channel_id: str, after: datetime.datetime) -> str:  # noqa: F821
    """Read recent messages from a channel after a given datetime"""
    return "mocked"


@mcp.tool
async def send_message(channel_id: str, content: str) -> str:
    """Send a message to a specific channel"""
    return "mocked"


@mcp.tool
async def add_reaction(channel_id: str, message_id: str, emoji: str) -> str:
    """Add a reaction to a message"""
    return "mocked"


@mcp.tool
async def remove_reaction(channel_id: str, message_id: str, emoji: str) -> str:
    """Remove a reaction from a message"""
    return "mocked"


@mcp.tool
async def get_user_info(user_id: str) -> str:
    """Get information about a Discord user"""
    user_info = {
        "id": "mocked",
        "name": "mocked",
        "discriminator": "mocked",
        "bot": "mocked",
        "created_at": "mocked",
    }
    return (
        "User information:\n"
        + f"Name: {user_info['name']}#{user_info['discriminator']}\n"
        + f"ID: {user_info['id']}\n"
        + f"Bot: {user_info['bot']}\n"
        + f"Created: {user_info['created_at']}"
    )


@mcp.tool
async def list_servers() -> str:
    """Get a list of all Discord servers the bot has access to with their details such as name, id, member count, and creation date."""
    servers = []
    servers.append(
        {
            "id": "mocked",
            "name": "mocked",
            "member_count": "mocked",
            "created_at": "mocked",
        }
    )

    return f"Available Servers ({len(servers)}):\n" + "\n".join(
        f"{s['name']} (ID: {s['id']}, Members: {s['member_count']})" for s in servers
    )
