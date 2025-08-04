"""
Mock MCP for a Discord server
"""

import datetime
from typing import Dict, Optional

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


def format_timestamp_for_llm(dt: datetime.datetime) -> str:
    """Format timestamp in a human-readable way that's optimal for LLMs"""
    # Convert to UTC if it has timezone info, otherwise assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    else:
        dt = dt.astimezone(datetime.timezone.utc)

    # Format as "January 15, 2024 at 2:30 PM UTC"
    return dt.strftime("%B %d, %Y at %I:%M %p UTC")


def extract_message_data(message_data: dict) -> dict:
    """Extract common message data from a mock Discord message"""
    return {
        "id": message_data["id"],
        "author": message_data["author"],
        "content": message_data["content"],
        "is_sent_by_fireworks_employee": message_data.get("is_sent_by_fireworks_employee", False),
        "timestamp": message_data["timestamp"],
        "reactions": message_data.get("reactions", []),
        "reply_info": message_data.get("reply_info"),
    }


def format_reaction(r: dict) -> str:
    """Format a single reaction for display"""
    return f"{r['emoji']}({r['count']})"


def format_message_lines(message_data: dict, message_idx: int, indent: str = "  ") -> list:
    """Format message data into lines for display"""
    message_lines = [
        f"{indent}Message {message_idx} (id: {message_data['id']}):",
        f"{indent}  Author: {message_data['author']}{' (Fireworks Employee)' if message_data['is_sent_by_fireworks_employee'] else ''}",
        f"{indent}  Timestamp: {message_data['timestamp']}",
        f"{indent}  Content: {message_data['content'] or '[No content]'}",
    ]

    # Only add reactions line if there are reactions
    if message_data["reactions"]:
        message_lines.append(
            f"{indent}  Reactions: {', '.join([format_reaction(r) for r in message_data['reactions']])}"
        )

    if message_data["reply_info"]:
        reply_info = message_data["reply_info"]
        message_lines.append(
            f"{indent}  Reply to: message(id={reply_info['reply_to_message_id']}) by {reply_info['reply_to_author']} at {reply_info['reply_to_timestamp']}"
        )

    return message_lines


@mcp.tool
async def list_servers() -> str:
    """Get a list of all Discord servers the bot has access to with their details such as name, id, member count, and creation date."""
    servers = [
        {
            "id": "1234567890123456789",
            "name": "Fireworks AI Community",
            "member_count": 15420,
            "created_at": "2023-01-15T10:30:00+00:00",
        },
        {
            "id": "9876543210987654321",
            "name": "AI Development Hub",
            "member_count": 8234,
            "created_at": "2023-03-22T14:15:00+00:00",
        },
        {
            "id": "5556667778889990001",
            "name": "Machine Learning Enthusiasts",
            "member_count": 5678,
            "created_at": "2023-06-10T09:45:00+00:00",
        },
    ]

    return f"Available Servers ({len(servers)}):\n" + "\n".join(
        f"{s['name']} (ID: {s['id']}, Members: {s['member_count']})" for s in servers
    )


@mcp.tool
async def get_channels(server_id: str) -> ServerChannelsResponse:
    """Get a list of all channels in a Discord server"""
    # Mock channel data for the Fireworks AI Community server
    if server_id == "1234567890123456789":
        channels = {
            "1111111111111111111": ChannelInfo(
                name="general",
                type="text",
                position=0,
                topic="General discussion about AI and machine learning",
                nsfw=False,
                slowmode_delay=0,
                created_at="2023-01-15T10:30:00+00:00",
            ),
            "2222222222222222222": ChannelInfo(
                name="announcements",
                type="text",
                position=1,
                topic="Important announcements and updates",
                nsfw=False,
                slowmode_delay=0,
                created_at="2023-01-15T10:30:00+00:00",
            ),
            "3333333333333333333": ChannelInfo(
                name="help-support",
                type="text",
                position=2,
                topic="Get help with AI-related questions",
                nsfw=False,
                slowmode_delay=5,
                created_at="2023-01-15T10:30:00+00:00",
            ),
            "4444444444444444444": ChannelInfo(
                name="voice-chat",
                type="voice",
                position=3,
                user_limit=10,
                bitrate=64000,
                created_at="2023-01-15T10:30:00+00:00",
            ),
            "5555555555555555555": ChannelInfo(
                name="fireworks-team",
                type="text",
                position=4,
                topic="Internal team discussions",
                nsfw=False,
                slowmode_delay=0,
                created_at="2023-01-15T10:30:00+00:00",
            ),
        }
        return ServerChannelsResponse(
            server_name="Fireworks AI Community",
            server_id="1234567890123456789",
            channels=channels,
            total_channels=len(channels),
        )
    else:
        # Default mock response for other server IDs
        channels = {
            "9999999999999999999": ChannelInfo(
                name="general",
                type="text",
                position=0,
                topic="General discussion",
                nsfw=False,
                slowmode_delay=0,
                created_at="2023-01-01T00:00:00+00:00",
            )
        }
        return ServerChannelsResponse(
            server_name="Mock Server",
            server_id=server_id,
            channels=channels,
            total_channels=len(channels),
        )


@mcp.tool
async def read_messages(channel_id: str, after: datetime.datetime) -> str:
    """Read recent messages from a channel after a given datetime"""
    after_str = f" after {format_timestamp_for_llm(after)}" if after else ""

    # Mock messages for different channels
    if channel_id == "1111111111111111111":  # general channel
        messages = [
            {
                "id": "1000000000000000001",
                "author": "Alice#1234",
                "content": "Has anyone tried the new Fireworks model? It's amazing!",
                "is_sent_by_fireworks_employee": False,
                "timestamp": format_timestamp_for_llm(
                    datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=2)
                ),
                "reactions": [{"emoji": "👍", "count": 5}, {"emoji": "🔥", "count": 3}],
                "reply_info": None,
            },
            {
                "id": "1000000000000000002",
                "author": "Bob#5678",
                "content": "Yes! The performance is incredible. What are you using it for?",
                "is_sent_by_fireworks_employee": False,
                "timestamp": format_timestamp_for_llm(
                    datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1, minutes=30)
                ),
                "reactions": [{"emoji": "👍", "count": 2}],
                "reply_info": {
                    "reply_to_message_id": "1000000000000000001",
                    "reply_to_author": "Alice#1234",
                    "reply_to_content": "Has anyone tried the new Fireworks model? It's amazing!",
                    "reply_to_timestamp": format_timestamp_for_llm(
                        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=2)
                    ),
                },
            },
            {
                "id": "1000000000000000003",
                "author": "Fireworks Team#9999",
                "content": "We're glad you're enjoying it! Let us know if you have any questions.",
                "is_sent_by_fireworks_employee": True,
                "timestamp": format_timestamp_for_llm(
                    datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
                ),
                "reactions": [{"emoji": "❤️", "count": 8}],
                "reply_info": None,
            },
        ]
        channel_name = "general"
    elif channel_id == "3333333333333333333":  # help-support channel
        messages = [
            {
                "id": "2000000000000000001",
                "author": "Charlie#1111",
                "content": "I'm having trouble with the API rate limits. Can someone help?",
                "is_sent_by_fireworks_employee": False,
                "timestamp": format_timestamp_for_llm(
                    datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=3)
                ),
                "reactions": [],
                "reply_info": None,
            },
            {
                "id": "2000000000000000002",
                "author": "Fireworks Support#8888",
                "content": "Hi Charlie! Can you share your current usage and the specific error you're seeing?",
                "is_sent_by_fireworks_employee": True,
                "timestamp": format_timestamp_for_llm(
                    datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=2, minutes=45)
                ),
                "reactions": [{"emoji": "✅", "count": 1}],
                "reply_info": {
                    "reply_to_message_id": "2000000000000000001",
                    "reply_to_author": "Charlie#1111",
                    "reply_to_content": "I'm having trouble with the API rate limits. Can someone help?",
                    "reply_to_timestamp": format_timestamp_for_llm(
                        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=3)
                    ),
                },
            },
        ]
        channel_name = "help-support"
    else:
        # Default mock response for other channel IDs
        messages = [
            {
                "id": "9999999999999999999",
                "author": "MockUser#0000",
                "content": "This is a mock message for testing purposes.",
                "is_sent_by_fireworks_employee": False,
                "timestamp": format_timestamp_for_llm(datetime.datetime.now(datetime.timezone.utc)),
                "reactions": [],
                "reply_info": None,
            }
        ]
        channel_name = "mock-channel"

    # Format messages in a more LLM-friendly, readable way
    formatted = []
    for idx, m in enumerate(messages, 1):
        message_data = extract_message_data(m)
        message_lines = format_message_lines(message_data, idx, "  ")
        formatted.append("\n".join(message_lines))

    result = f"Retrieved {len(messages)} messages{after_str} in {channel_name}:\n\n" + "\n".join(formatted)

    return result


if __name__ == "__main__":
    mcp.run()
