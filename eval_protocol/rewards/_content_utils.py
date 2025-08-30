"""
Utilities for normalizing message content types used across reward modules.

`Message.content` may be a `str` or a list of OpenAI-style content parts.
These helpers convert such values into plain strings suitable for text
processing without triggering type checker errors.
"""

from typing import Any, List, Optional, Union

from ..models import ChatCompletionContentPartTextParam


def to_text(
    content: Optional[Union[str, List[ChatCompletionContentPartTextParam]]]
) -> str:
    """Return plain text from a Message.content-like value.

    - If content is None, returns "".
    - If content is a string, returns it unchanged.
    - If content is a list of ChatCompletionContentPartTextParam, joins their
      text fields with a single space.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Join any text parts conservatively with a space
    try:
        return " ".join(part.text for part in content)
    except Exception:
        # Best-effort fallback if structure is unexpected
        return ""


def to_text_any(content: Any) -> str:
    """Best-effort conversion of arbitrary content values to text.

    Handles:
    - None -> ""
    - str -> unchanged
    - List[ChatCompletionContentPartTextParam] -> join texts
    - List[dict-like] with key 'text' -> join their 'text' values
    - Other -> "" (avoids surprising stringification of complex objects)
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # List of typed content parts
    if isinstance(content, list) and all(
        hasattr(p, "text") and isinstance(getattr(p, "text"), str) for p in content
    ):
        return " ".join(getattr(p, "text") for p in content)

    # List of dicts with 'text' entries
    if isinstance(content, list) and all(isinstance(p, dict) and "text" in p for p in content):
        try:
            return " ".join(str(p.get("text", "")) for p in content)
        except Exception:
            return ""

    return ""

