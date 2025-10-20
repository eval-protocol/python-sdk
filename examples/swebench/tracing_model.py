"""
Custom model classes for integrating mini-swe-agent with eval-protocol's tracing infrastructure.

## Why This File Exists

mini-swe-agent is an autonomous agent that makes 20-100+ LLM API calls per SWE-bench instance
(e.g., reading files, editing code, running tests). To debug agent behavior and display results
in eval-protocol's UI, we need to capture and analyze every LLM call.

This file bridges mini-swe-agent (which uses LitellmModel) with the Fireworks tracing proxy
(which requires specific URL patterns and SDK usage).

## Problem Without This File

By default, mini-swe-agent would:
- Call Fireworks API directly (no tracing)
- Agent conversations invisible in eval-protocol UI
- Can't debug why agent failed
- No cost tracking per call
- Model names get mangled by litellm routing

## What These Classes Do

### FireworksCompatibleModel (Base)
- Extends mini-swe-agent's LitellmModel
- Handles Fireworks API compatibility:
  * Strips non-standard message fields that Fireworks API rejects
  * Adds stop sequences to prevent common agent failure modes
  * Applies temperature/reasoning overrides from wrapper script
- Used when tracing isn't needed (direct Fireworks API calls)

### TracingFireworksModel (For eval-protocol integration)
- Extends FireworksCompatibleModel
- Routes ALL LLM calls through Fireworks tracing proxy instead of direct API
- Uses OpenAI SDK (not litellm) to preserve full model names
"""

import sys
import os

from minisweagent.models.litellm_model import LitellmModel


class FireworksCompatibleModel(LitellmModel):
    """
    Fireworks-compatible wrapper for LitellmModel.
    """

    def __init__(self, **kwargs):
        model_id = os.environ.get("FIREWORKS_MODEL_ID")
        if model_id:
            kwargs["model_name"] = model_id

        if "model_kwargs" not in kwargs:
            kwargs["model_kwargs"] = {}

        # CRITICAL: Set drop_params to False so stop sequences aren't stripped!
        kwargs["model_kwargs"]["drop_params"] = False

        # Get existing stop sequences
        existing_stop = kwargs["model_kwargs"].get("stop", [])
        if isinstance(existing_stop, str):
            existing_stop = [existing_stop]
        elif existing_stop is None:
            existing_stop = []

        # Add stop sequences (only the non-natural ones)
        # stop_sequences = existing_stop + [
        #     # ASCII versions
        #     "<|User|>",
        #     "<|Assistant|>",
        #     # Full-width PIPE versions (U+FF5C)
        #     "<｜User|>",  # \uff5c
        #     "<｜Assistant|>",
        #     "```<｜",
        #     "<｜User",
        #     "<｜Ass",
        #     # Full-width LETTER L versions (U+FF4C)
        #     "<ｌUser|>",  # \uff4c
        #     "<ｌAssistant|>",
        #     "```<ｌ",
        #     "<ｌUser",
        #     "<ｌAss",
        # ]
        # kwargs["model_kwargs"]["stop"] = stop_sequences
        kwargs["model_kwargs"]["max_tokens"] = 1024  # Reduce to 1024 to save tokens

        if "temperature" not in kwargs["model_kwargs"]:
            kwargs["model_kwargs"]["temperature"] = 0.0

        # Apply per-run overrides injected by the wrapper (no environment variables)
        overrides = globals().get("WRAPPER_MODEL_OVERRIDES")
        if isinstance(overrides, dict):
            if overrides.get("reasoning") in ("low", "medium", "high"):
                kwargs["model_kwargs"]["reasoning_effort"] = overrides["reasoning"]
            if overrides.get("temperature") is not None:
                try:
                    kwargs["model_kwargs"]["temperature"] = float(overrides["temperature"])
                except Exception:
                    pass
            if overrides.get("max_tokens") is not None:
                try:
                    kwargs["model_kwargs"]["max_tokens"] = int(overrides["max_tokens"])
                except Exception:
                    pass

        super().__init__(**kwargs)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        """Remove non-standard fields before sending to Fireworks API."""
        # Keep only standard OpenAI-compatible fields
        clean_messages = []
        for msg in messages:
            clean_msg = {"role": msg["role"], "content": msg["content"]}
            if "tool_calls" in msg:
                clean_msg["tool_calls"] = msg["tool_calls"]
            if "name" in msg:
                clean_msg["name"] = msg["name"]
            clean_messages.append(clean_msg)

        # IMPORTANT: Ensure drop_params stays False in the actual query
        kwargs_with_stop = kwargs.copy()
        if "drop_params" not in kwargs_with_stop:
            kwargs_with_stop["drop_params"] = False

        return super()._query(clean_messages, **kwargs_with_stop)


class TracingFireworksModel(FireworksCompatibleModel):
    """Routes LLM calls through tracing using OpenAI SDK (preserves model name)."""

    def _query(self, messages, **kwargs):
        """Use OpenAI SDK directly to preserve model name through tracing."""
        from openai import OpenAI
        import traceback

        tracing_url = os.environ.get("TRACING_BASE_URL", "")
        api_key = os.environ.get("FIREWORKS_API_KEY", "")

        if not tracing_url:
            print("⚠️  No TRACING_BASE_URL - using parent litellm")
            return super()._query(messages, **kwargs)

        print("\n🔗 OpenAI SDK Call:")
        print(f"   URL: {tracing_url[:60]}...")
        print(f"   Model: {self.config.model_name}")

        try:
            client = OpenAI(base_url=tracing_url, api_key=api_key)

            # Build OpenAI-compatible params
            openai_kwargs = {}
            if self.config.model_kwargs.get("stop"):
                openai_kwargs["stop"] = self.config.model_kwargs["stop"]
                print(f"   Stop sequences: {len(openai_kwargs['stop'])}")
            if self.config.model_kwargs.get("max_tokens"):
                openai_kwargs["max_tokens"] = self.config.model_kwargs["max_tokens"]
            if self.config.model_kwargs.get("temperature") is not None:
                openai_kwargs["temperature"] = self.config.model_kwargs["temperature"]

            # CRITICAL: Clean messages - remove 'extra' fields that OpenAI API doesn't accept!
            clean_messages = []
            for msg in messages:
                clean_msg = {"role": msg["role"], "content": msg["content"]}
                # Preserve standard fields only
                if "name" in msg:
                    clean_msg["name"] = msg["name"]
                if "tool_calls" in msg:
                    clean_msg["tool_calls"] = msg["tool_calls"]
                clean_messages.append(clean_msg)

            print(f"   Messages: {len(clean_messages)} (cleaned)")
            print("   Making call...")

            # OpenAI SDK call
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=clean_messages,  # ← Use cleaned messages!
                **openai_kwargs,
            )

            print("   ✅ Call succeeded!")
            print(f"   Response ID: {response.id}")
            print(f"   Tokens: {response.usage.total_tokens if response.usage else 'N/A'}\n")

            return response

        except Exception as e:
            print("\n❌ ERROR in TracingFireworksModel._query:")
            print(f"   {type(e).__name__}: {e}")
            traceback.print_exc()
            raise
