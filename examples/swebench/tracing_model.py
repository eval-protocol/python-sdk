"""
TracingFireworksModel - Routes through tracing using OpenAI SDK.
"""

import sys
import os

sys.path.insert(0, "/Users/shrey/Documents/cookbook-internal/recipes/eval/swe_bench")

from run_swe_agent_fw import FireworksCompatibleModel


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
