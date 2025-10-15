#!/usr/bin/env python3
"""
Fireworks-compatible wrapper for mini-swe-agent SWE-bench evaluations.

This script handles Fireworks API compatibility by stripping non-standard fields
that mini-swe-agent adds for internal tracking.

Requires fully qualified Fireworks model paths:
- Serverless models: fireworks_ai/accounts/fireworks/models/{model_name}
- Deployed models: fireworks_ai/accounts/{account}/deployedModels/{model_name}

Usage:
    python run_swe_agent_fw.py <fully_qualified_model_path> [options]

Examples:
    # Serverless models
    python run_swe_agent_fw.py fireworks_ai/accounts/fireworks/models/kimi-k2-instruct --instances 10 --workers 5
    python run_swe_agent_fw.py fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct --subset full --workers 8

    # Deployed models
    python run_swe_agent_fw.py fireworks_ai/accounts/cognition/deployedModels/swe-1-mtp-tc1huggf --single 0
    python run_swe_agent_fw.py fireworks_ai/accounts/fireworks/models/kimi-k2-instruct --test

Requirements:
    - mini-swe-agent: pip install mini-swe-agent
    - Fireworks API key: Set via 'mini-extra config set FIREWORKS_API_KEY <key>'
"""

import argparse
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# Import required dependencies
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
import litellm


class FireworksCompatibleModel(LitellmModel):
    """
    Fireworks-compatible wrapper for LitellmModel.
    """

    def __init__(self, **kwargs):
        if model_id := os.environ.get('FIREWORKS_MODEL_ID'):
            kwargs['model_name'] = model_id
        print(f"kwargs: {kwargs}")
        if 'model_kwargs' not in kwargs:
            kwargs['model_kwargs'] = {}
        
        # CRITICAL: Set drop_params to False so stop sequences aren't stripped!
        kwargs['model_kwargs']['drop_params'] = False
        
        # Get existing stop sequences
        existing_stop = kwargs['model_kwargs'].get('stop', [])
        if isinstance(existing_stop, str):
            existing_stop = [existing_stop]
        elif existing_stop is None:
            existing_stop = []
        
        # Add stop sequences (only the non-natural ones)
        stop_sequences = existing_stop + [
           # ASCII versions
            "<|User|>",
            "<|Assistant|>",
            
            # Full-width PIPE versions (U+FF5C)
            "<｜User|>",       # \uff5c
            "<｜Assistant|>",
            "```<｜",
            "<｜User",
            "<｜Ass",
            
            # Full-width LETTER L versions (U+FF4C) 
            "<ｌUser|>",      # \uff4c
            "<ｌAssistant|>",
            "```<ｌ",
            "<ｌUser",
            "<ｌAss",
        ]
        kwargs['model_kwargs']['stop'] = stop_sequences
        kwargs['model_kwargs']['max_tokens'] = 1024  # Reduce to 1024 to save tokens
        
        if 'temperature' not in kwargs['model_kwargs']:
            kwargs['model_kwargs']['temperature'] = 0.0

        # Apply per-run overrides injected by the wrapper (no environment variables)
        overrides = globals().get('WRAPPER_MODEL_OVERRIDES')
        if isinstance(overrides, dict):
            if overrides.get('reasoning') in ('low', 'medium', 'high'):
                kwargs['model_kwargs']['reasoning_effort'] = overrides['reasoning']
            if overrides.get('temperature') is not None:
                try:
                    kwargs['model_kwargs']['temperature'] = float(overrides['temperature'])
                except Exception:
                    pass
            if overrides.get('max_tokens') is not None:
                try:
                    kwargs['model_kwargs']['max_tokens'] = int(overrides['max_tokens'])
                except Exception:
                    pass
        
        super().__init__(**kwargs)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        """Remove non-standard fields before sending to Fireworks API."""
        # Keep only standard OpenAI-compatible fields
        clean_messages = []
        for msg in messages:
            clean_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
            if "tool_calls" in msg:
                clean_msg["tool_calls"] = msg["tool_calls"]
            if "name" in msg:
                clean_msg["name"] = msg["name"]
            clean_messages.append(clean_msg)
        
        # IMPORTANT: Ensure drop_params stays False in the actual query
        kwargs_with_stop = kwargs.copy()
        if 'drop_params' not in kwargs_with_stop:
            kwargs_with_stop['drop_params'] = False
        
        return super()._query(clean_messages, **kwargs_with_stop)

def __get_api_key():
    """Get Fireworks API key from environment or mini-swe-agent config."""
    # Environment variable takes precedence
    if api_key := os.environ.get('FIREWORKS_API_KEY'):
        return api_key

    # Try to get API key from mini-swe-agent's config system
    try:
        from minisweagent.config import get_config
        config = get_config()
        return config.get('FIREWORKS_API_KEY')
    except (ImportError, AttributeError, KeyError):
        # Fallback: check common config file locations
        config_paths = [
            Path.home() / ".config" / "mini-swe-agent" / ".env",
            Path.home() / "Library" / "Application Support" / "mini-swe-agent" / ".env"
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        for line in f:
                            if line.startswith('FIREWORKS_API_KEY='):
                                value = line.split('=', 1)[1].strip()
                                return value.strip("'\"")
                except (IOError, OSError):
                    continue

    return None


def __test_model(model_id):
    """Test model connectivity with a simple completion."""
    from litellm import completion

    # Verify API key exists
    api_key = __get_api_key()
    if not api_key:
        print("Error: FIREWORKS_API_KEY not found.")
        return False

    # Configure environment for litellm
    os.environ['FIREWORKS_API_KEY'] = api_key
    # Assume model_id is fully qualified
    model_name = model_id

    print(f"Testing model: {model_name}")

    try:
        # Send test completion
        response = completion(
            model=model_name,
            messages=[{"role": "user", "content": "Test message. Reply with OK."}],
            temperature=0.0,
            max_tokens=10
        )

        print(f"Success. Response: {response.choices[0].message.content}")
        print(f"Tokens used: {response.usage.total_tokens}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def __validate_environment():
    """Check for required API key."""
    if not __get_api_key():
        print("Warning: FIREWORKS_API_KEY not found.")
        print("Set it with: mini-extra config set FIREWORKS_API_KEY <key>")




def __build_command(args, wrapper_module_path):
    """Build mini-swe-agent command with appropriate arguments."""
    # Construct model class path
    wrapper_module = wrapper_module_path.stem
    model_class = f"{wrapper_module}.FireworksCompatibleModel"

    # Base command - assume model_id is fully qualified
    cmd = [
        sys.executable,
        "-m", "minisweagent.run.mini_extra",
        "swebench-single" if args.single is not None else "swebench",
        "--model", args.model_id,
        "--model-class", model_class,
        "--subset", args.subset,
        "--split", args.split
    ]
    if args.model_class:
        cmd.extend(["--model-class", args.model_class])
    print(f"Output: {args.output}")
    print(args.single)
    # Mode-specific arguments
    print(f"Output: {args.output}")
    print(args.single)
    # Mode-specific arguments
    if args.single is not None:
        # Use batch mode for a single index via slice and write to a per-row directory
        from pathlib import Path
        slice_spec = f"{args.single}:{args.single+1}"
        row_dir = str((Path(args.output) if args.output else Path.cwd()) / f"row_{args.single}")
        cmd = [
            sys.executable,
            "-m", "minisweagent.run.mini_extra",
            "swebench",
            "--model", args.model_id,
            "--model-class", model_class,
            "--subset", args.subset,
            "--split", args.split,
            "--slice", slice_spec,
            "--output", row_dir,
        ]
        if args.model_class:
            cmd.extend(["--model-class", args.model_class])
        print(f"DEBUG: Using batch mode with slice {slice_spec}, output={row_dir}")
    else:
        if args.instances:
            cmd.extend(["--slice", f"0:{args.instances}"])
        cmd.extend(["--workers", str(args.workers), "--output", args.output])

    return cmd

    


def main():
    parser = argparse.ArgumentParser(
        description='Run mini-swe-agent with Fireworks models on SWE-bench',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required model ID
    parser.add_argument('model_id', help='Fireworks model ID')
    parser.add_argument('--model-class', type=str, default=None, help='Optional mini-swe-agent model-class')
    # Execution options
    parser.add_argument('--instances', type=int, help='Number of instances to run')
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers (default: 1)')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--subset', default='verified', choices=['verified', 'lite', 'full'])
    parser.add_argument('--split', default='test', choices=['dev', 'test'])
    parser.add_argument('--single', type=int, metavar='INDEX', help='Run single instance')
    parser.add_argument('--exit-immediately', action='store_true')
    parser.add_argument('--test', action='store_true', help='Test model connectivity')
    parser.add_argument('--reasoning', type=str, choices=['low', 'medium', 'high'], default=None, help='Provider-specific reasoning effort')
    parser.add_argument('--temperature', type=float, default=None, help='Model temperature override')
    parser.add_argument('--max-tokens', type=int, default=None, help='Max tokens override')
    args = parser.parse_args()

    # Handle test mode
    if args.test:
        sys.exit(0 if _test_model(args.model_id) else 1)

    # Validate API key
    __validate_environment()

    # Set default output directory
    if args.output is None:
        safe_model_id = args.model_id.replace("/", "-").replace(":", "-")
        script_dir = Path(__file__).parent.resolve()
        args.output = str(script_dir / f'swebench-{safe_model_id}-results')

    # Create temporary module for importing FireworksCompatibleModel
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        with open(__file__, 'r') as current_file:
            f.write(current_file.read())
        # Inject per-run model overrides directly into the temp module
        f.write("\n# --- Injected by wrapper: per-run model overrides ---\n")
        f.write("WRAPPER_MODEL_OVERRIDES = {\n")
        f.write(f"    'reasoning': {repr(args.reasoning)},\n")
        f.write(f"    'temperature': {repr(args.temperature)},\n")
        f.write(f"    'max_tokens': {repr(args.max_tokens)},\n")
        f.write("}\n")
        temp_module_path = Path(f.name)

    try:
        # Configure environment
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{temp_module_path.parent}:{env.get('PYTHONPATH', '')}"
        # Pass the fully qualified model path to the subprocess
        env['FIREWORKS_MODEL_ID'] = args.model_id

        # Ensure API key is passed to subprocess
        api_key = __get_api_key()
        if api_key:
            env['FIREWORKS_API_KEY'] = api_key

        # No environment variables for model kwargs; overrides are injected into the temp module

        # Build command
        cmd = __build_command(args, temp_module_path)

        # Display configuration
        print(f"Model: {args.model_id}")
        print(f"Output: {args.output}")
        print(f"Workers: {args.workers}")
        if args.instances:
            print(f"Instances: {args.instances}")

        # Debug: Show the actual command being run
        print(f"Command: {' '.join(cmd)}")
        print(f"Model path in command: {cmd[cmd.index('--model') + 1] if '--model' in cmd else 'NOT FOUND'}")

        # Execute mini-swe-agent
        subprocess.run(cmd, env=env, check=True)

    finally:
        # Clean up temporary module
        if temp_module_path.exists():
            temp_module_path.unlink()


if __name__ == '__main__':
    main()
