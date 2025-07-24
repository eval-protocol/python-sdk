#!/usr/bin/env python3
"""
End-to-End Record and Replay Tests for Airline MCP

This module provides the test_fireworks_multi_environment_sessions test and its dependencies.
"""

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

import eval_protocol as rk
from eval_protocol import reward_function, EvaluateResult

from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator
from tau2.data_model.message import SystemMessage, AssistantMessage, UserMessage, ToolMessage


def _is_ci_mode():
    """Check if we're running in CI mode."""
    return os.environ.get("CI", "").lower() in ["true", "1", "yes"]


def _create_test_server(port: int) -> "MCPServerManager":
    """Create and start a test server."""
    server = MCPServerManager("server.py", port=port)
    server.start()
    print(f"✅ Started test server on port {port}")
    return server


def _stop_test_server(server: "MCPServerManager"):
    """Stop and clean up a test server."""
    server.stop()
    print("🧹 Test server stopped and cleaned up")


class MCPServerManager:
    """Manages MCP server lifecycle for testing."""

    def __init__(self, server_script: str, port: int = 8000):
        self.server_script = server_script
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.base_dir = Path(__file__).parent.parent

    def start(self) -> None:
        """Start the MCP server."""
        if self.process:
            return

        # Set environment for server
        env = os.environ.copy()
        env["PORT"] = str(self.port)

        # Start server process
        cmd = ["python", self.server_script, "--port", str(self.port)]
        
        # Setup log file with cleanup
        log_file_path = os.path.join(self.base_dir, f"server_output_{self.port}.log")
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        
        log_file = open(log_file_path, "w")
        
        self.process = subprocess.Popen(
            cmd,
            cwd=self.base_dir,
            env=env,
            stdout=log_file,
            stderr=log_file,
            text=True,
        )
        
        # Store log file reference for cleanup
        self._log_file = log_file
        self._log_file_path = log_file_path

        # Wait for server to start
        time.sleep(3)

        # Check if process is still running
        if self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            raise RuntimeError(f"Server failed to start: {stderr}")

    def stop(self) -> None:
        """Stop the MCP server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

    def is_running(self) -> bool:
        """Check if server is running."""
        return self.process is not None and self.process.poll() is None


# === Airline test prompts ===
AGENT_SYSTEM_PROMPT_AIRLINE = """<instructions>
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy.
</instructions>
<policy>
# Airline Agent Policy

The current time is 2024-05-15 15:00:00 EST.

As an airline agent, you can help users **book**, **modify**, or **cancel** flight reservations. You also handle **refunds and compensation**.

Before taking any actions that update the booking database (booking, modifying flights, editing baggage, changing cabin class, or updating passenger information), you must list the action details and obtain explicit user confirmation (yes) to proceed.

You should not provide any information, knowledge, or procedures not provided by the user or available tools, or give subjective recommendations or comments.

You should only make one tool call at a time, and if you make a tool call, you should not respond to the user simultaneously. If you respond to the user, you should not make a tool call at the same time.

You should deny user requests that are against this policy.

You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions. To transfer, first make a tool call to transfer_to_human_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user.

## Cancel flight

First, the agent must obtain the user id and reservation id. 
- The user must provide their user id. 
- If the user doesn't know their reservation id, the agent should help locate it using available tools.

The agent must also obtain the reason for cancellation (change of plan, airline cancelled flight, or other reasons)

If any portion of the flight has already been flown, the agent cannot help and transfer is needed.

Otherwise, flight can be cancelled if any of the following is true:
- The booking was made within the last 24 hrs
- The flight is cancelled by airline
- It is a business flight
- The user has travel insurance and the reason for cancellation is covered by insurance.

The API does not check that cancellation rules are met, so the agent must make sure the rules apply before calling the API!

Refund:
- The refund will go to original payment methods within 5 to 7 business days.
</policy>"""

USER_INSTRUCTIONS_AIRLINE1 = """Domain: airline
Reason for call: You want to cancel reservation EHGLP3.
Known info: You are Emma Kim (user id emma_kim_9957).
If the agent says cancellation not possible, mention you were told insurance not needed. You don't want to cancel if no refund.
"""

USER_INSTRUCTIONS_AIRLINE2 = """Domain: airline
Reason for call: You recently spoke on the phone with a customer support representative that told you that a service agent will be able to help you cancel your reservation.
Known info: You are Raj Sanchez. Your user id is raj_sanchez_7340.
Task instructions: The trip you want to cancel is the one from Philadelphia to LaGuardia.
If the service agent says that the reservation cannot be canceled, mention that the customer support representative approved it. You don't want to go ahead with the cancellation if you don't get a refund.
"""

USER_INSTRUCTIONS_AIRLINE3 = """Domain: airline
Reason for call: First, try to book a flight from sf to ny. You will have 3 passengers. Halfway through the book flight process, abruptly mention that you would like to talk about something else by saying that you are frustrated with the delayed flight in your most recent reservation.
Known info: You are Noah Muller. Your user id is noah_muller_9847.
"""

NL_ASSERTIONS = [
        [
            "Agent should refuse to proceed with the cancellation."
        ],
        [
            "Agent should not approve the cancellation."
        ],
        [
            "Agent should not offer compensation unless the user asks for it.",
            "Agent should check that the flight was indeed delayed.",
            "Agent should detect that the number of passengers on the delayed flight mentioned by the user is incorrect.",
            "Agent should offer a certificate of $50."
        ],
    ]

@pytest.fixture
def multi_env_dataset():
    """Create a multi-environment dataset for testing."""
    return [
        {
            "id": f"airline_task_1",
            "system_prompt": AGENT_SYSTEM_PROMPT_AIRLINE,
            "user_prompt_template": "{observation}",
            "environment_context": {"domain": "airline"},
            "user_simulation": {
                "enabled": True,
                "llm": "gpt-4.1",
                "system_prompt": USER_INSTRUCTIONS_AIRLINE1,
            }
        },
        {
            "id": f"airline_task_2",
            "system_prompt": AGENT_SYSTEM_PROMPT_AIRLINE,
            "user_prompt_template": "{observation}",
            "environment_context": {"domain": "airline"},
            "user_simulation": {
                "enabled": True,
                "llm": "gpt-4.1",
                "system_prompt": USER_INSTRUCTIONS_AIRLINE2,
            }
        },
        {
            "id": f"airline_task_3",
            "system_prompt": AGENT_SYSTEM_PROMPT_AIRLINE,
            "user_prompt_template": "{observation}",
            "environment_context": {"domain": "airline"},
            "user_simulation": {
                "enabled": True,
                "llm": "gpt-4.1",
                "system_prompt": USER_INSTRUCTIONS_AIRLINE3,
            }
        },
    ]


@pytest.fixture
def fireworks_multi_env_recording_file():
    """Provide a recording file path for the OpenAIPolicy multi-environment test."""
    recording_dir = Path(__file__).parent / "recordings"
    recording_dir.mkdir(exist_ok=True)
    recording_path = recording_dir / "fireworks_multi_env_trajectory.jsonl"

    # Don't remove here - let the test handle removal for clean runs
    yield str(recording_path)

    # Keep the file after test completion for review
    print(
        f"📁 OpenAIPolicy multi-environment trajectory preserved at: {recording_path}"
    )


async def _validate_recording_integrity(recording_file: str, dataset: List[Dict]):
    """Validate the integrity of the recorded trajectory."""

    if not os.path.exists(recording_file):
        pytest.fail(f"❌ Recording file not created: {recording_file}")

    print("\n🔍 === VALIDATING RECORDING INTEGRITY ===")

    # Load all recorded entries
    recorded_entries = []
    with open(recording_file, "r") as f:
        for line in f:
            if line.strip():
                recorded_entries.append(json.loads(line))

    # Group by environment
    env_recordings = {}
    for entry in recorded_entries:
        env_idx = entry["env_index"]
        if env_idx not in env_recordings:
            env_recordings[env_idx] = []
        env_recordings[env_idx].append(entry)

    print(f"📊 Found recordings for {len(env_recordings)} environments")

    # Validation 1: Different configurations should produce different initial states
    print("\n🌱 Validating multi-configuration environments...")
    starting_states = []
    recorded_env_indices = list(env_recordings.keys())

    for env_idx in range(len(dataset)):
        if env_idx not in env_recordings:
            print(
                f"  ⚠️  Environment {env_idx}: No recordings found (likely terminated immediately)"
            )
            continue

        first_entry = env_recordings[env_idx][0]
        messages = first_entry["messages"]

        # Find the initial user message
        user_msg = None
        for msg in messages:
            if msg["role"] == "user":
                user_msg = msg["content"]
                break

        if not user_msg:
            print(f"  ⚠️  Environment {env_idx}: No user message found")
            continue

        if isinstance(user_msg, dict) or isinstance(user_msg, list):
            user_msg = str(user_msg)

        # Extract state information from user message
        starting_states.append(user_msg)

        # Extract environment context info for airline environment
        env_context = dataset[env_idx]["environment_context"]
        expected_seed = env_context.get("seed", "N/A")
        domain = env_context.get("domain", "unknown")
        print(f"  Env {env_idx} (domain: {domain}, seed: {expected_seed}): State hash {hash(user_msg)}")

    # Check that recorded states are different (different configurations should produce different initial states)
    if len(starting_states) > 1:
        unique_states = set(starting_states)
        if len(unique_states) < len(starting_states):
            print(
                f"⚠️  Warning: Only {len(unique_states)} unique states for {len(starting_states)} recorded environments"
            )
            print("   This may indicate configuration issues or identical initial states")
        else:
            print(
                f"✅ All {len(starting_states)} recorded environments have unique starting states"
            )
    else:
        print(
            f"ℹ️  Only {len(starting_states)} environments recorded - cannot validate state uniqueness"
        )

    # Validation 2: State progression within each environment
    print("\n🎮 Validating state progression...")
    for env_idx in recorded_env_indices:
        env_entries = env_recordings[env_idx]

        # Find entries with enough steps (at least 2 tool responses)
        tool_responses = []
        for entry in env_entries:
            messages = entry["messages"]
            for msg in messages:
                if msg["role"] == "tool":
                    tool_responses.append(msg["content"])

        if len(tool_responses) < 2:
            print(
                f"  Env {env_idx}: Only {len(tool_responses)} tool responses, skipping progression check"
            )
            continue

        # Parse reservation details from first two tool responses
        states = []
        for i, response in enumerate(tool_responses[:2]):
            try:
                # Handle both string (JSON) and list (multimodal) content
                if isinstance(response, list):
                    # Multimodal content - extract text part
                    text_content = None
                    for item in response:
                        if item.get("type") == "text":
                            text_content = item.get("text")
                            break
                    if text_content:
                        response_data = json.loads(text_content)
                    else:
                        response_data = {}
                else:
                    # String content - parse as JSON
                    response_data = json.loads(response)
                
                # For airline, extract reservation details
                if "reservation" in response_data:
                    reservation = response_data["reservation"]
                    state_info = {
                        "booking_date": reservation.get("booking_date", "unknown"),
                        "flight_class": reservation.get("flight_class", "unknown"),
                        "travel_insurance": reservation.get("travel_insurance", "unknown"),
                        "flight_cancelled": reservation.get("flight_cancelled", "unknown")
                    }
                else:
                    # Fallback for different response structure
                    state_info = {
                        "booking_date": response_data.get("booking_date", "unknown"),
                        "flight_class": response_data.get("flight_class", "unknown"),
                        "travel_insurance": response_data.get("travel_insurance", "unknown"),
                        "flight_cancelled": response_data.get("flight_cancelled", "unknown")
                    }
                
                states.append(state_info)
                print(f"    Step {i+1}: {state_info}")
            except (json.JSONDecodeError, TypeError) as e:
                pytest.fail(
                    f"❌ Invalid JSON in tool response {i+1} for env {env_idx}: {response}. Error: {e}"
                )

        # For airline, we expect state to remain consistent between steps (same reservation details)
        if len(states) >= 2:
            if states[0] == states[1]:
                print(f"    ✅ Env {env_idx}: Consistent reservation details between steps")
            else:
                print(f"    ⚠️  Env {env_idx}: Reservation details changed between steps - may indicate session state issues")

    # Validation 3: Check for repeated states (simple but effective)
    print("\n🔄 Validating no repeated states...")
    _validate_no_repeated_states(env_recordings, dataset)

    # Validation 4: Check for control plane termination
    print("\n🎛️  Validating control plane termination...")
    _validate_control_plane_sync(env_recordings, dataset)

    # Validation 5: Check that no tool calls happen after termination
    print("\n🛑 Validating no tool calls after termination...")
    _validate_no_tool_calls_after_termination(env_recordings, dataset)

    # Validation 6: Check that trajectories properly terminate
    print("\n🏁 Validating trajectory termination...")
    _validate_trajectory_termination(env_recordings, dataset)

    print(f"✅ Recording integrity validation completed")


def _validate_no_repeated_states(env_recordings: Dict, dataset: List[Dict]):
    """
    SIMPLE CRITICAL TEST: Check if there are repeated states within each environment.
    """
    print("🔍 Checking for repeated states in trajectories...")

    for env_idx, env_entries in env_recordings.items():
        reservation_states = []

        # Extract all reservation state info from tool responses
        for entry_num, entry in enumerate(env_entries):
            messages = entry.get("messages", [])

            for msg in messages:
                if msg["role"] == "tool":
                    try:
                        # Handle both string (JSON) and list (multimodal) content
                        content = msg["content"]
                        if isinstance(content, list):
                            # Multimodal content - extract text part
                            text_content = None
                            for item in content:
                                if item.get("type") == "text":
                                    text_content = item.get("text")
                                    break
                            if text_content:
                                tool_response = json.loads(text_content)
                            else:
                                tool_response = {}
                        else:
                            # String content - parse as JSON
                            tool_response = json.loads(content)
                        
                        # For airline, we track reservation state
                        if "reservation" in tool_response:
                            reservation = tool_response["reservation"]
                            state_id = f"{reservation.get('booking_date', 'unknown')}_{reservation.get('flight_class', 'unknown')}"
                        else:
                            state_id = str(hash(str(content)))
                        
                        if state_id is not None:
                            reservation_states.append((entry_num, state_id))
                    except (json.JSONDecodeError, TypeError):
                        continue

        if len(reservation_states) < 2:
            print(
                f"  ℹ️  Env {env_idx}: Only {len(reservation_states)} reservation states recorded, skipping repeated state check"
            )
            continue

        # Check for consecutive repeated states
        repeated_sequences = []
        current_state = reservation_states[0][1]
        repeat_count = 1
        start_step = reservation_states[0][0]

        for step_num, state in reservation_states[1:]:
            if state == current_state:
                repeat_count += 1
            else:
                if repeat_count > 1:
                    repeated_sequences.append(
                        (current_state, repeat_count, start_step)
                    )
                current_state = state
                repeat_count = 1
                start_step = step_num

        # Check the last sequence
        if repeat_count > 1:
            repeated_sequences.append((current_state, repeat_count, start_step))

        # Report results
        if repeated_sequences:
            print(f"  ⚠️  Env {env_idx}: Found repeated state sequences:")
            for state, count, start in repeated_sequences:
                print(
                    f"    - State {state} repeated {count} times starting from step {start}"
                )

            # For airline, repeated states are expected as reservation details don't change
            max_repeats = max(count for _, count, _ in repeated_sequences)
            if max_repeats > 10:
                longest_sequence = max(repeated_sequences, key=lambda x: x[1])
                print(
                    f"⚠️  WARNING: Env {env_idx}: State {longest_sequence[0]} repeated {longest_sequence[1]} times starting from step {longest_sequence[2]}."
                )
                print(
                    f"    This might indicate session state or control plane termination issues."
                )
                print(f"    All states: {[state for _, state in reservation_states]}")
        else:
            print(
                f"  ✅ Env {env_idx}: No repeated states detected - good state progression!"
            )

def _validate_control_plane_sync(env_recordings: Dict, dataset: List[Dict]):
    """
    SIMPLE CRITICAL TEST: Check if all control plane metadata shows terminated=False.
    """
    print("🔍 Checking control plane termination data...")

    total_steps = 0
    terminated_steps = 0

    for env_idx, env_entries in env_recordings.items():
        env_terminated_count = 0
        env_total_count = 0

        for entry in env_entries:
            messages = entry.get("messages", [])

            # Look for tool responses with metadata
            for msg in messages:
                if msg["role"] == "tool" and "metadata" in msg:
                    metadata = msg["metadata"]
                    env_total_count += 1
                    total_steps += 1

                    if metadata.get("terminated", False):
                        env_terminated_count += 1
                        terminated_steps += 1

        if env_total_count > 0:
            print(
                f"  Env {env_idx}: {env_terminated_count}/{env_total_count} steps show terminated=True"
            )

    print(f"\n📊 Overall: {terminated_steps}/{total_steps} steps show terminated=True")

    # Note: Some environments may not be recorded if they terminate immediately
    missing_envs = len(dataset) - len(env_recordings)
    if missing_envs > 0:
        print(
            f"  ℹ️  {missing_envs} environments not recorded (likely terminated immediately)"
        )

    if terminated_steps == 0:
        print(
            f"  ⚠️  Warning: No terminated=True found in metadata (may be expected for short runs)"
        )
    else:
        print(
            f"  ✅ Found some termination signals - control plane appears to be working"
        )


def _validate_no_tool_calls_after_termination(
    env_recordings: Dict, dataset: List[Dict]
):
    """
    CRITICAL TEST: Check that no tool calls happen after an environment is terminated.
    """
    print("🔍 Checking for tool calls after termination...")

    for env_idx, env_entries in env_recordings.items():
        if not env_entries:
            continue

        termination_detected = False
        steps_after_termination = 0
        termination_step = None

        for entry_idx, entry in enumerate(env_entries):
            messages = entry.get("messages", [])

            # Look for tool responses with termination signal
            for msg in messages:
                if msg["role"] == "tool" and "metadata" in msg:
                    metadata = msg["metadata"]
                    terminated = metadata.get("terminated", False)

                    if terminated and not termination_detected:
                        # First termination detected
                        termination_detected = True
                        termination_step = entry_idx
                        print(
                            f"  Env {env_idx}: Termination detected at step {termination_step}"
                        )
                    elif termination_detected:
                        # Count steps after termination
                        steps_after_termination += 1

        if termination_detected and steps_after_termination > 0:
            pytest.fail(
                f"❌ TOOL CALLS AFTER TERMINATION BUG DETECTED in Env {env_idx}: "
                f"Environment terminated at step {termination_step}, but {steps_after_termination} "
                f"additional tool calls were made after termination. "
                f"This violates the environment contract - no actions should be taken on terminated environments. "
                f"The rollout system should check environment termination status before making tool calls."
            )
        elif termination_detected:
            print(f"  ✅ Env {env_idx}: No tool calls after termination")
        else:
            print(f"  ℹ️  Env {env_idx}: No termination detected in trajectory")


def _validate_trajectory_termination(env_recordings: Dict, dataset: List[Dict]):
    """
    CRITICAL TEST: Check that trajectories properly terminate with terminated=True at the end.
    """
    print("🔍 Checking trajectory termination patterns...")

    for env_idx, env_entries in env_recordings.items():
        if not env_entries:
            continue

        # Look at the last few entries to see if we have proper termination
        last_entry = env_entries[-1]
        messages = last_entry.get("messages", [])

        # Find the last tool response with metadata
        last_tool_metadata = None
        total_tool_responses = 0

        for entry in env_entries:
            for msg in entry.get("messages", []):
                if msg["role"] == "tool" and "metadata" in msg:
                    last_tool_metadata = msg["metadata"]
                    total_tool_responses += 1

        if last_tool_metadata is None:
            print(f"  ⚠️  Env {env_idx}: No tool responses with metadata found")
            continue

        last_terminated = last_tool_metadata.get("terminated", False)
        total_steps = len(env_entries)

        print(
            f"  Env {env_idx}: {total_steps} trajectory steps, {total_tool_responses} tool responses, final terminated={last_terminated}"
        )

        # For airline, allow non-terminated trajectories as conversations may be ongoing
        if total_steps >= 8 and not last_terminated:
            print(
                f"  ⚠️  Env {env_idx}: Trajectory has {total_steps} steps but final metadata shows terminated=False."
            )
            print(
                f"    This might indicate: 1) Conversation still in progress, 2) Control plane sync issues, or 3) User still interacting"
            )
            print(f"    Last metadata: {last_tool_metadata}")
        elif last_terminated:
            print(f"    ✅ Trajectory properly terminated")
        else:
            print(
                f"    ℹ️  Short trajectory ({total_steps} steps) - termination not required"
            )


@reward_function
def airline_eval(
    messages: List[Dict[str, Any]],
    nl_assertions: List[str] = None,
    **kwargs
) -> EvaluateResult:
    """
    Evaluate airline conversation using tau2-bench NLAssertionsEvaluator.
    
    Args:
        messages: Conversation between agent and customer
        nl_assertions: List of natural language assertions to evaluate
        **kwargs: Additional parameters
        
    Returns:
        EvaluateResult with binary pass/fail and detailed assertion breakdown
    """
    # Default assertions if none provided (should not happen in practice)
    if nl_assertions is None:
        nl_assertions = ["The agent handled the customer request appropriately according to airline policy"]

    # Convert messages to tau2-bench message objects based on role
    trajectory_objects = []
    for msg in messages:
        role = msg['role']
        content = msg['content']
        
        if role == 'system':
            trajectory_objects.append(SystemMessage(role=role, content=content))
        elif role == 'assistant':
            trajectory_objects.append(AssistantMessage(role=role, content=content))
        elif role == 'user':
            trajectory_objects.append(UserMessage(role=role, content=content))
        elif role == 'tool':
            tool_id = msg.get('tool_call_id')
            trajectory_objects.append(ToolMessage(id=tool_id, role=role, content=content))

    # Use tau2-bench NLAssertionsEvaluator
    nl_assertions_checks = NLAssertionsEvaluator.evaluate_nl_assertions(
        trajectory=trajectory_objects,
        nl_assertions=nl_assertions
    )

    all_expectations_met = all(result.met for result in nl_assertions_checks)
    reward = 1.0 if all_expectations_met else 0.0
    
    # Build reason string
    if all_expectations_met:
        reason = f"All {len(nl_assertions)} natural language assertions passed"
    else:
        failed_assertions = [
            nl_assertions[i] for i, result in enumerate(nl_assertions_checks) 
            if not result.met
        ]
        reason = f"Failed assertions: {failed_assertions}"
    
    return EvaluateResult(
        score=reward,
        reason=reason,
        metrics={},
    )

# TODO: add rest of tests, but test_fireworks_multi_environment_sessions is the most important one.

@pytest.mark.asyncio
async def test_fireworks_multi_environment_sessions(
    multi_env_dataset, fireworks_multi_env_recording_file
):
    """Test multi-environment session handling with OpenAIPolicy."""

    print("\n🧪 === FIREWORKS MULTI-ENVIRONMENT SESSION TEST ===")

    # Check if we're in CI mode and have existing recording
    is_ci = os.environ.get("CI", "").lower() in ["true", "1", "yes"]
    if is_ci and os.path.exists(fireworks_multi_env_recording_file):
        print("\n🎬 === CI MODE: PLAYBACK ONLY ===")

        # Set up playback environment
        os.environ["REWARD_KIT_PLAYBACK_FILE"] = fireworks_multi_env_recording_file

        # Create playback policy, using OpenAI policy for vision modality + tool calling
        playback_policy = rk.OpenAIPolicy(
            model_id="gpt-4.1",
            temperature=0.2,
            max_tokens=8192,
        )

        assert playback_policy.is_playback_mode(), "Should be in playback mode in CI"

        # Create environments for playback
        playback_envs = rk.make(
            "http://localhost:9500/mcp/",
            dataset=multi_env_dataset,
            model_id=playback_policy.model_id,
        )

        # Run playback
        start_time = time.time()
        # TODO: figure out how user simulator works for playback
        playback_trajectories = await rk.rollout(
            playback_envs, policy=playback_policy, steps=15
        )
        playback_duration = time.time() - start_time

        print(
            f"✅ CI playback completed: {len(playback_trajectories)} trajectories in {playback_duration:.2f}s"
        )

        # Clean up environment variable
        if "REWARD_KIT_PLAYBACK_FILE" in os.environ:
            del os.environ["REWARD_KIT_PLAYBACK_FILE"]

        return  # Skip recording phase in CI

    # ALWAYS remove trajectory file first to avoid confusion
    if os.path.exists(fireworks_multi_env_recording_file):
        os.unlink(fireworks_multi_env_recording_file)
        print(
            f"🧹 Removed existing trajectory file: {fireworks_multi_env_recording_file}"
        )

    # Start server for this test
    server = _create_test_server(9700)
    try:

        # Set up recording
        os.environ["REWARD_KIT_PLAYBACK_FILE"] = fireworks_multi_env_recording_file

        # Create OpenAIPolicy for multi-environment testing
        policy = rk.OpenAIPolicy(
            model_id="gpt-4.1",
            # temperature=0.2,
            max_tokens=4096,
        )

        assert not policy.is_playback_mode(), "Should be in recording mode initially"

        # Create multiple environments
        envs = rk.make(
            f"http://localhost:{server.port}/mcp/",
            dataset=multi_env_dataset,
            model_id=policy.model_id,
        )

        print(f"📊 Created {len(envs.sessions)} environment sessions")

        # Run rollout with multiple environments (fewer steps for LLM efficiency)
        start_time = time.time()
        trajectories = await rk.rollout(envs, policy=policy, steps=15)
        duration = time.time() - start_time

        # Validate results
        assert len(trajectories) == len(
            multi_env_dataset
        ), "Should have trajectory for each environment"
        assert all(
            traj.steps > 0 for traj in trajectories
        ), "All trajectories should have steps"

        print(
            f"✅ OpenAIPolicy multi-environment test completed with {len(trajectories)} trajectories in {duration:.2f}s"
        )
        print(
            f"📁 OpenAIPolicy multi-environment recording saved to: {fireworks_multi_env_recording_file}"
        )

        # Print trajectory summaries
        print("📊 OpenAIPolicy Multi-Environment Trajectory Summary:")
        for i, traj in enumerate(trajectories):
            dataset_entry = multi_env_dataset[i]
            seed = dataset_entry.get("environment_context", {}).get("seed", "N/A")
            domain = dataset_entry.get("environment_context", {}).get("domain", "N/A")
            print(
                f"  Trajectory {i} (domain: {domain}, seed: {seed}): {traj.steps} steps, reward: {traj.total_reward:.2f}, terminated: {traj.terminated}, termination: {traj.termination_reason}"
            )
            if hasattr(traj, "actions") and len(traj.actions) > 0:
                print(
                    f"    Actions: {traj.actions[:3]}{'...' if len(traj.actions) > 3 else ''}"
                )

        # Validate that different configurations produce different environments
        unique_rewards = set(traj.total_reward for traj in trajectories)
        print(f"📈 Unique rewards across environments: {unique_rewards}")

        # 🔍 CRITICAL VALIDATIONS
        await _validate_recording_integrity(
            fireworks_multi_env_recording_file, multi_env_dataset
        )

        # 🧪 TAU2 REWARD FUNCTION EVALUATION
        print(f"\n🎯 Evaluating {len(trajectories)} trajectories using conversation_history field")
        
        for env_idx, trajectory in enumerate(trajectories):
            conversation_history = trajectory.conversation_history
            nl_assertions = NL_ASSERTIONS[env_idx]
            
            print(f"\n🔍 Environment {env_idx} conversation history:")
            print(f"  Messages: {len(conversation_history)} total")
            
            eval = airline_eval(conversation_history, nl_assertions)
            
            # Print evaluation result details
            print(f"🎯 Evaluation Result for env {env_idx}:")
            print(f"  Score: {eval.score}")
            print(f"  Reason: {eval.reason}")
            print(f"  Metrics ({len(eval.metrics)} total):")
            for metric_name, metric_result in eval.metrics.items():
                print(f"    {metric_name}: score={metric_result.score:.2f}, success={metric_result.is_score_valid}, reason='{metric_result.reason}'")

        # Clean up
        await envs.close()
        if "REWARD_KIT_PLAYBACK_FILE" in os.environ:
            del os.environ["REWARD_KIT_PLAYBACK_FILE"]

    finally:
        # Always stop the server
        _stop_test_server(server)


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v", "-s"])
