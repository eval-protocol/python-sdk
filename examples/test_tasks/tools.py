"""
Tools for the test tasks.
"""

from eval_protocol.agent.tool_registry import ToolRegistry

# Create tool registry
R = ToolRegistry("test_task_tools")


@R.tool(
    description="Add an item to the list",
    parameters={"item": {"type": "string", "description": "The item to add"}},
)
def add_item(item, resource):
    """Add an item to the resource's items list."""
    state = resource.get_state()
    state["items"].append(item)
    resource.set_state(state)
    return {"status": "success", "message": f"Added item: {item}"}


@R.tool(description="Increment the counter", parameters={})
def increment_counter(resource):
    """Increment the resource's counter."""
    state = resource.get_state()
    state["counter"] += 1
    resource.set_state(state)
    return {
        "status": "success",
        "message": f"Counter incremented to {state['counter']}",
    }


@R.tool(
    description="Set the status",
    parameters={"status": {"type": "string", "description": "The new status"}},
)
def set_status(status, resource):
    """Set the resource's status."""
    state = resource.get_state()
    state["status"] = status
    resource.set_state(state)
    return {"status": "success", "message": f"Status set to: {status}"}


@R.tool(description="Get the current state", parameters={})
def get_state(resource):
    """Get the current state of the resource."""
    state = resource.get_state()
    return {"status": "success", "state": state}
