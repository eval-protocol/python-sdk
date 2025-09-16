# LangSmith Bootstrap Scripts

These scripts are ONLY for dumping synthetic traces into LangSmith to exercise the adapter and quickstart examples.

- `dump_traces_langsmith.py`: emits simple @traceable runs and an optional mini LangGraph echo flow.
- `emit_tool_calls.py`: emits runs that include assistant tool calls and a tool response message.

Usage:
1) Set your API key:

```bash
export LANGSMITH_API_KEY=...
export LANGSMITH_TRACING=true
export LS_PROJECT=ep-langgraph-examples
```

2) Run emitters:

```bash
python examples/langsmith/dump_traces_langsmith.py
python examples/langsmith/emit_tool_calls.py
```

These are not production examples; they exist to seed LangSmith with traces that the adapter can consume.
