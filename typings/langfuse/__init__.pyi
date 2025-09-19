from typing import Any

class LangfuseTraceAPI:
    def list(self, *args: Any, **kwargs: Any) -> Any: ...
    def get(self, *args: Any, **kwargs: Any) -> Any: ...

class LangfuseApi:
    trace: LangfuseTraceAPI

class LangfuseClient:
    api: LangfuseApi

    def create_score(self, *args: Any, **kwargs: Any) -> Any: ...

class ObservationsView:
    ...

class Trace:
    ...

class TraceWithFullDetails:
    ...

class Traces:
    ...

class Langfuse:
    ...

def get_client(*args: Any, **kwargs: Any) -> LangfuseClient: ...
