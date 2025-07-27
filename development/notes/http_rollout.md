# Remote Rollout Server API

Eval Protocol can collect reinforcement learning trajectories from an external HTTP service.
The service exposes three simple endpoints used by `RemoteHttpRolloutClient`:

## `POST /start_episode`
Returns an `episode_id` and the initial observation.

## `POST /step`
Request body:
```json
{
  "episode_id": "string",
  "action": {"any": "payload"}
}
```
Returns a JSON object:
```json
{
  "observation": {"any": "payload"},
  "is_done": false
}
```
representing the new observation after the action and whether the episode has ended.

## `POST /end_episode`
Request body:
```json
{"episode_id": "string"}
```
Signals that the episode is complete.

The Eval Protocol pipeline is responsible for invoking an
OpenAI-compatible API between steps and feeding the resulting assistant messages
back into the rollout. This illustrates how an environment can interact with an
LLM at every step while keeping model calls in the pipeline.

A concrete example of this is the [Frozen Lake Example](./frozen_lake_plan.md), which uses a remote HTTP rollout server to play the Frozen Lake game.
