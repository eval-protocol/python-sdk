# eval-protocol

This is the TypeScript SDK for [eval-protocol](https://evalprotocol.io).

It includes helpful Zod schemas. TypeScript types, and helper functions for
creating a remote server for
[RemoteRolloutProcessor](https://evalprotocol.io/tutorial/remote-rollout-processor).

> **Deprecated:** The hosted Fireworks tracing gateway (https://tracing.fireworks.ai) was shut down on October 28, 2026. `RemoteRolloutProcessor` and the logging helpers in this package (`FireworksTransport`, `logger`, `createRolloutLogger`) depend on it and will be removed in a future release.

Installation

```bash
npm install eval-protocol
```
