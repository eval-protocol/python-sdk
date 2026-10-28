# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- (Your new features here)

### Changed
- `FireworksTracingHttpHandler` is no longer attached automatically by `init_external_logging_from_env()` (used by `@evaluation_test`) when `FIREWORKS_API_KEY` is set; set `FW_TRACING_GATEWAY_BASE_URL` (or `GATEWAY_URL`) to opt in.

### Deprecated
- The hosted Fireworks tracing gateway (https://tracing.fireworks.ai) was shut down on October 28, 2026. These features depend on it, now emit a `DeprecationWarning`, and will be removed in a future release: `RemoteRolloutProcessor`, `GithubActionRolloutProcessor`, `FireworksTracingAdapter`, `FireworksTracingHttpHandler`, and the Fireworks backend of `ep logs`. In the TypeScript package, `FireworksTransport`, `logger`, and `createRolloutLogger` are marked `@deprecated`, and `FireworksTransport` warns once when it sends to the hosted URL. Until removal they only work with a self-hosted gateway that implements the same API, configured explicitly.

### Removed
- (Your now removed features here)

### Fixed
- Pin minimum `datasets` version to `>=3.0.0` to avoid import failures with `pyarrow>=21` (AttributeError: `pyarrow.PyExtensionType`). This ensures compatibility when installing via pip/uv in clean CI environments.

### Security
- (Your vulnerabilities patched here)

## [Internal Note - Action Required]
- Versions `0.2.5` through `0.2.11` have been released but are not yet documented in this changelog.
- These entries need to be backfilled by reviewing the git commit history for notable changes in each respective version.
- Please update release dates for all versions, including for `0.2.4` below.

## [0.2.4] - YYYY-MM-DD
### Added
- Initial setup of project structure and core functionalities.
- (List key features or changes for this version if known, otherwise this is a placeholder)

---

*Note: Future entries should be added at the top, under `[Unreleased]`.*
*When a new version is released, update the `[Unreleased]` section to `[Version.Number] - YYYY-MM-DD` and create a new `[Unreleased]` section above it.*
