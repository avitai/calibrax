# Contributing

This section is the contributor-facing entry point for Calibrax. Root project
files describe repository policy; these pages describe how to make changes that
fit the current code and documentation system.

## Start Here

- [Project contributing guide](https://github.com/avitai/calibrax/blob/main/CONTRIBUTING.md) - local setup, workflow,
  and pull request expectations.
- [Adding a Metric](adding-a-metric.md) - the implementation, registration,
  testing, and documentation path for new metrics.
- [Documentation Design Framework](example_documentation_design.md) - the
  detailed reference for example structure, progressive disclosure, and docs
  quality standards.

## Local Commands

Run commands from the repository root after activating the Calibrax environment:

```bash
source activate.sh
uv run pytest
uv run pre-commit run --all-files
uv run mkdocs build --strict --clean
```

Use targeted tests during development, but run the full verification stack
before opening a pull request that changes behavior, dependencies, or docs.

## Current Metric Architecture

The registry currently contains 111 Tier 0 pure-function metrics across 17
domains. Tier 1-3 APIs, optional plugin metrics, and metric-learning losses are
part of the package architecture, but they are not all registry entries today.
