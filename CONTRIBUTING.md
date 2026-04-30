# Contributing

Thanks for helping improve Calibrax. This repository is still early, so keep
changes small, tested, and explicit about user-visible behavior.

## Local Setup

```bash
git clone https://github.com/avitai/calibrax.git
cd calibrax
./setup.sh
source activate.sh
uv pip install -e ".[dev,test,stats,docs,publication,changepoint,mlflow,wandb]"
uv run pre-commit install
```

Run all project commands after activating the local environment:

```bash
source activate.sh
uv run pytest
uv run pre-commit run --all-files
uv run mkdocs build --strict --clean
```

## Contribution Workflow

1. Create a focused branch from `main`.
2. Add or update tests before changing behavior.
3. Keep docs and README claims aligned with the current code.
4. Run the relevant targeted checks plus the full verification stack before
   opening a pull request.
5. Use the pull request checklist and call out any intentionally skipped checks.

For metric additions, follow
[Adding a Metric](docs/contributing/adding-a-metric.md). For documentation
style, use the
[Documentation Design Framework](docs/contributing/example_documentation_design.md)
as the standards reference.

## Pull Request Expectations

- Functional changes include tests.
- Metric changes include numerical-equivalence coverage when a reliable
  reference implementation exists.
- Documentation changes build with strict MkDocs.
- Dependency or workflow changes explain the maintenance impact.

## Reporting Security Issues

Do not open a public issue for suspected vulnerabilities. Follow
[SECURITY.md](SECURITY.md) instead.
