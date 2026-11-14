# Contributing

Thanks for helping improve Calibrax. This repository is still early, so keep
changes small, tested, and explicit about user-visible behavior.

## Local Setup

`setup.sh` detects the backend (CUDA on Linux with a visible NVIDIA GPU, Metal on Apple
Silicon, otherwise CPU), syncs the `dev` and `test` extras plus the backend extra with `uv`,
and writes the managed environment file `.calibrax.env` that `activate.sh` loads. A
user-owned `.env` is never modified.

```bash
git clone https://github.com/avitai/calibrax.git
cd calibrax
./setup.sh
source activate.sh
uv run pre-commit install
```

| Flag | Effect |
| --- | --- |
| `--backend <auto\|cpu\|cuda12\|metal>` | Choose the backend policy; `auto` is the default |
| `--python <version>` | Create the environment with a specific Python version |
| `--extra <name>` | Sync an additional extra (repeatable), e.g. `--extra docs` for the documentation toolchain |
| `--recreate` | Remove the existing `.venv` before syncing |
| `--force-clean` | Remove `.venv`, `.calibrax.env` and repo-local test artifacts |
| `--dry-run` | Print the resolved backend and the `uv` commands without changing files |

`./setup.sh --help` prints the same table. Do not follow `setup.sh` with `uv pip install`:
the next `uv run` re-syncs the environment from the lock and reverts it; add extras with
`--extra` instead.

## Verifying a Change

Run everything after `source activate.sh`, from the repository root. CI runs the same
commands.

```bash
uv run pytest tests/ --cov=src/calibrax --cov-report=term-missing   # tests, with coverage
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
uv run pyright --warnings src/
uv run pre-commit run --all-files      # every gate, including import-linter, pydoclint,
                                       # interrogate and bandit
uv run mkdocs build --strict --clean   # needs `./setup.sh --extra docs`
uv run python scripts/derive_status.py --check   # README claims against the registry
```

Examples follow substrax's [Writing examples](https://substrax.readthedocs.io/en/latest/guides/writing-examples/) guide. The script is
the only file you author; `substrax-notebooks` writes its notebook and its page:

```bash
uv run substrax-notebooks convert examples/<topic>/<name>.py   # its clean notebook
uv run substrax-notebooks page examples/<topic>/<name>.py      # run it and write its page
uv run substrax-notebooks test                                 # the shared example tests
```

## Contribution Workflow

1. Create a focused branch from `main`.
2. Add or update tests before changing behavior.
3. Keep docs and README claims aligned with the current code.
4. Run the relevant targeted checks plus the full verification stack before
   opening a pull request.
5. Use the pull request checklist and call out any intentionally skipped checks.

For metric additions, follow
[Adding a Metric](docs/contributing/adding-a-metric.md). For examples,
follow substrax's [Writing examples](https://substrax.readthedocs.io/en/latest/guides/writing-examples/) guide.

## Pull Request Expectations

- Functional changes include tests.
- Metric changes include numerical-equivalence coverage when a reliable
  reference implementation exists.
- Documentation changes build with strict MkDocs.
- Dependency or workflow changes explain the maintenance impact.

## Reporting Security Issues

Do not open a public issue for suspected vulnerabilities. Follow
[SECURITY.md](SECURITY.md) instead.
