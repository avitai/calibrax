# Releasing Calibrax

Calibrax publishes from GitHub releases through `.github/workflows/publish.yml`.
The workflow builds the package, runs `twine check`, and publishes with PyPI
trusted publishing. GitHub generated release notes are the release-note
automation path; there is no separate tag-push release workflow.

## Release Checklist

1. Activate the local environment.

   ```bash
   source activate.sh
   ```

2. Bump the package version in `src/calibrax/__init__.py`.
3. Update `CHANGELOG.md` by moving unreleased entries under the new version and
   date.
4. Run the release checks.

   ```bash
   uv run pytest
   uv run pre-commit run --all-files
   uv run mkdocs build --strict --clean
   rm -rf dist/
   uv build
   uv run twine check dist/*
   ```

5. Commit the version and changelog updates.
6. Create and push an annotated tag from the exact release commit.

   ```bash
   target_sha=$(git rev-parse HEAD)
   git tag -a vX.Y.Z -m "calibrax X.Y.Z"
   git push origin main vX.Y.Z
   ```

7. Create the GitHub release with generated notes.

   ```bash
   gh release create vX.Y.Z --target "$target_sha" --generate-notes
   ```

8. Confirm the `Publish to PyPI` workflow completed successfully. That workflow
   is triggered by the published GitHub release and performs the PyPI upload.

## TestPyPI

Use the manual `workflow_dispatch` path in `publish.yml` with
`target=testpypi` when validating trusted publishing setup before a real
release.

## PyPI Trusted Publishing

PyPI must trust:

- Owner: `avitai`
- Repository: `calibrax`
- Workflow: `publish.yml`
- Environment: `pypi`

If PyPI rejects the publish with `invalid-publisher`, verify the trusted
publisher registration before looking for repository secrets. The expected
publisher identity is:

```text
repo:avitai/calibrax:environment:pypi
```

For TestPyPI, the expected environment is `testpypi`.
