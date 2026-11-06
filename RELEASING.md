# Releasing Calibrax

Calibrax publishes through `.github/workflows/publish.yml`.
No commit or tag push creates a release by itself. Release timing and versioning
stay under operator control. The manual `target=github-release` workflow path
creates a GitHub Release for an explicit existing tag with
`softprops/action-gh-release@v3` and `generate_release_notes: true`, then
publishes to PyPI with trusted publishing. Publishing an existing GitHub Release
also runs the PyPI upload path.

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

5. Commit the version and changelog updates on a branch and open a pull request. `main` changes
   only through a pull request whose checks passed, merged by rebase:

   ```bash
   gh pr create --base main --fill
   gh pr merge --rebase   # once its checks have passed
   ```

6. Run the macOS workflow on the release commit `main` now holds and wait for it to pass. No push
   or pull request runs on macOS, so this is the release's macOS check. A rebase merge gives the
   release commit a new SHA, so `main` is pulled first, and the run watched is the dispatch on that
   commit; `--exit-status` makes a failed run fail the step (`gh run watch` exits 0 without it).

   ```bash
   git switch main
   git pull --ff-only
   target_sha=$(git rev-parse HEAD)
   gh workflow run macos.yml --ref main
   sleep 15   # the dispatched run takes a few seconds to appear
   run_id=$(gh run list --workflow macos.yml --event workflow_dispatch --branch main \
     --commit "$target_sha" --limit 1 --json databaseId --jq '.[0].databaseId')
   gh run watch "$run_id" --exit-status
   ```

7. Create and push an annotated tag on that commit. Only the tag is pushed: `main` itself changes
   only through pull requests.

   ```bash
   git tag -a vX.Y.Z -m "calibrax X.Y.Z" "$target_sha"
   git push origin vX.Y.Z
   ```

8. Run `Publish to PyPI` from the tag, with `target=github-release` and `version_tag=vX.Y.Z`:

   ```bash
   gh workflow run publish.yml --ref vX.Y.Z -f target=github-release -f version_tag=vX.Y.Z
   ```

   The workflow verifies that the tag exists, creates the GitHub Release with
   generated release notes, then publishes to PyPI.

## Manual Release Recovery

If the manual generated-release workflow is interrupted before creating the
GitHub Release, use GitHub generated notes manually from the exact tagged
commit.

   ```bash
   gh release create vX.Y.Z --target "$target_sha" --generate-notes
   ```

Publishing that release triggers the same PyPI upload workflow.

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
