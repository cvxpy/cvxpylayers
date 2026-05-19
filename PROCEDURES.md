# Procedures for making a new cvxpylayers release

This file provides the procedures for releasing a new version of cvxpylayers.
The process involves tagging a new release in the commit history,
packaging and deploying the updated source code, and publishing release notes.

## Versioning

cvxpylayers uses [hatch-vcs](https://github.com/ofek/hatch-vcs) to derive the
package version from git tags. Concretely:

- A tagged commit `vX.Y.Z` produces version `X.Y.Z`.
- Untagged commits produce a [PEP 440 dev version](https://peps.python.org/pep-0440/#developmental-releases)
  of the form `X.Y.Z.devN+gHASH` where `N` is the number of commits since the
  previous tag.

There is no version constant to bump by hand. The single source of truth is the
git tag. At build time, hatch-vcs writes the resolved version to
`src/cvxpylayers/_version.py` (gitignored), which the package imports at runtime.

## Defining a new release

### Incrementing the MINOR version number

Let's say we're releasing 1.2.0.

1. Starting from `master`, checkout a new branch called `release/1.2.x` and push
   it. The branch protections / CI on `release/**` are configured in
   `.github/workflows/build.yml`.
2. On `release/1.2.x`, tag the head commit as `v1.2.0` and push the tag:
   ```
   git tag -a v1.2.0 -m "v1.2.0"
   git push origin v1.2.0
   ```
   This triggers the PyPI deploy (see below).
3. Future patches to the 1.2 line happen on `release/1.2.x` via cherry-picks
   from master — see the next section.
4. Continue feature development on `master`. No version bump is required there;
   hatch-vcs will produce `1.2.0.devN+gHASH` versions automatically until the
   next tag.

### Incrementing the MICRO version number (a.k.a., releasing a patch)

Let's say we're releasing cvxpylayers 1.2.1.

1. Create a new branch `patch/1.2.1` from `release/1.2.x`. Go through all
   commits merged into `master` since `v1.2.0` and cherry-pick the ones that
   belong in the patch: `git cherry-pick abc123`. Open a pull request against
   `release/1.2.x` listing the included commits.
2. After the PR merges, tag the head of `release/1.2.x` as `v1.2.1` and push
   the tag:
   ```
   git tag -a v1.2.1 -m "v1.2.1"
   git push origin v1.2.1
   ```

## Deploying a release to PyPI

Deployments to PyPI are automatically triggered for every tagged commit by the
GitHub Actions workflow at `.github/workflows/build.yml`. The progress of the
deploy can be inspected by opening the workflow run marked with `v*` from the
[actions tab](https://github.com/cvxpy/cvxpylayers/actions).

After a successful deployment, verify the result on
[PyPI](https://pypi.org/project/cvxpylayers/) — both the source distribution
and the wheel should be present, with the correct version.

If the action fails intermittently (e.g., a dependency install times out), it
can be retriggered from the actions tab.

## Creating a release on GitHub

Go to the [Releases](https://github.com/cvxpy/cvxpylayers/releases) tab and
click "Draft a new release". Select the previously created tag and write
release notes.

For minor releases, this includes a summary of new features and deprecations.
We additionally list the PRs contained in the release and their contributors.
For patch releases, list the cherry-picked fixes.

To generate the list of PRs and contributors, use the `tools/release_notes.py`
script:

```
python tools/release_notes.py v1.2.0  # minor release
python tools/release_notes.py v1.2.1  # patch release
```

For minor releases, the script automatically excludes PRs that were
cherry-picked into the previous release branch's patch releases. For patch
releases, it compares against the previous patch tag on the same release
branch.

Take care to select "set as the latest release" only for minor releases or
patches to the most recent minor release. Patches to older release lines
should not become the "latest".
