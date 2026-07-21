---
name: libtorch-bump
description: Check for a new stable pytorch/pytorch release and prepare a safe dotnet/TorchSharp libtorch patch-version bump locally. Use when asked to update libtorch, check the pinned libtorch version, prepare a dependency bump, or assess a new PyTorch release.
---

# LibTorch version bump

Detect a new stable libtorch release and prepare either a patch-level code change or a major or minor upgrade report.

## Hard rules

1. Do not modify `src/Native/` or `src/TorchSharp/PInvoke/`.
2. Automate only patch updates within the current `major.minor`. Major or minor changes require human-driven bindings work.
3. Do not duplicate an existing bump PR or a recently closed major-version tracking issue.
4. Never change the macOS x64 conditional pins. They currently use `2.2.2` and `2.2.2.0`.
5. Do not commit, push, create a PR, or create an issue without explicit user approval.

## Workflow

1. Read the unconditional `<LibTorchVersion>` from `build/Dependencies.props` and both unconditional `<LibTorchPackageVersion>` entries from `Directory.Build.props`.
2. Confirm each unconditional entry is followed by the macOS x64 conditional override and record the unchanged override values.
3. Query the latest stable `pytorch/pytorch` release:

   ```bash
   url='repos/pytorch/pytorch/releases?per_page=20'
   gh api "$url"
   ```

   Select the newest tag matching `^v[0-9]+\.[0-9]+\.[0-9]+$`. Ignore release candidates and prereleases.
4. If the pin matches the latest stable release, report that no change is needed.
5. Search existing PRs for `[libtorch-bump] <new-version>` and stop when an open PR or a PR merged in the last 30 days already covers it.
6. When the major or minor component changes:
   - summarize the upstream release notes and likely native ABI impact
   - search open and recently closed issues for `<!-- libtorch-bump:major:<new-major.minor> -->`
   - draft one tracking issue when no match exists
7. For a patch update:
   - replace the unconditional `<LibTorchVersion>` in `build/Dependencies.props`
   - replace both unconditional `<LibTorchPackageVersion>` entries in `Directory.Build.props`
   - leave every conditional macOS x64 override unchanged
8. Verify the four package-version entries and two libtorch-version entries with `grep`.
9. Run:

   ```bash
   dotnet restore
   dotnet build TorchSharp.sln --no-restore --configuration Debug
   ```

10. Present the local diff and build result. After explicit approval, create a branch named `libtorch-bump/<new-version>` and prepare a draft PR with the `dependencies` label.

## PR description

Include the upstream release URL, exact properties changed, confirmation that macOS x64 overrides remain unchanged, and the build result. Ask reviewers to verify the CUDA matrix before merging.
