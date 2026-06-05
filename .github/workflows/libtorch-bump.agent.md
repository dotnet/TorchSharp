---
description: |
  Monthly. Detects a new stable libtorch release in pytorch/pytorch and opens
  a scoped PR for patch-level pin updates only. Files a tracking issue instead
  of a PR when the bump crosses a major or minor version (those can change the
  native ABI). No-ops when nothing changed or a bump PR for the same version is
  already open.

on:
  schedule: every 30d
  roles: [admin, maintain, write]

if: github.repository == 'dotnet/TorchSharp'

timeout-minutes: 30

permissions: read-all

network:
  allowed:
    - defaults
    - github
    - dotnet

checkout:
  ref: main
  fetch-depth: 1

tools:
  github:
    toolsets: [repos, issues, pull_requests]
  bash: true
  edit:

safe-outputs:
  noop:
    report-as-issue: false
  create-pull-request:
    title-prefix: "[libtorch-bump] "
    labels: [dependencies]
    draft: true
    max: 1
    allowed-files:
      - "build/Dependencies.props"
      - "Directory.Build.props"
  create-issue:
    title-prefix: "[libtorch-bump] "
    labels: [dependencies]
    max: 1
---

# LibTorch Version Bump

Detect a new stable libtorch release in `pytorch/pytorch` and open a scoped PR updating the pinned version. If a clean bump cannot be proposed, file an issue or `noop`.

## Hard rules

1. Do not modify `src/Native/` or `src/TorchSharp/PInvoke/`. Bindings work belongs in a separate human-driven PR.
2. Only auto-bump within the same `major.minor` (patch-level). Any change to the major or minor component files a tracking issue instead, because libtorch minor releases frequently change the native ABI and need human-driven bindings work.
3. One PR per run. If a bump PR for the same version is already open (search by title prefix `[libtorch-bump] <version>`), `noop`.
4. **Issue idempotency.** If a major-bump issue with title containing `<major.minor>` and the `[libtorch-bump]` prefix already exists open OR was closed in the last 90 days, `noop` on the issue path.
5. **Never touch the mac x64 conditional pin.** Both `build/Dependencies.props` and `Directory.Build.props` carry a separate override gated on `'$(TargetArchitecture)' == 'x64' and '$(TargetOS)' == 'mac'` (currently `2.2.2` / `2.2.2.0`). Leave those lines alone. Replace only the unconditional `<LibTorchVersion>` (one in `build/Dependencies.props`) and `<LibTorchPackageVersion>` (two in `Directory.Build.props`) lines.

## Steps

1. **Read pin.** Get the unconditional `<LibTorchVersion>` from `build/Dependencies.props` (e.g. `2.10.0`) and `<LibTorchPackageVersion>` from `Directory.Build.props` (e.g. `2.10.0.0`). Note that `Directory.Build.props` defines `<LibTorchPackageVersion>` **twice** unconditionally (once near the top, once inside the `libtorch-` package `PropertyGroup`); both carry the same value and both must be bumped. Confirm the conditional mac x64 overrides are present on the next line after each unconditional entry; record them but never modify them.
2. **Latest stable.** Bind the URL first, then call: `url="repos/pytorch/pytorch/releases?per_page=20"; gh api "$url"` filtered to `^v\d+\.\d+\.\d+$` (no pre-release, no rc).
3. **Compare.** Pin equals latest: `noop`.
4. **Idempotency check (PR).** Search existing PRs: `gh pr list --repo dotnet/TorchSharp --state all --limit 50 --search "[libtorch-bump] <new-version> in:title"`. If any open or merged in the last 30 days matches: `noop`.
5. **Major or minor change.** `major` or `minor` differs from the current pin: file a tracking issue summarizing the upstream release notes and likely breaking changes (do not open a PR). Body MUST contain the marker `<!-- libtorch-bump:major:<new-major.minor> -->`. Before filing, search open and recently-closed issues for that marker; on hit: `noop`. Stop.
6. **Patch bump.** Same `major.minor`, patch differs:
   - Branch `libtorch-bump/<new>`.
   - In `build/Dependencies.props`, replace the unconditional `<LibTorchVersion>OLD</LibTorchVersion>` line. Do NOT touch the next line (the conditional mac x64 override).
   - In `Directory.Build.props`, replace **both** unconditional `<LibTorchPackageVersion>OLD.0</LibTorchPackageVersion>` lines (one near the top, one in the `libtorch-` package `PropertyGroup`). Do NOT touch the conditional mac x64 override that follows each of them.
   - Verify with `grep -n LibTorchVersion build/Dependencies.props` (two entries: one unconditional updated, one conditional unchanged) and `grep -n LibTorchPackageVersion Directory.Build.props` (four entries: two unconditional updated, two conditional unchanged).
   - `dotnet restore && dotnet build TorchSharp.sln --no-restore --configuration Debug`.
   - Open a draft PR. Body lists: source release link, exact lines updated, mac x64 override preserved, build outcome (success or first 50 lines of the first error). Remind reviewers to verify CUDA matrix legs are green before merging.
