---
description: |
  Monthly. Detects a new stable libtorch release in pytorch/pytorch and opens
  a scoped PR updating the pinned version. Files an issue instead of a PR
  when the bump crosses a major version. No-ops when nothing changed or a
  bump PR for the same version is already open.

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
    labels: [build, area-packaging, untriaged]
    draft: true
    max: 1
    allowed-files:
      - "Directory.Packages.props"
      - "linux_cuda.txt"
      - "windows_cuda.txt"
      - "pkg/**"
      - "build/**"
  create-issue:
    title-prefix: "[libtorch-bump] "
    labels: [build, area-packaging, untriaged]
    max: 1
---

# LibTorch Version Bump

Detect a new stable libtorch release in `pytorch/pytorch` and open a scoped PR updating the pinned version. If a clean bump cannot be proposed, file an issue or `noop`.

## Hard rules

1. Do not modify `src/Native/` or `src/TorchSharp/PInvoke/`. Bindings work belongs in a separate human-driven PR.
2. Do not bump across a libtorch major version. File an issue instead.
3. One PR per run. If a bump PR for the same version is already open, `noop`.

## Steps

1. **Read pin.** Get current `libtorch-cpu` Version from `Directory.Packages.props`.
2. **Latest stable.** `gh api repos/pytorch/pytorch/releases` filtered to `^v\d+\.\d+\.\d+$` (no pre-release).
3. **Compare.** Pin equals latest: `noop`.
4. **Major change.** `major.minor` differs: file an issue summarizing the upstream release notes and likely breaking changes. Stop.
5. **Patch bump.** Same `major.minor`:
   - Branch `libtorch-bump/<new>`.
   - Replace exact version strings in the allowed file set. Verify no stale references remain.
   - `dotnet restore && dotnet build TorchSharp.sln --no-restore --configuration Debug`.
   - Open a draft PR. Body lists: source release link, files updated, build outcome (success or first 50 lines of the first error). Remind reviewers to verify CUDA matrix legs are green before merging.
