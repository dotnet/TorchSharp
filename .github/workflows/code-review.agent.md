---
description: |
  First-pass review on every PR. Posts one comment with correctness, style,
  and packaging-impact analysis. Read-only otherwise.

on:
  pull_request:
    types: [opened, synchronize, ready_for_review]
  roles: [admin, maintain, write]

if: |
  github.repository == 'dotnet/TorchSharp' &&
  github.event.pull_request.draft == false

timeout-minutes: 15

permissions: read-all

network:
  allowed:
    - defaults
    - github

tools:
  github:
    toolsets: [repos, pull_requests, search]
  web-fetch:

checkout:
  fetch-depth: 50

safe-outputs:
  noop:
    report-as-issue: false
  add-comment:
    target: "triggering"
    max: 1
    hide-older-comments: true
---

# Code Review

Review PR #${{ github.event.pull_request.number }} and post one comment using the format below. Skip drafts.

## Hard rules

1. **Read-only.** No approvals, no labels, no commits.
2. **One comment per run.** If previous comment is identical, `noop`.
3. **High signal only.** Do not comment on style, formatting, comment wording, or trivial naming. Only comment on bugs, missing null checks, P/Invoke signature mismatches, memory ownership, thread safety, ABI breaks, and API consistency with the rest of TorchSharp.
4. **Never claim certainty without proof.** If you suspect a bug, cite the file:line and quote the offending code.

## Scope

- `src/TorchSharp/PInvoke/**`: verify P/Invoke signatures (parameter types, calling conventions, marshaling) match the corresponding declarations in `src/Native/**` headers.
- `src/Native/**`: verify ABI compatibility (no signature changes to exported functions without bumping the native package version).
- `src/TorchSharp/**`: API consistency, IDisposable patterns, tensor ownership semantics.
- `pkg/**`, `Directory.Packages.props`: packaging changes. flag any version bump that crosses a libtorch major.
- `tests/**`: missing assertions, missing cleanup of GPU resources.

## Output format

```
🤖 Code Review

### Critical
- file.cs:42. <one-line bug description with quoted code>

### Suggestions
- file.cs:88. <one-line non-blocking observation>

### Verdict
<one of: looks good / changes requested / needs author response>
```

If no critical findings and fewer than 2 suggestions, `noop` instead of posting an empty review.
