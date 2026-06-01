---
description: |
  Classify newly opened issues. Applies one of {bug, enhancement,
  question, documentation, Missing Feature} based on the title and body.
  Adds platform labels if mentioned. Posts no comments.

on:
  issues:
    types: [opened, reopened]
  roles: [admin, maintain, write]

if: github.repository == 'dotnet/TorchSharp'

timeout-minutes: 5

permissions: read-all

network:
  allowed:
    - defaults
    - github

tools:
  github:
    toolsets: [repos, issues, search]
  bash: true

safe-outputs:
  noop:
    report-as-issue: false
  add-labels:
    allowed:
      - bug
      - enhancement
      - question
      - documentation
      - "Missing Feature"
      - cuda
      - cpu
      - linux
      - windows
      - macos
---

# Issue Triage

Classify issue #${{ github.event.issue.number }}.

## Hard rules

1. **Exactly one classification label.** Pick one of `bug`, `enhancement`, `question`, `documentation`, `Missing Feature`.
2. **Platform labels are optional and additive.** Add `cuda`/`cpu`/`linux`/`windows`/`macos` only when the title or body explicitly mentions them.
3. **Never apply labels that already exist on the issue.**
4. **Do not comment.** Labels only. If unclassifiable, `noop`.

## Heuristics

- **bug**: error, exception, crash, stack trace, "doesn't work", reproducer included.
- **enhancement**: "would be nice", "support for", "improve", concrete improvement to existing API.
- **Missing Feature**: "TorchSharp doesn't have X", "PyTorch has X but TorchSharp doesn't", parity gap.
- **question**: "how do I", "is it possible", usage question, no obvious bug or gap.
- **documentation**: "docs missing", "example wrong", README/docs issue only.

## Process

1. Fetch issue title and first 2000 chars of body.
2. Apply the heuristics above to pick exactly one classification label.
3. Scan title + body for platform keywords; add matching platform labels.
4. Apply via `add-labels` safe-output.
