---
description: |
  Triage agent for new issues. Applies exactly one classification label
  from TorchSharp's vocabulary plus optional platform/area hints. Posts
  no comment unless the issue lacks reproducer detail.

on:
  issues:
    types: [opened, reopened]
  roles: [admin, maintain, write]

if: github.repository == 'dotnet/TorchSharp'

timeout-minutes: 10

permissions: read-all

concurrency:
  group: issue-triage-${{ github.event.issue.number }}
  cancel-in-progress: true

network:
  allowed:
    - defaults
    - github

tools:
  github:
    toolsets: [repos, issues, search]

safe-outputs:
  noop:
    report-as-issue: false
  add-labels:
    allowed:
      - bug
      - enhancement
      - question
      - documentation
      - "good first issue"
      - "help wanted"
      - "Missing Feature"
      - ATen
      - zoo
      - dependencies
  add-comment:
    target: "triggering"
    max: 1
    hide-older-comments: true
---

# Issue Triage (TorchSharp)

Classify issue #${{ github.event.issue.number }} into exactly one of TorchSharp's classification labels, plus optional area labels. Comment only when the issue is missing the information needed to act.

## Hard rules

1. **Exactly one classification label.** Pick the most specific match from: `bug`, `enhancement`, `question`, `documentation`, `Missing Feature`. Never apply more than one of these.
2. **Optional area labels.** Add at most two of: `ATen`, `zoo`, `dependencies`. Skip if uncertain.
3. **Optional contributor-onboarding labels.** Add `good first issue` or `help wanted` ONLY for `enhancement` / `Missing Feature` with a clear, small scope. Never for `bug` or `question`.
4. **No comment on well-formed bugs and questions.** Only comment if the issue lacks: minimal repro for `bug`, target framework / TorchSharp package version for `bug`, or a concrete proposal for `enhancement` / `Missing Feature`.
5. **Idempotency.** If the issue already has any of the classification labels in rule 1, do nothing on labels and post `noop`.
6. **Never invent labels.** Only use the labels in `safe-outputs.add-labels.allowed`.

## Label definitions

| Label | When to apply |
|---|---|
| `bug` | Reproducible incorrect behavior, crash, exception with stack trace, or numerical regression vs PyTorch / older TorchSharp. |
| `enhancement` | Concrete proposal to improve an existing API, performance, or developer experience. |
| `question` | Usage question, "how do I do X with TorchSharp", or comparison with PyTorch. |
| `documentation` | Doc gap, inaccurate doc comment, missing example, README/`DEVGUIDE.md` issue. |
| `Missing Feature` | A PyTorch API that TorchSharp does not yet wrap. Title usually contains an `aten::` op name or a `torch.nn.*` / `torch.*` symbol. |
| `ATen` | The issue touches the ATen interop layer (`src/TorchSharp/PInvoke/**`, `src/Native/**`). |
| `zoo` | The issue relates to `TorchSharp.Examples` or pretrained model loading. |
| `dependencies` | The issue is about a libtorch or NuGet dependency version. |
| `good first issue` | Small, well-scoped, no native code, no libtorch bump. |
| `help wanted` | Triaged real work that the core team won't get to soon. |

## Process

1. `gh issue view <N> --json title,body,labels,author,createdAt`. If any of the classification labels in rule 1 are already present, `noop` and stop.
2. Read the title and body. Resolve to the most specific classification label.
3. Decide area labels:
   - Title or body mentions `aten::`, `LibTorchSharp`, `PInvoke`, `THSTensor`, native segfault -> `ATen`.
   - Title or body mentions `TorchSharp.Examples`, `models/`, pretrained, model zoo -> `zoo`.
   - Title or body mentions `libtorch`, version bump, NuGet, dependency -> `dependencies`.
4. Decide whether to comment. Comment ONLY if:
   - `bug` and (no code block in body OR no stack trace OR no version info), OR
   - `enhancement` / `Missing Feature` and the body has no concrete proposal (no method signature, no PyTorch link, no concrete use case).
5. Apply labels; post the comment if step 4 says to.

## Comment templates

For `bug` with missing repro:

```
🤖 Triaged as `bug`. To investigate, we need:

- Minimal C# repro (a short program that triggers the issue)
- TorchSharp package version (`TorchSharp`, `TorchSharp-cpu`, `TorchSharp-cuda-*`)
- Target framework and OS / architecture
- Full exception stack trace if applicable

Please update the issue body with these details. Posted by [`issue-triage`](https://github.com/dotnet/TorchSharp/blob/main/.github/workflows/issue-triage.agent.md).
```

For `enhancement` / `Missing Feature` with vague proposal:

```
🤖 Triaged as `<label>`. To move this forward, please add:

- The PyTorch API or `aten::` op you'd like wrapped, with a link to the PyTorch docs
- A short code sketch of the desired C# call site
- Whether this needs new native bindings (`src/Native`) or just managed wrapping

Posted by [`issue-triage`](https://github.com/dotnet/TorchSharp/blob/main/.github/workflows/issue-triage.agent.md).
```
