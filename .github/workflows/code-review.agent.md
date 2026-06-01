---
description: |
  First-pass code review on every non-draft PR. Posts one comment with
  TorchSharp-specific findings (P/Invoke signatures, native ABI,
  IDisposable ownership, packaging surface). Read-only; high-signal only.

on:
  pull_request:
    types: [opened, synchronize, ready_for_review]
  roles: [admin, maintain, write]

if: |
  github.repository == 'dotnet/TorchSharp' &&
  github.event.pull_request.draft == false &&
  github.event.pull_request.head.repo.full_name == github.repository

timeout-minutes: 20

permissions: read-all

concurrency:
  group: code-review-${{ github.event.pull_request.number }}
  cancel-in-progress: true

network:
  allowed:
    - defaults
    - github

tools:
  github:
    toolsets: [repos, pull_requests]
  bash: ["git", "find", "ls", "cat", "grep", "head", "tail", "wc", "jq", "tee", "sed", "awk", "tr", "cut", "sort", "uniq", "xargs", "echo", "test", "mkdir", "basename", "dirname"]

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

# Code Review (TorchSharp)

Review PR #${{ github.event.pull_request.number }} and post one comment using the template below. Skip drafts and fork PRs (`if:` gate already handles this; treat any drift as a defect).

## Hard rules

1. **Read-only.** No approvals, no labels, no commits, no review-requests.
2. **One comment per run.** If your last comment on this PR contains the same `<!-- code-review:<head-sha> -->` marker as the current head sha, post `noop`. Always include `<!-- code-review:<head-sha> -->` at the top of the comment.
3. **High signal only.** Do not comment on style, formatting, line length, naming taste, or `var` vs explicit types. Only comment on bugs, missing null checks, native ABI mismatches, undisposed tensors, breaking public API, threading, packaging breakage, and missing tests for new behavior.
4. **Never claim certainty without proof.** Cite `file:line` and quote the offending code.
5. **`noop` if you find nothing critical, fewer than 2 suggestions, and tests look adequate.** Empty reviews are worse than silence.

## Scope

| Path | What to look for |
|---|---|
| `src/Native/**` | C++ shim signatures vs `src/TorchSharp/PInvoke/**`. parameter count, order, types. ABI changes that need a libtorch pin bump. |
| `src/TorchSharp/PInvoke/**` | `[DllImport]` signature mismatch with the C++ shim. Marshalling correctness (`IntPtr` vs `byte*`, `bool` blittability). |
| `src/TorchSharp/Tensor/**` | Tensor ownership / `IDisposable` correctness. Any new `tensor` allocation must either be returned, disposed, or wrapped in a `using`. |
| `src/TorchSharp/NN/**` | Module subclasses must call `RegisterComponents()` if they hold submodules or parameters. |
| `pkg/**`, `Directory.Packages.props`, `Directory.Build.props` | Packaging changes. Flag any version bump that crosses a libtorch major; flag changes to `TorchSharp-cpu` / `TorchSharp-cuda-*` package layouts. |
| `src/TorchVision/**`, `src/TorchAudio/**` | New transforms must have unit tests in `test/TorchSharpTest` or `test/TorchVisionTest`. |
| `test/**` | Missing assertions, missing dispose, missing edge cases (empty tensor, CPU vs CUDA, dtype mismatch). |

## Process

1. `gh pr view <PR> --json title,body,baseRefName,headRefOid,labels,additions,deletions,changedFiles,files`. Save head sha.
2. If `changedFiles > 80` OR `additions + deletions > 3000` -> comment "PR too large for first-pass review; please consider splitting" and `noop` for the rest.
3. `gh pr diff <PR>`. For each changed file in the scope table above, read the file at the head ref and apply the scope rules.
4. Search for previous bot comments matching `<!-- code-review:<head-sha> -->` to enforce one-per-sha.

## Output format

```
<!-- code-review:<head-sha> -->
🤖 Code Review

### Critical

- `file.cs:42` - <one-line bug description>
  ```csharp
  <quoted offending code, max 5 lines>
  ```
  <one-line why it's wrong>

### Suggestions

- `file.cs:88` - <one-line non-blocking observation>

### Tests

- <one line on test coverage of new behavior, or "Tests look adequate.">

### Verdict

<one of: looks good / changes requested / needs author response>

---

Posted by [`code-review`](https://github.com/dotnet/TorchSharp/blob/main/.github/workflows/code-review.agent.md). One comment per head sha; force-push to re-trigger.
```

If no Critical, fewer than 2 Suggestions, and tests are adequate -> `noop`.
