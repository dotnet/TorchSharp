---
description: |
  Weekly. Reads `.github/SCOPE.md`. For each "Out of scope" entry, finds
  matching open issues, posts a closing comment, and closes as
  `not planned`. Capped at 5 closes per run. Skips active discussion and
  protected labels. Noop if `.github/SCOPE.md` is missing.

on:
  schedule: weekly
  workflow_dispatch:
  roles: [admin, maintain, write]

if: |
  github.repository == 'dotnet/TorchSharp' &&
  hashFiles('.github/SCOPE.md') != ''

timeout-minutes: 20

permissions: read-all

concurrency:
  group: out-of-scope-closer
  cancel-in-progress: false

network:
  allowed:
    - defaults
    - github

tools:
  github:
    toolsets: [repos, issues, search]
  bash: ["git", "find", "ls", "cat", "grep", "head", "tail", "wc", "jq", "tee", "sed", "awk", "tr", "cut", "sort", "uniq", "xargs", "echo", "date", "test"]

checkout:
  fetch-depth: 1

safe-outputs:
  noop:
    report-as-issue: false
  add-comment:
    target: "*"
    max: 5
    hide-older-comments: false
  close-issue:
    max: 5
---

# Out-of-Scope Closer (TorchSharp)

Read `.github/SCOPE.md` and close open issues that match its explicit "Out of scope" entries. Capped at 5 closes per run to force review cadence.

## Hard rules

1. **`.github/SCOPE.md` must exist.** If missing, `noop` and stop.
2. **Cap 5 closes per run.** On cap, record `skipped: cap reached` and stop.
3. **Never close an issue without first posting the closing comment.** The comment cites the scope entry. Order matters.
4. **Skip protected labels: `bug`, `Known Build Error`, `blocking-clean-ci`, `help wanted`, `good first issue`.** A scope match against any of these is a false positive; skip silently.
5. **Skip active discussion.** Any issue with a non-bot comment in the last 30 days: skip silently.
6. **Skip issues opened in the last 14 days.** New issues deserve a triage pass first.
7. **Match only against explicit `## Out of scope` entries in `SCOPE.md`.** Each entry must be a bullet point with a clear keyword or symbol. Fuzzy matches that require interpretation: skip.
8. **Idempotency.** If the issue already has a bot close comment containing `<!-- out-of-scope-closer -->`, skip.

## Expected `SCOPE.md` shape

The file is a maintainer-owned document. Only the section starting with `## Out of scope` (exact heading) is consumed. Lines under it that start with `- ` are treated as entries. Each entry is a literal substring or `regex: <pattern>`. Example:

```markdown
## Out of scope

- training on TPU
- federated learning
- regex: ^(Add|Implement) (Quantization|FX) (graph|tracing)
```

Anything outside that section is ignored.

## Process

1. Read `.github/SCOPE.md`:
   ```bash
   test -f .github/SCOPE.md || { echo "SCOPE.md missing - noop"; exit 0; }
   ```
2. Parse the `## Out of scope` section into `/tmp/gh-aw/agent/scope-entries.txt` (one entry per line, literal or `regex:` prefix).
3. For each entry, search open issues:
   ```bash
   gh issue list --repo dotnet/TorchSharp --state open --limit 100 \
     --search "$query in:title,body" \
     --json number,title,labels,createdAt,updatedAt,comments
   ```
4. For each match, apply the filters in rules 4-7 in order. First failure -> skip.
5. For each surviving match (up to 5):
   - Post the closing comment (template below).
   - Close as `not planned`.
   - Append `<issue#> <entry>` to `/tmp/gh-aw/agent/closed.txt`.
6. At end of run, print the closed list to the agent log.

## Comment template

```
<!-- out-of-scope-closer -->
🤖 Closing as out of scope.

This issue matches the following entry in [`.github/SCOPE.md`](https://github.com/dotnet/TorchSharp/blob/main/.github/SCOPE.md):

> <quoted scope entry>

If you believe this is in scope, please reply with the rationale and we will reopen.

Posted by [`out-of-scope-closer`](https://github.com/dotnet/TorchSharp/blob/main/.github/workflows/out-of-scope-closer.agent.md).
```
