---
description: |
  Daily. Reads `.github/SCOPE.md`. For each non-draft open PR, categorizes
  status and applies one of `merge-ready` / `needs-author-response` /
  `needs-rebase`. Posts at most one ping per category per PR per
  fortnight. Labels and comments only; never rebases or pushes.

on:
  schedule: daily
  workflow_dispatch:
  roles: [admin, maintain, write]

if: github.repository == 'dotnet/TorchSharp'

timeout-minutes: 20

permissions: read-all

concurrency:
  group: pr-shepherd
  cancel-in-progress: false

network:
  allowed:
    - defaults
    - github

tools:
  github:
    toolsets: [repos, pull_requests, search]
  bash: ["git", "find", "ls", "cat", "grep", "head", "tail", "wc", "jq", "tee", "sed", "awk", "tr", "cut", "sort", "uniq", "xargs", "echo", "date", "test"]

checkout:
  fetch-depth: 1

safe-outputs:
  noop:
    report-as-issue: false
  add-labels:
    allowed: [merge-ready, needs-author-response, needs-rebase]
    max: 10
  add-comment:
    target: "*"
    max: 5
    hide-older-comments: false
---

# PR Shepherd (TorchSharp)

Walk non-draft open PRs once a day. For each PR, decide one of:
- `merge-ready` -> apply label, post short summary, hand off to maintainers.
- `needs-author-response` -> label and ping author once per fortnight at most.
- `needs-rebase` -> label and post one-line conflict note.
- `noop` -> no signal change.

Labels and comments only. Never push, rebase, approve, merge, close, or reopen.

## Hard rules

1. **No writes to code.** No `push-to-pull-request-branch`, no `create-pull-request`.
2. **Cap 5 comments per run.** On cap, record `skipped: cap reached` and stop.
3. **Cap one ping per PR per fortnight per category.** Look for `<!-- pr-shepherd:<category>:<sha> -->` in your prior bot comments. If your last marker for the same category was within 14 days, skip the comment (still update label if needed).
4. **Read `.github/SCOPE.md` if present.** If a PR's title or diff path is listed under `## Out of scope`, skip the PR (do not label, do not comment).
5. **Skip drafts. Skip fork PRs from external authors. Skip PRs opened <3 days ago.** Bot-author PRs (`dotnet-maestro[bot]`, `dependabot[bot]`, `app/copilot-swe-agent`) are exempt from the 3-day rule.
6. **Skip protected labels: `do-not-merge`, `WIP`, `blocked`.**
7. **Never request reviewers.** This repo has no CODEOWNERS; maintainers self-assign.
8. **Idempotency.** Always include `<!-- pr-shepherd:<category>:<head-sha> -->` at the top of every comment.

## Categories

| Category | Conditions (all must hold) | Action |
|---|---|---|
| `merge-ready` | `mergeable == MERGEABLE`, all required checks `SUCCESS`, has `>= 1` approval, no `CHANGES_REQUESTED` review since head sha, no merge conflicts. | Label `merge-ready`. Comment: short summary linking the approval and check run. |
| `needs-author-response` | Most recent review is `CHANGES_REQUESTED` OR most recent non-bot comment requests author action AND author has not pushed since. PR is `>= 14d` old. | Label `needs-author-response`. Comment: ping author by `@handle`, link the unresolved review, ask for a status update. |
| `needs-rebase` | `mergeable == CONFLICTING`. | Label `needs-rebase`. Comment: list the conflicting paths from `gh pr view --json mergeStateStatus,files`. |
| `noop` | None of the above, OR conditions hold but the same-category marker is within 14 days. | No label change, no comment. |

When a PR transitions to `merge-ready` from a different category, remove the old category label only if it was applied by this workflow (check that the most recent label-event for that label was by the workflow's bot identity). Otherwise leave it alone.

## Process

1. `gh pr list --repo dotnet/TorchSharp --state open --limit 50 --json number,title,author,isDraft,mergeable,mergeStateStatus,reviewDecision,headRefOid,createdAt,updatedAt,labels,headRepository,files`.
2. For each PR, apply rule 5 filters. Skip silently on any fail.
3. Read `.github/SCOPE.md` (rule 4 gate).
4. Resolve category via the table above. If `noop`, move on.
5. For non-`noop`: fetch prior bot comments, locate the most recent `<!-- pr-shepherd:<category>:* -->` marker, apply rule 3.
6. Apply label change first, then post comment if rule 3 permits.

## Comment templates

`merge-ready`:

```
<!-- pr-shepherd:merge-ready:<head-sha> -->
🤖 PR Shepherd: this PR looks ready to merge.

- Required checks: all `SUCCESS`.
- Approvals: <count> (<@reviewer> on <date>).
- No `CHANGES_REQUESTED` review against the current head.

Maintainers, this is yours to land.

Posted by [`pr-shepherd`](https://github.com/dotnet/TorchSharp/blob/main/.github/workflows/pr-shepherd.agent.md).
```

`needs-author-response`:

```
<!-- pr-shepherd:needs-author-response:<head-sha> -->
🤖 PR Shepherd: this PR has unanswered review feedback.

@<author>, please take a look at <review url> from <reviewer> (<N> days ago). Reply, push a fix, or close if you no longer plan to land this.

Posted by [`pr-shepherd`](https://github.com/dotnet/TorchSharp/blob/main/.github/workflows/pr-shepherd.agent.md).
```

`needs-rebase`:

```
<!-- pr-shepherd:needs-rebase:<head-sha> -->
🤖 PR Shepherd: this PR has merge conflicts with `main`.

Conflicting paths:
- `<path1>`
- `<path2>`

Please rebase or merge `main` into the branch.

Posted by [`pr-shepherd`](https://github.com/dotnet/TorchSharp/blob/main/.github/workflows/pr-shepherd.agent.md).
```
