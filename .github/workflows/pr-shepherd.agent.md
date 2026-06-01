---
description: |
  Daily. Walks open PRs that are aligned with repo goals (per
  .github/SCOPE.md) and pushes them forward: requests review when missing,
  pings stalled authors at 14d, labels merge-ready candidates. Read-only
  on the code; never rebases or pushes commits.

on:
  schedule: daily
  roles: [admin, maintain, write]

if: github.repository == 'dotnet/TorchSharp'

timeout-minutes: 20

permissions: read-all

network:
  allowed:
    - defaults
    - github

checkout:
  ref: main
  fetch-depth: 1

tools:
  github:
    toolsets: [repos, pull_requests, issues, search]
  bash: true

safe-outputs:
  noop:
    report-as-issue: false
  add-comment:
    target: "*"
    max: 10
    hide-older-comments: true
  add-labels:
    allowed: [merge-ready, needs-author-response, needs-rebase]
    max: 10
  add-reviewer:
    max: 5
---

# PR Shepherd

Walk every non-draft open PR and apply one of: `merge-ready` label, `needs-author-response` ping, `needs-rebase` ping, request-review for missing reviewers. Aligned-with-scope PRs only.

## Hard rules

1. **No commits, no rebases, no pushes.** This workflow only labels, comments, requests reviews.
2. **Read `.github/SCOPE.md` first.** If missing, `noop`. PRs that fall under SCOPE.md "Out of scope" are skipped here (the out-of-scope-closer handles them).
3. **One action per PR per run.** Don't both nudge the author and request review in the same cycle.
4. **No nudges newer than 7 days old.** Don't spam.

## Categories

For each non-draft open PR:

1. **`merge-ready`**. all required checks green, has at least 1 approval, no `changes-requested` review, no merge conflicts. Label and `noop`.
2. **`needs-author-response`**. reviewer left `changes-requested` or comment ≥ 14 days ago, author has not pushed since. Label + comment pinging author by `@handle`.
3. **`needs-rebase`**. merge conflict with main. Label + comment.
4. **Missing reviewer**. no reviewers requested, no human review yet, PR open ≥ 3 days. Request review from the file owners (parse CODEOWNERS) or fall back to repo maintainers.
5. **Otherwise**. skip.

## Comment formats

**author-response nudge:**
```
🤖 Friendly nudge. @<author>, this PR has unanswered review feedback from <date>. Could you take a look?
```

**rebase nudge:**
```
🤖 This PR has a merge conflict with main. Could you rebase?
```

## Scope filter

Before processing any PR, check whether the PR's stated goal matches `.github/SCOPE.md` "In scope" entries. If it matches "Out of scope", skip silently (the out-of-scope-closer will handle it). If unclear, treat as in-scope and proceed.
