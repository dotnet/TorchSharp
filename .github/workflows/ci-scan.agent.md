---
description: |
  Daily. Scans recent CI failures across open PRs, identifies recurring
  failure signatures, and files Known Build Error issues so future PR
  CI flags them as ignorable. Read-only on code; never edits.

on:
  schedule: daily
  roles: [admin, maintain, write]

if: github.repository == 'dotnet/TorchSharp'

timeout-minutes: 30

permissions: read-all

network:
  allowed:
    - defaults
    - github

tools:
  github:
    toolsets: [repos, pull_requests, actions, issues, search]
  bash: true

safe-outputs:
  noop:
    report-as-issue: false
  create-issue:
    title-prefix: "[ci-scan] "
    labels: [build, "Known Build Error", untriaged]
    max: 3
  add-comment:
    target: "*"
    max: 5
    hide-older-comments: true
---

# CI Failure Scanner

Identify recurring CI failures across recent runs and file Known Build Error issues so PR CI can flag them as already-known.

## Hard rules

1. **Read-only on code.** Workflow does not push, rebase, or label PRs.
2. **A failure is "recurring" if it has hit ≥ 3 distinct PRs in the last 14 days.** Below that bar, do nothing.
3. **One issue per signature per run.** Before filing, search open issues for the same normalized signature. If found, comment on it with the new occurrences and stop.
4. **Cap 3 new issues per run.** Force review cadence.
5. **Never file an issue without a quoted failing log line.** No vague "tests fail sometimes".

## Process

1. List failed workflow runs in the last 14 days: `gh run list --status failure --created '>14d' --json databaseId,name,headBranch,event,conclusion,createdAt --limit 200`.
2. For each failure, fetch the failing job logs: `gh run view <id> --log-failed`. Cap at 500 lines per job.
3. Normalize each first-error line (strip paths, timestamps, run IDs, GUIDs).
4. Group by normalized signature. Count distinct PRs affected per signature.
5. For each signature with ≥ 3 distinct PRs:
   - Search for open issues with label `Known Build Error` matching the signature.
   - If found: comment with new occurrence count, list of recent PRs, latest run link.
   - If not found and budget allows: file a new issue using the template below.
6. Stop after 3 new issues.

## Issue template

```
[ci-scan] <short signature>

**Signature** (normalized):
`<one line>`

**Failing line** (raw, one example):
`<one line>`

**Affected PRs in last 14 days** (count: N):
- #1234 (run: <url>)
- #1235 (run: <url>)
...

**Reproducer**: TBD by triage.

🤖 Filed by ci-scan agent.
```
