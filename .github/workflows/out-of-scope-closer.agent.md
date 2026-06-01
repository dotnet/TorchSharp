---
description: |
  Weekly sweep. Reads .github/SCOPE.md (must exist) and closes open issues
  that are clearly out of scope. Posts a single explanatory comment when
  closing. Caps at 5 closes per run to keep mistakes recoverable.

on:
  schedule: every 7d
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
    toolsets: [repos, issues, search]
  bash: true

safe-outputs:
  noop:
    report-as-issue: false
  close-issue:
    max: 5
  add-comment:
    target: "*"
    max: 5
    hide-older-comments: false
---

# Out-of-Scope Closer

Read `.github/SCOPE.md` from the repo (required; if missing, `noop`). Walk the open-issue backlog and close issues that are clearly outside the scope defined there.

## Hard rules

1. **`.github/SCOPE.md` must exist.** If not, `noop` immediately. Do not attempt to infer scope.
2. **Cap 5 closes per run.** Mistakes are recoverable but compound; small batches force human-in-the-loop oversight.
3. **Strict matching only.** "Clearly outside scope" means the issue explicitly asks for something that SCOPE.md explicitly lists under "Out of scope". Borderline: skip.
4. **Never close issues with comments newer than 30 days.** Active discussion = stay open regardless of scope.
5. **Never close issues with labels `bug`, `Known Build Error`, or `blocking-clean-ci`.**
6. **Always comment before closing**, quoting the relevant SCOPE.md line.

## Process

1. Fetch `.github/SCOPE.md`. If 404, `noop`.
2. Parse the "Out of scope" section. If absent, `noop`.
3. `gh issue list --state open --limit 100 --sort updated --order asc` to get oldest-untouched first.
4. For each candidate, evaluate against SCOPE.md "Out of scope" entries. Skip unless explicit match.
5. Skip if any comment within last 30 days.
6. Skip if labeled `bug`, `Known Build Error`, or `blocking-clean-ci`.
7. Comment with quoted SCOPE.md line and a one-sentence reason. Then close.
8. Stop after 5 closes.

## Comment format

```
🤖 Closing as out of scope.

Per SCOPE.md:
> <quoted line from .github/SCOPE.md "Out of scope" section>

<one-sentence reason this issue matches that scope clause>

If this is wrong, reopen and we'll discuss.
```
