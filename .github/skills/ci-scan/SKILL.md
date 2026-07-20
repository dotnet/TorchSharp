---
name: ci-scan
description: Analyze recent dotnet/TorchSharp CI failures locally, identify recurring actionable signatures, deduplicate them against existing ci-scan issues, and draft up to three tracking issues. Use when asked to scan TorchSharp CI, investigate recurring main-branch failures, or run the former ci-scan agent locally.
---

# TorchSharp CI scan

Run the TorchSharp CI failure scanner from a local Copilot CLI session. This skill keeps the agent out of scheduled CI until repository billing and authentication are available.

## Inputs

| Input | Required | Description |
|---|---|---|
| Build ID | No | Specific AzDO build to analyze. Otherwise select the source build using the playbook. |
| Apply approved drafts | No | Defaults to false. GitHub mutations require explicit approval in the current conversation. |

## Workflow

1. Confirm the working repository is `dotnet/TorchSharp`.
2. Verify local GitHub access with `gh auth status`.
3. Create local state under `/tmp/torchsharp-ci-scan/`.
4. Read [`references/playbook.md`](references/playbook.md) and [`references/ci-scan.instructions.md`](references/ci-scan.instructions.md).
5. Follow the playbook once for the selected build window.
6. Treat every issue-writing instruction as a request to prepare a draft. Do not create issues, add labels, or post comments while analyzing.
7. Present each proposed issue with its exact title, body, `bug` label, evidence links, and dedup result. Cap the run at three drafts.
8. Apply drafts only after the user explicitly approves the exact GitHub writes. Never close or modify unrelated issues.

## Validation

- Every failure signature has exactly one outcome.
- Every draft passes the occurrence, follow-up, specificity, sanitization, match-count, and dedup gates.
- Build breaks use the `[ci-scan] Build break: ` title prefix.
- The final response includes the full tally table and all skipped reasons.
