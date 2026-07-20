# CI scan playbook

This playbook contains the detailed TorchSharp CI failure-scanning methodology. The local execution and approval rules in [`../SKILL.md`](../SKILL.md) override all issue-writing examples below.

# CI Failure Scanner

You are a CI triage agent for `dotnet/TorchSharp`. Each local run walks the last ~25 completed builds of AzDO definition 174 (`dotnet.TorchSharp`) on `main`, classifies failures, and drafts one `[ci-scan]` tracking issue for every recurring actionable signature.

TorchSharp CI has no Build Analysis or Known Build Error system, so a `[ci-scan]` issue is a plain human-facing tracker. Draft conservatively and only for recurring real failures.

Use the [`ci-scan-feedback`](../../ci-scan-feedback/SKILL.md) skill to score recent scanner artifacts and maintainer feedback against the [Rubric](ci-scan.instructions.md#rubric), then draft targeted methodology edits locally.

## Read this first

Read [`ci-scan.instructions.md`](ci-scan.instructions.md) once at the start and follow its sections by name.

```bash
mkdir -p /tmp/torchsharp-ci-scan/coverage
cat .github/skills/ci-scan/references/ci-scan.instructions.md
```

The TorchSharp profile has `Helix present: no` and `Build Analysis present: no`. Skip every profile-gated Helix or Build Analysis step and read failing lines from AzDO task logs directly.

## Hard rules

1. **Read-only by default.** Draft every GitHub write and show it to the user. Never call a mutating command without explicit approval for the exact write.
2. **Cap 3 issue drafts per run.** On cap, record `cap reached` and stop drafting.
3. **Labels: only `bug`.** Area triage is owned by maintainers.
4. **Every issue title starts with `[ci-scan] `.** Build breaks add the `Build break: ` summary prefix.
5. **One signature = one issue across all legs.** Deduplicate on the signature alone, never on `leg|signature`. Follow [Search existing issues](ci-scan.instructions.md#search-existing) and the [Same-run dedup cache](ci-scan.instructions.md#dedup-cache).
6. **Skip infrastructure noise and weak signatures.** A signature must appear in at least two of the last ~10 builds, unless a shared build or package step blocks all downstream legs.
7. **All state stays under `/tmp/torchsharp-ci-scan/`.**
8. **AzDO REST is anonymous.** Stay on `https://dev.azure.com/dotnet/0e144272-85b9-44a0-bd6c-3800a7b687cb/_apis/build/...`. Do not use `_apis/test/...` or `vstmr.dev.azure.com`.
9. **Sanitize every embedded log excerpt** using [Sanitization](ci-scan.instructions.md#sanitization).

## Step 1 - Select the source build

Choose `source` and `follow_up` using [Source build selection and follow-up gate](ci-scan.instructions.md#source-selection).

```bash
url='https://dev.azure.com/dotnet/0e144272-85b9-44a0-bd6c-3800a7b687cb/_apis/build/builds?definitions=174&branchName=refs/heads/main&statusFilter=completed&resultFilter=succeeded,failed,partiallySucceeded&%24top=25&api-version=7.1'
curl -s "$url" | tee /tmp/torchsharp-ci-scan/builds.json | jq -r '.value[] | "\(.id) \(.result) \(.finishTime)"' | head -25
```

If no scannable build exists, report the matching skip reason and stop.

## Step 2 - Walk the timeline

```bash
src_id=<source build id>
url="https://dev.azure.com/dotnet/0e144272-85b9-44a0-bd6c-3800a7b687cb/_apis/build/builds/${src_id}/timeline?api-version=7.1"
curl -s "$url" | tee /tmp/torchsharp-ci-scan/timeline.json | jq '.records | length'
```

Reconstruct `Stage -> Phase -> Job -> Task` using `parentId`. A failed record with non-null `log.id` is a leaf.

| Leg | Where the signature comes from |
|---|---|
| `Ubuntu_x64` | xUnit test log or compile error |
| `MacOS_arm64` | xUnit test log or compile error |
| `Windows_x64_NetCore` | xUnit test log or compile error |
| `Windows_x64_NetFX` | xUnit test log or compile error |
| `Windows_arm64` | compile error |
| `Linux_Native_Build_For_Packages` | native CMake or compiler error |
| `Windows_Native_Build_For_Packages` | native CMake or compiler error |
| `Windows_arm64_Native_Build_For_Packages` | native CMake or compiler error |
| `MacOS_arm64_Native_Build_For_Packages` | native CMake or compiler error |
| `Build_TorchSharp_And_libtorch_cpu_Packages` | managed build or packaging error |
| `Build_libtorch_cuda_linux_Packages`, `Build_libtorch_cuda_win_Packages` | usually large-download infrastructure noise |
| `Push_*`, `CodeSign_*` | infrastructure noise |

## Step 3 - Classify each failure

Classify every failed leaf using [Failure classification](ci-scan.instructions.md#classification). Save the canonical task log to `/tmp/torchsharp-ci-scan/failure.log`, count occurrences by distinct build ID, and apply the stability gate from [Occurrence counting and window widening](ci-scan.instructions.md#occurrence-counting).

## Step 4 - Apply the follow-up gate

For each stable signature, inspect the later completed build. Skip signatures that disappeared or were fixed by a merged PR after the source build.

## Step 5 - Deduplicate

Run [Search existing issues](ci-scan.instructions.md#search-existing), then the [Same-run dedup cache](ci-scan.instructions.md#dedup-cache). On a match, draft nothing and record `existing-issue #<n>` or `dup of drafted-issue draft-<n> earlier in this run`.

## Step 6 - Draft the tracking issue

For each signature that clears every gate, pass the [match-count gate](ci-scan.instructions.md#new-issue-template) and prepare one issue draft using the [New-issue template](ci-scan.instructions.md#new-issue-template). Apply only the `bug` label. Prefix build-break titles with `Build break: `.

## Step 7 - Tally

Append one outcome line per signature to `/tmp/torchsharp-ci-scan/coverage/dotnet.TorchSharp.txt`:

```text
<sig-short>  <outcome>  <reason-if-skipped>
```

`<outcome>` is `drafted-issue draft-<n>`, `existing-issue #<n>`, or `skipped: <reason>`.

```text
| total-signatures | issues-drafted | reused-existing | skipped-with-reason |
```
