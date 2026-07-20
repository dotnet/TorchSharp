# CI scan shared instructions

Reusable methodology for [`ci-scan`](../SKILL.md) and
[`ci-scan-feedback`](../../ci-scan-feedback/SKILL.md).

## Table of contents

- [Repository profile](#repo-profile)
- [Environment constraints](#environment-constraints)
- [Data sources](#data-sources)
- [Source build selection and follow-up gate](#source-selection)
- [Failure classification](#classification)
- [Occurrence counting and window widening](#occurrence-counting)
- [Search existing issues](#search-existing)
- [Same-run dedup cache](#dedup-cache)
- [Signature specificity](#signature-specificity)
- [Bad vs good signatures](#bad-vs-good)
- [Sanitization](#sanitization)
- [New-issue template](#new-issue-template)
- [Recognized skip-reason vocabulary](#skip-reasons)
- [Rubric](#rubric)
- [Output discipline](#output-discipline)

<a id="repo-profile"></a>
## Repository profile

| Key | Value |
|---|---|
| Repo | `dotnet/TorchSharp` |
| AzDO org / project | `dotnet` / `0e144272-85b9-44a0-bd6c-3800a7b687cb` |
| Pipeline | `dotnet.TorchSharp`, definition id **174** |
| Branch scanned | `refs/heads/main` |
| Helix present | **no** |
| Build Analysis present | **no** |
| Issue model | plain `[ci-scan]` tracking issues |
| Issue labels | `bug` |
| Build-break title prefix | `[ci-scan] Build break: ` |

Skip every step marked for Helix or Build Analysis. TorchSharp tests run inline on AzDO agents and
the tracking issues are human-facing.

<a id="environment-constraints"></a>
## Environment constraints

- Bind URLs containing `?` or `&` to a shell variable before calling `curl`.
- Percent-encode OData `$top` and `$skip` as `%24top` and `%24skip`.
- Prefer `tee` when saving evidence so command output remains visible.
- Treat each bash call as a fresh subshell.
- Store all state under `/tmp/torchsharp-ci-scan/`.
- Use anonymous AzDO REST on the profile host. Do not add authentication headers.
- Do not use `_apis/test/...` or `vstmr.dev.azure.com` because they redirect to sign-in.

```bash
url='https://dev.azure.com/dotnet/0e144272-85b9-44a0-bd6c-3800a7b687cb/_apis/build/builds?definitions=174&branchName=refs/heads/main&statusFilter=completed&resultFilter=succeeded,failed,partiallySucceeded&%24top=25&api-version=7.1'
curl -s "$url" | tee /tmp/torchsharp-ci-scan/builds.json | jq -r '.value[0] | "\(.id) \(.result)"'
```

<a id="data-sources"></a>
## Data sources

- **Build list** uses definition 174, `refs/heads/main`, completed results, and the latest 25 builds.
- **Timeline** uses `/builds/{id}/timeline?api-version=7.1`. Reconstruct the hierarchy through
  `parentId`. Failed records with non-null `log.id` are leaves worth reading.
- **Task log** comes from each leaf record's `log.url`.
- **GitHub issues** are the only cross-run dedup source because TorchSharp has no Build Analysis.

<a id="source-selection"></a>
## Source build selection and follow-up gate

1. Pick the most recent failed or partially succeeded build that has a completed build with a
   strictly later `finishTime`.
2. Record one of these selection-time skip reasons when applicable:
   - `stale build window (>14d)`
   - `no follow-up build yet, defer to next run`
   - `no failed build in 7d`
3. For every stable signature, inspect the later build:
   - If the later build succeeded or does not contain the signature, record
     `signature absent from follow-up build #<id>`.
   - If the later build still contains the signature, continue.
   - For build breaks, search merged PRs after the source build that touch the failing file or cite
     the diagnostic. On a match, record `fix already merged after source build`.

<a id="classification"></a>
## Failure classification

Save the canonical failing log before extracting the signature:

```bash
log_url='<AzDO task log URL>'
curl -s "$log_url" | tee /tmp/torchsharp-ci-scan/failure.log | tail -200
```

1. **Build break.** The failing task is compile, restore, native build, CMake, or packaging and the
   test task did not run. Use the compiler, linker, CMake, or packaging diagnostic line.
2. **Phase or stage failure.** When no failed job leaf exists, inspect the phase log and the latest
   non-succeeded child task. Treat a compile-shaped diagnostic as a build break.
3. **Test failure.** Use the first `[FAIL]`, `Failed:`, or assertion line. The signature is the test
   method FQN plus the first assertion or exception message line.
4. **Infrastructure failure.** Agent disconnects, offline pools, queue timeouts, transient network
   failures, signing, publishing, and large libtorch download failures have no stable product
   signature. Record `infra noise - no stable signature`.

Shared native or package build failures can block all downstream legs. A compile failure isolated to
one test leg still needs recurrence.

<a id="occurrence-counting"></a>
## Occurrence counting and window widening

Count distinct build IDs containing the signature. Multiple legs or retries from the same build
count once.

A signature is stable when:

- it appears in at least two distinct builds from the recent window, or
- it is a build break in a shared build or package step that blocks every downstream leg.

If a signature appears in every sampled build, widen the list in increments of ten up to forty
additional builds. Stop at the first build where the signature is absent. The next build is the
first observed occurrence. If no gap appears, report the oldest scanned build and state that the
true origin may predate the window.

<a id="search-existing"></a>
## Search existing issues

Search before drafting:

```bash
sig_short='<distinctive sanitized substring, at most 80 characters>'
gh issue list --repo dotnet/TorchSharp --state open \
  --search "[ci-scan] $sig_short in:title,body" \
  --json number,title,url | tee /tmp/torchsharp-ci-scan/existing.json
```

Try a second distinctive substring when the first search misses. Include closed issues in a second
pass when a recent closure may already track the failure. Verify the same failure family and failing
line before accepting a match.

On a confirmed match, record `existing-issue #<n>` and draft nothing.

<a id="dedup-cache"></a>
## Same-run dedup cache

Deduplicate on the normalized signature, not `leg|signature`.

```bash
signature_norm=$(printf '%s' "<signature>" | tr -d '\t\n\r')
test -f /tmp/torchsharp-ci-scan/drafted.tsv \
  && cut -f1 /tmp/torchsharp-ci-scan/drafted.tsv \
  | grep -Fxq -- "$signature_norm"
printf '%s\t%s\n' "$signature_norm" "draft-<n>" \
  >> /tmp/torchsharp-ci-scan/drafted.tsv
```

On a cache hit, record `dup of drafted-issue draft-<n> earlier in this run`.

<a id="signature-specificity"></a>
## Signature specificity

Keep the most distinctive stable substring:

- test method FQN plus the assertion or exception stem
- compiler, linker, or CMake diagnostic code plus the offending symbol

Reject bare exit codes, generic exception types without messages, test-run summary lines, and text
that also appears on passing or skipped lines.

<a id="bad-vs-good"></a>
## Bad vs good signatures

| Failing line | Bad signature | Good signature |
|---|---|---|
| `[FAIL] TorchSharp.Test.TensorTests.TestArithmetic : Assert.Equal() Failure: Expected 0.81, got 0.79` | `Assert.Equal() Failure` | `TensorTests.TestArithmetic : Assert.Equal() Failure: Expected 0.81` |
| `src/Foo.cs(120,5): error CS0246: The type or namespace name 'Bar' could not be found` | `error CS0246` | `Foo.cs: error CS0246: The type or namespace name 'Bar' could not be found` |
| `Process terminated. Exit code 134.` on one leg | `Exit code 134` | Use only when the same stable stack frame recurs |
| `Test run failed` | `Test run failed` | Reject and locate the underlying failure |

<a id="sanitization"></a>
## Sanitization

Replace volatile values while keeping the diagnostic readable:

- absolute paths become repository-relative paths
- GUIDs and build numbers become `<id>`
- machine names and IP addresses become `<host>`
- ports become `<port>`
- timestamps and durations become `<time>`
- process IDs become `<pid>`

Keep diagnostic codes, symbols, assertion text, and `[FAIL]` markers verbatim.

<a id="new-issue-template"></a>
## New-issue template

Before drafting, verify the literal match is present in the saved failure log:

```bash
grep -F -c -- "<literal match substring>" /tmp/torchsharp-ci-scan/failure.log
```

The count must be at least one. Otherwise record
`signature did not match failure.log (N=0)`.

````markdown
## Signature

`<one-line normalized failure>`

## Failing line (raw)

```text
<one sanitized failure line>
```

## Match signature (literal substring)

```text
<exact verified substring>
```

## Category

<Build break | Test failure>

## Affected legs

- `<leg>` with task log URL

## First build it occurred

- Build: `<AzDO build URL>`
- Finished: `<UTC timestamp>`
- Commit: `<SHA>`
- Occurrences in the scanned window: `<n>`

## Reasoning

<why this is a recurring product failure rather than infrastructure noise>

---

Prepared with the repository-local [`ci-scan`](https://github.com/dotnet/TorchSharp/blob/main/.github/skills/ci-scan/SKILL.md)
skill. Comment here to flag a false positive or add context for the local
[`ci-scan-feedback`](https://github.com/dotnet/TorchSharp/blob/main/.github/skills/ci-scan-feedback/SKILL.md)
skill.
````

Apply the `bug` label. Build-break titles use
`[ci-scan] Build break: <signature>`.

<a id="skip-reasons"></a>
## Recognized skip-reason vocabulary

- `cap reached`
- `< 2 occurrences and not blocking`
- `infra noise - no stable signature`
- `signature absent from follow-up build #<id>`
- `stale build window (>14d)`
- `no follow-up build yet, defer to next run`
- `no failed build in 7d`
- `fix already merged after source build`
- `dup of drafted-issue draft-<n> earlier in this run`
- `existing-issue #<n>`
- `suspected infra outage`
- `signature did not match failure.log (N=<count>)`
- `weak signature`

<a id="rubric"></a>
## Rubric

- **Title scoped to one failure shape.**
- **Classification matches the failure.** Real failures use `bug`; build breaks use the title prefix.
- **Match block is specific.**
- **Occurrence count is honest.**
- **Follow-up gate is respected.**

<a id="output-discipline"></a>
## Output discipline

- Walk definition 174 once per run.
- Write one outcome line per signature to
  `/tmp/torchsharp-ci-scan/coverage/dotnet.TorchSharp.txt`.
- Do not create aggregate outage issues.
- Do not apply area labels.
- Do not add comments to existing trackers.
- Change scanner behavior in the skill, not in issue bodies.
- Include the Step 7 summary table in the final response.
