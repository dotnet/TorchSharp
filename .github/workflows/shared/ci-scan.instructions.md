# CI scan - shared instructions (TorchSharp)

Reusable methodology shared by [`ci-scan`](../ci-scan.agent.md) (the scanner) and
[`ci-scan-feedback`](../ci-scan-feedback.agent.md) (the reviewer). The scanner reads this file
once at the start of a run and follows the named sections by reference; the feedback workflow
reads only the [Recognized skip-reason vocabulary](#skip-reasons) and
[Rubric](#rubric) sections. Keeping the heavy, slow-changing detail here keeps each agent prompt
short enough to stay fully in context (progressive disclosure) while still giving the agent the
same depth on every run.

Read it with `cat .github/workflows/shared/ci-scan.instructions.md` as the first bash step. Each
section below has an HTML anchor so a prompt can cite it precisely (for example
"follow [Same-run dedup cache](#dedup-cache)").

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

These are the only repo-specific values the methodology depends on. The TorchSharp copy of this
file differs from this one only in this table and in the profile-gated lines flagged inline; every
other section is identical so the two scanners behave the same way.

| Key | Value |
|---|---|
| Repo | `dotnet/TorchSharp` |
| AzDO org / project | `dotnet` / `0e144272-85b9-44a0-bd6c-3800a7b687cb` |
| Pipeline | `dotnet.TorchSharp`, definition id **174** |
| Branch scanned | `refs/heads/main` |
| Helix present | **no** (tests run inline on the AzDO agent; there are no Helix work items) |
| Build Analysis present | **no** (no `Build_Analysis_KnownIssues_v1` attachment; issues are human-facing trackers) |
| Issue model | plain `[ci-scan]` tracking issues for maintainers, not machine-consumed KBEs |
| Issue labels | `bug` (`Build break: ` title prefix for compile-time breaks) |
| Title prefix | `[ci-scan] ` |

`Helix present` and `Build Analysis present` gate two pieces of behavior, flagged inline below as
**(profile: Helix)** and **(profile: Build Analysis)**. When a key is `no`, skip the gated step.

<a id="environment-constraints"></a>
## Environment constraints

These look like permission errors but are physical properties of the sandbox. Do not retry around
them; switch to the working pattern immediately.

- **Pre-bind every URL that contains `?` or `&` to a shell variable on its own line, then
  `curl -s "$url"`.** Inline query strings are rejected as "Permission denied" even when quoted,
  because the tool approver treats `?`/`&` as interactive prompts.

  ```bash
  url='https://dev.azure.com/dotnet/0e144272-85b9-44a0-bd6c-3800a7b687cb/_apis/build/builds?definitions=174&branchName=refs/heads/main&statusFilter=completed&resultFilter=succeeded,failed,partiallySucceeded&%24top=25&api-version=7.1'
  curl -s "$url" | tee /tmp/gh-aw/agent/builds.json | jq -r '.value[0] | "\(.id) \(.result)"'
  ```

- **OData `$top`/`$skip` must be percent-encoded as `%24top` / `%24skip`** inside the URL.
- **No `>` or `-o` redirection.** Use `| tee /path/to/file`.
- **Each bash call is a fresh subshell.** Nothing persists except files under `/tmp/gh-aw/agent/`.
  Write intermediate state there (`mkdir -p /tmp/gh-aw/agent/coverage` first).
- **AzDO REST is anonymous.** Never add auth headers; stay on the `dotnet/0e144272-85b9-44a0-bd6c-3800a7b687cb` host.

<a id="data-sources"></a>
## Data sources

- **AzDO REST** - `https://dev.azure.com/dotnet/0e144272-85b9-44a0-bd6c-3800a7b687cb/_apis/build/...`, anonymous.
  - List builds: `?definitions=174&branchName=refs/heads/main&statusFilter=completed&resultFilter=succeeded,failed,partiallySucceeded&%24top=25&api-version=7.1`. The list is sorted DESC by queue time, so the later-in-wall-clock `follow_up` appears in the array *before* the older `source` build it follows.
  - Timeline: `/builds/{id}/timeline?api-version=7.1` returns a flat `records[]`; reconstruct the `Stage -> Phase -> Job -> Task` tree via `parentId`. A failed record with a non-null `log.id` is a leaf worth reading.
  - Task log: each leaf record exposes `log.url`; `curl -s "$log_url"` returns plain text.
- **Helix REST** **(profile: Helix - not present in this repo, skip).** TorchSharp tests run inline on the AzDO agent, so there are no Helix work items; read the failing line from the AzDO task log directly.
- **Build Analysis attachment** **(profile: Build Analysis - not present in this repo, skip).** TorchSharp has no `Build_Analysis_KnownIssues_v1` attachment; the only dedup path is [Search existing issues](#search-existing).

<a id="source-selection"></a>
## Source build selection and follow-up gate

Pick the build to scan, then prove the failure is still live before filing.

1. **Select `source`.** From the build list, pick the most recent build with
   `result in {failed, partiallySucceeded}` that has at least one COMPLETED build with a strictly
   later `finishTime`. That later build is the `follow_up` anchor. Because the list is sorted DESC
   by queue time, `follow_up` sits in the array *before* `source`.
2. **Skip reasons at selection time:**
   - `source.finishTime` older than 14 days: record `stale build window (>14d)`.
   - No `follow_up` (the source is the absolute latest completed build): record
     `no follow-up build yet, defer to next run`.
   - No qualifying failed build in the last 7 days: record `no failed build in 7d`.
3. **Follow-up gate (run per signature, after classification).** Inspect `follow_up`:
   - `follow_up.result == succeeded`, or it failed/partiallySucceeded WITHOUT this signature:
     record `signature absent from follow-up build #<id>` and emit nothing.
   - `follow_up` contains the signature: proceed to filing.
   - For build breaks, also search merged PRs touching the failing source file (or citing the
     error code) with `merged:>=<source.finishTime>`. On a match record
     `fix already merged after source build` and emit nothing.

<a id="classification"></a>
## Failure classification

Classification decides WHERE to read the signature text from, not whether to file. Save the
canonical failing log to `/tmp/gh-aw/agent/failure.log` before extracting, because the
[match-count gate](#new-issue-template) greps it for the verbatim signature.

```bash
log_url='<console URL from the AzDO task log>'
curl -s "$log_url" | tee /tmp/gh-aw/agent/failure.log | tail -200
```

1. **Build break.** Failing task is a compile/restore/native step (`Build`, `Restore`, `Compile`,
   `Configure CMake`, `Build native`) and the test task is absent or `skipped`. Read the signature
   from the failing compile log: the `CSxxxx` diagnostic, linker error, native compiler error, or
   cmake error line. A build break in a SHARED build/package step (a native build leg or
   `Build_TorchSharp_And_libtorch_cpu_Packages`) blocks every downstream leg, so it may be filed on
   first sight (see the [stability gate](#occurrence-counting)); a compile error confined to a
   single test leg still needs the recurrence count. Prefix the title summary with `Build break: `.
2. **Phase/stage-only failure with no failed job underneath.** A compile break aggregated at the
   phase level (no leaf Job record). Open the Phase log plus the latest log of any non-succeeded
   child Task and treat it as a build break.
3. **Test failure.** Failing task is the test run (`Run Tests`, `dotnet test`, an `xunit` task).
   Locate the first `[FAIL]` / `Failed:` / `Assert.*` line. The signature is the test method FQN
   plus the first line of the assertion/exception message.
4. **Helix-routed test failure** **(profile: Helix - not present, skip).** TorchSharp has no Helix;
   read the failing line from the AzDO task log in case 3.
5. **Dead-lettered Helix work item** **(profile: Helix - not present, skip).**
6. **Infra-shaped job failure.** `Initialize job` failed, agent disconnect, `Pool is offline`,
   queue-capacity timeout, transient network, libtorch download/push/sign failure (`Push_*`,
   `CodeSign_*`, `Build_libtorch_cuda_*` legs). No stable signature: skip emission and record
   `infra noise - no stable signature`.

For each (1)/(2)/(3) signature compute the tuple `(category, leg, signature)` and proceed to
[occurrence counting](#occurrence-counting).

<a id="occurrence-counting"></a>
## Occurrence counting and window widening

Count how many of the last ~10 completed builds of definition 174 contain the signature. Multiple
legs, retries, or work items of the SAME build id count as ONE occurrence, never two.

**Stability gate.** A signature is stable when it appears in `>= 2` distinct builds in the window,
OR it is a build break that fails every leg of the source build (block-everyone severity worth
filing on first sight). A one-off that appears in a single build is not stable: record
`< 2 occurrences and not blocking` and let the next run revisit.

**Window widening.** If the signature appears in *every* sampled build (100% in the ~10-build
window), the true first occurrence likely predates the window. Widen the build list
(`&%24skip=10`, `&%24skip=20`, ...) up to ~40 additional builds and stop as soon as you find a
build where the signature is absent. Report the build immediately after that gap as
`First build it occurred`. If you hit the cap without finding a gap, set `First build it occurred`
to the oldest build scanned and add the note `Persistent across the entire scanned window; true
origin may predate <oldest-build-date>.`

<a id="search-existing"></a>
## Search existing issues

Before filing, search for an already-open issue covering the same signature. This is the primary
dedup mechanism, and since TorchSharp has no Build Analysis it is the ONLY cross-run dedup. Search
the `[ci-scan]` issue space (the issue model named in the [profile](#repo-profile)):

```bash
sig_short='<most distinctive sanitized substring of the signature, <= 80 chars>'
gh issue list --repo dotnet/TorchSharp --state open \
  --search "[ci-scan] $sig_short in:title,body" --json number,title,url | tee /tmp/gh-aw/agent/existing.json
```

The same failure can be recorded in different wordings, so do not conclude "no existing issue" from
one query:

- Search the most distinctive single substring (the assertion stem or `CSxxxx` + symbol), not the
  whole joined line.
- If the first search misses, try a second distinctive substring (for example the test FQN alone).
- Include closed issues in a second pass (`--state all`) when a recent closure may be the right
  tracker; a freshly closed "fixed" issue means do not re-file unless the failure clearly recurs
  after the fix.

On a confirmed match record `existing-issue #<n>` and emit nothing. Verify the candidate actually
matches the same test/family AND the same failing line before trusting it; a coincidental substring
hit is not a match. When two candidates look equally plausible and you cannot disambiguate, record
`existing-issue #<n>` for the closest and note the ambiguity in the tally rather than filing a
duplicate.

<a id="dedup-cache"></a>
## Same-run dedup cache

A single failure surfaces on many legs of the same build, so dedupe on the signature alone, NOT on
`leg|signature` (that would file one issue per leg). Cache filed signatures in
`/tmp/gh-aw/agent/filed.tsv` as `<key>\t<aw_id>` where `key = <signature_norm>` and
`<signature_norm>` is the signature with tab/CR/newline stripped (raw signatures are copied
verbatim from logs and may carry whitespace that would corrupt the TSV).

```bash
signature_norm=$(printf '%s' "<signature>" | tr -d '\t\n\r')
test -f /tmp/gh-aw/agent/filed.tsv && cut -f1 /tmp/gh-aw/agent/filed.tsv | grep -Fxq -- "$signature_norm"   # dup if exit 0
printf '%s\t%s\n' "$signature_norm" "aw_<id>" >> /tmp/gh-aw/agent/filed.tsv                                   # append after every emit
```

On a cache hit record `dup of filed-issue #aw_<id> earlier in this run` and stop for that
signature. Append the key after every issue emission. Run the [specificity](#signature-specificity)
check before appending: a signature too generic to be specific must be rejected up front, never
cached, because a broad key collapses unrelated failures into one issue.

<a id="signature-specificity"></a>
## Signature specificity

A signature must be specific enough that a maintainer searching the tracker matches the real
failure and nothing else. Prefer the most distinctive stable substring of the failing line:

- Keep the test method FQN plus the assertion/exception stem.
- Keep `CSxxxx` / linker / cmake error codes and the offending symbol.
- Strip everything volatile per [Sanitization](#sanitization): absolute paths, GUIDs, machine
  names, timestamps, ports, PIDs, durations, and run-specific numeric ids.
- Never use a bare exit code, a generic exception type with no message, or a phrase that also
  appears on `[PASS]` / `[SKIP]` lines of the same log.

<a id="bad-vs-good"></a>
## Bad vs good signatures

| Failing line | Bad signature (too broad) | Good signature (specific + stable) |
|---|---|---|
| `[FAIL] TorchSharp.Test.TensorTests.TestArithmetic : Assert.Equal() Failure: Expected 0.81, got 0.79` | `Assert.Equal() Failure` | `TensorTests.TestArithmetic : Assert.Equal() Failure: Expected 0.81` |
| `D:\a\1\s\src\Foo.cs(120,5): error CS0246: The type or namespace name 'Bar' could not be found` | `error CS0246` | `Foo.cs: error CS0246: The type or namespace name 'Bar' could not be found` |
| `Process terminated. Exit code 134. Stack: ...` on one leg only | `Exit code 134` | `Process terminated. Exit code 134.` only if the same stack frame recurs; otherwise treat as infra noise |
| `Test run failed` summary line | `Test run failed` | (reject - find the underlying `[FAIL]` line instead) |

<a id="sanitization"></a>
## Sanitization

Sanitize every log excerpt before embedding it in an issue body. Replace, do not delete, so the
line stays readable:

- Absolute paths -> repo-relative (`D:\a\1\s\src\Foo.cs` -> `src/Foo.cs`).
- GUIDs, build numbers -> `<id>`.
- Machine/agent names, IP addresses, ports -> `<host>` / `<port>`.
- Timestamps, durations, PIDs -> `<time>` / `<pid>`.
- Keep the diagnostic code, symbol names, assertion text, and `[FAIL]` marker verbatim - those are
  the signature.

<a id="new-issue-template"></a>
## New-issue template

Emit one `create-issue` per stable signature when every gate passes and the per-run cap allows.

**Match-count gate.** Before emitting, confirm the literal match block is a verbatim substring of
`/tmp/gh-aw/agent/failure.log`:

```bash
grep -F -c -- "<literal match substring>" /tmp/gh-aw/agent/failure.log   # must be >= 1
```

If the count is `0`, do not emit; record `signature did not match failure.log (N=0)` instead.

````markdown
## Signature

`<one-line normalized failure>`

## Failing line (raw)

```
<one [FAIL] or compile-error line, sanitized>
```

## Match signature (literal substring)

```
<exact substring a maintainer search should match on - the literal verified by the match-count gate>
```

## Category

<Build break | Test failure>

## Affected legs (in the source build)

- `<leg display name>` (task log: `<url>`)
- ...

## First build it occurred

- Build: `<azdo build url>`
- Finished: `<UTC timestamp>`
- Commit: `<sha>`
- Occurrences in the scanned window: `<n>`
- Computed within the scanned window; may not be the true origin.

## Reasoning

<why this is a real failure and not flake; cite the source line and the occurrence count>

---

Filed by [`ci-scan`](https://github.com/dotnet/TorchSharp/blob/main/.github/workflows/ci-scan.agent.md),
which scans the `dotnet.TorchSharp` pipeline (definition 174) on `main` and converts recurring failures into
`[ci-scan]` tracking issues. Comment here to flag a false positive or add context;
[`ci-scan-feedback`](https://github.com/dotnet/TorchSharp/blob/main/.github/workflows/ci-scan-feedback.agent.md)
reads that feedback daily and proposes edits to the scanner prompt.
````

**(profile: Build Analysis - not present in this repo).** Keep the `## Match signature (literal substring)`
heading; the block is the literal a maintainer search keys on, not a Build Analysis match. Apply the `bug`
label, and for build breaks prefix the title summary with `Build break: ` (final title
`[ci-scan] Build break: <signature>`) so the feedback build-break signal can detect it.

<a id="skip-reasons"></a>
## Recognized skip-reason vocabulary

Every skipped signature MUST carry a reason, and the reason SHOULD reuse one of these phrasings so
[`ci-scan-feedback`](../ci-scan-feedback.agent.md) can aggregate the tally stably. The list is not
exhaustive; new reasons should follow the same short, lower-case shape.

- `cap reached`
- `< 2 occurrences and not blocking`
- `infra noise - no stable signature`
- `signature absent from follow-up build #<id>`
- `stale build window (>14d)`
- `no follow-up build yet, defer to next run`
- `no failed build in 7d`
- `fix already merged after source build`
- `dup of filed-issue #aw_<id> earlier in this run`
- `existing-issue #<n>`
- `suspected infra outage`
- `signature did not match failure.log (N=<count>)`
- `weak signature`

<a id="rubric"></a>
## Rubric

[`ci-scan-feedback`](../ci-scan-feedback.agent.md) scores each `[ci-scan]` issue against these
criteria. A failing criterion is a candidate signal for a prompt edit.

- **Title scoped to a single failure shape** - a test FQN + assertion stem or a single compile
  error, not a list of legs.
- **Classification matches the failure** - a real test/build break carries the `bug` label and
  build breaks use the `Build break: ` title prefix; infra noise should never have been filed at all.
- **Match block is specific** - the literal-substring block is a stable substring of the real
  failing line per [Signature specificity](#signature-specificity), not a bare method name,
  generic exception, exit code, or a phrase shared with `[PASS]`/`[SKIP]` lines.
- **Occurrence count is honest** - the figure is consistent with the cited build and the
  [occurrence rules](#occurrence-counting).
- **Follow-up gate respected** - the issue cites a real failing build and was not filed for a
  signature already absent from the follow-up build.

<a id="output-discipline"></a>
## Output discipline

- Each definition gets exactly one walk-through per run. Do not revisit.
- One signature = one outcome line in `/tmp/gh-aw/agent/coverage/<pipeline>.txt`.
- No meta / aggregate / outage issues. Every issue is keyed to a single `(category, signature)`.
- Do not add `area-*` labels; area triage is owned elsewhere.
- Do not pile comments on an existing `[ci-scan]` tracker; one issue per signature is the model.
- Do not propose alternative workflow designs in issue bodies. To change the scanner, edit the
  prompt or comment on a filed issue so the feedback workflow can pick it up.
- The final agent log MUST include the Step 7 summary table.
