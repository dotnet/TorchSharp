"""
Issue triage bot for TorchSharp.

Classifies new GitHub issues using an LLM and applies the appropriate label.
Posts a polite comment acknowledging the issue.
Skips issues that already have triage labels (manually set by maintainers).
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request

GITHUB_API = "https://api.github.com"
INFERENCE_API = "https://models.inference.ai.azure.com"
MODEL = "gpt-4o-mini"

TRIAGE_LABELS = {"bug", "Missing Feature", "question", "enhancement", "breaking-change"}

SYSTEM_PROMPT = """\
You are an issue triage bot for TorchSharp, a .NET binding for PyTorch.

Classify the following GitHub issue into exactly ONE of these categories:
- bug: Something is broken, crashes, throws an unexpected error, or produces wrong results.
- Missing Feature: A PyTorch API or feature that is not yet available in TorchSharp.
- question: The user is asking for help, guidance, or clarification on how to use TorchSharp.
- enhancement: A suggestion to improve existing functionality (not a missing PyTorch API).
- breaking-change: The issue reports or requests a change that would break existing public API.

Respond with ONLY a JSON object in this exact format, no other text:
{"label": "<one of: bug, Missing Feature, question, enhancement, breaking-change>", "reason": "<one sentence explanation>"}
"""

COMMENT_TEMPLATES = {
    "bug": (
        "Thank you for reporting this issue! 🙏\n\n"
        "I've triaged this as a **bug**. {reason}\n\n"
        "A maintainer will review this soon. In the meantime, please make sure you've "
        "included a minimal code sample to reproduce the issue and the TorchSharp version you're using.\n\n"
        "*This comment was generated automatically by the issue triage bot.*"
    ),
    "Missing Feature": (
        "Thank you for opening this issue! 🙏\n\n"
        "I've triaged this as a **missing feature** request. {reason}\n\n"
        "If you haven't already, it would be very helpful to include a link to the "
        "corresponding PyTorch documentation and a Python code example.\n\n"
        "*This comment was generated automatically by the issue triage bot.*"
    ),
    "question": (
        "Thank you for reaching out! 🙏\n\n"
        "I've triaged this as a **question**. {reason}\n\n"
        "A maintainer or community member will try to help as soon as possible. "
        "Please make sure to include the TorchSharp version and a code sample for context.\n\n"
        "*This comment was generated automatically by the issue triage bot.*"
    ),
    "enhancement": (
        "Thank you for the suggestion! 🙏\n\n"
        "I've triaged this as an **enhancement** request. {reason}\n\n"
        "A maintainer will review this when they get a chance.\n\n"
        "*This comment was generated automatically by the issue triage bot.*"
    ),
    "breaking-change": (
        "Thank you for reporting this! 🙏\n\n"
        "I've triaged this as a potential **breaking change**. {reason}\n\n"
        "A maintainer will review this carefully.\n\n"
        "*This comment was generated automatically by the issue triage bot.*"
    ),
}


def github_request(method, path, body=None):
    """Make an authenticated request to the GitHub API."""
    token = os.environ["GITHUB_TOKEN"]
    url = f"{GITHUB_API}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode(errors="replace") if e.fp else ""
        raise RuntimeError(f"GitHub API {method} {path} failed ({e.code}): {error_body}") from e


def sanitize_reason(reason):
    """Sanitize LLM-generated reason to prevent markdown injection."""
    reason = reason[:200]
    reason = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", reason)  # Strip links
    reason = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", "", reason)  # Strip images
    return reason.strip()


def classify_issue(title, body):
    """Call the LLM to classify the issue."""
    token = os.environ["GITHUB_TOKEN"]
    user_message = f"Issue title: {title}\n\nIssue body:\n{body or '(empty)'}"
    max_length = 4000
    if len(user_message) > max_length:
        user_message = user_message[:max_length] + "\n\n[Truncated]"

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.0,
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{INFERENCE_API}/chat/completions", data=data, method="POST"
    )
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode(errors="replace") if e.fp else ""
        raise RuntimeError(f"LLM API call failed ({e.code}): {error_body}") from e

    content = result["choices"][0]["message"]["content"].strip()

    # Parse the JSON response, stripping markdown fences if present
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        content = json_match.group(0)

    parsed = json.loads(content)
    label = parsed["label"]
    reason = sanitize_reason(parsed.get("reason", ""))

    if label not in TRIAGE_LABELS:
        print(f"::warning::LLM returned unknown label '{label}', defaulting to 'question'")
        label = "question"
        reason = reason or "Could not determine the issue type."

    return label, reason


def main():
    required_vars = ["GITHUB_TOKEN", "REPO", "ISSUE_NUMBER"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    repo = os.environ["REPO"]
    issue_number = os.environ["ISSUE_NUMBER"]

    # Fetch the issue
    issue = github_request("GET", f"/repos/{repo}/issues/{issue_number}")
    existing_labels = {lbl["name"] for lbl in issue.get("labels", [])}

    # Skip if the issue already has a triage label (manually set by maintainer)
    overlap = existing_labels & TRIAGE_LABELS
    if overlap:
        print(f"Issue #{issue_number} already has triage label(s): {overlap}. Skipping.")
        return

    title = issue.get("title", "")
    body = issue.get("body", "")

    print(f"Classifying issue #{issue_number}: {title}")
    label, reason = classify_issue(title, body)
    print(f"Classification: {label} — {reason}")

    # Add the label
    github_request("POST", f"/repos/{repo}/issues/{issue_number}/labels", {"labels": [label]})
    print(f"Added label '{label}' to issue #{issue_number}")

    # Post a comment
    comment_body = COMMENT_TEMPLATES[label].format(reason=reason)
    github_request("POST", f"/repos/{repo}/issues/{issue_number}/comments", {"body": comment_body})
    print(f"Posted triage comment on issue #{issue_number}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"::error::Triage failed: {e}")
        sys.exit(1)
