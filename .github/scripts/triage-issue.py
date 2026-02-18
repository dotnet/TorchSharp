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
import time
import urllib.error
import urllib.request

GITHUB_API = "https://api.github.com"
INFERENCE_API = "https://models.github.ai/inference"
MODEL = "gpt-4o-mini"

TRIAGE_LABELS = {"bug", "Missing Feature", "question"}

SYSTEM_PROMPT = """\
You are an issue triage bot for TorchSharp, a .NET binding for PyTorch.

Classify the following GitHub issue into exactly ONE of these categories:
- bug: Something is broken, crashes, throws an unexpected error, or produces wrong results.
- Missing Feature: A PyTorch API or feature that is not yet available in TorchSharp.
- question: The user is asking for help, guidance, or clarification on how to use TorchSharp.

Respond with ONLY a JSON object in this exact format, no other text:
{"label": "<one of: bug, Missing Feature, question>", "reason": "<one sentence explanation>"}
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
    # Limit length to avoid excessively long comments.
    reason = reason[:200]

    # Strip markdown links: [text](url) -> text
    reason = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", reason)

    # Strip markdown images entirely: ![alt](url) -> ""
    reason = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", "", reason)

    # Remove fenced code blocks with triple backticks to prevent block injection.
    reason = re.sub(r"```.*?```", "", reason, flags=re.DOTALL)

    # Remove any remaining standalone backticks used for inline code.
    reason = reason.replace("`", "")

    # Strip simple HTML tags such as <script>, <b>, etc.
    reason = re.sub(r"<[^>]+>", "", reason)

    # Escape markdown special characters so the text is rendered literally.
    reason = re.sub(r"([\\*_{}\[\]()>#+\-!])", r"\\\1", reason)

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

    # Retry with exponential backoff
    max_retries = 3
    result = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            f"{INFERENCE_API}/chat/completions", data=data, method="POST"
        )
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"::warning::LLM API attempt {attempt + 1} failed ({e}); retrying in {wait}s")
                time.sleep(wait)
            else:
                error_detail = ""
                if isinstance(e, urllib.error.HTTPError) and e.fp:
                    error_detail = e.read().decode(errors="replace")
                raise RuntimeError(f"LLM API call failed after {max_retries} attempts: {error_detail or e}") from e

    # Validate response structure
    if not isinstance(result, dict):
        raise RuntimeError("LLM API returned unexpected response format: top-level JSON is not an object.")

    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM API returned unexpected response format: missing or empty 'choices' array.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError("LLM API returned unexpected response format: first choice is not an object.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("LLM API returned unexpected response format: missing or invalid 'message' in first choice.")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM API returned unexpected response format: missing or invalid 'content' in message.")

    content = content.strip()

    # Parse the JSON response, stripping markdown fences if present
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        content = json_match.group(0)

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"::warning::Failed to parse LLM JSON response ({e}); defaulting to 'question'")
        return "question", "Could not parse LLM response; defaulting to 'question'."

    if not isinstance(parsed, dict):
        print("::warning::LLM JSON content is not an object, defaulting to 'question'")
        return "question", "Could not determine the issue type."

    label = parsed.get("label")
    if not isinstance(label, str) or not label:
        print("::warning::LLM response missing 'label', defaulting to 'question'")
        label = "question"

    reason_raw = parsed.get("reason", "")
    reason = sanitize_reason(reason_raw if isinstance(reason_raw, str) else "")

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
