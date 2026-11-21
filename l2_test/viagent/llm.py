"""LLM client wrapper (OpenAI-compatible API)."""
from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from openai import OpenAI


class LLMClient:
    def __init__(self, model: str | None = None, temperature: float = 0.0):
        self.api_key = os.environ.get("API_KEY")
        self.base_url = os.environ.get("BASE_URL")
        self.model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def complete(self, system: str, user: str, max_tokens: int = 16000,
                 timeout: float = 300.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def complete_json(self, system: str, user: str, max_tokens: int = 16000,
                      retries: int = 2, timeout: float = 300.0) -> dict:
        """Request a JSON object, with robust extraction and retries."""
        # deepseek json mode: needs 'json' in prompt
        user_prompt = user + "\n\nRespond ONLY with a single valid JSON object."
        last_err = None
        for attempt in range(retries + 1):
            content = ""
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
            if content.strip():
                try:
                    return _extract_json(content)
                except ValueError as e:
                    last_err = e
            # if json mode returned empty/truncated, retry with plain mode
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
            if content.strip():
                try:
                    return _extract_json(content)
                except ValueError as e:
                    last_err = e
            # retry with a hint about the failure
            hint = "Your previous response was not valid JSON. Return ONLY valid JSON."
            if attempt < retries:
                user_prompt = user + f"\n\n{hint}\nRespond ONLY with a single valid JSON object."
        raise RuntimeError(f"LLM JSON failed after {retries+1} attempts: {last_err}")


def _repair_json(text: str) -> str:
    """Best-effort repairs for common LLM JSON mistakes."""
    # remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # collapse literal newlines inside strings (common with 'note' fields)
    text = re.sub(r"(\"[^\"]*?)\n([^\"]*?\")", r"\1 \2", text, flags=re.DOTALL)
    return text


def _extract_json(text: str) -> dict:
    text = text.strip()
    # strip code fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    candidates = []
    # full text
    candidates.append(text)
    # balanced braces block (outermost)
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    for cand in candidates:
        for repaired in (cand, _repair_json(cand)):
            try:
                obj = json.loads(repaired)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    # tolerant fallback: extract key fields from truncated output
    partial = _extract_partial(text)
    if partial:
        return partial
    raise ValueError(f"could not parse JSON from LLM output: {text[:600]}")


def _extract_partial(text: str) -> dict | None:
    """Extract top-level fields from a possibly truncated JSON blob.

    Uses regex for scalars and a light parser for secure_lines / rejected_lines.
    Returns a dict with whatever was recoverable, or None if nothing usable.
    """
    out = {}
    m = re.search(r'"spectype"\s*:\s*"([^"]+)"', text)
    if m:
        out["spectype"] = m.group(1)
    for key, cast in (("redshift", float), ("redshift_err", float), ("vi_quality", float)):
        m = re.search(rf'"{key}"\s*:\s*(-?[\d.]+)', text)
        if m:
            try:
                out[key] = cast(m.group(1))
            except ValueError:
                pass

    def parse_line_entries(key):
        entries = []
        # find the array block for this key
        m = re.search(rf'"{key}"\s*:\s*\[(.*)', text, re.DOTALL)
        if not m:
            return entries
        body = m.group(1)
        # split on top-level '{'...'}'
        for block in re.findall(r"\{[^{}]*\}", body):
            e = {}
            for k, cast in (("line", str), ("kind", str), ("w_obs", float),
                            ("rest", float), ("sig", float), ("z_from_line", float),
                            ("note", str), ("reason", str)):
                mm = re.search(rf'"{k}"\s*:\s*("([^"]*)"|(-?[\d.]+))', block)
                if mm:
                    val = mm.group(2) if mm.group(2) is not None else mm.group(3)
                    if k in ("w_obs", "rest", "sig", "z_from_line"):
                        try:
                            e[k] = float(val)
                        except ValueError:
                            pass
                    else:
                        e[k] = val
            if "line" in e:
                entries.append(e)
        return entries

    out["secure_lines"] = parse_line_entries("secure_lines")
    out["rejected_lines"] = parse_line_entries("rejected_lines")
    m = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if m:
        out["reasoning"] = m.group(1)
    if "spectype" in out or "redshift" in out or out.get("secure_lines"):
        return out
    return None
