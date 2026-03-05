#!/usr/bin/env python3
"""src/llm_client.py

Multi-backend LLM client with response caching.

Supported backends (detected from env vars):
  - anthropic   : ANTHROPIC_API_KEY
  - openai      : OPENAI_API_KEY
  - gemini      : GEMINI_API_KEY
  - dry_run     : fallback, returns placeholder text

Responses are cached to data/llm_cache.json to avoid re-running.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """You are an expert rheumatologist providing differential diagnosis consultation. 
You have deep knowledge of rheumatic diseases, their clinical presentations, laboratory findings, 
imaging features, and pathognomonic signs. Provide clinically rigorous, evidence-based reasoning."""

# ── Explainability task ────────────────────────────────────────────────────
# Goal: grounded, traceable reasoning. Each finding cited with its diagnostic why.
DDX_PROMPT_TEMPLATE = """{rag_section}Patient Case:
{chart_text}

Provide a structured differential diagnosis in exactly this format:

PRIMARY DIAGNOSIS: [diagnosis name]
Confidence: [High / Medium / Low]
Supporting findings: [2-3 specific findings from the case that most strongly support this]
Findings against: [1-2 findings that argue against this diagnosis, or "None identified"]

DIFFERENTIAL DIAGNOSIS:
1. [Diagnosis] ([probability]%) — for: [specific finding] | against: [specific finding]
2. [Diagnosis] ([probability]%) — for: [specific finding] | against: [specific finding]
3. [Diagnosis] ([probability]%) — for: [specific finding] | against: [specific finding]

REASONING:
[3-5 sentences. Walk through the clinical logic: which findings narrow the differential,
which are most discriminating, and why the primary diagnosis ranks above the alternatives.
Be specific — cite lab values, timecourses, and examination findings by name.]

NEXT STEP: [The single investigation or result that would most change or confirm this
diagnosis, and what you expect it to show.]"""

# ── Teaching task ──────────────────────────────────────────────────────────
# Goal: pedagogical completeness. Covers WHAT each finding is, WHY it matters, HOW to assess.
TEACHING_PROMPT_TEMPLATE = """{rag_section}Patient Case:
{chart_text}

You are a rheumatology attending teaching a medical student about this case.

Identify the most likely diagnosis and teach the student about this presentation.
For each key clinical finding in the vignette:
  1. WHAT it is — define it clearly for a student
  2. WHY it matters — what diseases does it point toward or away from, and why
  3. HOW to assess it — how a clinician evaluates this in practice

Then briefly note one or two findings that could suggest an alternative diagnosis,
and explain why you favour your primary diagnosis over that alternative.

Structure your response as:

PRIMARY DIAGNOSIS: [diagnosis name]

TEACHING POINTS:
Finding: [finding name]
  What: [definition/description]
  Why it matters: [diagnostic significance]
  How to assess: [clinical assessment method]
[Repeat for each key finding]

DIFFERENTIAL TEACHING:
[1-2 paragraphs on the main alternative diagnosis and the distinguishing features]"""

RAG_SECTION_TEMPLATE = """Relevant clinical knowledge for this case:

{context}

"""


def _hash_key(chart_text: str, rag_context: str | None, model: str, task: str = "explainability") -> str:
    payload = f"{model}||{task}||{rag_context or ''}||{chart_text}"
    return hashlib.md5(payload.encode()).hexdigest()


class LLMCache:
    def __init__(self, path: str | Path = "data/llm_cache.json") -> None:
        self.path = Path(path)
        self._data: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")


def _detect_backend() -> str:
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    return "dry_run"


def _call_anthropic(prompt: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _call_openai(prompt: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.1,
    )
    return resp.choices[0].message.content or ""


def _call_gemini(prompt: str, model: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    m = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT,
    )
    resp = m.generate_content(prompt)
    return resp.text


def _call_dry_run(prompt: str, _model: str) -> str:
    return (
        "PRIMARY DIAGNOSIS: [DRY RUN - No API key detected]\n"
        "Confidence: N/A\n"
        "Supporting findings: Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY "
        "to get real responses.\n"
        "Findings against: N/A\n\n"
        "DIFFERENTIAL DIAGNOSIS:\n"
        "1. Placeholder diagnosis A (33%) — for: N/A | against: N/A\n"
        "2. Placeholder diagnosis B (33%) — for: N/A | against: N/A\n"
        "3. Placeholder diagnosis C (34%) — for: N/A | against: N/A\n\n"
        "REASONING: No real analysis performed. Set an API key to enable real LLM responses.\n\n"
        "NEXT STEP: Set an API key to enable real LLM responses."
    )

class LLMClient:
    """
    Multi-backend LLM client with caching.
    
    Auto-detects backend from environment variables.
    Recommended backends by cost/quality:
      - anthropic: claude-3-5-haiku-20241022 (fast, cheap, good)
      - gemini:    gemini-1.5-flash (free tier available)
      - openai:    gpt-4o-mini (cheap)
    """

    DEFAULT_MODELS = {
        "anthropic": "claude-haiku-4-5-20251001",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "dry_run": "dry_run",
    }

    def __init__(
        self,
        cache_path: str | Path = "data/llm_cache.json",
        backend: str | None = None,
        model: str | None = None,
        retry_delay: float = 2.0,
        max_retries: int = 3,
    ) -> None:
        self.backend = backend or _detect_backend()
        self.model = model or self.DEFAULT_MODELS.get(self.backend, "dry_run")
        self.cache = LLMCache(cache_path)
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        print(f"  [LLM] Backend: {self.backend} | Model: {self.model}")

    def generate(self, chart_text: str, rag_context: str | None = None, task: str = "explainability") -> str:
        """Generate a response, using cache if available.

        task: "explainability" (structured DDx) or "teaching" (pedagogical walkthrough)
        """
        cache_key = _hash_key(chart_text, rag_context, self.model, task)
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # Build RAG section (shared across both prompts)
        rag_section = ""
        if rag_context and rag_context.strip():
            rag_section = RAG_SECTION_TEMPLATE.format(context=rag_context.strip())

        # Select prompt template by task
        if task == "teaching":
            prompt = TEACHING_PROMPT_TEMPLATE.format(
                rag_section=rag_section,
                chart_text=chart_text.strip(),
            )
        else:
            prompt = DDX_PROMPT_TEMPLATE.format(
                rag_section=rag_section,
                chart_text=chart_text.strip(),
            )

        # Call with retry
        response = ""
        for attempt in range(self.max_retries):
            try:
                if self.backend == "anthropic":
                    response = _call_anthropic(prompt, self.model)
                elif self.backend == "openai":
                    response = _call_openai(prompt, self.model)
                elif self.backend == "gemini":
                    response = _call_gemini(prompt, self.model)
                else:
                    response = _call_dry_run(prompt, self.model)
                break
            except Exception as e:
                print(f"  [LLM] Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    response = f"[ERROR: {e}]"

        self.cache.set(cache_key, response)
        return response

    def backend_info(self) -> dict[str, str]:
        return {"backend": self.backend, "model": self.model}