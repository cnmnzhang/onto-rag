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

DDX_PROMPT_TEMPLATE = """{rag_section}
Patient Case:
{chart_text}

Provide a structured differential diagnosis in exactly this format:

PRIMARY DIAGNOSIS: [diagnosis name]
Supporting evidence: [2-3 sentences citing specific findings from the case]
Ontology grounding: [ONLY if RAG context provided above: name the specific ontology concept(s) that support this diagnosis, e.g. "Per ontology: Sm antibody fulfills a diagnostic criterion for SLE"]
Confidence: [High / Medium / Low]

DIFFERENTIAL DIAGNOSIS:
1. [Diagnosis] ([probability]%) — [what supports it] | [what argues against it]
2. [Diagnosis] ([probability]%) — [what supports it] | [what argues against it]
3. [Diagnosis] ([probability]%) — [what supports it] | [what argues against it]

KEY DISTINGUISHING FEATURES: [what test or finding would most help confirm or exclude the primary diagnosis]

REASONING CHAIN:
- Finding → [specific finding from case]
- Ontology class → [relevant ontology category, e.g. "Inflammatory Polyarthritis"]
- Supports → [diagnosis name] because [one sentence]"""

RAG_SECTION_TEMPLATE = """The following structured ontology knowledge may be relevant to this case:

{context}

Use the above ontology information where relevant to inform your reasoning.

"""


def _hash_key(chart_text: str, rag_context: str | None, model: str) -> str:
    payload = f"{model}||{rag_context or ''}||{chart_text}"
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
    # Deterministic placeholder for testing pipeline without API keys
    return (
        "PRIMARY DIAGNOSIS: [DRY RUN - No API key detected]\n"
        "Supporting evidence: This is a placeholder response. Set ANTHROPIC_API_KEY, "
        "OPENAI_API_KEY, or GEMINI_API_KEY to get real responses.\n"
        "Confidence: N/A\n\n"
        "DIFFERENTIAL DIAGNOSIS:\n"
        "1. Placeholder diagnosis A (33%) — No real analysis performed.\n"
        "2. Placeholder diagnosis B (33%) — No real analysis performed.\n"
        "3. Placeholder diagnosis C (34%) — No real analysis performed.\n\n"
        "KEY DISTINGUISHING FEATURES: Set an API key to enable real LLM responses."
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

    def generate(self, chart_text: str, rag_context: str | None = None) -> str:
        """Generate a DDx response, using cache if available."""
        cache_key = _hash_key(chart_text, rag_context, self.model)
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # Build prompt
        rag_section = ""
        if rag_context and rag_context.strip():
            rag_section = RAG_SECTION_TEMPLATE.format(context=rag_context.strip())

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
