"""
LLM Interface for Disease Differential Diagnosis
Supports OpenAI, Hugging Face (Qwen2.5-1.5B-Instruct), Google Gemini, and dry-run modes
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, TYPE_CHECKING

from .label_alias import LabelNormalizer

if TYPE_CHECKING:
    from .onto_config import OntologyConfig


class LLMInterface:
    """
    Unified interface for LLM predictions with multiple backend support.

    Backends:
    - Google Gemini (gemini-3-flash-preview) - FREE with API key
    - OpenAI (gpt-3.5-turbo)
    - Hugging Face (Qwen2.5-1.5B-Instruct)
    - Dry-run (deterministic heuristics)
    """

    def __init__(
        self,
        allowed_labels: List[str],
        cache_file: str = "llm_cache.json",
        config: Optional["OntologyConfig"] = None
    ):
        self.none_label = "NONE"
        self.allowed_labels = set(allowed_labels + [self.none_label])
        self._allowed_uris = sorted([l for l in self.allowed_labels if l != self.none_label])
        self.label_normalizer = LabelNormalizer(self._allowed_uris, none_label=self.none_label)
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.config = config

        # Determine backend
        self.backend = self._detect_backend()
        self._initialize_backend()

    def _detect_backend(self) -> str:
        """Detect which LLM backend to use."""
        google_key = os.getenv("GOOGLE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        use_hf = os.getenv("USE_HUGGINGFACE", "").lower() in ["true", "1", "yes"]

        # Priority: Gemini (free) > HF (free local) > OpenAI (paid) > dry-run
        if google_key and not use_hf:
            return "gemini"
        elif use_hf:
            return "huggingface"
        elif openai_key:
            return "openai"
        else:
            return "dry_run"

    def _initialize_backend(self):
        """Initialize the selected backend."""
        if self.backend == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
                self.client = genai.GenerativeModel(model_name)
                print(f"✓ Google Gemini initialized ({model_name})")
            except ImportError:
                print("✗ google-generativeai package not installed, falling back to dry-run mode")
                print("  Install with: pip install google-generativeai")
                self.backend = "dry_run"
            except Exception as e:
                print(f"✗ Error initializing Gemini: {e}")
                print("  Falling back to dry-run mode")
                self.backend = "dry_run"

        elif self.backend == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                print("✓ OpenAI client initialized")
            except ImportError:
                print("✗ OpenAI package not installed, falling back to dry-run mode")
                self.backend = "dry_run"

        elif self.backend == "huggingface":
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM

                model_name = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
                print(f"Loading Hugging Face model: {model_name}...")

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )

                # Move to GPU if available
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                if self.device == "cpu" and hasattr(self.model, "to"):
                    self.model = self.model.to(self.device)

                print(f"✓ Hugging Face model loaded on {self.device}")

            except ImportError as e:
                print(f"✗ Hugging Face packages not installed: {e}")
                print("  Falling back to dry-run mode")
                self.backend = "dry_run"
            except Exception as e:
                print(f"✗ Error loading Hugging Face model: {e}")
                print("  Falling back to dry-run mode")
                self.backend = "dry_run"

        if self.backend == "dry_run":
            print("✓ Running in DRY-RUN mode (using deterministic keyword heuristics)")

    def _load_cache(self) -> Dict:
        """Load cached responses from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        """Save cache to disk."""
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _hash_prompt(self, prompt: str) -> str:
        """Create hash key for prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def _dry_run_predict(self, chart_text: str, rag_context: Optional[str] = None) -> Dict:
        """Deterministic heuristic for dry-run mode."""
        text_lower = chart_text.lower()
        disease_labels = [l for l in self.allowed_labels if l != self.none_label]
        disease_labels_list = list(disease_labels)

        # Use config keywords if available, otherwise use generic fallback
        if self.config:
            positive_keywords = [kw.lower() for kw in self.config.positive_keywords]
            negative_keywords = [kw.lower() for kw in self.config.negative_keywords]
            disease_name = self.config.disease_name
        else:
            # Generic fallback keywords
            positive_keywords = ["malignant", "carcinoma", "cancer", "tumor", "suspicious"]
            negative_keywords = ["normal", "benign", "no abnormalities"]
            disease_name = "disease"

        # Heuristic rules
        # Check for strong positive indicators
        strong_positive = any(kw in text_lower for kw in positive_keywords[:3])  # Top 3 positive keywords
        if strong_positive:
            return {
                "predicted_label": disease_labels_list[0] if disease_labels_list else self.none_label,
                "ddx_top3": [
                    {
                        "label": lbl,
                        "rationale": f"Heuristic candidate based on {disease_name} keywords",
                    }
                    for lbl in (disease_labels_list[:3] if disease_labels_list else [self.none_label])
                ],
                "evidence": [],
            }

        # Check for moderate positive indicators
        moderate_positive = any(kw in text_lower for kw in positive_keywords[3:])
        if moderate_positive:
            pred_label = disease_labels_list[1] if len(disease_labels_list) > 1 else (disease_labels_list[0] if disease_labels_list else "NONE")
            return {
                "predicted_label": pred_label,
                "ddx_top3": [
                    {
                        "label": lbl,
                        "rationale": f"Heuristic candidate based on possible {disease_name} indicators",
                    }
                    for lbl in (disease_labels_list[:3] if disease_labels_list else [self.none_label])
                ],
                "evidence": [],
            }

        # Check for negative indicators
        if any(kw in text_lower for kw in negative_keywords):
            return {
                "predicted_label": self.none_label,
                "ddx_top3": [{"label": self.none_label, "rationale": "Heuristic: normal/benign findings"}],
                "evidence": [],
            }

        # Default to NONE if no clear indicators
        return {
            "predicted_label": self.none_label,
            "ddx_top3": [{"label": self.none_label, "rationale": f"Heuristic: no clear {disease_name} indicators"}],
            "evidence": [],
        }

    def _error_response(self, message: str) -> Dict:
        return {
            "predicted_label": self.none_label,
            "ddx_top3": [{"label": self.none_label, "rationale": message[:200]}],
            "evidence": [],
        }

    def _predict_gemini(self, system_prompt: str, user_prompt: str) -> Dict:
        """Generate prediction using Google Gemini."""
        try:
            # Gemini doesn't have separate system/user roles in the same way
            # Combine them into a single prompt
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
            response = self.client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": 200,
                }
            )
            response_text = response.text

            # Extract JSON if wrapped in code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            # Find JSON object in response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                response_text = response_text[start_idx:end_idx]

            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"JSON parse error from Gemini: {e}")
            print(f"Raw response: {response_text[:200]}")
            return self._error_response(f"Parse error: {response_text[:100]}")
        except Exception as e:
            print(f"Gemini error: {e}")
            return self._error_response(f"Error: {str(e)}")

    def _predict_openai(self, system_prompt: str, user_prompt: str) -> Dict:
        """Generate prediction using OpenAI."""
        try:
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=200
            )
            response_text = completion.choices[0].message.content
            return json.loads(response_text)
        except Exception as e:
            print(f"OpenAI error: {e}")
            return self._error_response(f"Error: {str(e)}")

    def _predict_huggingface(self, system_prompt: str, user_prompt: str) -> Dict:
        """Generate prediction using Hugging Face model."""
        try:
            import torch

            temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
            top_p = float(os.getenv("LLM_TOP_P", "1.0"))
            do_sample = os.getenv("LLM_DO_SAMPLE", "false").lower() in ["1", "true", "yes"]

            # Construct prompt for chat model
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            generation_kwargs = {
                "max_new_tokens": 220,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                )

            # Decode
            response_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Try to parse JSON from response
            response_text = response_text.strip()

            # Extract JSON if wrapped in code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            # Find JSON object in response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                response_text = response_text[start_idx:end_idx]

            return json.loads(response_text)

        except json.JSONDecodeError as e:
            print(f"JSON parse error from HF model: {e}")
            print(f"Raw response: {response_text[:200]}")
            return self._error_response(f"Parse error: {response_text[:100]}")
        except Exception as e:
            print(f"Hugging Face error: {e}")
            return self._error_response(f"Error: {str(e)}")

    def predict(self, chart_text: str, rag_context: Optional[str] = None) -> Dict:
        """Generate prediction with optional RAG context."""
        disease_name = self.config.disease_name if self.config else "disease"

        # Format allowed labels for clarity
        allowed_labels_list = self._allowed_uris
        allowed_labels_formatted = "\n".join([f"  - {label}" for label in allowed_labels_list])

        system_prompt = f"""You are a clinical diagnosis assistant. Given a patient chart,
predict the most likely {disease_name} diagnosis from the allowed label set.

ALLOWED LABELS (use EXACT string, including full IRI):
{allowed_labels_formatted}
  - NONE

CRITICAL RULES:
1. Your predicted_label MUST be one of the exact labels listed above
2. Use the FULL IRI starting with \"http://\" - do NOT shorten or modify it
3. If uncertain, use \"NONE\" rather than inventing a label
4. Output ONLY valid JSON with this schema:
{{
    \"predicted_label\": \"<exact_label_or_NONE>\",
    \"ddx_top3\": [
        {{\"label\": \"<exact_label_or_NONE>\", \"rationale\": \"<short reason>\"}},
        {{\"label\": \"<exact_label_or_NONE>\", \"rationale\": \"<short reason>\"}},
        {{\"label\": \"<exact_label_or_NONE>\", \"rationale\": \"<short reason>\"}}
    ],
    \"evidence\": [\"<retrieved_doc_id_if_available>\"]
}}
5. ddx_top3 is optional but if present it must contain at most 3 items.
6. evidence is optional and should list retrieved doc identifiers only when available.

Example valid output:
{{
    \"predicted_label\": \"NONE\",
    \"ddx_top3\": [{{\"label\": \"NONE\", \"rationale\": \"No clear {disease_name} indicators\"}}],
    \"evidence\": []
}}"""

        user_prompt = f"Patient Chart:\n{chart_text}"
        if rag_context:
            user_prompt = f"{rag_context}\n\n{user_prompt}"

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        prompt_hash = self._hash_prompt(full_prompt)

        # Check cache
        if prompt_hash in self.cache:
            return self.cache[prompt_hash]

        # Generate response based on backend
        if self.backend == "gemini":
            response = self._predict_gemini(system_prompt, user_prompt)
        elif self.backend == "openai":
            response = self._predict_openai(system_prompt, user_prompt)
        elif self.backend == "huggingface":
            response = self._predict_huggingface(system_prompt, user_prompt)
        else:  # dry_run
            response = self._dry_run_predict(chart_text, rag_context)

        # Validate and coerce
        response = self._validate_response(response)

        # Cache result
        self.cache[prompt_hash] = response
        self._save_cache()

        return response

    def _validate_response(self, response: Dict) -> Dict:
        """Validate response and coerce labels to allowed URI set or NONE."""
        raw_predicted = response.get("predicted_label", self.none_label)
        predicted = self.label_normalizer.normalize_label(raw_predicted)
        if predicted == self.none_label and str(raw_predicted).strip().upper() != self.none_label:
            print(f"  ⚠️  Invalid label coerced to {self.none_label}: {str(raw_predicted)[:80]}")

        ddx_input = response.get("ddx_top3")
        if ddx_input is None:
            # Backward compatibility with older field name.
            legacy = response.get("top3_labels") or []
            ddx_input = [{"label": x, "rationale": "legacy_top3"} for x in legacy]

        validated_ddx: list[dict[str, str]] = []
        if isinstance(ddx_input, list):
            for item in ddx_input:
                if len(validated_ddx) >= 3:
                    break
                if isinstance(item, dict):
                    raw_label = item.get("label", self.none_label)
                    rationale = str(item.get("rationale", "")).strip()
                else:
                    raw_label = item
                    rationale = ""
                normalized = self.label_normalizer.normalize_label(raw_label)
                validated_ddx.append(
                    {
                        "label": normalized,
                        "rationale": rationale[:240],
                    }
                )

        evidence = response.get("evidence")
        if isinstance(evidence, list):
            evidence_out = [str(x).strip() for x in evidence if str(x).strip()][:6]
        else:
            evidence_out = []

        return {
            "predicted_label": predicted,
            "ddx_top3": validated_ddx,
            "evidence": evidence_out,
        }

    def get_backend_info(self) -> Dict:
        """Get information about the current backend."""
        return {
            "backend": self.backend,
            "cache_size": len(self.cache),
            "cache_file": self.cache_file
        }
