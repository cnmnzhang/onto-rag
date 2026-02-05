"""
LLM Interface for Thyroid Cancer Differential Diagnosis
Supports OpenAI, Hugging Face (Qwen2.5-1.5B-Instruct), Google Gemini, and dry-run modes
"""

import os
import json
import hashlib
import random
from typing import Dict, List, Optional, Set


class LLMInterface:
    """
    Unified interface for LLM predictions with multiple backend support.

    Backends:
    - Google Gemini (gemini-1.5-flash) - FREE with API key
    - OpenAI (gpt-3.5-turbo)
    - Hugging Face (Qwen2.5-1.5B-Instruct)
    - Dry-run (deterministic heuristics)
    """

    def __init__(self, allowed_labels: List[str], cache_file: str = "llm_cache.json"):
        self.allowed_labels = set(allowed_labels + ["NONE"])
        self.cache_file = cache_file
        self.cache = self._load_cache()

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
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
            except:
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
        cancer_labels = [l for l in self.allowed_labels if l != "NONE"]
        cancer_labels_list = list(cancer_labels)

        # Heuristic rules
        if "malignant" in text_lower or "fna reveals malignant" in text_lower:
            return {
                "predicted_label": cancer_labels_list[0] if cancer_labels_list else "NONE",
                "top3_labels": cancer_labels_list[:3] if cancer_labels_list else ["NONE"],
                "rationale": "Heuristic: malignant keywords detected"
            }
        elif any(kw in text_lower for kw in ["thyroid mass", "thyroid nodule", "fna cytology shows suspicious", "atypical cells"]):
            pred_label = cancer_labels_list[1] if len(cancer_labels_list) > 1 else (cancer_labels_list[0] if cancer_labels_list else "NONE")
            return {
                "predicted_label": pred_label,
                "top3_labels": cancer_labels_list[:3] if cancer_labels_list else ["NONE"],
                "rationale": "Heuristic: thyroid nodule/suspicious cytology detected"
            }
        elif "normal thyroid" in text_lower or "no thyroid" in text_lower or "tsh" in text_lower:
            return {
                "predicted_label": "NONE",
                "top3_labels": ["NONE"],
                "rationale": "Heuristic: normal thyroid findings"
            }
        else:
            return {
                "predicted_label": "NONE",
                "top3_labels": ["NONE"],
                "rationale": "Heuristic: no clear thyroid cancer indicators"
            }

    def _predict_gemini(self, system_prompt: str, user_prompt: str) -> Dict:
        """Generate prediction using Google Gemini."""
        try:
            # Gemini doesn't have separate system/user roles in the same way
            # Combine them into a single prompt
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            response = self.client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": 0.3,
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
            return {
                "predicted_label": "NONE",
                "top3_labels": ["NONE"],
                "rationale": f"Parse error: {response_text[:100]}"
            }
        except Exception as e:
            print(f"Gemini error: {e}")
            return {
                "predicted_label": "NONE",
                "top3_labels": ["NONE"],
                "rationale": f"Error: {str(e)}"
            }

    def _predict_openai(self, system_prompt: str, user_prompt: str) -> Dict:
        """Generate prediction using OpenAI."""
        try:
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            response_text = completion.choices[0].message.content
            return json.loads(response_text)
        except Exception as e:
            print(f"OpenAI error: {e}")
            return {
                "predicted_label": "NONE",
                "top3_labels": ["NONE"],
                "rationale": f"Error: {str(e)}"
            }

    def _predict_huggingface(self, system_prompt: str, user_prompt: str) -> Dict:
        """Generate prediction using Hugging Face model."""
        try:
            import torch

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
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            response_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Try to parse JSON from response
            # The model might wrap it in markdown or add text
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
            # Try to extract label from text
            return {
                "predicted_label": "NONE",
                "top3_labels": ["NONE"],
                "rationale": f"Parse error: {response_text[:100]}"
            }
        except Exception as e:
            print(f"Hugging Face error: {e}")
            return {
                "predicted_label": "NONE",
                "top3_labels": ["NONE"],
                "rationale": f"Error: {str(e)}"
            }

    def predict(self, chart_text: str, rag_context: Optional[str] = None) -> Dict:
        """
        Generate prediction with optional RAG context.

        Args:
            chart_text: Patient chart text
            rag_context: Optional retrieved ontology context

        Returns:
            Dict with predicted_label, top3_labels, and rationale
        """
        # Build prompt
        system_prompt = f"""You are a clinical diagnosis assistant. Given a patient chart,
predict the most likely thyroid cancer diagnosis from the allowed label set.

ALLOWED LABELS: {', '.join(sorted([l for l in self.allowed_labels if l != 'NONE']))}, NONE

Output valid JSON only:
{{"predicted_label": "<label>", "top3_labels": ["<label1>", "<label2>", "<label3>"], "rationale": "<brief explanation>"}}

If no thyroid cancer is evident, return "NONE" as the predicted_label.
Use the full IRI (http://...) for thyroid cancer labels, not just the short name."""

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
        """Validate response and coerce invalid labels to NONE."""
        predicted = response.get("predicted_label", "NONE")

        if predicted not in self.allowed_labels:
            print(f"  Invalid label '{predicted}' coerced to NONE")
            response["predicted_label"] = "NONE"

        # Validate top3
        top3 = response.get("top3_labels", [])
        validated_top3 = [l for l in top3 if l in self.allowed_labels][:3]
        # Pad with NONE if needed
        while len(validated_top3) < 3:
            validated_top3.append("NONE")
        response["top3_labels"] = validated_top3

        return response

    def get_backend_info(self) -> Dict:
        """Get information about the current backend."""
        return {
            "backend": self.backend,
            "cache_size": len(self.cache),
            "cache_file": self.cache_file
        }
