# TCO RAG Comparison - Usage Guide

This project compares No-RAG vs RAG(TCO) for thyroid cancer differential diagnosis using synthetic patient charts.

**🆕 Now with Google Colab support!** See [COLAB_SETUP.md](COLAB_SETUP.md) for complete Colab instructions.

## Installation

### 1. Install Dependencies

```bash
# Create and activate virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Set Environment Variables

The BioPortal API key is already in the `.env` file. For LLM backends, set:

```bash
# For OpenAI (optional)
export OPENAI_API_KEY="your-openai-api-key"

# For Hugging Face (free option)
export USE_HUGGINGFACE="true"

# Optional: specify HF model (default: Qwen/Qwen2.5-1.5B-Instruct)
export HF_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
```

## Running the Script

### Option 1: Google Gemini (Free, Cloud) **RECOMMENDED**

Uses Gemini 1.5 Flash via Google AI API (free tier):

```bash
export GOOGLE_API_KEY="your-gemini-key"
python rag_exp.py
```

Get a free API key: https://aistudio.google.com/app/apikey

**Pros:**
- Free (generous quota: 15 req/min, 1M tokens/day)
- Fast (optimized flash model)
- Good quality predictions
- No local compute requirements
- No credit card required

**Cons:**
- Requires internet connection
- Data sent to Google

### Option 2: Hugging Face (Free, Local)

Uses Qwen2.5-1.5B-Instruct model locally (no API costs):

```bash
export USE_HUGGINGFACE="true"
python rag_exp.py
```

**Pros:**
- Free (no API costs)
- Runs locally
- Full privacy (no data sent to external APIs)

**Cons:**
- Requires ~3GB disk space for model
- Slower than OpenAI API
- Requires GPU for reasonable speed (CPU works but is slow)

### Option 3: OpenAI (Paid, Fast)

Uses GPT-3.5-turbo via OpenAI API:

```bash
export OPENAI_API_KEY="your-key"
# Don't set USE_HUGGINGFACE or set it to false
python rag_exp.py
```

**Pros:**
- Fast
- High quality predictions
- No local compute requirements

**Cons:**
- Costs money (~$0.01-0.05 for full run)
- Requires internet connection
- Data sent to OpenAI

### Option 4: Dry-Run (No LLM Required)

Uses deterministic keyword heuristics:

```bash
# Don't set any LLM-related environment variables
python rag_exp.py
```

**Pros:**
- Free
- Fast
- No dependencies on transformers/openai
- Good for testing pipeline

**Cons:**
- Simple heuristics (not real ML)
- Lower quality predictions

## LLM Interface Module

The LLM logic is extracted to `llm_interface.py` for modularity. It provides:

- **Automatic backend detection** (Hugging Face > OpenAI > Dry-run)
- **Prompt caching** (saves all responses to `llm_cache.json`)
- **Response validation** (coerces invalid outputs to NONE)
- **Unified API** regardless of backend

### Using the LLM Interface Standalone

```python
from llm_interface import LLMInterface

# Initialize with allowed TCO labels
llm = LLMInterface(
    allowed_labels=[
        "http://purl.obolibrary.org/obo/TCO_0000123",
        "http://purl.obolibrary.org/obo/TCO_0000456",
        # ... more TCO IRI
    ],
    cache_file="my_cache.json"
)

# Get prediction
chart_text = "45-year-old M with thyroid nodule..."
prediction = llm.predict(chart_text, rag_context=None)

print(prediction)
# {
#   "predicted_label": "http://...",
#   "top3_labels": ["http://...", "http://...", "NONE"],
#   "rationale": "Patient presents with..."
# }

# Check backend
info = llm.get_backend_info()
print(f"Using backend: {info['backend']}")
print(f"Cache size: {info['cache_size']} entries")
```

## Backend Comparison

| Backend | Cost | Speed | Quality | Setup | Best For |
|---------|------|-------|---------|-------|----------|
| **Google Gemini** ⭐ | Free | Very Fast | Excellent | API key | Colab, general use |
| **Hugging Face** | Free | Medium-Fast | Good | Model download | Local, privacy, GPU |
| **OpenAI** | ~$0.02 | Fast | Excellent | API key | Production |
| **Dry-run** | Free | Fastest | Basic | None | Testing pipeline |

**Recommendation:** Start with **Google Gemini** (free, no setup hassle) or **Hugging Face** (if you have a GPU and want local inference).

## Caching

All LLM responses are cached in `llm_cache.json` with MD5 hash keys:

- **Cache key:** MD5 hash of full prompt (system + user + RAG context)
- **Cache hit:** Returns cached response immediately (no API call)
- **Cache miss:** Generates new response and saves to cache
- **Persistence:** Cache survives script restarts

This means:
- First run is slow (generates all responses)
- Subsequent runs are instant (uses cache)
- Changing prompts creates new cache entries
- Useful for iterating without API costs

To clear cache:
```bash
rm llm_cache.json
```

## Output Artifacts

The script generates:

1. **synthetic_charts.csv** - 120 patient charts with gold labels
2. **tco_corpus.jsonl** - 8 TCO classes with metadata (JSON Lines)
3. **results.json** - Evaluation metrics and metadata
4. **examples.md** - Error examples from both conditions
5. **llm_cache.json** - Cached LLM responses

## Expected Performance

### Dry-Run Mode
- No-RAG: ~25-35% agreement (simple heuristics)
- RAG(TCO): ~25-35% agreement (heuristics don't use RAG context)

### Hugging Face (Qwen2.5-1.5B-Instruct)
- No-RAG: ~40-60% agreement
- RAG(TCO): ~50-70% agreement (expected +10-20pp improvement)

### OpenAI (GPT-3.5-turbo)
- No-RAG: ~50-70% agreement
- RAG(TCO): ~60-80% agreement (expected +10-20pp improvement)

## Troubleshooting

### Hugging Face Model Download Fails

```bash
# Set HF cache directory if low on disk space in home
export HF_HOME="/path/to/large/disk"

# Use smaller model
export HF_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
```

### Out of Memory (OOM) with Hugging Face

```python
# Edit llm_interface.py and change:
torch_dtype=torch.float32  # Instead of float16
```

Or use a smaller model:
```bash
export HF_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
```

### OpenAI Rate Limit

The script has basic rate limit handling (1 second wait on 429 errors). For large runs, you can:

1. Use the cache (re-run uses cached responses)
2. Process charts in batches
3. Add longer delays between requests

### Sentence-Transformers Not Available

The script automatically falls back to TF-IDF for retrieval. This works fine and is deterministic.

## Architecture

```
rag_exp.py
├── Load proposal
├── Connect to BioPortal TCO API
├── Build retrieval corpus (8 TCO classes)
├── Generate 120 synthetic charts
├── Initialize LLM Interface ──> llm_interface.py
│   ├── Detect backend (HF/OpenAI/Dry-run)
│   ├── Load model/client
│   └── Setup caching
├── Run No-RAG predictions (chart → LLM)
├── Run RAG predictions (chart → retrieval → LLM)
├── Evaluate (percent agreement, confusion matrix)
└── Save artifacts
```

## Development

To modify the LLM behavior:

1. **Edit prompt:** Modify `_predict_openai()` or `_predict_huggingface()` in `llm_interface.py`
2. **Add new backend:** Add new methods and update `_detect_backend()`
3. **Change validation:** Edit `_validate_response()` in `llm_interface.py`
4. **Adjust caching:** Modify `_hash_prompt()` or `_save_cache()`

To modify the evaluation:

1. **Edit metrics:** Modify `calculate_agreement()` in `rag_exp.py`
2. **Add visualizations:** Add plotting after evaluation section
3. **Change corpus size:** Modify `select_thyroid_cancer_classes(n=8)` parameter

## License

This is a course project for BIME 550. Use for educational purposes.
