# Google Colab Setup Guide

## Quick Start for Colab

### Step 1: Open Notebook in Colab

Upload `tco_rag_comparison.ipynb` to Google Colab or open it directly from GitHub.

### Step 2: Add Colab Setup Cell (Insert at Top)

Insert this cell **before Section 1** in the notebook:

```python
# ============================================================================
# GOOGLE COLAB SETUP (Run this cell first!)
# ============================================================================

# Detect Colab environment
try:
    import google.colab
    IN_COLAB = True
    print("✓ Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("✓ Running locally")

if IN_COLAB:
    print("\n" + "="*60)
    print("COLAB ENVIRONMENT DETECTION")
    print("="*60)

    # Check GPU
    print("\nGPU Information:")
    gpu_info = !nvidia-smi
    gpu_info_str = '\n'.join(gpu_info)
    if gpu_info_str.find('failed') >= 0:
        print('✗ Not connected to a GPU')
        print('  To enable: Runtime > Change runtime type > Hardware accelerator > GPU')
        print('  Recommended: T4 GPU (free tier)')
    else:
        # Find GPU name
        for line in gpu_info:
            if any(x in line for x in ['Tesla', 'T4', 'P100', 'V100', 'A100', 'K80']):
                print(f'✓ GPU detected')
                # Extract GPU name (simplified)
                if 'T4' in line:
                    print('  Model: Tesla T4')
                elif 'P100' in line:
                    print('  Model: Tesla P100')
                elif 'V100' in line:
                    print('  Model: Tesla V100')
                elif 'K80' in line:
                    print('  Model: Tesla K80')
                break

    # Check RAM
    print("\nRAM Information:")
    import psutil
    ram_gb = psutil.virtual_memory().total / 1e9
    print(f'✓ Available RAM: {ram_gb:.1f} GB')

    if ram_gb < 12:
        print('  Runtime: Standard (12.7 GB)')
    elif ram_gb < 20:
        print('  Runtime: High-RAM')
        print('  To change: Runtime > Change runtime type > Runtime shape')
    else:
        print('  Runtime: Premium High-RAM (52 GB)')

    print("\n" + "="*60)
    print("INSTALLING DEPENDENCIES")
    print("="*60)

    # Install packages
    print("\nInstalling required packages...")
    !pip install -q pandas numpy requests scikit-learn psutil
    !pip install -q sentence-transformers  # For better retrieval
    !pip install -q google-generativeai    # For Gemini (FREE)
    print("✓ Dependencies installed")

    print("\n" + "="*60)
    print("API KEY SETUP")
    print("="*60)

    # Setup API keys
    import os

    # Google API Key (for Gemini - FREE)
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n🔑 GOOGLE API KEY (Required for FREE Gemini)")
        print("   Get your free API key:")
        print("   → https://aistudio.google.com/app/apikey")
        print()
        google_key = input("   Enter your Google API key (or press Enter to skip): ").strip()
        if google_key:
            os.environ["GOOGLE_API_KEY"] = google_key
            print("   ✓ Google API key set")
        else:
            print("   ⚠ Skipped - will run in dry-run mode")
    else:
        print("✓ GOOGLE_API_KEY already set")

    # BioPortal API Key
    if not os.getenv("BIOPORTAL_API_KEY"):
        print("\n🔑 BIOPORTAL API KEY (Required for TCO ontology access)")
        print("   Default key available in project .env file")
        print("   Or get your own: https://bioportal.bioontology.org/account")
        print()
        use_default = input("   Use default key from project? (Y/n): ").strip().lower()
        if use_default != 'n':
            # Default key from project
            os.environ["BIOPORTAL_API_KEY"] = "98d19152-8c21-4a0c-bd50-c09b46543947"
            print("   ✓ Using default BioPortal API key")
        else:
            bp_key = input("   Enter your BioPortal API key: ").strip()
            if bp_key:
                os.environ["BIOPORTAL_API_KEY"] = bp_key
                print("   ✓ BioPortal API key set")
    else:
        print("✓ BIOPORTAL_API_KEY already set")

    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\n✓ Ready to run the notebook")
    print("✓ Using Google Gemini (FREE) for LLM inference")
    print("✓ Proceed with Section 1 below")
    print("\n" + "="*60)

else:
    print("\nRunning locally - use standard setup:")
    print("  pip install -r requirements.txt")
    print("  export GOOGLE_API_KEY='your-key'")
    print("  export BIOPORTAL_API_KEY='your-key'")
```

### Step 3: Run the Setup Cell

Click the play button on the setup cell. It will:
- ✓ Detect your GPU (T4, P100, V100, etc.)
- ✓ Check available RAM (12-52 GB)
- ✓ Install all required packages
- ✓ Prompt for API keys

### Step 4: Get Google Gemini API Key (FREE)

1. Visit: https://aistudio.google.com/app/apikey
2. Click "Create API key"
3. Copy the key
4. Paste it when prompted in Step 3

**Why Gemini?**
- ✓ Free (generous quota)
- ✓ Fast (flash model)
- ✓ Good quality
- ✓ No credit card required

### Step 5: Run the Notebook

After setup completes, run all cells normally. The notebook will automatically use Google Gemini for inference.

## Colab-Specific Tips

### Enable GPU (Recommended)

For faster Hugging Face models (if you want to use HF instead of Gemini):

1. Runtime > Change runtime type
2. Hardware accelerator > **GPU**
3. GPU type > **T4** (free tier)
4. Save

Then add this to setup cell:
```python
os.environ["USE_HUGGINGFACE"] = "true"
```

### Enable High-RAM Runtime

For large models:

1. Runtime > Change runtime type
2. Runtime shape > **High-RAM**
3. Save

### Using Different LLM Backends in Colab

**Option 1: Google Gemini (Recommended for Colab)**
```python
# Already default - no changes needed!
# Just need GOOGLE_API_KEY
```

**Option 2: Hugging Face (Requires GPU)**
```python
import os
os.environ["USE_HUGGINGFACE"] = "true"
os.environ["HF_MODEL"] = "Qwen/Qwen2.5-1.5B-Instruct"  # Or smaller: 0.5B
```

**Option 3: OpenAI (Requires API key + $)**
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
# Don't set USE_HUGGINGFACE or GOOGLE_API_KEY
```

**Option 4: Dry-Run (No LLM needed)**
```python
# Don't set any API keys
# Uses deterministic heuristics
```

## Alternative: Use Python Script in Colab

You can also run the Python script directly in Colab:

```python
# Cell 1: Clone/upload files
!git clone https://github.com/your-repo/tco-rag.git
%cd tco-rag

# Cell 2: Setup
!pip install -q -r requirements.txt
import os
os.environ["GOOGLE_API_KEY"] = "your-key-here"
os.environ["BIOPORTAL_API_KEY"] = "98d19152-8c21-4a0c-bd50-c09b46543947"

# Cell 3: Run
!python rag_exp.py
```

## Expected Runtime in Colab

| Configuration | Runtime | Cost |
|---------------|---------|------|
| Gemini + TF-IDF | ~5-8 min | Free |
| Gemini + Embeddings | ~6-10 min | Free |
| HF (T4 GPU) + Embeddings | ~15-25 min | Free |
| HF (CPU only) | ~45-90 min | Free |

## Colab Quotas

### Free Tier
- **GPU:** T4 (16GB VRAM)
- **RAM:** 12.7 GB standard, 52 GB high-RAM (limited)
- **Session:** 12 hours max
- **Gemini API:** 15 requests/min, 1M tokens/day (generous!)

### Tips to Stay Within Quota
1. Use caching (automatic in notebook)
2. Start with Gemini (free, fast)
3. Only use GPU if needed (HF models)
4. Save artifacts to Google Drive

## Saving Results in Colab

Add this cell at the end to save results to Google Drive:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy artifacts to Drive
!cp synthetic_charts.csv /content/drive/MyDrive/
!cp tco_corpus.jsonl /content/drive/MyDrive/
!cp results.json /content/drive/MyDrive/
!cp examples.md /content/drive/MyDrive/
!cp llm_cache.json /content/drive/MyDrive/

print("✓ Artifacts saved to Google Drive")
```

## Troubleshooting

### "Quota exceeded" for Gemini
- Wait a few minutes
- Or use smaller dataset (modify n_thyroid=30, n_none=30)

### "Out of RAM" in Colab
- Switch to High-RAM runtime
- Or use TF-IDF instead of embeddings

### "GPU not available"
- Check Runtime > Change runtime type > Hardware accelerator
- Or use Gemini (doesn't need GPU)

### Slow execution
- Use Gemini instead of HF (much faster)
- Enable GPU for HF models
- Reduce dataset size for testing

## Complete Colab Notebook Template

Here's a minimal working example for Colab:

```python
# === CELL 1: Colab Setup ===
# [Insert full setup cell from Step 2 above]

# === CELL 2: Download Files ===
# Option A: If files are on GitHub
!git clone https://github.com/your-repo/bime-550.git
%cd bime-550

# Option B: Upload files manually
# from google.colab import files
# uploaded = files.upload()  # Upload PROPOSAL.md, src/classes/llm_interface.py, etc.

# === CELLS 3+: Run Normal Notebook ===
# [Rest of notebook cells from tco_rag_comparison.ipynb]
```

## Advanced: Custom Colab Runtime

For power users:

```python
# Use custom model
os.environ["GEMINI_MODEL"] = "gemini-1.5-pro"  # Better quality, slower

# Use smaller HF model
os.environ["HF_MODEL"] = "Qwen/Qwen2.5-0.5B-Instruct"  # Faster, fits in RAM

# Adjust retrieval
# Edit in notebook: retriever = EmbeddingRetriever(corpus, top_k=5)
```

## Resources

- **Gemini API Docs:** https://ai.google.dev/docs
- **Colab FAQ:** https://research.google.com/colaboratory/faq.html
- **BioPortal API:** https://data.bioontology.org/documentation
- **Sentence Transformers:** https://www.sbert.net/

## Support

If you encounter issues:
1. Check the error message
2. Verify API keys are set correctly
3. Ensure you're using a GPU runtime (for HF)
4. Try Gemini instead (simpler, faster)
5. Check Colab quota/limits
