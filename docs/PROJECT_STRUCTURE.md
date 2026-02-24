# Project Structure

## 📁 Directory Layout

```
bime/550/
├── src/                          # Source code
│   ├── __init__.py              # Package init
│   ├── rag_exp.py               # Main experiment script
│   ├── classes/                 # Core modules/classes
│   │   ├── __init__.py          # Convenience re-exports
│   │   ├── llm_interface.py     # LLM backend abstraction
│   │   ├── onto_config.py       # Ontology configurations
│   │   ├── retrievers.py        # Retrieval implementations
│   │   ├── corpus.py            # BioPortal fetch + corpus normalization
│   │   └── label_alias.py       # Label normalization helpers
│   ├── synthetic_data.py        # Chart generation
│   ├── colab_setup.py           # Colab utilities
│   ├── demo_config_swap.py      # Config demo script
│   └── test_quick.py            # Quick test script
│
├── data/                         # Data files (git-ignored)
│   ├── .gitkeep                 # Keep folder in git
│   ├── synthetic_charts.csv     # Generated charts
│   ├── tco_corpus.jsonl         # Ontology corpus cache
│   ├── results.json             # Evaluation metrics
│   ├── examples.md              # Error examples
│   └── llm_cache.json           # LLM response cache
│
├── docs/                         # Documentation
│   ├── README.md                # Main README
│   ├── README_USAGE.md          # Usage guide
│   ├── COLAB_SETUP.md           # Google Colab guide
│   ├── CONFIG_MIGRATION_SUMMARY.md  # Config system docs
│   └── PROJECT_STRUCTURE.md     # This file
│
├── run.py                        # Simple Python runner
├── run_experiment.sh             # Shell runner script
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (git-ignored)
├── .gitignore                    # Git ignore rules
└── PROPOSAL.md                   # Project proposal

```

## 🚀 Running the Experiment

### Option 1: Python Runner (Recommended)
```bash
python run.py
```

### Option 2: Direct Execution
```bash
python src/rag_exp.py
```

### Option 3: Shell Script
```bash
./run_experiment.sh [gemini|huggingface|openai|dryrun]
```

## 📝 Quick Test Mode

Edit `src/rag_exp.py` line 38:
```python
TEST_MODE = True   # 3 charts for testing
TEST_MODE = False  # 120 charts for full run
```

## 🔧 Configuration

All source files are in `src/` and import from each other directly:
```python
from classes.llm_interface import LLMInterface
from classes.onto_config import get_config
from classes.retrievers import create_retriever
```

All data files are written to `../data/` relative to `src/`:
- Charts: `../data/synthetic_charts.csv`
- Corpus: `../data/tco_corpus.jsonl`
- Results: `../data/results.json`
- Cache: `../data/llm_cache.json`

## 📦 Package Structure

The `src/` directory is a proper Python package with:
- `__init__.py` for package initialization
- Modular imports between files
- Clean separation of concerns

## 🔄 Switching Diseases

Edit `src/rag_exp.py` line 34:
```python
CONFIG = get_config("tco")         # Thyroid cancer (default)
CONFIG = get_config("diabetes")    # Diabetes
CONFIG = get_config("lung_cancer") # Lung cancer
```

## 📊 Output Files

All outputs go to `data/`:
1. `synthetic_charts.csv` - Generated patient charts
2. `tco_corpus.jsonl` - Ontology class documents
3. `results.json` - Evaluation metrics
4. `examples.md` - Error analysis examples
5. `llm_cache.json` - Cached LLM responses

## 🧪 Development Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env  # Edit with your API keys

# 3. Run quick test
python run.py  # With TEST_MODE=True

# 4. Check outputs
ls -la data/

# 5. Run full experiment
# Edit src/rag_exp.py: TEST_MODE = False
python run.py
```

## 🎯 Key Design Decisions

1. **src/ for code** - Clean separation of source from data
2. **data/ for outputs** - All generated files in one place
3. **Root for config** - .env, requirements.txt at project root
4. **Relative paths** - All paths relative to script location
5. **Python package** - src/ is importable as a package

## 🔐 Environment Variables

Create `.env` at project root:
```env
BIOPORTAL_API_KEY=your-key-here
GOOGLE_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key (optional)
```

## 📚 Documentation

- **README.md** - Project overview and quickstart
- **README_USAGE.md** - Detailed usage instructions
- **COLAB_SETUP.md** - Google Colab setup guide
- **CONFIG_MIGRATION_SUMMARY.md** - Config system documentation
- **PROJECT_STRUCTURE.md** - This file

## 🎓 Benefits of This Structure

1. **Clean separation** - Code vs data vs docs
2. **Easy navigation** - Everything has its place
3. **Git-friendly** - data/ is ignored, structure is tracked
4. **Professional** - Follows Python packaging standards
5. **Scalable** - Easy to add new modules or datasets
