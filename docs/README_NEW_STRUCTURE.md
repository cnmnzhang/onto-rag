# 🎯 Quick Start (New Structure)

## Project Structure

```
.
├── src/           # All Python source code
├── data/          # All generated data files  
├── run.py         # Main runner script
└── .env           # Environment variables
```

## Running the Experiment

```bash
# Quick way
python run.py

# Or with shell script
./run_experiment.sh gemini
```

## Test Mode (3 charts)

The experiment is currently in **TEST MODE** for quick testing.

Edit `src/rag_exp.py` line 38 to change:
```python
TEST_MODE = True   # ← Currently this (3 charts, ~30 seconds)
TEST_MODE = False  # ← Change to this (120 charts, ~10 minutes)
```

## Outputs

All generated files go to `data/`:
- `data/synthetic_charts.csv` - Patient charts
- `data/tco_corpus.jsonl` - Ontology corpus
- `data/results.json` - Metrics
- `data/examples.md` - Error examples

## Configuration

Change disease at `src/rag_exp.py` line 34:
```python
CONFIG = get_config("tco")       # Thyroid (current)
CONFIG = get_config("diabetes")  # Switch to diabetes
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for details.
