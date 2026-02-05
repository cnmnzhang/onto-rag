#!/bin/bash
# Runner script for TCO RAG Comparison
# Usage: ./run_experiment.sh [gemini|huggingface|openai|dryrun]

set -e

# Load .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Determine mode
MODE=${1:-gemini}

echo "=================================="
echo "TCO RAG Comparison Experiment"
echo "=================================="
echo "Mode: $MODE"
echo

case $MODE in
    gemini|google)
        echo "Using Google Gemini (gemini-1.5-flash)"
        echo "FREE with API key from https://aistudio.google.com/app/apikey"
        if [ -z "$GOOGLE_API_KEY" ]; then
            echo "Warning: GOOGLE_API_KEY not set"
            echo "Set it with: export GOOGLE_API_KEY='your-key'"
            echo "Or will fall back to dry-run mode"
        fi
        unset USE_HUGGINGFACE
        unset OPENAI_API_KEY
        ;;

    huggingface|hf)
        echo "Using Hugging Face (Qwen2.5-1.5B-Instruct)"
        echo "This will download ~3GB model on first run"
        export USE_HUGGINGFACE="true"
        unset OPENAI_API_KEY
        unset GOOGLE_API_KEY
        ;;

    openai)
        echo "Using OpenAI (GPT-3.5-turbo)"
        if [ -z "$OPENAI_API_KEY" ]; then
            echo "Error: OPENAI_API_KEY not set"
            echo "Set it with: export OPENAI_API_KEY='your-key'"
            exit 1
        fi
        unset USE_HUGGINGFACE
        unset GOOGLE_API_KEY
        ;;

    dryrun|dry)
        echo "Using Dry-Run Mode (deterministic heuristics)"
        unset USE_HUGGINGFACE
        unset OPENAI_API_KEY
        unset GOOGLE_API_KEY
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [gemini|huggingface|openai|dryrun]"
        exit 1
        ;;
esac

echo
echo "Starting experiment..."
echo

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected"
    echo "Consider running: source .venv/bin/activate"
    echo
fi

# Run the experiment
python tco_rag_comparison.py

echo
echo "=================================="
echo "Experiment Complete!"
echo "=================================="
echo
echo "Generated artifacts:"
echo "  - synthetic_charts.csv"
echo "  - tco_corpus.jsonl"
echo "  - results.json"
echo "  - examples.md"
echo "  - llm_cache.json"
echo
echo "Check results.json for metrics summary"
