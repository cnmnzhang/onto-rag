"""Stage-based official evaluation pipeline."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from classes.corpus import ensure_corpus
from classes.llm_interface import LLMInterface
from classes.onto_config import get_config
from classes.retrievers import create_retriever
from config.paths import FETCH_LABEL_URIS_SCRIPT_PATH, PROJECT_ROOT
from eval_types import RunConfig
from rag_context import build_rag_context
from schemas import (
    CoercionsSection,
    EvaluationResultsDocument,
    LabelSet,
    MetricsSection,
    PredictionRow,
    RunSection,
)


@dataclass(frozen=True)
class PreparedRuntime:
    cfg: RunConfig
    config: Any
    label_set: LabelSet
    allowed: set[str]
    embedding_model: str
    model_cache_dir: Path
    retriever: Any
    llm: LLMInterface
    dataset_df: pd.DataFrame
    excluded_df: pd.DataFrame


@dataclass(frozen=True)
class PredictionBatch:
    rows: list[PredictionRow]
    gold: list[str]
    pred_no_rag: list[str]
    pred_rag: list[str]
    coerced_no_rag: int
    coerced_rag: int


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit_hash() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def _sanitize_model_name_for_path(model_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(model_name).strip())
    return safe.strip("-") or "default"


def _model_name_for_backend(llm: LLMInterface) -> str:
    if llm.backend == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    if llm.backend == "huggingface":
        return os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    if llm.backend == "openai":
        return "gpt-3.5-turbo"
    return "dry_run"


def _ensure_label_set_exists(*, cfg: RunConfig, ontology_acronym: str) -> None:
    if cfg.label_set_path.exists():
        return

    fetch_script = PROJECT_ROOT / FETCH_LABEL_URIS_SCRIPT_PATH
    if not fetch_script.exists():
        raise RuntimeError(
            "Label set is missing and fetch script was not found at "
            f"{fetch_script}."
        )

    cfg.label_set_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(fetch_script),
        "--ontology",
        str(ontology_acronym),
        "--output",
        str(cfg.label_set_path),
    ]
    print(
        f"Label set not found at {cfg.label_set_path}; generating via fetch_label_uris..."
    )
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to auto-generate label set. "
            f"Command exited with code {proc.returncode}: {' '.join(cmd)}"
        )
    if not cfg.label_set_path.exists():
        raise RuntimeError(
            "fetch_label_uris completed but label set file was not created: "
            f"{cfg.label_set_path}"
        )
    print(f"✓ Generated label set: {cfg.label_set_path}")


def _exact_agreement(gold: list[str], pred: list[str]) -> float:
    if not gold:
        return 0.0
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    return 100.0 * correct / len(gold)


def _coerce(pred: str, allowed: set[str], none_label: str) -> tuple[str, bool]:
    if pred in allowed:
        return pred, False
    print(f"⚠️  Invalid prediction label coerced to {none_label}: {pred[:80]}")
    return none_label, True


def _ensure_case_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "case_id" in out.columns:
        out["case_id"] = out["case_id"].astype(str)
    elif "chart_id" in out.columns:
        out["case_id"] = out["chart_id"].apply(lambda x: f"chart-{int(x)}")
    else:
        out["case_id"] = [f"case-{i:04d}" for i in range(len(out))]
    return out


def _predict_case(
    llm: LLMInterface,
    *,
    text: str,
    rag_context: str | None,
    allowed: set[str],
    none_label: str,
) -> tuple[str, str, str, int]:
    out = llm.predict(text, rag_context=rag_context)
    pred, did_coerce = _coerce(str(out.get("predicted_label", none_label)), allowed, none_label)
    ddx = json.dumps(out.get("ddx_top3") or [], ensure_ascii=False)
    evidence = json.dumps(out.get("evidence") or [], ensure_ascii=False)
    return pred, ddx, evidence, int(did_coerce)


def _paired_ttest_correctness(gold: list[str], pred_no_rag: list[str], pred_rag: list[str]) -> dict[str, Any]:
    ok_no_rag = np.asarray([1.0 if g == p else 0.0 for g, p in zip(gold, pred_no_rag)], dtype=float)
    ok_rag = np.asarray([1.0 if g == p else 0.0 for g, p in zip(gold, pred_rag)], dtype=float)
    diff = ok_rag - ok_no_rag

    n = int(diff.size)
    mean_delta = float(np.mean(diff)) if n else 0.0
    improved_cases = int(np.sum(diff > 0))
    worse_cases = int(np.sum(diff < 0))
    unchanged_cases = int(np.sum(diff == 0))

    out: dict[str, Any] = {
        "available": True,
        "method": "paired_ttest_on_binary_correctness",
        "n_cases": n,
        "improved_cases": improved_cases,
        "worse_cases": worse_cases,
        "unchanged_cases": unchanged_cases,
        "mean_delta_correctness": mean_delta,
        "mean_delta_agreement_points": 100.0 * mean_delta,
    }

    if n < 2:
        out.update(
            {
                "available": False,
                "reason": "need_at_least_2_cases",
                "degrees_freedom": 0,
                "t_stat": None,
                "p_value": None,
            }
        )
        return out

    if np.allclose(diff, diff[0]):
        if float(diff[0]) == 0.0:
            t_stat = 0.0
            p_value = 1.0
        else:
            t_stat = float("inf") if diff[0] > 0 else float("-inf")
            p_value = 0.0
        out.update(
            {
                "degrees_freedom": int(n - 1),
                "t_stat": t_stat,
                "p_value": p_value,
                "library": "degenerate_closed_form",
            }
        )
        return out

    try:
        from scipy.stats import ttest_rel  # type: ignore

        res = ttest_rel(ok_rag, ok_no_rag)
        out.update(
            {
                "degrees_freedom": int(n - 1),
                "t_stat": float(res.statistic),
                "p_value": float(res.pvalue),
                "library": "scipy.stats.ttest_rel",
            }
        )
        return out
    except Exception:
        std_diff = float(np.std(diff, ddof=1))
        if std_diff <= 0.0:
            out.update(
                {
                    "degrees_freedom": int(n - 1),
                    "t_stat": None,
                    "p_value": None,
                    "library": "manual_no_scipy",
                    "reason": "zero_variance_diff",
                }
            )
            return out
        se = std_diff / np.sqrt(n)
        out.update(
            {
                "degrees_freedom": int(n - 1),
                "t_stat": float(mean_delta / se),
                "p_value": None,
                "library": "manual_no_scipy",
                "reason": "scipy_unavailable",
            }
        )
        return out


def prepare_runtime(cfg: RunConfig) -> PreparedRuntime:
    os.environ.setdefault("PYTHONHASHSEED", str(cfg.seed))
    np.random.seed(cfg.seed)

    config = get_config(cfg.ontology_key)
    _ensure_label_set_exists(cfg=cfg, ontology_acronym=config.acronym)
    label_set = LabelSet.from_path(cfg.label_set_path)
    allowed = label_set.allowed_set()

    embedding_model = os.getenv("EMBEDDING_MODEL", cfg.embedding_model).strip() or "all-MiniLM-L6-v2"
    corpus = ensure_corpus(
        config=config,
        label_ids=label_set.labels,
        output_path=cfg.corpus_path,
        prefer_bioportal=True,
    )

    model_cache_dir = cfg.retriever_cache_dir / _sanitize_model_name_for_path(embedding_model)
    retriever = create_retriever(
        corpus,
        top_k=cfg.top_k,
        prefer_embeddings=cfg.prefer_embeddings,
        cache_dir=str(model_cache_dir),
        model_name=embedding_model,
    )
    llm = LLMInterface(list(label_set.labels), cache_file=str(cfg.llm_cache_path), config=config)

    df = pd.read_csv(cfg.dataset_path)
    required = {"chart_text", "gold_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    df = _ensure_case_ids(df)
    is_valid_gold = df["gold_label"].astype(str).isin(allowed)
    excluded_df = df.loc[~is_valid_gold, ["case_id", "gold_label"]]
    df = df.loc[is_valid_gold].copy()
    if df.empty:
        raise RuntimeError(
            "No rows have gold_label in the allowed set ∪ NONE. "
            "Update the selected --label-set or the dataset gold labels."
        )

    return PreparedRuntime(
        cfg=cfg,
        config=config,
        label_set=label_set,
        allowed=allowed,
        embedding_model=embedding_model,
        model_cache_dir=model_cache_dir,
        retriever=retriever,
        llm=llm,
        dataset_df=df,
        excluded_df=excluded_df,
    )


def run_prediction_stage(runtime: PreparedRuntime) -> PredictionBatch:
    rows: list[PredictionRow] = []
    pred_no_rag: list[str] = []
    pred_rag: list[str] = []
    coerced_no_rag = 0
    coerced_rag = 0
    none_label = runtime.label_set.none_label

    for _, row in runtime.dataset_df.iterrows():
        text = str(row["chart_text"])
        case_id = str(row["case_id"])
        gold_label = str(row["gold_label"])
        print(f"Evaluating case_id={case_id}")

        p0, ddx0, evidence0, did0 = _predict_case(
            runtime.llm,
            text=text,
            rag_context=None,
            allowed=runtime.allowed,
            none_label=none_label,
        )
        coerced_no_rag += did0
        pred_no_rag.append(p0)

        ctx = build_rag_context(
            text,
            runtime.retriever,
            runtime.config,
            top_k=runtime.cfg.top_k,
            max_chars=runtime.cfg.max_context_chars,
        )
        p1, ddx1, evidence1, did1 = _predict_case(
            runtime.llm,
            text=text,
            rag_context=ctx,
            allowed=runtime.allowed,
            none_label=none_label,
        )
        coerced_rag += did1
        pred_rag.append(p1)

        rows.append(
            PredictionRow(
                case_id=case_id,
                gold_label=gold_label,
                pred_no_rag=p0,
                pred_rag=p1,
                ddx_no_rag=ddx0,
                ddx_rag=ddx1,
                evidence_no_rag=evidence0,
                evidence_rag=evidence1,
            )
        )

    gold = [str(x) for x in runtime.dataset_df["gold_label"].tolist()]
    return PredictionBatch(
        rows=rows,
        gold=gold,
        pred_no_rag=pred_no_rag,
        pred_rag=pred_rag,
        coerced_no_rag=coerced_no_rag,
        coerced_rag=coerced_rag,
    )


def assemble_results(runtime: PreparedRuntime, batch: PredictionBatch) -> EvaluationResultsDocument:
    agreement_no_rag = _exact_agreement(batch.gold, batch.pred_no_rag)
    agreement_rag = _exact_agreement(batch.gold, batch.pred_rag)
    ttest = _paired_ttest_correctness(batch.gold, batch.pred_no_rag, batch.pred_rag)

    excluded_gold_rows = None
    excluded_gold_label_counts = None
    if len(runtime.excluded_df) > 0:
        excluded_gold_rows = int(len(runtime.excluded_df))
        excluded_gold_label_counts = (
            runtime.excluded_df["gold_label"].astype(str).value_counts().to_dict()
        )

    run = RunSection(
        timestamp=_now_iso(),
        git_commit=_git_commit_hash(),
        model_backend=runtime.llm.backend,
        model_name=_model_name_for_backend(runtime.llm),
        k=runtime.cfg.top_k,
        seed=runtime.cfg.seed,
        n_cases=int(len(runtime.dataset_df)),
        ontology_key=runtime.cfg.ontology_key,
        dataset_path=str(runtime.cfg.dataset_path),
        label_set_path=str(runtime.cfg.label_set_path),
        corpus_path=str(runtime.cfg.corpus_path),
        retriever_cache_dir=str(runtime.cfg.retriever_cache_dir),
        retriever_model_cache_dir=str(runtime.model_cache_dir),
        embedding_model=runtime.embedding_model,
        retrieval_mode="embeddings" if runtime.cfg.prefer_embeddings else "tfidf",
        excluded_gold_rows=excluded_gold_rows,
        excluded_gold_label_counts=excluded_gold_label_counts,
    )
    metrics = MetricsSection(
        agreement_no_rag=agreement_no_rag,
        agreement_rag=agreement_rag,
        n_cases=int(len(runtime.dataset_df)),
    )
    coercions = CoercionsSection(
        coerced_invalid_labels_no_rag=batch.coerced_no_rag,
        coerced_invalid_labels_rag=batch.coerced_rag,
    )
    return EvaluationResultsDocument(
        run=run,
        metrics=metrics,
        inferential={"paired_ttest_correctness": ttest},
        coercions=coercions,
    )

