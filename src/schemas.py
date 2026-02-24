"""Typed artifact schemas used by evaluation and data-generation scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class LabelSet:
    labels: tuple[str, ...]
    none_label: str = "NONE"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelSet":
        raw_labels = [str(x).strip() for x in (payload.get("labels") or []) if str(x).strip()]
        if not raw_labels:
            raise ValueError("label_set.labels is empty")
        if len(set(raw_labels)) != len(raw_labels):
            raise ValueError("label_set.labels contains duplicates")
        none_label = str(payload.get("none_label") or "NONE").strip() or "NONE"
        return cls(labels=tuple(raw_labels), none_label=none_label)

    @classmethod
    def from_path(cls, path: str | Path) -> "LabelSet":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Label set JSON must be an object: {path}")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "none_label": self.none_label,
        }

    def allowed_set(self) -> set[str]:
        return set(self.labels) | {self.none_label}


@dataclass(frozen=True)
class PredictionRow:
    case_id: str
    gold_label: str
    pred_no_rag: str
    pred_rag: str
    ddx_no_rag: str
    ddx_rag: str
    evidence_no_rag: str
    evidence_rag: str

    def to_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "gold_label": self.gold_label,
            "pred_no_rag": self.pred_no_rag,
            "pred_rag": self.pred_rag,
            "ddx_no_rag": self.ddx_no_rag,
            "ddx_rag": self.ddx_rag,
            "evidence_no_rag": self.evidence_no_rag,
            "evidence_rag": self.evidence_rag,
        }


def predictions_to_frame(rows: list[PredictionRow]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in rows])


@dataclass(frozen=True)
class RunSection:
    timestamp: str
    git_commit: str | None
    model_backend: str
    model_name: str
    k: int
    seed: int
    n_cases: int
    ontology_key: str
    dataset_path: str
    label_set_path: str
    corpus_path: str
    retriever_cache_dir: str
    retriever_model_cache_dir: str
    embedding_model: str
    retrieval_mode: str
    excluded_gold_rows: int | None = None
    excluded_gold_label_counts: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "model_backend": self.model_backend,
            "model_name": self.model_name,
            "k": int(self.k),
            "seed": int(self.seed),
            "n_cases": int(self.n_cases),
            "ontology_key": self.ontology_key,
            "dataset_path": self.dataset_path,
            "label_set_path": self.label_set_path,
            "corpus_path": self.corpus_path,
            "retriever_cache_dir": self.retriever_cache_dir,
            "retriever_model_cache_dir": self.retriever_model_cache_dir,
            "embedding_model": self.embedding_model,
            "retrieval_mode": self.retrieval_mode,
        }
        if self.excluded_gold_rows is not None:
            out["excluded_gold_rows"] = int(self.excluded_gold_rows)
        if self.excluded_gold_label_counts is not None:
            out["excluded_gold_label_counts"] = dict(self.excluded_gold_label_counts)
        return out


@dataclass(frozen=True)
class MetricsSection:
    agreement_no_rag: float
    agreement_rag: float
    n_cases: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "agreement_no_rag": float(self.agreement_no_rag),
            "agreement_rag": float(self.agreement_rag),
            "n_cases": int(self.n_cases),
        }


@dataclass(frozen=True)
class CoercionsSection:
    coerced_invalid_labels_no_rag: int
    coerced_invalid_labels_rag: int

    def to_dict(self) -> dict[str, int]:
        return {
            "coerced_invalid_labels_no_rag": int(self.coerced_invalid_labels_no_rag),
            "coerced_invalid_labels_rag": int(self.coerced_invalid_labels_rag),
        }


@dataclass(frozen=True)
class EvaluationResultsDocument:
    run: RunSection
    metrics: MetricsSection
    inferential: dict[str, Any]
    coercions: CoercionsSection

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run.to_dict(),
            "metrics": self.metrics.to_dict(),
            "inferential": dict(self.inferential),
            "coercions": self.coercions.to_dict(),
        }

