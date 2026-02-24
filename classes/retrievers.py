"""Retrievers for ontology document lookup.

Primary path: sentence-transformers embeddings (optionally FAISS).
Fallback: scikit-learn TF-IDF cosine.

Embeddings/FAISS are optional dependencies. When available and a `cache_dir`
is provided, we cache embeddings/index to speed up reruns.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import hashlib
import json
from pathlib import Path

import numpy as np


class FaissRetriever:
    def __init__(
        self,
        corpus: List[Dict],
        top_k: int = 3,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        cache_dir: Optional[str] = None,
        cache_key: Optional[str] = None,
    ):
        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer

        self.corpus = corpus
        self.top_k = top_k
        print(f"Loading sentence transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.documents = [_doc_text(doc) for doc in corpus]
        cache_paths = _faiss_cache_paths(cache_dir, cache_key, model_name)

        if cache_paths is not None:
            index_path, meta_path = cache_paths
            if index_path.exists() and meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    if meta.get("n_docs") == len(self.documents):
                        self.index = faiss.read_index(str(index_path))
                        print(f"✓ Loaded FAISS index cache: {index_path}")
                        return
                except Exception:
                    pass

        print("Encoding corpus documents...")
        embeddings = self.model.encode(self.documents, convert_to_tensor=False).astype("float32")
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"✓ Encoded and indexed {len(self.documents)} documents")

        if cache_paths is not None:
            index_path, meta_path = cache_paths
            try:
                index_path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.index, str(index_path))
                meta_path.write_text(json.dumps({"n_docs": len(self.documents), "model": model_name}))
                print(f"✓ Saved FAISS index cache: {index_path}")
            except Exception:
                pass

    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve top-k most similar documents using FAISS."""
        import faiss  # type: ignore

        query_embedding = self.model.encode([query], convert_to_tensor=False).astype("float32")
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, self.top_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            results.append({
                **self.corpus[idx],
                "similarity_score": float(score)
            })
        return results


class EmbeddingRetriever:
    def __init__(
        self,
        corpus: List[Dict],
        top_k: int = 3,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        cache_dir: Optional[str] = None,
        cache_key: Optional[str] = None,
    ):
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        self.corpus = corpus
        self.top_k = top_k
        self._cosine_similarity = cosine_similarity
        print(f"Loading sentence transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.documents = [_doc_text(doc) for doc in corpus]

        cache_paths = _embedding_cache_paths(cache_dir, cache_key, model_name)
        if cache_paths is not None:
            emb_path, meta_path = cache_paths
            if emb_path.exists() and meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    if meta.get("n_docs") == len(self.documents):
                        self.doc_embeddings = np.load(str(emb_path))
                        print(f"✓ Loaded embedding cache: {emb_path}")
                        return
                except Exception:
                    pass

        print("Encoding corpus documents...")
        self.doc_embeddings = self.model.encode(self.documents, convert_to_tensor=False)
        print(f"✓ Encoded {len(self.documents)} documents")

        if cache_paths is not None:
            emb_path, meta_path = cache_paths
            try:
                emb_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(emb_path), self.doc_embeddings)
                meta_path.write_text(json.dumps({"n_docs": len(self.documents), "model": model_name}))
                print(f"✓ Saved embedding cache: {emb_path}")
            except Exception:
                pass

    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve top-k most similar documents using embeddings."""
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        similarities = self._cosine_similarity(
            query_embedding.reshape(1, -1),
            self.doc_embeddings
        )[0]
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                **self.corpus[idx],
                "similarity_score": float(similarities[idx])
            })
        return results


class TFIDFRetriever:
    def __init__(self, corpus: List[Dict], top_k: int = 3):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self.corpus = corpus
        self.top_k = top_k
        self._cosine_similarity = cosine_similarity
        self.documents = [_doc_text(doc) for doc in corpus]
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        print("Vectorizing corpus with TF-IDF...")
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)
        print(f"✓ Vectorized {len(self.documents)} documents")

    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve top-k most similar documents."""
        query_vector = self.vectorizer.transform([query])
        similarities = self._cosine_similarity(query_vector, self.doc_vectors)[0]
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                **self.corpus[idx],
                "similarity_score": float(similarities[idx])
            })
        return results


def create_retriever(
    corpus: List[Dict],
    top_k: int = 3,
    prefer_embeddings: bool = True,
    *,
    cache_dir: Optional[str] = None,
    model_name: str = "all-MiniLM-L6-v2",
):
    """Create an embedding or TF-IDF retriever based on availability.

    If `cache_dir` is provided, embeddings (and FAISS index when available) are
    cached to speed up reruns.
    """

    cache_key = _corpus_fingerprint(corpus)
    if prefer_embeddings:
        try:
            import faiss  # type: ignore  # noqa: F401
            import sentence_transformers  # noqa: F401
            print("✓ Using FAISS + sentence-transformers for retrieval")
            return FaissRetriever(
                corpus,
                top_k=top_k,
                model_name=model_name,
                cache_dir=cache_dir,
                cache_key=cache_key,
            )
        except Exception:
            pass
        try:
            import sentence_transformers  # noqa: F401
            print("✓ Using sentence-transformers for retrieval")
            return EmbeddingRetriever(
                corpus,
                top_k=top_k,
                model_name=model_name,
                cache_dir=cache_dir,
                cache_key=cache_key,
            )
        except Exception:
            pass

    print("✓ Using TF-IDF for retrieval (sentence-transformers not available)")
    return TFIDFRetriever(corpus, top_k=top_k)


def _doc_text(doc: Dict) -> str:
    return str(doc.get("text") or doc.get("document_text") or "")


def _corpus_fingerprint(corpus: List[Dict]) -> str:
    h = hashlib.md5()
    for doc in corpus:
        doc_id = str(doc.get("tco_id") or doc.get("doc_id") or doc.get("id") or "")
        h.update(doc_id.encode("utf-8"))
        h.update(b"\n")
        h.update(_doc_text(doc).encode("utf-8"))
        h.update(b"\n\n")
    return h.hexdigest()


def _embedding_cache_paths(cache_dir: Optional[str], cache_key: Optional[str], model_name: str):
    if not cache_dir or not cache_key:
        return None
    base = Path(cache_dir)
    model_safe = model_name.replace("/", "-")
    emb_path = base / f"emb_{cache_key}_{model_safe}.npy"
    meta_path = base / f"emb_{cache_key}_{model_safe}.json"
    return emb_path, meta_path


def _faiss_cache_paths(cache_dir: Optional[str], cache_key: Optional[str], model_name: str):
    if not cache_dir or not cache_key:
        return None
    base = Path(cache_dir)
    model_safe = model_name.replace("/", "-")
    index_path = base / f"faiss_{cache_key}_{model_safe}.index"
    meta_path = base / f"faiss_{cache_key}_{model_safe}.json"
    return index_path, meta_path
