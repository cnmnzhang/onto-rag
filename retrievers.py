"""
Retrievers for ontology document lookup.
"""

from typing import Dict, List

import numpy as np


class EmbeddingRetriever:
    def __init__(self, corpus: List[Dict], top_k: int = 3, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        self.corpus = corpus
        self.top_k = top_k
        self._cosine_similarity = cosine_similarity
        print(f"Loading sentence transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.documents = [doc["document_text"] for doc in corpus]
        print("Encoding corpus documents...")
        self.doc_embeddings = self.model.encode(self.documents, convert_to_tensor=False)
        print(f"✓ Encoded {len(self.documents)} documents")

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
        self.documents = [doc["document_text"] for doc in corpus]
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


def create_retriever(corpus: List[Dict], top_k: int = 3, prefer_embeddings: bool = True):
    """Create an embedding or TF-IDF retriever based on availability."""
    if prefer_embeddings:
        try:
            import sentence_transformers  # noqa: F401
            print("✓ Using sentence-transformers for retrieval")
            return EmbeddingRetriever(corpus, top_k=top_k)
        except Exception:
            pass

    print("✓ Using TF-IDF for retrieval (sentence-transformers not available)")
    return TFIDFRetriever(corpus, top_k=top_k)
