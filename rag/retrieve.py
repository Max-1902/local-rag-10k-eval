#!/usr/bin/env python3
"""Query a Chroma collection and return the top-k matching chunks, using
hybrid search: dense embedding similarity fused with BM25 keyword scoring
via reciprocal rank fusion. Keyword scoring helps on queries naming a
specific line item or table (e.g. "other income (expense), net") where
small dense embedders struggle to distinguish similar financial language."""

import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
STORE_DIR = Path(__file__).parent / "store"
RRF_K = 60  # reciprocal rank fusion constant

_model = None
_client = None
_bm25_cache: dict[str, tuple] = {}  # collection -> (BM25Okapi, ids, documents, metadatas)


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(STORE_DIR))
    return _client


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_bm25(collection: str):
    if collection not in _bm25_cache:
        client = _get_client()
        coll = client.get_collection(collection)
        all_docs = coll.get()
        ids = all_docs["ids"]
        documents = all_docs["documents"]
        metadatas = all_docs["metadatas"]
        tokenized = [_tokenize(d) for d in documents]
        bm25 = BM25Okapi(tokenized)
        _bm25_cache[collection] = (bm25, ids, documents, metadatas)
    return _bm25_cache[collection]


def retrieve(query: str, collection: str, k: int = 5, pool: int = 20) -> list[dict]:
    """Hybrid retrieval: fetch `pool` candidates from each of dense search and
    BM25, fuse their rankings with reciprocal rank fusion, return top-k."""
    model = _get_model()
    client = _get_client()
    coll = client.get_collection(collection)

    # dense ranking
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    dense_results = coll.query(query_embeddings=query_embedding, n_results=pool)
    dense_ids = dense_results["ids"][0]

    # BM25 ranking
    bm25, all_ids, all_docs, all_metas = _get_bm25(collection)
    doc_by_id = dict(zip(all_ids, all_docs))
    meta_by_id = dict(zip(all_ids, all_metas))
    scores = bm25.get_scores(_tokenize(query))
    bm25_ranked = sorted(zip(all_ids, scores), key=lambda x: x[1], reverse=True)[:pool]
    bm25_ids = [doc_id for doc_id, _ in bm25_ranked]

    # reciprocal rank fusion
    fused_scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, doc_id in enumerate(bm25_ids):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)

    top_ids = sorted(fused_scores.keys(), key=lambda i: fused_scores[i], reverse=True)[:k]

    hits = []
    for doc_id in top_ids:
        meta = meta_by_id[doc_id]
        hits.append(
            {
                "id": doc_id,
                "text": doc_by_id[doc_id],
                "page": meta.get("page"),
                "source": meta.get("source"),
                "score": fused_scores[doc_id],
            }
        )
    return hits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query a Chroma collection")
    parser.add_argument("query")
    parser.add_argument("--collection", required=True)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    for hit in retrieve(args.query, args.collection, args.k):
        print(f"[page {hit['page']}] score={hit['score']:.4f}")
        print(hit["text"][:300].replace("\n", " "))
        print()
