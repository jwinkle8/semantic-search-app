#!/usr/bin/env python3
"""Precompute chunk embeddings, and load them into an in-memory Chroma index at app startup.

Embeddings are computed once, offline, and committed next to the chunk file:

    data/nps_embeddings.npy    float32 array, one normalized 384-d vector per chunk, in chunk order
    data/nps_embeddings.json   which model produced it and the SHA-256 of the chunk file it matches

At startup the app calls build_collection(), which loads those vectors into Chroma instead of
re-embedding 11k chunks on a small cloud CPU (see "Design decisions" in README.md). If the
saved embeddings don't match the current chunk file or model, it warns and falls back to
embedding everything, so a stale file can make startup slow but never wrong.

Both chunks and queries are embedded with Chroma's built-in ONNX build of all-MiniLM-L6-v2,
so there is a single embedding implementation and no PyTorch dependency. Its 79 MB model is
downloaded to ~/.cache/chroma on first use.

    python build_index.py --embed   # recompute and save embeddings (after rebuilding the chunks)
    python build_index.py           # simulate app startup: load saved embeddings, time it, smoke-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

ROOT = Path(__file__).resolve().parent
CHUNKS_PATH = ROOT / "data" / "nps_chunks.json"
EMBEDDINGS_PATH = ROOT / "data" / "nps_embeddings.npy"
EMBEDDINGS_META_PATH = ROOT / "data" / "nps_embeddings.json"
COLLECTION = "national_parks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2 (ONNX, chromadb ONNXMiniLM_L6_V2)"
EMBED_BATCH = 256


def get_embedder() -> ONNXMiniLM_L6_V2:
    # Pin the CPU provider so a Mac (which could pick CoreML) computes exactly what the cloud does.
    return ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])


def embed(texts: list[str], embedder: ONNXMiniLM_L6_V2) -> np.ndarray:
    """Embed texts as L2-normalized float32 rows, so cosine similarity == dot product."""
    parts = [np.asarray(embedder(texts[i:i + EMBED_BATCH]), dtype=np.float32)
             for i in range(0, len(texts), EMBED_BATCH)]
    vectors = np.vstack(parts)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _load_chunks(chunks_path: Path) -> tuple[dict, str]:
    raw = Path(chunks_path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def save_embeddings(chunks_path: Path = CHUNKS_PATH) -> None:
    data, chunks_sha = _load_chunks(chunks_path)
    start = time.time()
    vectors = embed(data["texts"], get_embedder())
    print(f"Embedded {len(vectors)} chunks with {EMBEDDING_MODEL} in {time.time() - start:.0f}s")
    np.save(EMBEDDINGS_PATH, vectors)
    EMBEDDINGS_META_PATH.write_text(json.dumps({
        "embedding_model": EMBEDDING_MODEL,
        "chromadb_version": chromadb.__version__,
        "shape": list(vectors.shape),
        "dtype": str(vectors.dtype),
        "normalized": True,
        "chunks_file": str(Path(chunks_path).relative_to(ROOT)),
        "chunks_sha256": chunks_sha,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    print(f"Saved {EMBEDDINGS_PATH.relative_to(ROOT)} ({EMBEDDINGS_PATH.stat().st_size / 1e6:.1f} MB) "
          f"and {EMBEDDINGS_META_PATH.relative_to(ROOT)}")


def load_embeddings(chunks_sha: str, num_chunks: int) -> np.ndarray | None:
    """Return the saved vectors if they match this chunk file and model; otherwise None."""
    if not (EMBEDDINGS_PATH.exists() and EMBEDDINGS_META_PATH.exists()):
        problem = "no saved embeddings"
    else:
        meta = json.loads(EMBEDDINGS_META_PATH.read_text())
        if meta["chunks_sha256"] != chunks_sha:
            problem = "saved embeddings were built from a different chunk file"
        elif meta["embedding_model"] != EMBEDDING_MODEL:
            problem = f"saved embeddings use {meta['embedding_model']!r}, not {EMBEDDING_MODEL!r}"
        else:
            vectors = np.load(EMBEDDINGS_PATH)
            if vectors.shape[0] == num_chunks:
                return vectors
            problem = f"saved embeddings have {vectors.shape[0]} rows for {num_chunks} chunks"
    print(f"WARNING: {problem}; embedding all chunks now (slow). "
          "Run `python build_index.py --embed` and commit the result.")
    return None


def build_collection(chunks_path: Path = CHUNKS_PATH):
    """Load chunks and their embeddings into an in-memory Chroma collection.

    Returns (collection, embedder). Embed queries with the same embedder:
    embed([query], embedder).
    """
    data, chunks_sha = _load_chunks(chunks_path)
    ids, texts, metadatas = data["ids"], data["texts"], data["metadatas"]

    embedder = get_embedder()  # still needed to embed queries
    embeddings = load_embeddings(chunks_sha, len(ids))
    source = "precomputed"
    if embeddings is None:
        embeddings, source = embed(texts, embedder), "embedded at startup"

    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    # Ephemeral clients share state within a process; start clean if this runs twice.
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(
        COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},  # distance = 1 - cosine similarity
        embedding_function=None,  # we supply our own vectors
        metadata={
            "embedding_model": EMBEDDING_MODEL,
            "embeddings_source": source,
            "num_chunks": len(ids),
            "chunks_sha256": chunks_sha,
            "chunks_built_at": data["info"]["built_at"],
            "index_built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    batch = client.get_max_batch_size()
    for i in range(0, len(ids), batch):
        collection.add(ids=ids[i:i + batch], documents=texts[i:i + batch],
                       metadatas=metadatas[i:i + batch], embeddings=embeddings[i:i + batch])
    assert collection.count() == len(ids), "stored count does not match chunk file"
    return collection, embedder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embed", action="store_true",
                        help="recompute and save embeddings for the current chunk file")
    args = parser.parse_args()
    if args.embed:
        save_embeddings()

    start = time.time()
    collection, embedder = build_collection()
    print(f"Index ready: {collection.count()} chunks ({collection.metadata['embeddings_source']}) "
          f"in {time.time() - start:.1f}s (chunks sha256 {collection.metadata['chunks_sha256'][:12]}…)")

    for query in ["kid-friendly short hike to a waterfall",
                  "quiet place to see the stars away from city lights"]:
        res = collection.query(query_embeddings=embed([query], embedder), n_results=3)
        print(f"\nSmoke test: {query!r}")
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            print(f"  {1 - dist:.2f}  {meta['title']} ({meta.get('park_name') or 'no park'})")


if __name__ == "__main__":
    main()
