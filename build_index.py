#!/usr/bin/env python3
"""Build the Chroma search index in memory from data/nps_chunks.json.

The Hugging Face Space calls build_collection() once at startup, so no database files are
committed to the repo (see "Design decisions -> Index storage" in README.md for the trade-off).

Run directly to time a full build and smoke-test it, the same way the Space will:

    python build_index.py

The collection's metadata records which embedding model and which chunk file (by SHA-256)
produced it, so a running app can report exactly which data version it is serving.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
CHUNKS_PATH = ROOT / "data" / "nps_chunks.json"
COLLECTION = "national_parks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENCODE_BATCH = 64


def build_collection(chunks_path: Path = CHUNKS_PATH,
                     model: SentenceTransformer | None = None):
    """Embed every chunk and load it into an in-memory Chroma collection.

    Returns (collection, model). Queries must be embedded with this same model,
    normalized, e.g. model.encode([query], normalize_embeddings=True).
    """
    raw = Path(chunks_path).read_bytes()
    data = json.loads(raw)
    ids, texts, metadatas = data["ids"], data["texts"], data["metadatas"]

    model = model or SentenceTransformer(EMBEDDING_MODEL)
    start = time.time()
    # Normalized vectors: cosine similarity == dot product, and distances are comparable.
    embeddings = model.encode(texts, batch_size=ENCODE_BATCH, normalize_embeddings=True,
                              show_progress_bar=True)
    print(f"Embedded {len(texts)} chunks with {EMBEDDING_MODEL} in {time.time() - start:.0f}s")

    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    # Ephemeral clients share state within a process; start clean if this runs twice.
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(
        COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},  # distance = 1 - cosine similarity
        embedding_function=None,  # we supply our own vectors; don't download Chroma's default model
        metadata={
            "embedding_model": EMBEDDING_MODEL,
            "num_chunks": len(ids),
            "chunks_sha256": hashlib.sha256(raw).hexdigest(),
            "chunks_built_at": data["info"]["built_at"],
            "index_built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    batch = client.get_max_batch_size()
    for i in range(0, len(ids), batch):
        collection.add(ids=ids[i:i + batch], documents=texts[i:i + batch],
                       metadatas=metadatas[i:i + batch],
                       embeddings=embeddings[i:i + batch].tolist())
    assert collection.count() == len(ids), "stored count does not match chunk file"
    return collection, model


def main() -> None:
    start = time.time()
    collection, model = build_collection()
    print(f"Index ready: {collection.count()} chunks in {time.time() - start:.0f}s total "
          f"(chunks sha256 {collection.metadata['chunks_sha256'][:12]}…)")

    for query in ["kid-friendly short hike to a waterfall",
                  "quiet place to see the stars away from city lights"]:
        res = collection.query(query_embeddings=model.encode([query], normalize_embeddings=True).tolist(),
                               n_results=3)
        print(f"\nSmoke test: {query!r}")
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            print(f"  {1 - dist:.2f}  {meta['title']} ({meta.get('park_name') or 'no park'})")


if __name__ == "__main__":
    main()
