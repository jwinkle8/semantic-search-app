# National Parks Semantic Search

Search U.S. National Park Service parks, things to do, and campgrounds by describing what you
want, such as "kid-friendly short hike to a waterfall", and filter by state or result type.

**Live app: https://nps-semantic-search.streamlit.app/**

## How it works

1. **Extract:** [build_dataset.py](build_dataset.py) downloads parks, things to do, and campgrounds
   from the [NPS Data API](https://www.nps.gov/subjects/developer/) (public domain) and caches
   each raw page under `data/raw/`.
2. **Transform and chunk:** it strips HTML, joins each record to its park, and splits text into
   chunks of up to 200 words with a context header such as
   `Yosemite National Park (CA) | Thing to do: …`. It then runs data-quality checks and writes
   `data/nps_chunks.json` (11,043 chunks).
3. **Embed and index:** [build_index.py](build_index.py)` --embed` precomputes one vector per
   chunk with `all-MiniLM-L6-v2` and saves them to
   `data/nps_embeddings.npy`. At startup the app loads those vectors into an in-memory Chroma
   collection (cosine similarity) in about 6 s.
4. **Search:** [streamlit_app.py](streamlit_app.py) is a Streamlit app that embeds the query,
   filters by state and result type, and shows one result per place or activity. The index is
   cached with `st.cache_resource`, so it loads once per server and is shared by all visitors.

## Run locally

```bash
# Optional: rebuild the dataset from the NPS API (free key: https://www.nps.gov/subjects/developer/get-started.htm),
# then recompute the embeddings to match it (~5 min on CPU) and commit both
NPS_API_KEY=... .venv/bin/python build_dataset.py
.venv/bin/python build_index.py --embed

python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

# serve locally at http://localhost:8501
# index may take ~10s to load on startup
.venv/bin/streamlit run streamlit_app.py
```

## Deploying

The app runs on [Streamlit Community Cloud](https://share.streamlit.io), which deploys straight
from this GitHub repository.


Community Cloud installs `requirements.txt` and redeploys when the branch changes, so every
deployed version is a commit on `main`. The app sleeps after 12 hours without traffic and the next
visitor wakes it.

## Design decisions

### Index storage: commit  embeddings, build index at startup
---

Chunk embeddings are computed once, offline, and committed as
`data/nps_embeddings.npy` (11,043 × 384 float32, 17 MB). A small `data/nps_embeddings.json`
records the model and the SHA-256 of the chunk file they were computed from. At startup,
[build_index.py](build_index.py) loads the vectors into an in-memory Chroma collection; no
database files are committed.

The first version embedded every chunk at startup, because Hugging Face Spaces
required Git LFS for any file over 10 MB. When the app moved to Streamlit Community Cloud instead of Hugging Face Spaces, that limit went away, and the new
host's small resource allowance (0.078 to 2 CPU cores, 690 MB to 2.7 GB of memory) made
embedding at startup a liability.

To change the data, run
`build_index.py --embed` (~5 min on CPU) and commit the new vectors with them.

The chunk file and its vectors are versioned together in Git. The collection's
metadata records the model, the chunk file's SHA-256, and whether the vectors were precomputed,
so the running app can report exactly which data version it is serving.

### Embedding runtime: Chroma's ONNX model instead of PyTorch
---

Both chunks and queries are embedded with Chroma's built-in ONNX version of
`all-MiniLM-L6-v2` (`ONNXMiniLM_L6_V2`, pinned to the CPU), not with sentence-transformers and
PyTorch. It's the same model in a different runtime.

Even with precomputed vectors, the app needed the model to embed each query, and
importing PyTorch and sentence-transformers alone took ~400 MB of memory.

Offline embedding is about 3.5× slower, which only matters when the data
changes. The 79 MB ONNX model downloads to `~/.cache/chroma` the first time the app starts on a
fresh machine (~13 s on the dev laptop).
