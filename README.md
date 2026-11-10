# National Parks Semantic Search

Search U.S. National Park Service parks, things to do, and campgrounds by describing what you
want, such as "kid-friendly short hike to a waterfall", and filter by state or result type.

Source code and the full data pipeline: https://github.com/jwinkle8/semantic-search-app

## How it works

1. **Extract:** [build_dataset.py](build_dataset.py) downloads parks, things to do, and campgrounds
   from the [NPS Data API](https://www.nps.gov/subjects/developer/) (public domain) and caches
   each raw page under `data/raw/`.
2. **Transform and chunk:** it strips HTML, joins each record to its park, and splits text into
   chunks of up to 200 words with a context header such as
   `Yosemite National Park (CA) | Thing to do: …`. It then runs data-quality checks and writes
   `data/nps_chunks.json` (11,043 chunks).
3. **Embed and index:** [build_index.py](build_index.py)` --embed` precomputes one vector per
   chunk with `all-MiniLM-L6-v2` (Chroma's built-in ONNX version) and saves them to
   `data/nps_embeddings.npy`. At startup the app loads those vectors into an in-memory Chroma
   collection (cosine similarity) in about 6 s.
4. **Search:** [streamlit_app.py](streamlit_app.py) is a Streamlit app that embeds the query,
   filters by state and result type, and shows one result per place or activity. The index is
   cached with `st.cache_resource`, so it loads once per server and is shared by all visitors.

## Run locally

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run streamlit_app.py   # serves http://localhost:8501; the index loads (~10 s) on first visit

# Optional: rebuild the dataset from the NPS API (free key: https://www.nps.gov/subjects/developer/get-started.htm),
# then recompute the embeddings to match it (~5 min on CPU) and commit both
NPS_API_KEY=... .venv/bin/python build_dataset.py
.venv/bin/python build_index.py --embed
```

## Deploying

The app runs on [Streamlit Community Cloud](https://share.streamlit.io), which deploys straight
from this GitHub repository:

1. Sign in at https://share.streamlit.io with GitHub and click **Create app**, then
   **Yup, I have an app**.
2. Repository `jwinkle8/semantic-search-app`, branch `main`, main file path `streamlit_app.py`.
3. Under **Advanced settings**, set **Python version** to **3.12** (the version the pinned
   requirements were tested with), then deploy.

Community Cloud installs `requirements.txt` and redeploys when the branch changes, so every
deployed version is a commit on `main`. Apps sleep after 12 hours without traffic; the next
visitor wakes them.

The app was first built for Hugging Face Spaces with Gradio. Hugging Face now requires a paid
plan to host Gradio Spaces, so with the instructor's approval it moved to Streamlit Community
Cloud. The Gradio version is in the Git history.

## Design decisions

### Index storage: commit precomputed embeddings, build the Chroma index at startup

**Decision.** Chunk embeddings are computed once, offline, and committed as
`data/nps_embeddings.npy` (11,043 × 384 float32, 17 MB). A small `data/nps_embeddings.json`
records the model and the SHA-256 of the chunk file they were computed from. At startup,
[build_index.py](build_index.py) loads the vectors into an in-memory Chroma collection; no
database files are committed.

**History.** The first version embedded every chunk at startup, because Hugging Face Spaces
required Git LFS for any file over 10 MB. When the app moved to Streamlit Community Cloud
(Hugging Face now requires a paid plan for Gradio Spaces), that limit went away, and the new
host's small resource allowance (0.078 to 2 CPU cores, 690 MB to 2.7 GB of memory) made
embedding at startup a liability.

| | Commit prebuilt `chromadb/` | Embed at startup | Precomputed embeddings (chosen) |
|---|---|---|---|
| Repo footprint | 142 MB index (`chroma.sqlite3` is 116 MB, 98 MB after `chroma vacuum`) | Nothing beyond the chunk file | +17 MB vectors |
| Git hosting | Over GitHub's 100 MB per-file limit, so it needs Git LFS, and every rebuild adds another ~110 MB version | Plain Git | Plain Git |
| Startup on the dev laptop (Apple Silicon Mac, CPU) | Seconds | 83 s (75 s of it embedding) | ~6 s |
| Consistency | The index can drift from the chunk file if someone forgets to rebuild | Always rebuilt from the chunk file | Checked at startup: if the vectors don't match the chunk file's SHA-256 or the model, the app warns and re-embeds, so a stale file is slow but never wrong |

**Why a prebuilt index is so large.** The vectors themselves are only about 17 MB. Local Chroma
(`chromadb` 1.5.9) stores each chunk's text about four times in SQLite: as a metadata row, in a
string index, and in both tables of a full-text keyword index. Turning off the full-text index
through Chroma's `Schema` had no effect in local mode.

**What it costs.** Changing the data takes one extra step: after rebuilding the chunks, run
`build_index.py --embed` (~5 min on CPU) and commit the new vectors with them.

**Versioning.** The chunk file and its vectors are versioned together in Git. The collection's
metadata records the model, the chunk file's SHA-256, and whether the vectors were precomputed,
so the running app can report exactly which data version it is serving.

### Embedding runtime: Chroma's ONNX model instead of PyTorch

**Decision.** Both chunks and queries are embedded with Chroma's built-in ONNX version of
`all-MiniLM-L6-v2` (`ONNXMiniLM_L6_V2`, pinned to the CPU), not with sentence-transformers and
PyTorch. It's the same model in a different runtime.

**Why.** Even with precomputed vectors, the app needed the model to embed each query, and
importing PyTorch and sentence-transformers alone took ~400 MB of memory. Measured on the dev
laptop:

| | sentence-transformers + PyTorch | Chroma ONNX (chosen) |
|---|---|---|
| Peak memory: load index + 2 queries | ~920 MB | 627 MB |
| Vectors | (baseline) | Identical: minimum cosine similarity 1.000000 across all 11,043 chunks |
| Precomputing all embeddings (offline) | 77 s | 274 s |
| Dependencies | torch (555 MB Linux wheel, plus NVIDIA libraries unless the CPU-only build is used), sentence-transformers | None beyond `chromadb` |

**What it costs.** Offline embedding is about 3.5× slower, which only matters when the data
changes. The 79 MB ONNX model downloads to `~/.cache/chroma` the first time the app starts on a
fresh machine (~13 s on the dev laptop).
