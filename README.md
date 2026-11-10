---
title: National Parks Semantic Search
emoji: 🏞️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.10"
app_file: app.py
pinned: false
short_description: Search U.S. national parks by meaning, not keywords
---

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
3. **Index:** at startup, [build_index.py](build_index.py) embeds the chunks with
   `all-MiniLM-L6-v2` into an in-memory Chroma collection (cosine similarity).
4. **Search:** [app.py](app.py) is a Gradio app that embeds the query, filters by state and
   result type, and shows one result per place or activity.

## Run locally

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python app.py                 # builds the index (~80 s), then serves http://127.0.0.1:7860

# Optional: rebuild the dataset from the NPS API (free key: https://www.nps.gov/subjects/developer/get-started.htm)
NPS_API_KEY=... .venv/bin/python build_dataset.py
```

## Deploying

```bash
.venv/bin/hf auth login   # once, with a Hugging Face token that has write access
./deploy.sh
```

[deploy.sh](deploy.sh) uploads only the files the app needs with `hf upload`, creating the Space
on first run, and labels each Space commit with the GitHub commit it came from. It refuses to
deploy uncommitted changes. `git push` to the Hub isn't used because the Git history contains
earlier versions of the 16 MB chunk file, which is over Hugging Face's 10 MB limit for plain Git
files. `hf upload` stores large files in Hugging Face's Xet storage automatically.

## Design decisions

### Index storage: build the Chroma index at startup

**Decision.** The app builds its Chroma index in memory when it starts
([build_index.py](build_index.py)): it embeds every chunk in `data/nps_chunks.json` with
`all-MiniLM-L6-v2` and loads them into an in-memory collection. No database files are committed.

**Alternative considered: commit a prebuilt `chromadb/` folder.** This is what the starter
notebook suggests, and it makes the app start instantly. It was rejected because of size:

| | Commit prebuilt `chromadb/` | Build at startup (chosen) |
|---|---|---|
| Repo footprint | 142 MB index (`chroma.sqlite3` is 116 MB, 98 MB after `chroma vacuum`) | Nothing beyond the chunk file |
| Git hosting | Over GitHub's 100 MB per-file limit, and Hugging Face requires Git LFS for files over 10 MB, so it needs Git LFS on both, plus LFS quota, and every rebuild adds another ~110 MB version | Plain Git |
| Startup time | Seconds (load from disk) | 83 s on the dev laptop (Intel Mac, CPU): 75 s embedding + ~8 s loading. Hugging Face free CPU: *to be measured at deploy* |
| Consistency | The index can drift out of sync with the chunk file if someone forgets to rebuild | Always rebuilt from the committed chunk file |

**Why the index is so large.** The vectors themselves are only about 17 MB (11,043 × 384 dimensions).
Local Chroma (`chromadb` 1.5.9) stores each chunk's text about four times in SQLite: as a
metadata row, in a string index, and in both tables of a full-text keyword index. Turning off
the full-text index through Chroma's `Schema` had no effect in local mode.

**What it costs.** Search doesn't work until the build finishes. Free Hugging Face Spaces also
go to sleep after a period of inactivity, so the first visitor after a sleep waits for the rebuild.
Peak memory during the build is about 0.9 GB.

**Versioning.** The chunk file is the versioned data, and the index is rebuilt from it. The
collection's metadata records the embedding model and the chunk file's SHA-256, so the running
app can report exactly which data version it is serving.

**Revisit if** startup time becomes a problem. Options, cheapest first:
- Commit precomputed embeddings instead of the database. 11,043 × 384 float16 vectors come to
  about 8.5 MB, under Hugging Face's 10 MB limit, and startup would skip the embedding step.
- Use paid persistent storage on the Space.
- Host the index on a separate Chroma server.
