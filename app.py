"""National Parks Semantic Search: a Gradio app over NPS park, activity, and campground text.

At startup it builds the Chroma index in memory (see build_index.py and the README's
"Index storage" decision), then serves searches against it.

    python app.py
"""

from __future__ import annotations

import html

import gradio as gr

from build_index import build_collection

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AS": "American Samoa", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "GU": "Guam", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "MP": "Northern Mariana Islands", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "PR": "Puerto Rico", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VI": "U.S. Virgin Islands", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}
ANY_STATE = "Any state"

# UI label -> doc_types in the index. The three park doc types are shown as one "Parks" result.
RESULT_TYPES = {
    "Parks": ["park", "park_features", "park_weather"],
    "Things to do": ["thing_to_do"],
    "Campgrounds": ["campground"],
}
BADGES = {
    "park": "Park", "park_features": "Park · activities", "park_weather": "Park · weather",
    "thing_to_do": "Thing to do", "campground": "Campground",
}
SNIPPET_CHARS = 360

# ---------------------------------------------------------------------------- startup

collection, model = build_collection()

# Chroma can't substring-match "DC,MD,PA,VA", so map each state to its park codes once and
# filter searches with park_code $in [...]. Chunks with no park can't be placed in a state.
_metas = collection.get(include=["metadatas"])["metadatas"]
PARKS_BY_STATE: dict[str, set[str]] = {}
for m in _metas:
    for state in filter(None, m.get("states", "").split(",")):
        PARKS_BY_STATE.setdefault(state, set()).add(m["park_code"])
STATE_CHOICES = [ANY_STATE] + [f"{STATE_NAMES.get(s, s)} ({s})"
                               for s in sorted(PARKS_BY_STATE, key=lambda s: STATE_NAMES.get(s, s))]
DATA_VERSION = (f"{collection.count():,} chunks · data built {collection.metadata['chunks_built_at'][:10]}"
                f" · sha256 {collection.metadata['chunks_sha256'][:12]}")

# ---------------------------------------------------------------------------- search


def build_where(state_choice: str, result_types: list[str]) -> dict | None:
    clauses = []
    doc_types = [t for label in result_types for t in RESULT_TYPES[label]]
    if len(doc_types) < sum(len(v) for v in RESULT_TYPES.values()):
        clauses.append({"doc_type": {"$in": doc_types}})
    if state_choice and state_choice != ANY_STATE:
        code = state_choice.rsplit("(", 1)[1].rstrip(")")
        clauses.append({"park_code": {"$in": sorted(PARKS_BY_STATE[code])}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def search(query: str, state_choice: str, result_types: list[str], num_results: int):
    """Return (results, matches) for a query; results are deduplicated to one per place/activity."""
    query = (query or "").strip()
    if not query or not result_types:
        return []
    num_results = int(num_results)
    where = build_where(state_choice, result_types)
    # Over-fetch: several chunks of the same document often match, and we show each document once.
    res = collection.query(
        query_embeddings=model.encode([query], normalize_embeddings=True).tolist(),
        n_results=min(num_results * 4, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    results, seen = [], set()
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        group = "park" if meta["doc_type"].startswith("park") else meta["doc_type"]
        key = (group, meta["park_code"], meta["title"])
        if key in seen:
            continue
        seen.add(key)
        results.append({"text": doc, "meta": meta, "similarity": 1 - dist})
        if len(results) == num_results:
            break
    return results


def snippet(text: str) -> str:
    body = text.split("\n\n", 1)[1] if "\n\n" in text else text  # drop the context header
    body = " ".join(body.split())
    if len(body) <= SNIPPET_CHARS:
        return body
    return body[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"


def render(results: list[dict], query: str, result_types: list[str]) -> str:
    if not (query or "").strip():
        return "<p class='nps-empty'>Describe what you're looking for, e.g. a kind of hike, a place to camp, or a piece of history.</p>"
    if not result_types:
        return "<p class='nps-empty'>Pick at least one result type.</p>"
    if not results:
        return "<p class='nps-empty'>No matches. Try another state or more result types.</p>"
    cards = []
    for r in results:
        m = r["meta"]
        e = html.escape
        title = e(m["title"])
        url = m.get("url", "")
        if url.startswith("https://"):
            title = f"<a href='{e(url)}' target='_blank' rel='noopener'>{title}</a>"
        where = ""
        if m.get("park_name") and m["park_name"] != m["title"]:
            where = e(m["park_name"])
        if m.get("states"):
            where = f"{where} · {e(m['states'].replace(',', ', '))}" if where else e(m["states"].replace(",", ", "))
        cards.append(
            "<div class='nps-card'>"
            f"<div class='nps-top'><span class='nps-badge'>{e(BADGES.get(m['doc_type'], m['doc_type']))}</span>"
            f"<span class='nps-score'>match {r['similarity']:.2f}</span></div>"
            f"<div class='nps-title'>{title}</div>"
            + (f"<div class='nps-where'>{where}</div>" if where else "")
            + f"<p class='nps-snippet'>{e(snippet(r['text']))}</p></div>"
        )
    return "".join(cards)


def run(query: str, state_choice: str, result_types: list[str], num_results: int) -> str:
    return render(search(query, state_choice, result_types, num_results), query, result_types)


# ---------------------------------------------------------------------------- UI

CSS = """
.nps-card { border: 1px solid var(--border-color-primary); border-radius: 10px;
            padding: 12px 14px; margin-bottom: 10px; background: var(--background-fill-secondary); }
.nps-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.nps-badge { font-size: 0.75rem; font-weight: 600; padding: 2px 8px; border-radius: 999px;
             background: var(--color-accent-soft); color: var(--body-text-color); }
.nps-score { font-size: 0.75rem; color: var(--body-text-color-subdued); }
.nps-title { font-size: 1.05rem; font-weight: 600; margin-top: 6px; }
.nps-where { font-size: 0.85rem; color: var(--body-text-color-subdued); }
.nps-snippet { margin: 6px 0 0; line-height: 1.45; }
.nps-empty { color: var(--body-text-color-subdued); }
"""

with gr.Blocks(title="National Parks Semantic Search") as demo:
    gr.Markdown(
        "# National Parks Semantic Search\n"
        "Search U.S. National Park Service parks, things to do, and campgrounds **by meaning**, "
        "not just keywords. Try describing the experience you want."
    )
    with gr.Row():
        query = gr.Textbox(label="What are you looking for?", scale=4, autofocus=True,
                           placeholder="e.g. quiet place to see the stars away from city lights")
        button = gr.Button("Search", variant="primary", scale=1)
    with gr.Row():
        state = gr.Dropdown(STATE_CHOICES, value=ANY_STATE, label="State or territory")
        types = gr.CheckboxGroup(list(RESULT_TYPES), value=list(RESULT_TYPES), label="Result types")
        count = gr.Slider(3, 20, value=8, step=1, label="Number of results")
    output = gr.HTML(render([], "", list(RESULT_TYPES)))
    gr.Examples(
        examples=[
            ["kid-friendly short hike to a waterfall", ANY_STATE],
            ["campground with showers near the beach", ANY_STATE],
            ["learn about the civil rights movement", "Alabama (AL)"],
            ["see glaciers without a long hike", "Alaska (AK)"],
        ],
        inputs=[query, state],
    )
    gr.Markdown(f"<small>Data: [NPS Data API](https://www.nps.gov/subjects/developer/) "
                f"(public domain) · {DATA_VERSION} · embeddings: all-MiniLM-L6-v2</small>")

    inputs = [query, state, types, count]
    button.click(run, inputs, output)
    query.submit(run, inputs, output)

if __name__ == "__main__":
    demo.launch(css=CSS)
