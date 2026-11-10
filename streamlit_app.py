"""National Parks Semantic Search: a Streamlit app over NPS park, activity, and campground text.

On first run it loads the precomputed embeddings into an in-memory Chroma index (see
build_index.py and "Design decisions" in README.md), cached for every later session.

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import re

import streamlit as st

from build_index import build_collection, embed

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
BADGES = {  # doc_type -> (label, color)
    "park": ("Park", "green"), "park_features": ("Park · activities", "green"),
    "park_weather": ("Park · weather", "green"), "thing_to_do": ("Thing to do", "blue"),
    "campground": ("Campground", "orange"),
}
EXAMPLES = [
    ("kid-friendly short hike to a waterfall", ANY_STATE),
    ("campground with showers near the beach", ANY_STATE),
    ("learn about the civil rights movement", "AL"),
    ("see glaciers without a long hike", "AK"),
]
SNIPPET_CHARS = 360


# ---------------------------------------------------------------------------- index (cached)


@st.cache_resource(show_spinner="Loading the search index…")
def load_index():
    """Built once per server process and shared by every visitor."""
    collection, embedder = build_collection()
    # Chroma can't substring-match "DC,MD,PA,VA", so map each state to its park codes once and
    # filter searches with park_code $in [...]. Chunks with no park can't be placed in a state.
    parks_by_state: dict[str, list[str]] = {}
    for m in collection.get(include=["metadatas"])["metadatas"]:
        for state in filter(None, m.get("states", "").split(",")):
            parks_by_state.setdefault(state, []).append(m["park_code"])
    parks_by_state = {s: sorted(set(codes)) for s, codes in parks_by_state.items()}
    return collection, embedder, parks_by_state


# ---------------------------------------------------------------------------- search


def build_where(state: str, result_types: list[str], parks_by_state: dict) -> dict | None:
    clauses = []
    doc_types = [t for label in result_types for t in RESULT_TYPES[label]]
    if len(doc_types) < sum(len(v) for v in RESULT_TYPES.values()):
        clauses.append({"doc_type": {"$in": doc_types}})
    if state != ANY_STATE:
        clauses.append({"park_code": {"$in": parks_by_state[state]}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def search(query: str, state: str, result_types: list[str], num_results: int,
           collection, embedder, parks_by_state) -> list[dict]:
    """Top matches, deduplicated to one result per place or activity."""
    where = build_where(state, result_types, parks_by_state)
    query_vector = embed([query], embedder)
    # Over-fetch, because several chunks of one document often match and each document is shown
    # once; if that still yields too few distinct documents, fetch more (up to the whole index).
    fetch = num_results * 4
    while True:
        res = collection.query(query_embeddings=query_vector, n_results=min(fetch, collection.count()),
                               where=where, include=["documents", "metadatas", "distances"])
        results, seen = [], set()
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            group = "park" if meta["doc_type"].startswith("park") else meta["doc_type"]
            key = (group, meta["park_code"], meta["title"])
            if key in seen:
                continue
            seen.add(key)
            results.append({"text": doc, "meta": meta, "similarity": 1 - dist})
            if len(results) == num_results:
                return results
        if len(res["ids"][0]) < fetch or fetch >= collection.count():
            return results  # fewer matches exist than requested
        fetch *= 4


# ---------------------------------------------------------------------------- rendering


def md_escape(text: str) -> str:
    """Show NPS text literally: no stray bold/italics, links, or '$...$' rendered as math."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|$<>~])", r"\\\1", text)


def snippet(text: str) -> str:
    body = text.split("\n\n", 1)[1] if "\n\n" in text else text  # drop the context header
    body = " ".join(body.split())
    if len(body) <= SNIPPET_CHARS:
        return body
    return body[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"


def show_result(r: dict) -> None:
    m = r["meta"]
    label, color = BADGES.get(m["doc_type"], (m["doc_type"], "gray"))
    with st.container(border=True):
        top = st.columns([3, 1], vertical_alignment="center")
        top[0].badge(label, color=color)
        top[1].caption(f"match {r['similarity']:.2f}")
        title = md_escape(m["title"])
        url = m.get("url", "")
        st.markdown(f"**[{title}]({url})**" if url.startswith("https://") else f"**{title}**")
        where = []
        if m.get("park_name") and m["park_name"] != m["title"]:
            where.append(m["park_name"])
        if m.get("states"):
            where.append(m["states"].replace(",", ", "))
        if where:
            st.caption(md_escape(" · ".join(where)))
        st.markdown(md_escape(snippet(r["text"])))


def use_example(query: str, state: str) -> None:
    st.session_state.query = query
    st.session_state.state = state


# ---------------------------------------------------------------------------- page

st.set_page_config(page_title="National Parks Semantic Search", page_icon="🏞️", layout="centered")
collection, embedder, parks_by_state = load_index()
state_options = [ANY_STATE] + sorted(parks_by_state, key=lambda s: STATE_NAMES.get(s, s))

st.title("🏞️ National Parks Semantic Search")
st.write("Search U.S. National Park Service parks, things to do, and campgrounds **by meaning**, "
         "not just keywords. Describe the experience you want.")

query = st.text_input("What are you looking for?", key="query",
                      placeholder="e.g. quiet place to see the stars away from city lights")
filters = st.columns([2, 3, 2])
state = filters[0].selectbox(
    "State or territory", state_options, key="state",
    format_func=lambda s: s if s == ANY_STATE else f"{STATE_NAMES.get(s, s)} ({s})")
result_types = filters[1].pills("Result types", list(RESULT_TYPES), selection_mode="multi",
                                default=list(RESULT_TYPES), key="types")
num_results = filters[2].slider("Results", 3, 20, 8, key="num_results")

st.caption("Try an example:")
example_cols = st.columns(len(EXAMPLES))
for col, (text, ex_state) in zip(example_cols, EXAMPLES):
    hint = "" if ex_state == ANY_STATE else f" ({ex_state})"
    col.button(text + hint, on_click=use_example, args=(text, ex_state), width="stretch")

st.divider()
query = (query or "").strip()
if not query:
    st.info("Describe what you're looking for, e.g. a kind of hike, a place to camp, "
            "or a piece of history.")
elif not result_types:
    st.warning("Pick at least one result type.")
else:
    results = search(query, state, result_types, num_results, collection, embedder, parks_by_state)
    if not results:
        st.warning("No matches. Try another state or more result types.")
    for r in results:
        show_result(r)

meta = collection.metadata
st.caption(
    "Data: [NPS Data API](https://www.nps.gov/subjects/developer/) (public domain) · "
    f"{meta['num_chunks']:,} chunks · data built {meta['chunks_built_at'][:10]} · "
    f"sha256 {meta['chunks_sha256'][:12]} · embeddings: all\\-MiniLM\\-L6\\-v2 (ONNX) · "
    "[source on GitHub](https://github.com/jwinkle8/semantic-search-app)"
)
