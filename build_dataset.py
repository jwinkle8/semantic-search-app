#!/usr/bin/env python3
"""Build a U.S. National Park Service dataset for semantic search.

Source: the NPS Data API (https://www.nps.gov/subjects/developer/), public-domain text.
Get a free API key at https://www.nps.gov/subjects/developer/get-started.htm and run:

    export NPS_API_KEY=your_key_here
    python build_dataset.py               # fetch anything not cached, then chunk
    python build_dataset.py --chunk-only  # re-chunk from the cached raw pages (no network)
    python build_dataset.py --refresh     # ignore the cache and re-download everything

Pipeline (each stage writes files, so later stages rerun without re-fetching):

  1. Extract  /parks, /thingstodo, /campgrounds -> data/raw/<endpoint>/page_<start>.json
              Raw API pages are kept untouched so the transform can be rerun or changed.
  2. Transform  Strip HTML, join each record to its park (name, states, designation),
              and turn every record into a self-describing text document.
  3. Chunk    Long documents are split into overlapping word windows. Every chunk starts
              with a short header ("Yosemite National Park (CA) | Thing to do: ...") so it
              still makes sense, and embeds well, on its own.
              -> data/nps_chunks.json  {"ids": [...], "texts": [...], "metadatas": [...]}

The chunk file is column-oriented so it maps straight onto Chroma:

    data = json.load(open("data/nps_chunks.json"))
    collection.add(ids=data["ids"], documents=data["texts"], metadatas=data["metadatas"], ...)
"""

from __future__ import annotations

import argparse
import html
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

API_BASE = "https://developer.nps.gov/api/v1"
ENDPOINTS = ("parks", "thingstodo", "campgrounds")
PAGE_SIZE = 500
REQUEST_INTERVAL_S = 0.5

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_PATH = DATA_DIR / "nps_chunks.json"


# --------------------------------------------------------------------------- extract


class RateLimited(Exception):
    pass


def fetch_page(endpoint: str, start: int, api_key: str) -> dict:
    url = f"{API_BASE}/{endpoint}?" + urllib.parse.urlencode({"limit": PAGE_SIZE, "start": start})
    # Send the key as a header rather than in the URL so it never shows up in logs.
    req = urllib.request.Request(url, headers={"X-Api-Key": api_key,
                                               "User-Agent": "nps-semantic-search (class project)"})
    for attempt in range(4):
        try:
            # Big pages (500 things-to-do with long HTML descriptions) can be slow to stream.
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise RateLimited from exc
            if exc.code < 500 or attempt == 3:
                raise
            error = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as exc:
            if attempt == 3:
                raise
            error = exc
        print(f"    retrying in {2 ** attempt}s ({error})", file=sys.stderr)
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def extract(endpoint: str, api_key: str, refresh: bool) -> list[dict]:
    """Download every page of an endpoint (or reuse cached pages); return all records."""
    folder = RAW_DIR / endpoint
    folder.mkdir(parents=True, exist_ok=True)
    records, start, total = [], 0, None
    while total is None or start < total:
        path = folder / f"page_{start:05d}.json"
        if path.exists() and not refresh:
            page = json.loads(path.read_text())
        else:
            page = fetch_page(endpoint, start, api_key)
            path.write_text(json.dumps(page, ensure_ascii=False))
            time.sleep(REQUEST_INTERVAL_S)
        total = int(page["total"])
        if not page["data"]:
            break
        records += page["data"]
        start += len(page["data"])
    print(f"  {endpoint}: {len(records)} records")
    return records


def load_cached(endpoint: str) -> list[dict]:
    pages = sorted((RAW_DIR / endpoint).glob("page_*.json"))
    if not pages:
        sys.exit(f"No cached pages in {RAW_DIR / endpoint}; run without --chunk-only first.")
    return [rec for p in pages for rec in json.loads(p.read_text())["data"]]


# --------------------------------------------------------------------------- transform


class _TextExtractor(HTMLParser):
    BLOCK_TAGS = {"p", "br", "li", "div", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "ul", "ol"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")

    def handle_endtag(self, tag):
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def clean(text: str | None) -> str:
    """HTML -> plain text with paragraph breaks preserved."""
    if not text:
        return ""
    parser = _TextExtractor()
    parser.feed(text)
    out = html.unescape("".join(parser.parts)).replace("\xa0", " ")
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def names(items: list[dict]) -> str:
    return ", ".join(i["name"] for i in items or [] if i.get("name"))


def to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def section(label: str, text: str) -> str:
    return f"{label}: {text}" if text else ""


def park_docs(parks: list[dict]) -> list[dict]:
    """Each park yields an overview, an activities/topics list, and a weather doc (when present).

    The lists are kept out of the overview so big parks' 100-item topic lists don't
    crowd out (or split) the descriptive prose.
    """
    docs = []
    for p in parks:
        base = {"park_code": p["parkCode"], "url": p.get("url") or "",
                "latitude": to_float(p.get("latitude")), "longitude": to_float(p.get("longitude")),
                "title": p["fullName"]}
        docs.append({**base, "doc_type": "park", "source_id": p["id"],
                     "label": "Park overview", "body": clean(p.get("description"))})
        features = "\n\n".join(filter(None, [
            section("Activities", names(p.get("activities"))),
            section("Topics", names(p.get("topics"))),
        ]))
        if features:
            docs.append({**base, "doc_type": "park_features", "source_id": p["id"] + "-features",
                         "label": "Activities and topics", "body": features})
        weather = clean(p.get("weatherInfo"))
        if weather:
            docs.append({**base, "doc_type": "park_weather", "source_id": p["id"] + "-weather",
                         "label": "Weather and when to visit", "body": weather})
    return docs


def todo_docs(things: list[dict]) -> list[dict]:
    docs = []
    for t in things:
        related = t.get("relatedParks") or []
        details = [
            section("Duration", clean(t.get("durationDescription")) or t.get("duration", "")),
            section("Season", ", ".join(t.get("season") or [])),
            section("Time of day", ", ".join(t.get("timeOfDay") or [])),
            section("Activities", names(t.get("activities"))),
            section("Pets", clean(t.get("petsDescription"))),
            section("Fees", clean(t.get("feeDescription"))),
            section("Accessibility", clean(t.get("accessibilityInformation"))),
        ]
        body = "\n\n".join(filter(None, [
            clean(t.get("shortDescription")),
            clean(t.get("longDescription")),
            "\n".join(filter(None, details)),
        ]))
        if not body:
            continue
        docs.append({
            "doc_type": "thing_to_do", "source_id": t["id"], "title": t["title"],
            "label": f"Thing to do: {t['title']}", "body": body,
            "park_code": related[0]["parkCode"] if related else "",
            "all_park_codes": ",".join(r["parkCode"] for r in related),
            "url": t.get("url") or "",
            "latitude": to_float(t.get("latitude")), "longitude": to_float(t.get("longitude")),
        })
    return docs


def campground_docs(camps: list[dict]) -> list[dict]:
    docs = []
    for c in camps:
        amen = c.get("amenities") or {}
        features = []
        for key, label in [("toilets", "Toilets"), ("showers", "Showers"),
                           ("potableWater", "Drinking water"), ("cellPhoneReception", "Cell reception"),
                           ("campStore", "Camp store"), ("firewoodForSale", "Firewood for sale"),
                           ("foodStorageLockers", "Food storage lockers"), ("dumpStation", "Dump station")]:
            value = amen.get(key)
            value = ", ".join(value) if isinstance(value, list) else (value or "")
            if value:
                features.append(f"{label}: {value}")
        sites = c.get("campsites") or {}
        site_bits = [f"{n} {label}" for key, label in
                     [("totalSites", "total sites"), ("tentOnly", "tent-only"), ("rvOnly", "RV-only"),
                      ("electricalHookups", "with electric hookups"), ("group", "group sites"),
                      ("horse", "horse sites"), ("walkBoatTo", "walk-in or boat-in")]
                     if (n := str(sites.get(key) or "0")) != "0"]
        body = "\n\n".join(filter(None, [
            clean(c.get("description")),
            section("Campsites", ", ".join(site_bits)),
            "\n".join(features),
            section("Reservations", clean(c.get("reservationInfo"))),
            section("Weather", clean(c.get("weatherOverview"))),
            section("Regulations", clean(c.get("regulationsOverview"))),
        ]))
        if not body:
            continue
        docs.append({
            "doc_type": "campground", "source_id": c["id"], "title": c["name"],
            "label": f"Campground: {c['name']}", "body": body,
            "park_code": c.get("parkCode") or "", "url": c.get("url") or "",
            "latitude": to_float(c.get("latitude")), "longitude": to_float(c.get("longitude")),
        })
    return docs


# --------------------------------------------------------------------------- chunk


def windows(body: str, max_words: int, overlap: int) -> list[str]:
    """Split on paragraphs, packing them into chunks of <= max_words; window any huge paragraph."""
    paragraphs = []
    for para in body.split("\n\n"):
        words = para.split()
        if len(words) <= max_words:
            paragraphs.append(para)
            continue
        step = max(1, max_words - overlap)
        for start in range(0, len(words), step):
            paragraphs.append(" ".join(words[start:start + max_words]))
            if start + max_words >= len(words):
                break

    chunks, current, count = [], [], 0
    for para in paragraphs:
        n = len(para.split())
        if current and count + n > max_words:
            chunks.append("\n\n".join(current))
            current, count = [], 0
        current.append(para)
        count += n
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def build_chunks(parks: list[dict], things: list[dict], camps: list[dict],
                 max_words: int, overlap: int) -> dict:
    park_by_code = {p["parkCode"]: p for p in parks}
    docs = park_docs(parks) + todo_docs(things) + campground_docs(camps)
    docs = [d for d in docs if d["body"].strip()]  # a few parks have no description

    # The NPS API occasionally gives two *different* records the same id (seen in /thingstodo),
    # so keep both and disambiguate the second rather than silently dropping one.
    seen_sources: dict[str, int] = {}
    for doc in docs:
        key = f"{doc['doc_type']}-{doc['source_id']}"
        seen_sources[key] = seen_sources.get(key, 0) + 1
        if seen_sources[key] > 1:
            doc["source_id"] += f"-dup{seen_sources[key]}"

    ids, texts, metadatas = [], [], []
    for doc in docs:
        park = park_by_code.get(doc["park_code"], {})
        park_name = park.get("fullName", "")
        states = park.get("states", "")
        # Context header: lets a chunk like "The trail climbs 600 feet..." be matched to its park.
        where = f"{park_name} ({states.replace(',', ', ')})" if park_name else doc["title"]
        header = f"{where} | {doc['label']}"

        pieces = windows(doc["body"], max_words, overlap)
        for i, piece in enumerate(pieces):
            meta = {
                "doc_type": doc["doc_type"],
                "title": doc["title"],
                "park_code": doc["park_code"],
                "park_name": park_name,
                "states": states,
                "designation": park.get("designation", ""),
                "url": doc["url"],
                "latitude": doc["latitude"],
                "longitude": doc["longitude"],
                "all_park_codes": doc.get("all_park_codes", doc["park_code"]),
                "chunk_index": i,
                "num_chunks": len(pieces),
                "source": "NPS Data API",
            }
            ids.append(f"{doc['doc_type']}-{doc['source_id']}-{i:02d}")
            texts.append(f"{header}\n\n{piece}")
            # Chroma metadata values must be str/int/float/bool -- no None.
            metadatas.append({k: v for k, v in meta.items() if v is not None})

    counts = {}
    for m in metadatas:
        counts[m["doc_type"]] = counts.get(m["doc_type"], 0) + 1
    return {
        "ids": ids,
        "texts": texts,
        "metadatas": metadatas,
        "info": {
            "source": "NPS Data API (developer.nps.gov), public domain",
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "num_documents": len(docs),
            "num_chunks": len(ids),
            "chunks_by_type": counts,
            "chunking": {"strategy": "paragraph-packed word windows with context header",
                         "max_words": max_words, "overlap_words": overlap},
        },
    }


# --------------------------------------------------------------------------- validate


def validate(data: dict) -> None:
    """Data-quality gate: refuse to write a chunk file that Chroma would reject or that is broken."""
    ids, texts, metas = data["ids"], data["texts"], data["metadatas"]
    errors = []
    if not (len(ids) == len(texts) == len(metas)):
        errors.append(f"column lengths differ: {len(ids)} ids, {len(texts)} texts, {len(metas)} metadatas")
    dupes = len(ids) - len(set(ids))
    if dupes:
        errors.append(f"{dupes} duplicate ids")
    empty = sum(1 for t in texts if "\n\n" not in t or not t.split("\n\n", 1)[1].strip())
    if empty:
        errors.append(f"{empty} chunks have a header but no body text")
    bad_types = sum(1 for m in metas for v in m.values() if not isinstance(v, (str, int, float, bool)))
    if bad_types:
        errors.append(f"{bad_types} metadata values are not str/int/float/bool")
    if errors:
        sys.exit("Data-quality check failed:\n  - " + "\n  - ".join(errors))

    no_park = sum(1 for m in metas if not m["park_name"])
    print(f"Data-quality checks passed. Note: {no_park} chunks have no matching park "
          "(the NPS record lists no park, or one outside /parks).")


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunk-only", action="store_true", help="re-chunk cached raw pages; no network")
    parser.add_argument("--refresh", action="store_true", help="re-download every page")
    parser.add_argument("--max-words", type=int, default=200, help="max words per chunk body (default 200)")
    parser.add_argument("--overlap", type=int, default=40, help="word overlap when windowing (default 40)")
    args = parser.parse_args()

    if args.chunk_only:
        raw = {ep: load_cached(ep) for ep in ENDPOINTS}
    else:
        api_key = os.environ.get("NPS_API_KEY")
        if not api_key:
            print("NPS_API_KEY not set; using the shared DEMO_KEY (very low rate limit).\n"
                  "Get a free key: https://www.nps.gov/subjects/developer/get-started.htm",
                  file=sys.stderr)
            api_key = "DEMO_KEY"
        print("Extracting from the NPS API (cached pages are reused):")
        raw = {}
        try:
            for ep in ENDPOINTS:
                raw[ep] = extract(ep, api_key, args.refresh)
        except RateLimited:
            sys.exit("\nRate limited by api.data.gov. Pages downloaded so far are cached in "
                     f"{RAW_DIR}; set NPS_API_KEY (or wait an hour) and rerun to continue.")

    data = build_chunks(raw["parks"], raw["thingstodo"], raw["campgrounds"],
                        args.max_words, args.overlap)
    validate(data)
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    info = data["info"]
    print(f"\nWrote {CHUNKS_PATH.relative_to(DATA_DIR.parent)}: {info['num_chunks']} chunks "
          f"from {info['num_documents']} documents {info['chunks_by_type']}")


if __name__ == "__main__":
    main()
