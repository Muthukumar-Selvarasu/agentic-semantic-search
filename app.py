"""Semantic Search to Moment Search.

Compares a fixed 250-character chunk RAG pipeline with a moment-level RAG
pipeline over the same lecture transcript. Run with:

    python app.py

The same comparison is in live_test.ipynb for a grader who wants to run
cells and type a question.
"""

from __future__ import annotations

import os
import re
import socket
import sys
import textwrap
import threading
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, ".cache")
os.environ.setdefault("HF_HOME", os.path.join(CACHE, "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(CACHE, "hf", "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(CACHE, "hf", "transformers"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(CACHE, "st"))
os.environ.setdefault("TORCH_HOME", os.path.join(CACHE, "torch"))
os.environ.setdefault("XDG_CACHE_HOME", CACHE)
os.environ.setdefault("CHROMA_CACHE_DIR", os.path.join(CACHE, "chroma"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import chromadb
from sentence_transformers import SentenceTransformer

VIDEO_ID = "UF8uR6Z6KLc"
VIDEO_LABEL = "Steve Jobs, Stanford commencement address (2005)"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
TOP_K = 3
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PAUSE_SECONDS = 2.0
MAX_MOMENT_SECONDS = 120.0
CHROMA_PATH = os.path.join(ROOT, "chroma_db")
COLUMN_WIDTH = 46

TOPIC_CUES = (
    "now let's",
    "now let us",
    "moving on",
    "the next topic",
    "next topic",
    "switching to",
    "switching gears",
    "in this section",
    "let's turn",
    "let us turn",
    "another topic",
    "the first story",
    "the second story",
    "the third story",
    "my first story",
    "my second story",
    "my third story",
    "first story",
    "second story",
    "third story",
    "part one",
    "part two",
    "part three",
)


@dataclass
class Question:
    text: str
    probes: list[str]


@dataclass
class IndexedPassage:
    text: str
    start_time: float
    end_time: float
    metadata: dict = field(default_factory=dict)


FALLBACK_BLOCKS: list[tuple[str, float]] = [
    (
        "Welcome to this multi-topic lecture on retrieval for long-form video. "
        "We will compare two ways of indexing a transcript: fixed character windows "
        "and semantic moments. Hold the details of each method until we reach its "
        "section, because the point of the lecture is how boundaries change what a "
        "system can answer. Speakers do not think in character budgets. They think "
        "in ideas that open, develop, and close. A useful index follows those ideas.",
        0.0,
    ),
    (
        "Now let's talk about fixed-size chunking. A baseline pipeline cuts the "
        "transcript into windows of 250 characters with 50 characters of overlap, "
        "then embeds every window. Context loss happens when a definition begins in "
        "one window and the example that gives the definition its meaning finishes "
        "in the next window. The lecture example is context clipping: a retrieval "
        "unit slices the sentence about fixed windows breaking lecture answers, so "
        "the retriever returns the phrase fixed windows break lecture answers "
        "without the conclusion the listener needed. That conclusion says the answer "
        "is incomplete unless the window also includes the claim that an overlap of "
        "fifty characters cannot repair a sentence cut before its verb. Standard "
        "chunking fails on video transcripts because a speaker finishes an idea "
        "after the character budget has already closed the chunk, and the next chunk "
        "opens in the middle of the thought. The timestamps on those chunks look "
        "precise, but they point at fragments rather than the moment the idea was "
        "actually said.",
        3.2,
    ),
    (
        "Moving on to moment boundaries. A moment groups caption entries that belong "
        "to one idea. We open a new moment when the timestamp gap since the previous "
        "caption is a pause, and also when the next caption begins with a topic cue "
        "such as now let's or moving on. Each stored moment keeps three metadata "
        "fields: start time, end time, and moment summary. Start time is the first "
        "caption's start. End time is the last caption's start plus its duration. "
        "Moment summary is the opening sentence of that idea. Retrieving the moment "
        "returns the full context between those timestamps, so the definition and "
        "the example stay in one passage instead of being clipped at 250 characters. "
        "The extra tokens are intentional. They buy the surrounding sentences that a "
        "fixed window drops on the floor.",
        3.4,
    ),
    (
        "The next topic is evaluation. We ask the same two questions of both indexes "
        "and compare retrieved timestamps, whether the context is clipped, and the "
        "extractive answer. The moment index should name a single time range. The "
        "baseline names several short ranges that may start mid-sentence. Token "
        "overhead is higher for a full moment than for one small chunk, and that "
        "cost is the price of keeping the speaker's idea intact. When you read the "
        "two answers together, prefer the one whose timestamps cover the whole idea "
        "rather than the one whose windows merely share the query's words.",
        3.1,
    ),
]

FALLBACK_QUESTIONS = [
    Question(
        text=(
            "Why does fixed-size chunking cause context loss on a lecture transcript, "
            "and what does the context clipping example show?"
        ),
        probes=[
            "context clipping",
            "cut before its verb",
            "character budget",
        ],
    ),
    Question(
        text=(
            "How do timestamp pauses and topic cues define a moment, and what do "
            "start time, end time, and moment summary record?"
        ),
        probes=[
            "topic cue",
            "start time",
            "end time",
            "moment summary",
        ],
    ),
]

YOUTUBE_QUESTIONS = [
    Question(
        text=(
            "What did Steve Jobs learn about connecting the dots from dropping out "
            "and the calligraphy class?"
        ),
        probes=[
            "calligraphy",
            "connect the dots",
            "looking backwards",
        ],
    ),
    Question(
        text=(
            "What advice did Steve Jobs give about death, and how should that "
            "change the way you live?"
        ),
        probes=[
            "going to die",
            "time is limited",
            "don't waste it",
        ],
    ),
]


def format_ts(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_range(start: float, end: float) -> str:
    return f"{format_ts(start)}–{format_ts(end)}"


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def first_sentence(text: str, limit: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    match = re.search(r".+?[.!?]", cleaned)
    sentence = match.group(0).strip() if match else cleaned
    if len(sentence) > limit:
        return sentence[: limit - 1].rstrip() + "…"
    return sentence


def starts_with_topic_cue(text: str) -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(cue) for cue in TOPIC_CUES)


def sentence_edges(text: str) -> tuple[bool, bool]:
    stripped = text.strip()
    if not stripped:
        return False, False
    starts_ok = stripped[0].isupper() or stripped[0] in "\"'“”("
    ends_ok = stripped[-1] in ".?!\"'”"
    return starts_ok, ends_ok


def pack_captions(blocks: list[tuple[str, float]]) -> list[dict]:
    """Turn lecture paragraphs into caption-sized entries with timed pauses."""
    entries: list[dict] = []
    clock = 0.0
    for paragraph, pause_before in blocks:
        clock += pause_before
        words = paragraph.split()
        buffer: list[str] = []

        def flush() -> None:
            nonlocal clock, buffer
            text = " ".join(buffer)
            duration = max(1.6, len(text) / 14.0)
            entries.append(
                {
                    "text": text,
                    "start": round(clock, 2),
                    "duration": round(duration, 2),
                }
            )
            clock += duration
            buffer = []

        for word in words:
            if buffer and len(" ".join(buffer + [word])) > 90:
                flush()
            buffer.append(word)
        if buffer:
            flush()
    return entries


def normalize_entries(raw_items) -> list[dict]:
    entries: list[dict] = []
    for item in raw_items:
        if isinstance(item, dict):
            text = str(item.get("text", ""))
            start = float(item.get("start", 0) or 0)
            duration = float(item.get("duration", 0) or 0)
        else:
            text = str(getattr(item, "text", ""))
            start = float(getattr(item, "start", 0) or 0)
            duration = float(getattr(item, "duration", 0) or 0)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if duration <= 0:
            duration = max(1.0, len(text) / 14.0)
        entries.append({"text": text, "start": start, "duration": duration})
    entries.sort(key=lambda entry: (entry["start"], entry["duration"]))
    return entries


def _snippets_from_fetched(fetched) -> list[dict]:
    if hasattr(fetched, "to_raw_data"):
        return normalize_entries(fetched.to_raw_data())
    return normalize_entries(list(fetched))


def _fetch_youtube_sync(video_id: str) -> list[dict]:
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    fetched = None
    last_type_error: Exception | None = None
    if hasattr(api, "fetch"):
        for kwargs in (
            {"languages": ["en", "en-US", "en-GB"]},
            {"languages": ("en", "en-US", "en-GB")},
            {},
        ):
            try:
                fetched = api.fetch(video_id, **kwargs)
                break
            except TypeError as exc:
                last_type_error = exc
        if fetched is None and last_type_error is not None and not hasattr(api, "list"):
            raise last_type_error
    if fetched is None and hasattr(api, "list"):
        listing = api.list(video_id)
        transcript = listing.find_transcript(["en", "en-US", "en-GB"])
        fetched = transcript.fetch()
    if fetched is None and hasattr(YouTubeTranscriptApi, "get_transcript"):
        fetched = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    if fetched is None:
        raise RuntimeError("youtube-transcript-api returned no transcript object")
    entries = _snippets_from_fetched(fetched)
    if len(entries) < 5:
        raise RuntimeError(f"transcript too short ({len(entries)} captions)")
    return entries


def load_youtube_transcript(video_id: str, timeout: float = 25.0) -> list[dict]:
    holder: dict = {}

    def target() -> None:
        previous = socket.getdefaulttimeout()
        socket.setdefaulttimeout(20)
        try:
            holder["entries"] = _fetch_youtube_sync(video_id)
        except Exception as exc:  # network, blocking, or API shape
            holder["error"] = exc
        finally:
            socket.setdefaulttimeout(previous)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"YouTube transcript request exceeded {timeout:.0f}s")
    if "error" in holder:
        raise holder["error"]
    return holder["entries"]


def load_transcript() -> tuple[list[dict], str, list[Question]]:
    print(f"Fetching YouTube transcript for {VIDEO_LABEL}")
    print(f"https://www.youtube.com/watch?v={VIDEO_ID}")
    try:
        entries = load_youtube_transcript(VIDEO_ID)
    except Exception as exc:
        print(f"YouTube transcript unavailable ({type(exc).__name__}: {exc}).")
        print("Using the embedded multi-topic lecture transcript.")
        return pack_captions(FALLBACK_BLOCKS), "embedded lecture fallback", FALLBACK_QUESTIONS
    print(f"Downloaded {len(entries)} captions.")
    return entries, f"YouTube {VIDEO_ID}", YOUTUBE_QUESTIONS


def build_stream(entries: list[dict]) -> tuple[str, list[tuple[int, int, float, float]]]:
    parts: list[str] = []
    spans: list[tuple[int, int, float, float]] = []
    cursor = 0
    for entry in entries:
        if parts:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(entry["text"])
        cursor += len(entry["text"])
        spans.append((start, cursor, entry["start"], entry["start"] + entry["duration"]))
    return "".join(parts), spans


def time_for_char(spans: list[tuple[int, int, float, float]], index: int, end: bool) -> float:
    if not spans:
        return 0.0
    if index <= 0:
        return spans[0][3] if end else spans[0][2]
    if index >= spans[-1][1]:
        return spans[-1][3]
    for char_start, char_end, t0, t1 in spans:
        if char_start <= index < char_end:
            return t1 if end else t0
    return spans[-1][3]


def chunk_transcript(entries: list[dict]) -> list[IndexedPassage]:
    text, spans = build_stream(entries)
    if not text:
        return []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    passages: list[IndexedPassage] = []
    index = 0
    while index < len(text):
        window_end = min(index + CHUNK_SIZE, len(text))
        raw = text[index:window_end]
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        document = raw.strip()
        if document:
            char_start = index + leading
            char_end = window_end - trailing
            start_time = time_for_char(spans, char_start, end=False)
            end_time = time_for_char(spans, max(char_start, char_end - 1), end=True)
            passages.append(
                IndexedPassage(
                    text=document,
                    start_time=start_time,
                    end_time=end_time,
                    metadata={
                        "start_time": float(start_time),
                        "end_time": float(end_time),
                        "chunk_index": len(passages),
                        "char_start": int(char_start),
                        "char_end": int(char_end),
                    },
                )
            )
        if window_end >= len(text):
            break
        index += step
    return passages


def group_moments(entries: list[dict]) -> list[IndexedPassage]:
    if not entries:
        return []
    groups: list[tuple[list[dict], str]] = []
    current: list[dict] = []
    current_reason = "opening"

    def close() -> None:
        nonlocal current
        if current:
            groups.append((current, current_reason))
            current = []

    for entry in entries:
        if not current:
            current = [entry]
            continue
        previous = current[-1]
        previous_end = previous["start"] + previous["duration"]
        gap = entry["start"] - previous_end
        cue = starts_with_topic_cue(entry["text"])
        current_span = previous_end - current[0]["start"]
        projected = (entry["start"] + entry["duration"]) - current[0]["start"]
        reasons: list[str] = []
        if gap >= PAUSE_SECONDS:
            reasons.append(f"pause {gap:.1f}s")
        if cue:
            reasons.append("topic cue")
        if not reasons and projected > MAX_MOMENT_SECONDS and current_span >= 20:
            reasons.append("max duration")
        if reasons:
            close()
            current_reason = " + ".join(reasons)
            current = [entry]
        else:
            current.append(entry)
    close()

    moments: list[IndexedPassage] = []
    for group, reason in groups:
        text = " ".join(entry["text"] for entry in group)
        start_time = group[0]["start"]
        end_time = group[-1]["start"] + group[-1]["duration"]
        summary = first_sentence(text)
        moments.append(
            IndexedPassage(
                text=text,
                start_time=start_time,
                end_time=end_time,
                metadata={
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "moment_summary": summary,
                    "moment_index": len(moments),
                    "boundary": reason,
                },
            )
        )
    return moments


def recreate_collection(client: chromadb.PersistentClient, name: str):
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})


def store_passages(collection, passages: list[IndexedPassage], embeddings: list[list[float]], prefix: str) -> None:
    batch = 100
    for start in range(0, len(passages), batch):
        chunk = passages[start : start + batch]
        collection.add(
            ids=[f"{prefix}-{passage.metadata.get('chunk_index', passage.metadata.get('moment_index', start + offset))}" for offset, passage in enumerate(chunk)],
            documents=[passage.text for passage in chunk],
            embeddings=embeddings[start : start + len(chunk)],
            metadatas=[passage.metadata for passage in chunk],
        )


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def query_passages(collection, model: SentenceTransformer, query: str, count: int) -> list[dict]:
    if count <= 0:
        return []
    embedding = embed_texts(model, [query])
    result = collection.query(
        query_embeddings=embedding,
        n_results=min(TOP_K, count),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    documents = result.get("documents") or [[]]
    metadatas = result.get("metadatas") or [[]]
    distances = result.get("distances") or [[]]
    for document, metadata, distance in zip(documents[0], metadatas[0], distances[0]):
        hits.append(
            {
                "text": document,
                "metadata": metadata or {},
                "distance": float(distance),
            }
        )
    return hits


def probe_coverage(text: str, probes: list[str]) -> tuple[list[str], list[str]]:
    lowered = text.lower()
    hit = [probe for probe in probes if probe.lower() in lowered]
    missed = [probe for probe in probes if probe.lower() not in lowered]
    return hit, missed


def describe_quality(segments: list[str], probes: list[str], single_span: bool) -> tuple[str, int, int]:
    notes: list[str] = []
    clipped = 0
    for index, segment in enumerate(segments, start=1):
        starts_ok, ends_ok = sentence_edges(segment)
        problems = []
        if not starts_ok:
            problems.append("starts mid-sentence")
        if not ends_ok:
            problems.append("ends mid-sentence")
        if problems:
            clipped += 1
            notes.append(f"segment {index} {' and '.join(problems)}")
    if not notes and single_span:
        edge = "Intact single span"
    elif not notes:
        edge = "Sentence edges intact, but the context is split across segments"
    else:
        edge = "Clipped — " + "; ".join(notes)
    combined = "\n".join(segments)
    found: list[str] = []
    if not probes:
        detail = "Free-form query, no key-detail checklist"
    else:
        found, missing = probe_coverage(combined, probes)
        detail = f"Key details {len(found)}/{len(probes)}"
        if missing:
            detail += " (missing " + "; ".join(missing) + ")"
    tokens = estimate_tokens(combined)
    return f"{edge}. {detail}. About {tokens} tokens.", tokens, len(found)


def render_baseline_answer(hits: list[dict]) -> str:
    ordered = sorted(hits, key=lambda hit: int(hit["metadata"].get("char_start", 0)))
    blocks: list[str] = []
    previous_end = None
    for hit in ordered:
        char_start = int(hit["metadata"].get("char_start", 0))
        char_end = int(hit["metadata"].get("char_end", char_start))
        text = hit["text"].strip()
        if previous_end is not None and char_start > previous_end + 1:
            blocks.append("[...]")
        if blocks and blocks[-1] != "[...]":
            previous = blocks[-1]
            overlap = 0
            limit = min(len(previous), len(text), CHUNK_OVERLAP + 20)
            for size in range(limit, 8, -1):
                if previous.endswith(text[:size]):
                    overlap = size
                    break
            text = text[overlap:].lstrip()
            if text:
                blocks[-1] = (previous + " " + text).strip()
        elif text:
            blocks.append(text)
        previous_end = char_end if previous_end is None else max(previous_end, char_end)
    return "\n".join(block for block in blocks if block).strip()


def wrap_block(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        wrapped = textwrap.wrap(paragraph, width=width, break_long_words=False, break_on_hyphens=False)
        lines.extend(wrapped or [""])
    return lines or [""]


def side_by_side(left: str, right: str, width: int = COLUMN_WIDTH) -> str:
    left_lines = wrap_block(left, width)
    right_lines = wrap_block(right, width)
    height = max(len(left_lines), len(right_lines))
    left_lines += [""] * (height - len(left_lines))
    right_lines += [""] * (height - len(right_lines))
    gap = " | "
    rows = [left_line.ljust(width) + gap + right_line for left_line, right_line in zip(left_lines, right_lines)]
    return "\n".join(rows)


def pipeline_panel(
    timestamps: str,
    quality: str,
    answer: str,
) -> str:
    return (
        f"Retrieved timestamps:\n{timestamps}\n\n"
        f"Context quality:\n{quality}\n\n"
        f"Answer:\n{answer}"
    )


def spans_are_contiguous(hits: list[dict]) -> bool:
    ordered = sorted(hits, key=lambda hit: int(hit["metadata"].get("char_start", 0)))
    previous_end = None
    for hit in ordered:
        char_start = int(hit["metadata"].get("char_start", 0))
        char_end = int(hit["metadata"].get("char_end", char_start))
        if previous_end is not None and char_start > previous_end + 1:
            return False
        previous_end = char_end if previous_end is None else max(previous_end, char_end)
    return True


def baseline_view(hits: list[dict], question: Question) -> tuple[str, int, int]:
    segments = [hit["text"] for hit in hits]
    quality, tokens, covered = describe_quality(segments, question.probes, single_span=len(hits) == 1)
    if len(hits) > 1 and not spans_are_contiguous(hits):
        quality = "Non-contiguous windows. " + quality
    lines = []
    for rank, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        lines.append(
            f"{rank}. {format_range(float(meta['start_time']), float(meta['end_time']))} "
            f"(chunk {meta.get('chunk_index')})"
        )
    panel = pipeline_panel("\n".join(lines), quality, render_baseline_answer(hits))
    return panel, tokens, covered


def moment_view(hits: list[dict], question: Question) -> tuple[str, int, int]:
    if not hits:
        return pipeline_panel("none", "No moment retrieved.", ""), 0, 0
    primary = hits[0]
    meta = primary["metadata"]
    quality, tokens, covered = describe_quality([primary["text"]], question.probes, single_span=True)
    timestamp_lines = [
        f"1. {format_range(float(meta['start_time']), float(meta['end_time']))} "
        f"(moment {meta.get('moment_index')}, full context)"
    ]
    if len(hits) > 1:
        alternates = []
        for hit in hits[1:]:
            other = hit["metadata"]
            alternates.append(format_range(float(other["start_time"]), float(other["end_time"])))
        timestamp_lines.append("Also retrieved: " + "; ".join(alternates))
    summary = meta.get("moment_summary", "")
    answer = primary["text"].strip()
    if summary:
        answer = f"Moment summary: {summary}\n\n{answer}"
    panel = pipeline_panel("\n".join(timestamp_lines), quality, answer)
    return panel, tokens, covered


def print_inventory(moments: list[IndexedPassage]) -> None:
    print("Moments:")
    for moment in moments:
        meta = moment.metadata
        print(
            f"  M{meta['moment_index']:02d}  {format_range(moment.start_time, moment.end_time):<17}  "
            f"{meta['boundary']:<28}  {meta['moment_summary'][:72]}"
        )


def clipped_chunk_count(passages: list[IndexedPassage]) -> int:
    clipped = 0
    for passage in passages:
        starts_ok, ends_ok = sentence_edges(passage.text)
        if not starts_ok or not ends_ok:
            clipped += 1
    return clipped


@dataclass
class SearchIndex:
    entries: list[dict]
    source: str
    questions: list[Question]
    chunks: list[IndexedPassage]
    moments: list[IndexedPassage]
    model: SentenceTransformer
    baseline: object
    moment_collection: object


def build_index() -> SearchIndex:
    """Load the transcript, embed both indexes, and print the inventory."""
    print("=" * 96)
    print("Semantic Search to Moment Search")
    print("Baseline: 250-character chunks, 50-character overlap, top-3")
    print("Moment RAG: pause and topic-cue boundaries, full moment context")
    print("=" * 96)

    entries, source, questions = load_transcript()
    if len(entries) < 4:
        raise RuntimeError("Transcript did not contain enough captions to index.")

    chunks = chunk_transcript(entries)
    moments = group_moments(entries)
    if not chunks or len(moments) < 2:
        raise RuntimeError(f"Indexing failed (chunks={len(chunks)}, moments={len(moments)}).")

    print()
    print(f"Source: {source}")
    print(f"Captions: {len(entries)}")
    print(
        f"Baseline chunks: {len(chunks)} "
        f"({clipped_chunk_count(chunks)} clipped at a sentence edge) "
        f"in collection baseline_collection"
    )
    print(f"Moments: {len(moments)} in collection moment_collection")
    print_inventory(moments)
    print()
    print(f"Embedding with {EMBED_MODEL} and writing ChromaDB at {CHROMA_PATH}")

    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    baseline = recreate_collection(client, "baseline_collection")
    moment_collection = recreate_collection(client, "moment_collection")
    store_passages(baseline, chunks, embed_texts(model, [passage.text for passage in chunks]), "baseline")
    store_passages(
        moment_collection,
        moments,
        embed_texts(model, [passage.text for passage in moments]),
        "moment",
    )
    print(f"Stored {baseline.count()} baseline chunks and {moment_collection.count()} moments.")
    return SearchIndex(
        entries=entries,
        source=source,
        questions=questions,
        chunks=chunks,
        moments=moments,
        model=model,
        baseline=baseline,
        moment_collection=moment_collection,
    )


def verdict_for(left: str, right: str, b_hit: int, m_hit: int, b_tokens: int, m_tokens: int) -> str:
    baseline_clipped = "Clipped" in left
    moment_intact = "Intact single span" in right
    skipped = "Non-contiguous" in left
    if m_hit > b_hit and moment_intact:
        return (
            "Moment RAG recovered more of the idea's key details in one intact time range. "
            "Baseline windows are clipped"
            + (" and skip intervening transcript" if skipped else "")
            + ", so neighboring topics can enter the answer."
        )
    if moment_intact and baseline_clipped and m_hit >= b_hit:
        return (
            "Key-detail coverage is "
            + ("tied" if m_hit == b_hit else "higher for the moment")
            + ", and the baseline context is clipped"
            + (" across non-contiguous windows" if skipped else "")
            + ". Moment RAG answers from one continuous timestamp range."
        )
    if m_hit == b_hit and m_tokens > b_tokens:
        return (
            "Both answers cover the same key details. Moment RAG spends more "
            "tokens to keep one continuous time range instead of overlapping fragments."
        )
    return (
        "Baseline returns fixed windows. Moment RAG returns the enclosing moment, "
        "with its start time, end time, and summary."
    )


def compare_question(index: SearchIndex, question: Question, number: int | None = None) -> tuple[str, int, int, int, int, int]:
    """Print one side-by-side comparison and return its scorecard row."""
    baseline_hits = query_passages(index.baseline, index.model, question.text, index.baseline.count())
    moment_hits = query_passages(
        index.moment_collection, index.model, question.text, index.moment_collection.count()
    )
    left, b_tokens, b_hit = baseline_view(baseline_hits, question)
    right, m_tokens, m_hit = moment_view(moment_hits, question)
    print()
    print("=" * 96)
    if number is None:
        print("LIVE QUERY")
    else:
        print(f"QUESTION {number}")
    print(question.text)
    print("=" * 96)
    print(side_by_side("BASELINE SEMANTIC RAG", "MOMENT RAG"))
    print(side_by_side("-" * COLUMN_WIDTH, "-" * COLUMN_WIDTH))
    print(side_by_side(left, right))
    print()
    print("Verdict: " + verdict_for(left, right, b_hit, m_hit, b_tokens, m_tokens))
    label = "Live" if number is None else f"Q{number}"
    return label, b_tokens, m_tokens, b_hit, m_hit, len(question.probes)


def print_scorecard(rows: list[tuple[str, int, int, int, int, int]]) -> None:
    print("=" * 96)
    print("SCORECARD")
    print("=" * 96)
    header = f"{'Question':<10} {'Baseline tokens':>16} {'Moment tokens':>14} {'Baseline details':>18} {'Moment details':>16}"
    print(header)
    print("-" * len(header))
    for label, b_tokens, m_tokens, b_hit, m_hit, probe_count in rows:
        if probe_count:
            baseline_details = f"{b_hit}/{probe_count}"
            moment_details = f"{m_hit}/{probe_count}"
        else:
            baseline_details = "n/a"
            moment_details = "n/a"
        print(
            f"{label:<10} {b_tokens:>16} {m_tokens:>14} "
            f"{baseline_details:>18} {moment_details:>16}"
        )


def main() -> int:
    try:
        index = build_index()
    except RuntimeError as error:
        print(error)
        return 1

    questions = list(index.questions)
    user_query = " ".join(sys.argv[1:]).strip()
    if user_query:
        questions.append(Question(text=user_query, probes=[]))

    score_rows = [
        compare_question(index, question, number)
        for number, question in enumerate(questions, start=1)
    ]
    print()
    print_scorecard(score_rows)
    print()
    print("Comparison complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
