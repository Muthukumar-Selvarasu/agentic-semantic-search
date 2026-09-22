# Semantic Search

Project for the Semantic Search to Moment Search comparison: ingest a lecture transcript, index it with fixed-size chunking and with pause-and-cue moments, then evaluate both retrievers on the same questions.

Tickets are sequential. Start the next ticket only after the previous ticket's definition of done is checked off.

---

## 1. Ingestion & Mock Fallback

**Priority:** Urgent

**Estimate:** 3 points

**Description:** Build transcript ingestion for a real YouTube lecture and a deterministic fallback so the rest of the pipeline can run without a live caption API. The source must be caption entries with text, start time, and duration. Downstream chunking and moment grouping both read this same entry list.

Depends on: nothing. Blocks the baseline index, the moment index, and the evaluation script.

**Definition of Done:**

- [x] `youtube-transcript-api` downloads an English caption track for a real video id and normalizes each snippet to `text`, `start`, and `duration`.
- [x] A failed fetch (timeout, block, empty track, or API error) switches to an embedded multi-topic lecture of at least four caption groups, and the run continues.
- [x] The active source is printed (YouTube id or embedded fallback) along with the caption count.
- [x] Entries are ordered by start time, and any caption with a missing duration receives a positive duration before indexing.
- [x] Ingestion is covered by `requirements.txt` (`youtube-transcript-api`) and does not require a second manual download step.

---

## 2. Baseline Semantic Chunking Pipeline

**Priority:** High

**Estimate:** 5 points

**Description:** Index the ingested transcript as fixed character windows and serve top-3 semantic search over that index. This is the baseline the moment index is compared against. Windows are character cuts, including cuts that land mid-word or mid-sentence, because that is the failure mode under test.

Depends on: Ingestion & Mock Fallback. Blocks the evaluation script.

**Definition of Done:**

- [x] The transcript is split into 250-character windows with a 50-character overlap, in order, including a shorter final window when the tail is under 250 characters.
- [x] Each window is embedded with `sentence-transformers` and stored in ChromaDB collection `baseline_collection`, with `start_time` and `end_time` metadata mapped from the captions that overlap the window.
- [x] A query returns the top 3 windows by cosine similarity, and the formatted answer shows each window's timestamp range.
- [x] Re-running the script replaces the collection contents so ids do not accumulate across runs.
- [x] The run prints the chunk count and how many chunks break a sentence boundary.

---

## 3. Moment RAG Semantic Boundary Pipeline

**Priority:** High

**Estimate:** 5 points

**Description:** Group the same caption entries into semantic moments and retrieve the full moment instead of a fixed window. Boundaries come from timestamp pauses and topic cues. Each moment carries the metadata a video player needs in order to seek to the idea, not just the matching words.

Depends on: Ingestion & Mock Fallback. Can be built in parallel with the baseline ticket after ingestion, and blocks the evaluation script.

**Definition of Done:**

- [x] A new moment starts when the gap since the previous caption is at least 2 seconds, or when the next caption begins with a topic cue such as "now let's", "moving on", or "my second story".
- [x] Every stored moment in ChromaDB includes metadata `start_time`, `end_time`, and `moment_summary` (the moment's opening sentence).
- [x] The document stored for a moment is the full caption text between those timestamps, and retrieval answers from that full text for the top moment.
- [x] The run prints an inventory of moments with index, time range, boundary reason, and summary.
- [x] A long caption stream still yields multiple moments when pauses or topic cues are present, and a length guard prevents one silent stretch from becoming a single multi-minute document.

---

## 4. Side-by-Side Evaluation Script

**Priority:** High

**Estimate:** 3 points

**Description:** One command runs two benchmark questions through both indexes and prints a side-by-side comparison. The comparison has to make timestamps, context quality, and the answer text visible without opening Chroma or reading source. Questions must match the transcript that was actually loaded (YouTube lecture vs embedded fallback).

Depends on: Baseline Semantic Chunking Pipeline and Moment RAG Semantic Boundary Pipeline.

**Definition of Done:**

- [x] `python app.py` executes both pipelines for 2 benchmark questions in a single process and exits 0.
- [x] Each question prints baseline and moment columns with retrieved timestamps, a context-quality line (clipping, contiguity, key-detail coverage, token estimate), and the answer text.
- [x] Baseline answers preserve gaps between non-contiguous windows so a reader can see where transcript was skipped.
- [x] Moment answers include the moment summary plus the full moment body for the top hit, and list alternate moment ranges without mixing them into the answer.
- [x] A scorecard summarizes token estimates and key-detail hits for both pipelines on both questions.

---

## 5. Presentation & Loom Walkthrough Deliverable

**Priority:** Medium

**Estimate:** 2 points

**Description:** Turn the verified run into a narrative a reviewer can present in five slides or a short Loom. The talk track uses the numbers from the latest successful `python app.py` run: caption count, chunk count, moment count, timestamp ranges, and token estimates. The close states when a full moment is worth the extra tokens and when a later pause still hides the last sentence of an answer.

Depends on: Side-by-Side Evaluation Script.

**Definition of Done:**

- [x] `SUMMARY.md` contains a comparison table of baseline fixed chunking vs moment RAG on granularity, context loss, precision, and token overhead.
- [x] The findings section explains, with timestamps from the run, how a 250-character cut drops the rest of a spoken idea and how a pause or topic cue keeps that idea in one retrievable span.
- [x] A 5-slide outline is written in presentation order: problem, baseline, moment boundaries, side-by-side evidence, recommendation.
- [x] Slide 4 cites concrete retrieved ranges from the evaluation output, including one case where baseline context is non-contiguous and one case where the moment span stays intact.
- [x] The outline is short enough to record as a Loom without adding a new demo path beyond `python app.py`.
