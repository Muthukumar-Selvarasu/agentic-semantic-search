# Semantic Search to Moment Search

Executive summary of a verified run of `python app.py` (exit code 0) on the Stanford commencement address by Steve Jobs, 2005 (`UF8uR6Z6KLc`, 244 English captions). Both indexes use `sentence-transformers/all-MiniLM-L6-v2` and ChromaDB. The baseline collection is `baseline_collection` (61 windows). The moment collection is `moment_collection` (15 moments). If the caption API is blocked, the same script falls back to an embedded multi-topic lecture and asks questions written for that lecture. This run used the live YouTube track.

## Comparison

| Dimension | Baseline fixed chunking | Moment RAG |
| --- | --- | --- |
| Granularity | 250-character windows with a 50-character overlap. The 15-minute talk becomes 61 windows. Cuts are character offsets, so they land mid-word and mid-sentence. | 15 moments. A boundary opens on a pause of at least 2 seconds, on a topic cue ("the first story", "my second story", "my third story"), or on a 120-second cap that still splits on a caption edge. |
| Context loss | 61 of 61 chunks break a sentence edge. The calligraphy answer opens on the fragment "iceless later on" and joins three windows with `[...]` gaps. The death question's nearest window (11:44–12:02) sits at the start of the advice moment and closes before "Your time is limited, so don't waste it", which is character 744 of that 1,125-character moment. | The retrieved unit is the full text between `start_time` and `end_time`, plus `moment_summary`. Question 1 returns 04:47–05:33 as one intact span (the dropout, the calligraphy class, and the dots moral). Question 2 returns 09:21–10:59 as one intact span (the mirror test through the cancer diagnosis). |
| Precision | Top-3 merge pulls other stories into the answer. Question 1 includes 13:18–13:34, the Stewart Brand / Whole Earth passage, inside a calligraphy answer. Question 2 includes 07:54–08:12, NeXT and Laurene from the love-and-loss story, and matches 0 of 3 key phrases (`going to die`, `time is limited`, `don't waste it`). | The answer is the top moment only, so an alternate hit cannot leak into the text. Question 1 matches 3 of 3 key phrases inside moment 6. Question 2 matches `going to die` inside moment 10. The lines `time is limited` and `don't waste it` are intact inside moment 12 (11:41–12:58); that moment was outside the top 3, so the closer diagnosis moment was the one returned. |
| Token overhead | About 188 tokens for three windows on both questions. The bill is small because each window is capped, and the overlap repeats words the speaker already said. | Question 1 costs about 185 tokens for a short complete moment, the same budget as three fragments, spent on one time range. Question 2 costs about 375 tokens, roughly twice the baseline, to keep the diagnosis passage whole. |

Scorecard from the run:

| Question | Baseline tokens | Moment tokens | Baseline key details | Moment key details |
| --- | ---: | ---: | ---: | ---: |
| Connecting the dots and the calligraphy class | 188 | 185 | 3/3 | 3/3 |
| Death, and how that should change the way you live | 188 | 375 | 0/3 | 1/3 |

## Core findings

A lecture transcript is a sequence of ideas with pauses between them. Fixed chunking ignores those pauses and spends a 250-character budget that closes while the sentence is still going. On this talk every baseline window is clipped. The evaluation shows two concrete forms of that loss.

The first form is a cut in the middle of a word and a jump across the timeline. For the calligraphy question the baseline returns 04:51–05:07, 03:26–03:49, and 13:18–13:34. The formatted answer therefore starts mid-word, skips with `[...]`, and then quotes Stewart Brand from ten minutes later because "personal computers" and "publishing" are near the typography passage in embedding space. Moment RAG returns moment 6, 04:47–05:33, whose summary is "If I had never dropped out, I would have never dropped in on this calligraphy class." The same three key phrases are present, and they sit in one seekable range. Key-detail coverage is tied; the context around those details is the part that changes.

The second form is a window that finds the right neighborhood and still misses the sentence that answers the question. "Your time is limited, so don't waste it living someone else's life" lives in moment 12. A baseline hit lands on 11:44–12:02, the opening of that same moment, and the character cap ends the chunk before character 744, where the sentence actually is. The moment index stores that sentence together with the sentences that lead into it. Retrieval still has to rank moment 12 first. On this question it ranked the earlier death moment (09:21–10:59) first, so the answer contains "you are going to die" and the mirror test, and the closing imperative stays in the next pause-bounded moment. Pause boundaries fix clipping inside an idea. They also mean a story told across several pauses is several moments, and top-1 retrieval returns one of them.

The token trade is visible in the scorecard. A short moment (question 1) costs the same ~185 tokens as three clipped windows and removes the cross-topic jump. A longer moment (question 2) costs ~375 tokens and buys the uninterrupted diagnosis. That extra context is the price of handing a reader, or a later language model, a passage that starts and ends on the speaker's idea.

## Five-slide outline

### Slide 1 — Video answers are moments

- Title: Semantic Search to Moment Search
- A commencement talk is three stories plus a closing charge, spoken with pauses between them.
- The index under test is the Stanford 2005 address: 244 captions, one embedding model, two collections.
- Claim to test: a 250-character window will answer with a fragment; a pause-and-cue moment will answer with the idea.
- Demo path for the Loom is a single command, `python app.py`, which prints the side-by-side.

### Slide 2 — Baseline: fixed windows

- Join the captions, then cut every 250 characters with 50 characters of overlap.
- Embed each window and store it in `baseline_collection` with the caption times the window touches.
- Retrieve the top 3 windows and stitch them in time order, marking gaps with `[...]`.
- Result on this file: 61 windows, and all 61 break a sentence edge.
- Show the calligraphy answer opening on "iceless later on."

### Slide 3 — Moment RAG: pauses and topic cues

- Walk the captions in order. Open a new moment when the clock gap is at least 2 seconds, or when a caption starts with a topic cue such as "my third story is about death."
- A 120-second cap splits a very long stretch on a caption boundary so one silent region cannot become the whole video.
- Store the full moment text with `start_time`, `end_time`, and `moment_summary` in `moment_collection`.
- This talk yields 15 moments. Story cues land on "connecting the dots" (00:56), "love and loss" (05:39), and "death" (09:05).
- Retrieval returns that whole span. Alternate moments are listed and kept out of the answer text.

### Slide 4 — Side-by-side evidence

- Question 1, calligraphy and the dots. Baseline timestamps: 04:51–05:07, 03:26–03:49, 13:18–13:34 (Stewart Brand). Moment timestamp: 04:47–05:33, intact, 3/3 key details, ~185 tokens against ~188.
- Read the moment answer through "you can only connect them looking backwards" as one paragraph with no gap marker.
- Question 2, death and how to live. Baseline timestamps: 09:34–09:55, 11:44–12:02, 07:54–08:12 (love and loss), 0/3 key details, ~188 tokens. Moment timestamp: 09:21–10:59, intact, includes "you are going to die", ~375 tokens.
- Point at moment 12 (11:41–12:58): "Your time is limited, so don't waste it" is in the index, past the baseline window that ends at 12:02, and it was not the top-ranked moment.
- Scorecard line to leave on screen: Q1 tied on details with a cleaner span; Q2 moment 1/3 vs baseline 0/3.

### Slide 5 — What to ship

- Index lectures as moments when the user needs a seek time and a complete idea: same token cost on short moments, about 2× on a diagnosis-length moment.
- Keep fixed windows only as a debug view. On this transcript they are a reliable way to clip a sentence and to import a neighboring story.
- Next retrieval step, after boundary quality: rank a moment using the full idea, and consider the following moment when the question asks for the charge that comes after a pause. Moment 12 is the example to show.
- Close the Loom on the two ranges worth remembering: 04:47–05:33 for a whole answer, and 11:44–12:02 for a window that stopped thirty seconds early.
