# Implement Semantic Search to Moment Search

A retrieval comparison on one YouTube transcript. Part 1 is baseline semantic search over fixed chunks. Part 2 is Moment RAG: the same transcript, grouped into spoken moments, retrieved as a whole idea.

The verified source is [Steve Jobs, Stanford commencement address (2005)](https://www.youtube.com/watch?v=UF8uR6Z6KLc). It is one talk with three stories, pauses, and topic cues, so both indexes can be asked the same questions.

## Requirement

**Objective.** Explore Moment RAG by reviewing the approach, understanding its architecture, and implementing a comparable retrieval-augmented generation pipeline on a YouTube video transcript. Build a baseline RAG system with semantic search first, then a Moment RAG-style architecture, and compare the two.

**Task.** Select a YouTube video, extract its transcript, and use it as the knowledge source. The work has two stages.

### Part 1: Baseline Semantic Search RAG

Build a simple RAG pipeline using semantic search.

1. Extract or obtain the transcript from a YouTube video.
2. Split the transcript into meaningful chunks.
3. Generate embeddings for the transcript chunks.
4. Store the embeddings in a vector database or similarity search index.
5. Accept user queries and retrieve the most relevant transcript chunks.
6. Generate an answer using the retrieved context.

### Part 2: Moment RAG implementation

Review the Moment RAG architecture, then implement that approach on the same transcript.

1. Identify meaningful moments within the video transcript.
2. Structure transcript segments around those moments.
3. Improve retrieval by using moment-level context instead of only fixed-size chunks.
4. Compare Moment RAG retrieval results with the baseline semantic search RAG.
5. Explain how Moment RAG changes or improves the quality of answers.

### Required deliverables

Submit one of the following.

**Option 1: Brief presentation.** A short slide deck that includes:

1. The YouTube video selected and why you chose it.
2. A summary of your understanding of the Moment RAG architecture.
3. The baseline semantic search RAG implementation.
4. The Moment RAG-style implementation.
5. Comparison between the two approaches.
6. Key findings, limitations, and possible improvements.

**Option 2: Loom video.** A short walkthrough that includes:

1. Explanation of the selected video and transcript.
2. Demo of the baseline semantic search RAG.
3. Demo of the Moment RAG implementation.
4. Explanation of the architecture and design choices.
5. Comparison of outputs from both systems.
6. Final findings and reflections.

### Expected outcome

1. Understand how standard semantic-search-based RAG works.
2. Explain the limitations of simple transcript chunking.
3. Understand the Moment RAG architecture.
4. Implement a Moment RAG-style retrieval pipeline.
5. Compare retrieval quality between chunk-based and moment-based approaches.
6. Present the technical findings clearly.

## How it works

`app.py` loads one transcript and builds two ChromaDB indexes with the same embedding model, `sentence-transformers/all-MiniLM-L6-v2`.

```text
YouTube captions (or embedded lecture)
        |
        +-- 250-character windows, 50-character overlap --> baseline_collection
        |
        +-- pauses (>= 2s) and topic cues --> moment_collection
                    |
                    +-- each moment stores start_time, end_time, moment_summary
        |
        two queries (plus any question you pass on the command line)
        |
        side-by-side timestamps, context quality, and answers
```

**Ingestion.** `youtube-transcript-api` downloads the English captions for `UF8uR6Z6KLc`. If that request fails, the script continues with an embedded four-topic lecture and questions written for that lecture.

**Part 1, baseline.** Captions are joined into one string and cut every 250 characters with a 50-character overlap. Cuts are character offsets, so they can land mid-word. Each window is embedded and stored in `baseline_collection` with the caption times it touches. A query returns the top 3 windows. The answer is those windows in time order, with `[...]` where they are not contiguous.

**Part 2, Moment RAG.** A new moment starts when the gap since the previous caption is at least 2 seconds, or when the next caption begins with a topic cue such as "the first story" or "my second story". A 120-second cap splits a very long stretch on a caption boundary. The stored document is the full text between `start_time` and `end_time`. `moment_summary` is the opening sentence. The answer uses the top moment only. Other hits are listed and kept out of the answer text.

**Answers.** There is no separate chat-model API in this project. The answer is the retrieved context, formatted with timestamps and a context-quality line. That is the passage a later language model would read. Quality is judged by sentence edges, whether the windows are contiguous, a token estimate (`characters / 4`), and whether known phrases from the question appear.

**What the comparison showed.** On the calligraphy question both pipelines hit 3/3 key phrases. The baseline answer is clipped and includes a Stewart Brand passage from 13:18. Moment RAG returns one intact span, 04:47–05:33. On the death question the baseline hit 0/3 key phrases and mixed in the love-and-loss story. Moment RAG hit "you are going to die" inside 09:21–10:59. "Your time is limited, so don't waste it" sits in a later moment (11:41–12:58), so top-1 moment retrieval does not automatically include the next pause. The full write-up, scorecard, and five-slide outline are in `SUMMARY.md`.

## How to test

From this directory:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

`python app.py` must exit 0. The terminal prints the caption count, the chunk count, the moment inventory, a side-by-side for two benchmark questions, and a scorecard.

Ask your own question as well. It runs after the two benchmarks, through both indexes:

```bash
python app.py "What did Steve Jobs say about the calligraphy class?"
```

### Live notebook for a grader

`live_test.ipynb` is the same pipeline, split into cells so a grader can re-run it and type a question without editing `app.py`.

1. Use the `.venv` interpreter above (`ipykernel` is in `requirements.txt`).
2. Open `live_test.ipynb`.
3. Run All. The first code cell downloads the YouTube captions, builds both indexes, and prints the moment inventory. That cell takes about a minute the first time, while the embedding model downloads into `.cache/`.
4. Read the benchmark cell. It prints the same side-by-side and scorecard as `python app.py`.
5. Edit `LIVE_QUERY` in the last cell and run that cell again. The indexes stay loaded, so only the new question is retrieved.

```python
LIVE_QUERY = "What did Steve Jobs say about the calligraphy class?"
```

If YouTube blocks the caption request, the notebook prints the error and continues on the embedded lecture. The comparison still runs.

What to look for:

| Check | Baseline | Moment RAG |
| --- | --- | --- |
| Timestamps | Three short ranges | One start-to-end moment, plus alternate ranges |
| Context | Often starts or ends mid-sentence, with `[...]` gaps | One intact span and a moment summary |
| Answer | Stitched windows, which can include a neighboring story | The full top moment only |

The first run downloads the embedding model into `.cache/` and writes the indexes to `chroma_db/`. Both are local and are listed in `.gitignore`.

If YouTube blocks the caption request, the script prints the error and uses the embedded lecture. The comparison still runs, with questions written for that lecture.

## Requirement checklist

| Requirement | Where it is | Status |
| --- | --- | --- |
| Extract a YouTube transcript | `load_transcript()` in `app.py`, video `UF8uR6Z6KLc` | Done |
| Split into chunks | 250 characters, 50 overlap, `baseline_collection` | Done |
| Generate embeddings | `all-MiniLM-L6-v2` | Done |
| Store embeddings | ChromaDB | Done |
| Accept a query and retrieve chunks | Two benchmark questions, `python app.py "your question"`, and `LIVE_QUERY` in `live_test.ipynb` | Done |
| Answer from retrieved context | Formatted extractive answer from the top 3 chunks | Done |
| Identify moments | Pauses of at least 2 seconds and topic cues | Done |
| Structure segments around moments | `start_time`, `end_time`, `moment_summary` | Done |
| Retrieve moment-level context | Top moment's full text in `moment_collection` | Done |
| Compare the two approaches | Side-by-side output and `SUMMARY.md` | Done |
| Explain the quality change | Findings in `SUMMARY.md` | Done |
| Presentation outline | Five-slide outline in `SUMMARY.md` | Done |
| Recorded Loom file | The outline is the script. No video file is in this repo. | Outline only |

Linear project: [Semantic-Search](https://linear.app/fdem/project/semantic-search-64c96975b4d3). The five issues FDE-170 through FDE-174 are Done and follow this same split: transcript, Part 1, Part 2, comparison, presentation or Loom.
