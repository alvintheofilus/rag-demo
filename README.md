# rag-demo

A retrieval-augmented answering pipeline in one dependency-free Python file, built to show
the parts that decide whether a RAG system is trustworthy: **chunking, retrieval, a
relevance floor that refuses to answer, and citations on every claim.**

```bash
python3 rag.py ingest docs/                 # chunk + embed + index
python3 rag.py ask "how are duplicate webhooks avoided?"
python3 rag.py eval                         # hit@1 / hit@4 against a golden set
```

No install step, no model download, no API key. `index.json` is generated.

## What it demonstrates

| Concern | Where |
|---|---|
| Paragraph-aware chunking with overlap | `chunks()` |
| Embeddings with a swappable backend | `local_vec()` / `openai_embed()`, `RAG_EMBEDDER=openai` |
| Cosine retrieval + relevance floor | `retrieve()`, `answer()` |
| Grounded answering with `[n]` citations | `PROMPT`, `answer()` |
| Retrieval quality measured, not assumed | `evaluate()` + `questions.json` |

## Sample run

```
$ python3 rag.py eval
PASS hit@1: how many times does a failed webhook retry and what is the backoff
PASS hit@1: how are duplicate webhooks from the same provider avoided
PASS hit@1: where are the vectors stored
PASS hit@2: what happens when confidence is too low to answer
PASS hit@1: what is the biggest cause of wrong answers
PASS hit@1: how much latency does the reranker add

hit@1 5/6  hit@4 6/6
```

## Design decisions

- **Local embeddings by default.** Hashing + IDF vectors are weak but deterministic and free,
  so `eval` is reproducible on any machine and CI needs no secrets. Set `RAG_EMBEDDER=openai`
  to swap in `text-embedding-3-small` without touching the pipeline.
- **Refusing beats guessing.** If the top chunk scores below the floor, `ask` says so instead
  of answering from whatever scored highest. In an audit-heavy domain a wrong answer costs
  more than no answer.
- **Answering is optional.** Without `OPENAI_API_KEY`, `ask` returns the winning chunk
  labelled "extractive" rather than pretending a model wrote it.
- **Eval is part of the CLI.** A RAG change that cannot be measured is a guess.

## Limits (on purpose, documented rather than hidden)

- The local embedder has no semantic understanding: it matches vocabulary, so paraphrases miss.
- No reranker here; the production version added a cross-encoder and paid ~180 ms per query.
- Single-file on purpose — this is a readable reference, not a framework.

## Production notes

The system this demo distils ran on PostgreSQL + pgvector with BullMQ workers: webhooks
persisted before acknowledgement, idempotency keyed on `tenant_id + message_id`, constant-time
dedupe in Redis, audit log on every state change, and a dead-letter queue with owner alerts.
Those pieces live in private client repositories; this repository is a clean-room
reimplementation of the retrieval path using only its own sample documents.
