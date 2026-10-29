#!/usr/bin/env python3
"""Tiny, dependency-free RAG pipeline: chunk -> embed -> index -> retrieve -> answer with citations.

    python3 rag.py ingest docs/
    python3 rag.py ask "how are webhooks retried?"
    python3 rag.py eval

Embeddings are local by default (hashing + IDF, no model download, no API key).
Set RAG_EMBEDDER=openai (+ OPENAI_API_KEY) to swap in real embeddings;
set OPENAI_API_KEY on its own to get an LLM-written answer instead of the
extractive fallback. Both are optional: the demo runs offline.
"""
import json
import math
import os
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

INDEX = Path(__file__).with_name("index.json")
DIM = 512
WORD = re.compile(r"[a-z0-9]+")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
EMBEDDER = os.environ.get("RAG_EMBEDDER", "openai" if OPENAI_KEY else "local").strip()


# --------------------------------------------------------------------------- chunk
def chunks(path: Path, size=600, overlap=1):
    """Paragraph-packed chunks; `overlap` paragraphs are repeated for context continuity."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", path.read_text()) if p.strip()]
    out, buf = [], []
    for p in paras:
        buf.append(p)
        if sum(len(x) for x in buf) >= size:
            out.append(buf[:])
            buf = buf[-overlap:] if overlap else []
    if buf:
        out.append(buf)
    return [{"source": path.name, "text": "\n\n".join(b)} for b in out]


# --------------------------------------------------------------------------- embed
def local_vec(text, idf):
    """Signed hashing bag-of-words, IDF-weighted, L2-normalised. Deterministic, no deps."""
    vec = [0.0] * DIM
    for word, n in Counter(WORD.findall(text.lower())).items():
        h = hash_word(word)
        vec[h % DIM] += (1 + math.log(n)) * idf.get(word, 1.0) * (1 if h % 2 else -1)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def hash_word(word):
    h = 2166136261
    for ch in word:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


def openai_embed(texts):
    body = {"model": "text-embedding-3-small", "input": texts}
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)["data"]
    return [d["embedding"] for d in data]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


# --------------------------------------------------------------------------- index
def ingest(folder):
    files = sorted(Path(folder).glob("**/*"))
    docs = [c for f in files if f.is_file() for c in chunks(f)]
    idf = dict(Counter(w for d in docs for w in set(WORD.findall(d["text"].lower()))))
    n = len(docs)
    idf = {w: math.log(1 + n / c) for w, c in idf.items()}
    if EMBEDDER == "openai":
        vecs = openai_embed([d["text"] for d in docs])
    else:
        vecs = [local_vec(d["text"], idf) for d in docs]
    for d, v in zip(docs, vecs):
        d["vec"] = v
    INDEX.write_text(json.dumps({"embedder": EMBEDDER, "idf": idf, "docs": docs}))
    print(f"indexed {len(docs)} chunks from {len(set(d['source'] for d in docs))} files -> {INDEX.name}")


def load():
    if not INDEX.exists():
        sys.exit("no index; run: python3 rag.py ingest docs/")
    return json.loads(INDEX.read_text())


def embed_one(text, idx):
    return openai_embed([text])[0] if idx["embedder"] == "openai" else local_vec(text, idx["idf"])


def retrieve(question, k=4):
    idx = load()
    qv = embed_one(question, idx)
    scored = sorted(((cosine(qv, d["vec"]), d) for d in idx["docs"]), key=lambda x: -x[0])
    return scored[:k]


# --------------------------------------------------------------------------- answer
PROMPT = """Answer only from the numbered sources. Cite every claim like [1]. If the sources do
not contain the answer, say "not in the provided documents" and stop. No preamble.

{sources}

Question: {question}
Answer:"""


def answer(question, k=4):
    hits = retrieve(question, k)
    if not hits or hits[0][0] < 0.08:
        print("No chunk passed the relevance floor — refusing to answer.")
        return
    sources = "\n\n".join(f"[{i}] ({d['source']})\n{d['text']}" for i, (_, d) in enumerate(hits, 1))
    if OPENAI_KEY:
        body = {"model": "gpt-4o-mini", "temperature": 0,
                "messages": [{"role": "user", "content": PROMPT.format(sources=sources, question=question)}]}
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            reply = json.load(r)["choices"][0]["message"]["content"]
    else:
        # extractive fallback: the best chunk IS the answer, honestly labelled as such
        reply = f"(extractive answer — set OPENAI_API_KEY for a written one)\n\n{hits[0][1]['text']}\n\n[1]"
    print(reply.strip())
    print("\nSources: " + "; ".join(f"[{i}] {d['source']}" for i, (_, d) in enumerate(hits, 1)))


# --------------------------------------------------------------------------- eval
def evaluate(path="questions.json"):
    """hit@1 / hit@k against a golden set: {"question": ..., "expect": "substring in the right chunk"}"""
    golden = json.loads(Path(__file__).with_name(path).read_text())
    h1 = hk = 0
    for case in golden:
        hits = retrieve(case["question"], 4)
        where = [i for i, (_, d) in enumerate(hits) if case["expect"].lower() in d["text"].lower()]
        h1 += bool(where) and where[0] == 0
        hk += bool(where)
        print(f"{'PASS' if where else 'MISS'} hit@{where[0]+1 if where else '-'}: {case['question']}")
    print(f"\nhit@1 {h1}/{len(golden)}  hit@4 {hk}/{len(golden)}")
    return 0 if hk == len(golden) else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ask"
    if cmd == "ingest":
        ingest(sys.argv[2] if len(sys.argv) > 2 else "docs")
    elif cmd == "ask":
        answer(" ".join(sys.argv[2:]))
    elif cmd == "eval":
        sys.exit(evaluate())
    else:
        sys.exit(__doc__)
