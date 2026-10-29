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
from html import escape
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
    files = sorted(f for f in Path(folder).glob("**/*") if f.suffix.lower() in {".md", ".txt"})
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
    nl = chr(10) * 2
    sources = nl.join(f"[{i}] ({d['source']})" + chr(10) + d["text"] for i, (_, d) in enumerate(hits, 1))
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


# --------------------------------------------------------------------------- web ui
PAGE = """<!doctype html><meta charset="utf-8"><title>rag-demo</title>
<style>
 :root{color-scheme:dark}
 *{box-sizing:border-box}
 body{margin:0;background:#0b0f14;color:#c9d1d9;font:15px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}
 .wrap{max-width:860px;margin:0 auto;padding:40px 24px 80px}
 h1{font:600 22px/1.3 ui-monospace,Menlo,monospace;margin:0 0 6px}
 .sub{color:#8b949e;font:13px ui-monospace,Menlo,monospace;margin-bottom:28px}
 form{display:flex;gap:10px;margin-bottom:28px}
 input{flex:1;padding:13px 15px;border-radius:10px;border:1px solid #30363d;background:#0d1117;color:#e6edf3;font-size:15px}
 input:focus{outline:2px solid #1f6feb;border-color:transparent}
 button{padding:13px 20px;border-radius:10px;border:0;background:#238636;color:#fff;font-weight:600;cursor:pointer}
 button:hover{background:#2ea043}
 .card{background:#0d1117;border:1px solid #21262d;border-radius:12px;padding:20px 22px;margin-bottom:14px}
 .lbl{font:11px/1 ui-monospace,Menlo,monospace;letter-spacing:.09em;text-transform:uppercase;color:#6e7681;margin-bottom:12px}
 .ans{white-space:pre-wrap;margin:0;font-size:15px}
 cite{color:#58a6ff;font:600 11px ui-monospace,Menlo,monospace;vertical-align:super;padding:0 1px}
 .refused{color:#d29922}
 .src{display:flex;gap:12px;align-items:baseline;padding:9px 0;border-top:1px solid #161b22;font:13px ui-monospace,Menlo,monospace}
 .src:first-of-type{border-top:0}
 .n{color:#58a6ff;font-weight:700;min-width:18px}
 .file{color:#e6edf3}
 .score{margin-left:auto;color:#8b949e}
 .bar{height:3px;background:#238636;border-radius:2px;min-width:2px}
 .snip{color:#6e7681;font-size:12px;margin-top:2px}
 footer{margin-top:26px;color:#484f58;font:12px ui-monospace,Menlo,monospace}
 a{color:#58a6ff;text-decoration:none}
</style>
<div class="wrap">
  <h1>rag-demo</h1>
  <div class="sub">retrieval + citations over 3 documents &middot; no API key required</div>
  <form method="get" action="/">
    <input name="q" value="{{Q}}" placeholder="Ask something about the documents..." autofocus>
    <button>Ask</button>
  </form>
  {{BODY}}
  <footer>every claim is cited &middot; <a href="https://github.com/alvintheofilus/rag-demo">source</a></footer>
</div>"""


def ask_json(question, k=4):
    hits = retrieve(question, k)
    if not hits or hits[0][0] < 0.08:
        return {"refused": True, "answer": "Not in the provided documents.", "sources": []}
    nl = chr(10) * 2
    sources = nl.join(f"[{i}] ({d['source']})" + chr(10) + d["text"] for i, (_, d) in enumerate(hits, 1))
    if OPENAI_KEY:
        body = {"model": "gpt-4o-mini", "temperature": 0,
                "messages": [{"role": "user", "content": PROMPT.format(sources=sources, question=question)}]}
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            text = json.load(r)["choices"][0]["message"]["content"]
    else:
        # No LLM: answer with the best sentences of the winning chunk, still cited.
        # Crude prefix matching stands in for stemming; real embeddings replace this.
        q = [w for w in WORD.findall(question.lower()) if len(w) > 2]
        body = " ".join(l for l in hits[0][1]["text"].splitlines() if not l.startswith("#"))
        sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", body) if len(x.strip()) > 30]

        def hits_for(sent):
            words = WORD.findall(sent.lower())
            return sum(1 for t in q if any(w == t or (len(w) > 4 and len(t) > 4 and w[:5] == t[:5]) for w in words))

        ranked = sorted(sents, key=lambda x: (-hits_for(x), len(x)))
        keep = [x for x in sents if x in ranked[:2]] or sents[:1]
        text = " ".join(keep) + " [1]"
    return {"refused": False, "answer": text,
            "sources": [{"n": i, "source": d["source"], "score": round(sc, 3), "text": d["text"]}
                        for i, (sc, d) in enumerate(hits, 1) if sc >= 0.01]}


def snip(text, n=110):
    """Snippet without markdown headers, so a doc dump never shows up as an answer."""
    body = " ".join(l for l in text.splitlines() if not l.startswith("#"))
    return body[:n].strip() + ("..." if len(body) > n else "")


def render(result):
    if result["refused"]:
        return f'<div class="card"><div class="lbl">answer</div><p class="ans refused">{escape(result["answer"])}</p></div>'
    ans = escape(result["answer"])
    for i in range(1, len(result["sources"]) + 1):
        ans = ans.replace(f"[{i}]", f"<cite>[{i}]</cite>")
    rows = "".join(
        f'<div class="src"><span class="n">[{s["n"]}]</span>'
        f'<span><span class="file">{escape(s["source"])}</span>'
        f'<div class="snip">{escape(snip(s["text"]))}</div></span>'
        f'<span class="score">{s["score"]}</span></div>' for s in result["sources"])
    mode = "written by the model" if OPENAI_KEY else "extractive mode &middot; no API key"
    return (f'<div class="card"><div class="lbl">answer &middot; {mode}</div>'
            f'<p class="ans">{ans}</p></div>'
            f'<div class="card"><div class="lbl">retrieved sources</div>{rows}</div>')


def serve(port=8010):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlparse, parse_qs

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query).get("q", [""])[0]
            if u.path == "/api/ask":
                payload = json.dumps(ask_json(q)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            body = render(ask_json(q)) if q.strip() else ""
            page = PAGE.replace("{{Q}}", escape(q)).replace("{{BODY}}", body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

    print(f"serving on http://127.0.0.1:{port}/  (Ctrl-C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ask"
    if cmd == "ingest":
        ingest(sys.argv[2] if len(sys.argv) > 2 else "docs")
    elif cmd == "ask":
        answer(" ".join(sys.argv[2:]))
    elif cmd == "eval":
        sys.exit(evaluate())
    elif cmd == "serve":
        serve(int(sys.argv[2]) if len(sys.argv) > 2 else 8010)
    else:
        sys.exit(__doc__)
