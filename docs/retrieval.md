# Retrieval and citations

Documents are parsed to plain text, split into roughly 600 character chunks on paragraph
boundaries, and embedded. Chunk overlap is one paragraph so a sentence split across a
boundary is still retrievable from both sides.

Vectors live in pgvector on the same PostgreSQL instance as the application data, which
keeps backups and permissions in one place. At query time the top eight chunks are fetched
by cosine distance, then a cross-encoder reranks them and the best four reach the prompt.

The prompt refuses to answer when the highest score falls below a floor, and every sentence
in the response must carry a citation to a chunk id. That citation is what makes the system
auditable: a reviewer can click a claim and read the source paragraph.
