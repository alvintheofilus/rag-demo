# Known limits

OCR quality on scanned financial documents is the largest source of wrong answers, not the
model. Tables recovered from PDFs lose their column structure, so numeric questions over
tables are answered from text and should be treated as indicative only.

Long documents are summarised before chunking when they exceed 40 pages, which costs an
extra model call per page and can drop detail the user later asks about.

The reranker adds about 180 milliseconds per query on a two vCPU instance. For chat-style
use that is invisible; for bulk extraction of ten thousand documents it is the bottleneck.
