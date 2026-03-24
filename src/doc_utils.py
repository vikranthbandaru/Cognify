"""
doc_utils.py - Robust My Documents system for Cognify.

No dependency on Wikipedia data, wiki_data, or TOPICS.
Uses the shared LLM (llm.py) for generation but all retrieval
and indexing logic is completely self-contained.

Pipeline:
  PDF / URL
    > text extraction  (PyMuPDF for PDFs, trafilatura for URLs)
    > overlapping text chunking (300-word chunks, 50-word overlap)
    > PageIndex tree (heading-aware hierarchical structure)
    > stored in session_state.user_index

  Query
    > context-aware expansion (resolve pronouns)
    > keyword pre-filter over tree nodes (no LLM)
    > LLM navigates tree > picks relevant node IDs
    > retrieve full node text
    > LLM generates answer
"""

import re
import json
import uuid
import hashlib
import random
import traceback
from typing import Optional

import fitz                          # PyMuPDF
import requests
from bs4 import BeautifulSoup
from llm import chat

# Try importing trafilatura (it's in requirements.txt)
try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False


# ══════════════════════════════════════════════
# 1. TEXT EXTRACTION
# ══════════════════════════════════════════════

def _extract_pdf_structured(file_bytes: bytes) -> list[dict]:
    """
    Extract text from a PDF with heading detection using font size analysis.

    Returns list of {"level": int, "title": str, "text": str} dicts,
    where level 0 = top-level heading, 1 = sub-heading, 2 = body block.

    Falls back to page-based chunking if heading detection produces
    too few sections (e.g. uniform-font PDFs).
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    blocks_raw = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:   # skip images
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    blocks_raw.append({
                        "text":  text,
                        "size":  round(span.get("size", 10), 1),
                        "bold":  bool(span.get("flags", 0) & 2**4),
                        "page":  page_num + 1,
                    })

    total_pages = len(doc)
    doc.close()

    if not blocks_raw:
        return []

    # ── Try heading-based extraction first ────────────────
    sections = _heading_based_sections(blocks_raw)

    # ── Fallback: page-based chunking ─────────────────────
    # If heading detection produced ≤ 1 section but we have
    # substantial text, split by page instead.
    total_text_len = sum(len(b["text"]) for b in blocks_raw)
    if len(sections) <= 1 and total_text_len > 500:
        sections = _page_based_sections(blocks_raw, total_pages)

    return sections


def _heading_based_sections(blocks_raw: list[dict]) -> list[dict]:
    """Original heading detection via font-size analysis."""
    sizes = sorted({b["size"] for b in blocks_raw}, reverse=True)
    body_size = sorted([b["size"] for b in blocks_raw])[len(blocks_raw) // 2]
    heading_sizes = [s for s in sizes if s > body_size * 1.1][:3]

    def get_level(block):
        if block["size"] in heading_sizes:
            return heading_sizes.index(block["size"])
        if block["bold"] and block["size"] >= body_size:
            return 2
        return None   # body text

    sections = []
    current_heading = {"level": 0, "title": "Introduction", "text": ""}

    for b in blocks_raw:
        level = get_level(b)
        if level is not None:
            if current_heading["text"].strip():
                sections.append(current_heading)
            current_heading = {"level": level, "title": b["text"], "text": ""}
        else:
            current_heading["text"] += " " + b["text"]

    if current_heading["text"].strip():
        sections.append(current_heading)

    return sections


def _page_based_sections(blocks_raw: list[dict], total_pages: int) -> list[dict]:
    """Fallback: group text by page when heading detection fails."""
    page_texts = {}
    for b in blocks_raw:
        page = b["page"]
        if page not in page_texts:
            page_texts[page] = []
        page_texts[page].append(b["text"])

    sections = []
    for page_num in sorted(page_texts.keys()):
        text = " ".join(page_texts[page_num]).strip()
        if text:
            sections.append({
                "level": 0,
                "title": f"Page {page_num}",
                "text": text,
            })

    return sections


def _extract_url_structured(url: str) -> list[dict]:
    """
    Fetch a web page and extract content structured by headings.

    Uses trafilatura (purpose-built for web content extraction) as primary.
    Falls back to BeautifulSoup if trafilatura is unavailable or returns nothing.

    Returns list of {"level": int, "title": str, "text": str} dicts.
    """
    # ── Primary: trafilatura ──────────────────────────────
    if HAS_TRAFILATURA:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                # Extract with full settings for maximum content
                text = trafilatura.extract(
                    downloaded,
                    include_tables=True,
                    include_comments=False,
                    include_links=False,
                    favor_recall=True,       # prefer getting more content
                    output_format="txt",
                )
                if text and len(text.strip()) > 100:
                    return _split_text_into_sections(text, url)
        except Exception:
            pass  # fall through to BeautifulSoup

    # ── Fallback: BeautifulSoup ───────────────────────────
    return _extract_url_beautifulsoup(url)


def _split_text_into_sections(text: str, url: str) -> list[dict]:
    """
    Split trafilatura plain-text output into sections by detecting
    lines that look like headings (short, possibly uppercase, no trailing punctuation).
    """
    lines = text.split("\n")
    sections = []
    current = {"level": 0, "title": url, "text": ""}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Heuristic: short lines (< 80 chars) that don't end with sentence
        # punctuation are likely headings
        is_heading = (
            len(stripped) < 80
            and not stripped.endswith((".", ",", ";", ":", "?", "!"))
            and len(stripped.split()) < 12
            and len(stripped) > 2
        )

        if is_heading and current["text"].strip():
            sections.append(current)
            current = {"level": 1, "title": stripped, "text": ""}
        else:
            current["text"] += " " + stripped

    if current["text"].strip():
        sections.append(current)

    # If everything ended up in one section, do paragraph-based splitting
    if len(sections) <= 1 and len(text) > 800:
        return _paragraph_based_sections(text, url)

    return sections


def _paragraph_based_sections(text: str, url: str) -> list[dict]:
    """Split text into sections of ~3 paragraphs each."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    sections = []
    chunk_size = 3  # paragraphs per section
    for i in range(0, len(paragraphs), chunk_size):
        group = paragraphs[i:i + chunk_size]
        title = group[0][:60] + "..." if len(group[0]) > 60 else group[0]
        sections.append({
            "level": 0,
            "title": f"Section {i // chunk_size + 1}: {title}",
            "text": " ".join(group),
        })

    return sections if sections else [{"level": 0, "title": url, "text": text}]


def _extract_url_beautifulsoup(url: str) -> list[dict]:
    """BeautifulSoup fallback for URL extraction."""
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Cognify/2.0"})
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Could not fetch URL: {e}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove scripts, styles, navbars
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    heading_tags = {"h1": 0, "h2": 1, "h3": 2, "h4": 3, "h5": 3}
    content_tags = {"p", "li", "td", "th", "dd", "dt", "blockquote", "pre", "figcaption"}
    sections = []
    current = {"level": 0, "title": soup.title.string if soup.title else url, "text": ""}

    for element in soup.find_all(list(heading_tags.keys()) + list(content_tags)):
        tag = element.name
        text = element.get_text(" ", strip=True)
        if not text or len(text) < 3:
            continue
        if tag in heading_tags:
            if current["text"].strip():
                sections.append(current)
            current = {"level": heading_tags[tag], "title": text, "text": ""}
        else:
            current["text"] += " " + text

    if current["text"].strip():
        sections.append(current)

    return sections


# ══════════════════════════════════════════════
# 2. TEXT CHUNKING
# ══════════════════════════════════════════════

def _chunk_sections(sections: list[dict],
                    max_words: int = 400,
                    chunk_words: int = 300,
                    overlap_words: int = 50) -> list[dict]:
    """
    Split sections that exceed max_words into overlapping chunks.
    Each chunk inherits the parent section's title and level.
    Short sections pass through unchanged.
    """
    chunked = []

    for sec in sections:
        words = sec["text"].split()
        if len(words) <= max_words:
            chunked.append(sec)
            continue

        # Split into overlapping chunks
        start = 0
        part_num = 1
        while start < len(words):
            end = start + chunk_words
            chunk_text = " ".join(words[start:end])
            chunked.append({
                "level": sec["level"],
                "title": f"{sec['title']} (part {part_num})",
                "text": chunk_text,
            })
            start = end - overlap_words  # overlap
            part_num += 1

    return chunked


# ══════════════════════════════════════════════
# 3. PAGEINDEX TREE BUILDER
# ══════════════════════════════════════════════

def _make_node(title: str, text: str, level: int, children: list = None) -> dict:
    return {
        "id":       uuid.uuid4().hex[:10],
        "title":    title.strip()[:120],
        "level":    level,
        "text":     text.strip(),
        "summary":  text.strip()[:500] + ("..." if len(text) > 500 else ""),
        "children": children or [],
    }


def _build_tree(sections: list[dict]) -> dict:
    """
    Convert a flat list of {level, title, text} sections into a
    PageIndex-style hierarchical tree.
    """
    root = _make_node("Document Root", "", -1)
    stack = [root]   # stack[-1] is current parent

    for sec in sections:
        node = _make_node(sec["title"], sec["text"], sec["level"])

        # Pop stack until we find a parent at a higher level
        while len(stack) > 1 and stack[-1]["level"] >= sec["level"]:
            stack.pop()

        stack[-1]["children"].append(node)
        stack.append(node)

    return root


def _flatten_tree(node: dict) -> list[dict]:
    """Flatten a tree into a list of all nodes (for keyword search)."""
    result = []
    if node.get("text"):
        result.append(node)
    for child in node.get("children", []):
        result.extend(_flatten_tree(child))
    return result


def _tree_summary(node: dict, depth: int = 0, max_depth: int = 3) -> str:
    """
    Build a compact text representation of the tree for LLM navigation.
    Format: [id] Title: summary...
    """
    if depth > max_depth:
        return ""
    indent = "  " * depth
    line = f"{indent}[{node['id']}] {node['title']}: {node['summary'][:200]}\n"
    for child in node.get("children", []):
        line += _tree_summary(child, depth + 1, max_depth)
    return line


def _find_node_by_id(node: dict, target_id: str) -> Optional[dict]:
    """Recursively find a node by its ID."""
    if node.get("id") == target_id:
        return node
    for child in node.get("children", []):
        found = _find_node_by_id(child, target_id)
        if found:
            return found
    return None


# ══════════════════════════════════════════════
# 4. PUBLIC INDEXING API
# ══════════════════════════════════════════════

def index_pdf(file_bytes: bytes, filename: str) -> tuple[dict, str]:
    """
    Extract and build a PageIndex tree from a PDF.

    Returns:
        (doc_entry, status_message)
        doc_entry = {"source": str, "type": "pdf", "tree": dict, "node_count": int}
    """
    try:
        sections = _extract_pdf_structured(file_bytes)
        if not sections:
            return {}, "Could not extract text from this PDF. It may be scanned or image-only."

        # Chunk long sections before building tree
        sections = _chunk_sections(sections)

        tree = _build_tree(sections)
        flat = _flatten_tree(tree)
        entry = {"source": filename, "type": "pdf", "tree": tree, "node_count": len(flat)}
        return entry, f"Indexed '{filename}' - {len(flat)} sections across {len(sections)} headings."
    except Exception as e:
        return {}, f"Error processing PDF: {e}"


def index_url(url: str) -> tuple[dict, str]:
    """
    Fetch and build a PageIndex tree from a web page.

    Returns:
        (doc_entry, status_message)
        doc_entry = {"source": str, "type": "url", "tree": dict, "node_count": int}
    """
    try:
        sections = _extract_url_structured(url)
        if not sections:
            return {}, f"Could not extract content from: {url}"

        # Chunk long sections before building tree
        sections = _chunk_sections(sections)

        tree = _build_tree(sections)
        flat = _flatten_tree(tree)
        entry = {"source": url, "type": "url", "tree": tree, "node_count": len(flat)}
        return entry, f"Indexed {url} - {len(flat)} sections."
    except Exception as e:
        return {}, f"Error fetching URL: {e}"


# ══════════════════════════════════════════════
# 5. PAGEINDEX RAG PIPELINE (fully independent)
# ══════════════════════════════════════════════

_STOP = {
    "what", "is", "the", "a", "an", "are", "do", "does", "did",
    "can", "could", "would", "should", "tell", "me", "about",
    "of", "in", "on", "at", "to", "for", "with", "and",
    "or", "but", "not", "its", "it", "this", "that", "was", "were",
    "been", "be", "have", "has", "had", "will", "by", "from", "as",
}


def _keyword_score(node: dict, query_words: set) -> float:
    """
    Score a node by keyword overlap with query.
    Uses exact match + substring match for robustness.
    """
    title_lower = node.get("title", "").lower()
    text_lower  = node.get("text",  "").lower()
    title_words = set(title_lower.split())
    text_words  = set(text_lower.split())

    # Exact word matches
    title_hits  = len(query_words & title_words) * 3.0
    text_hits   = len(query_words & text_words)

    # Substring matches - catches abbreviations, partial words, compounds
    substring_hits = 0.0
    for qw in query_words:
        if len(qw) < 3:
            continue
        if qw in title_lower:
            substring_hits += 2.0
        elif qw in text_lower:
            substring_hits += 0.5

    return title_hits + text_hits + substring_hits


def _keyword_prefilter(flat_nodes: list[dict], query: str, top_n: int = 60) -> list[dict]:
    """
    Fast keyword pass to narrow nodes before LLM step.
    Falls back to proportional sampling if too few results.
    """
    words = set(query.lower().split()) - _STOP
    if not words:
        return flat_nodes[:top_n]

    scored = [(node, _keyword_score(node, words)) for node in flat_nodes]
    scored_hits = [(n, s) for n, s in scored if s > 0]
    scored_hits.sort(key=lambda x: -x[1])
    results = [n for n, _ in scored_hits[:top_n]]

    # Fallback: if keyword filter found < 5 results, pad with random nodes
    # to ensure the LLM has enough context to work with
    if len(results) < 5 and len(flat_nodes) > len(results):
        remaining = [n for n in flat_nodes if n not in results]
        pad_count = min(10, len(remaining))
        results.extend(random.sample(remaining, pad_count))

    return results


def _llm_select_nodes(query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
    """
    PageIndex step: LLM reasons over candidate node summaries, picks most relevant IDs.
    """
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates

    listing = "\n".join(
        f"[{n['id']}] {n['title']}: {n['summary'][:300]}"
        for n in candidates
    )
    prompt = (
        f"You are a retrieval assistant. Select the {top_k} most relevant sections "
        f"for answering this query.\n\n"
        f"Query: \"{query}\"\n\n"
        f"Sections:\n{listing}\n\n"
        f"Return ONLY the section IDs separated by commas, nothing else."
    )
    try:
        result = chat([{"role": "user", "content": prompt}], max_tokens=120)
        ids = [x.strip() for x in result.split(",") if x.strip()]
        id_set = {n["id"] for n in candidates}
        selected = [n for n in candidates if n["id"] in ids and n["id"] in id_set]
        return selected[:top_k] if selected else candidates[:top_k]
    except Exception:
        return candidates[:top_k]


def _expand_query(query: str, history: list[dict]) -> str:
    """Resolve pronouns/context from chat history into a self-contained query."""
    if not history:
        return query
    recent = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-4:]
    )
    prompt = (
        f"Rewrite this query to be fully self-contained using the conversation history.\n"
        f"Return ONLY the rewritten query.\n\n"
        f"History:\n{recent}\n\nQuery: {query}\nRewritten:"
    )
    try:
        return chat([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=120)
    except Exception:
        return query


def answer_from_docs(
    user_index: dict,
    question: str,
    history: list[dict],
    selected_sources: list[str] = None,
) -> tuple[str, list[dict]]:
    """
    Fully independent RAG pipeline for user documents.

    Args:
        user_index:       {source_name: doc_entry} from session state
        question:         User's question
        history:          Chat history [{"role": ..., "content": ...}]
        selected_sources: Limit search to these source names (None = all)

    Returns:
        (answer_string, source_nodes_used)
    """
    if not user_index:
        return "No documents are indexed yet. Upload a PDF or add a URL in the My Documents tab.", []

    try:
        # ── Step 1: Expand query using history ────────────────────
        expanded = _expand_query(question, history)

        # ── Step 2: Gather all nodes from selected sources ────────
        sources_to_search = (
            [s for s in selected_sources if s in user_index]
            if selected_sources else list(user_index.keys())
        )

        all_nodes = []
        for src in sources_to_search:
            entry = user_index[src]
            nodes = _flatten_tree(entry["tree"])
            all_nodes.extend(nodes)

        if not all_nodes:
            return "The selected documents have no indexed content.", []

        # ── Step 3: Keyword pre-filter ────────────────────────────
        candidates = _keyword_prefilter(all_nodes, expanded, top_n=60)

        # ── Step 4: LLM selects most relevant nodes (PageIndex) ───
        selected = _llm_select_nodes(expanded, candidates, top_k=10)

        if not selected:
            return "I couldn't find relevant sections in your documents for this question.", []

        # ── Step 5: Build context from selected nodes ─────────────
        context = "\n\n---\n\n".join(
            f"[{n['title']}]\n{n['text']}" for n in selected
        )

        recent_history = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history[-6:]
        )

        # ── Step 6: Generate answer ───────────────────────────────
        prompt = (
            "You are a knowledgeable assistant answering questions from user-provided documents. "
            "Use the document excerpts below as your primary source of information.\n\n"
            "Guidelines:\n"
            "- Synthesize information across multiple excerpts when relevant.\n"
            "- Quote or reference specific sections when possible.\n"
            "- If the excerpts contain partial information, provide what you can and note what's missing.\n"
            "- Only say you cannot answer if the excerpts are truly unrelated to the question.\n\n"
            f"Document excerpts:\n{context}\n\n"
            f"Conversation history:\n{recent_history}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        answer = chat([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=800)
        return answer, selected

    except Exception as e:
        traceback.print_exc()
        return f"An error occurred: {e}", []


def evaluate_doc_answer(question: str, answer: str) -> int:
    """Rate answer relevance 1-10 (independent of Wikipedia pipeline)."""
    prompt = (
        f"Rate how well this answer addresses the question. Score 1-10. "
        f"Reply with a single number only.\n\n"
        f"Question: {question}\nAnswer: {answer}\nScore:"
    )
    try:
        result = chat([{"role": "user", "content": prompt}], max_tokens=5)
        digits = "".join(c for c in result if c.isdigit())
        return min(10, max(1, int(digits[:2]))) if digits else 5
    except Exception:
        return 5


def source_id(source: str) -> str:
    """Stable short ID for a source name."""
    return hashlib.md5(source.encode()).hexdigest()[:8]
