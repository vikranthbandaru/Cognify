# Cognify
**Fully Local Wikipedia Intelligence | PageIndex Retrieval | Zero Cloud Dependencies**

Cognify is a locally-run Q&A system that answers questions from two knowledge sources: a built-in library of 50,000 Wikipedia article summaries across 10 topics, and any documents you upload yourself (PDFs and web pages). It runs entirely on your machine with a local LLM via Ollama. No API keys, no cloud subscriptions, no internet connection needed after the initial model download.

---

## What Changed: v1 vs Cognify

| Area | v1 (Data-Driven Wikipedia Q&A Chatbot) | Cognify |
|---|---|---|
| **Name** | Q&A with RAG (Information Retrieval Project 3) | Cognify |
| **Scope** | Wikipedia only | Wikipedia + user PDFs + web pages (My Documents) |
| **LLM** | OpenAI GPT-3.5-Turbo (cloud, paid per query) | Any local model via Ollama (free forever) |
| **Retrieval backend** | Apache Solr hosted on GCP (external server required) | Local JSON file + PageIndex trees, no server, no network |
| **Document Q&A** | Not supported | Full pipeline: PDF parsing (PyMuPDF), web extraction (trafilatura), overlapping chunking, PageIndex RAG |
| **Bug** | `filter_topics` variable used before assignment in else branch | Fixed |
| **Dependencies** | 104 packages (openai, pysolr, langchain, ragas, datasets, tiktoken...) | 10 packages |
| **Internet required** | Yes, Solr server + OpenAI API on every query | No, fully offline after setup |
| **Cost** | Pay-per-query (OpenAI API) | Free |
| **Navigation** | Sidebar radio: Home / About Our Team | Five tabs: Chat, My Documents, Analytics, LLM Setup, About |
| **My Documents** | Not available | Upload PDFs, paste URLs, builds a PageIndex tree, independent RAG pipeline |
| **Text chunking** | None | Overlapping 300-word chunks so no content is missed |
| **URL extraction** | None | trafilatura (boilerplate removal, table support) + BeautifulSoup fallback |
| **Visualizations** | 5 basic charts crammed into the sidebar | 9 charts in a dedicated Analytics tab |
| **Metric summary** | None | 4 metric cards: total queries, avg relevancy, topic queries, general queries |
| **Source articles** | Not shown | Collapsible panel after each RAG answer |
| **LLM management** | None | In-app model installer, cloud API config, status monitor |

---

## Why Cognify is Better

**v1 problems:**
- Required a live Apache Solr instance on a GCP VM to function at all. If the VM went down, the app broke.
- Charged money for every query via the OpenAI API.
- Shipped 104 Python packages, most of which were unused.
- Had dead functions (`classify_topics1`, `simulate_response`) that were never called.
- Showed analytics in a cramped sidebar with no interactivity.
- Never showed which Wikipedia articles it actually used to answer your question.
- Could only work with Wikipedia, no way to query your own documents.

**Cognify fixes all of that:**
- No external servers. Retrieval runs against a local JSON file and in-session PageIndex trees.
- No API costs. The LLM runs on your own hardware via Ollama.
- 10 packages total.
- Every RAG answer shows the exact Wikipedia articles or document sections it drew from.
- A full-page Analytics tab with 9 interactive charts.
- **My Documents:** Upload PDFs or paste URLs to build a personal knowledge base. Documents are parsed, chunked into overlapping sections, and indexed into a PageIndex tree. A separate RAG pipeline answers questions using only your documents.

---

## Architecture

```
Wikipedia RAG (Chat tab):
  User query
    > classify_query()                  casual or topic-related?
    |
    |-- casual ────────────────> generate_chitchat_response()  [1 LLM call]
    |
    |-- topic-related
          > classify_topics()           [1 LLM call]
          > keyword_search()            [local, 0 LLM calls]
            50K articles > 50 candidates
          > select_relevant_articles()  [1 LLM call]
            50 > 8 articles
          > answer_question_with_rag()  [1 LLM call]
            8 summaries > answer
          > evaluate_answer_relevance() [1 LLM call]
            > relevancy score 1-10

My Documents RAG (My Documents tab):
  Indexing:
    PDF/URL > text extraction > overlapping chunking (300w, 50w overlap)
      > PageIndex tree > stored in session

  Querying:
    User query
      > expand_query() (resolve pronouns from history)   [1 LLM call]
      > keyword_prefilter() > 60 candidates              [local]
      > llm_select_nodes() > 10 best sections            [1 LLM call]
      > generate answer from 10 sections                  [1 LLM call]
      > evaluate relevancy                                [1 LLM call]
```

Maximum 4 LLM calls per topic-related query. All retrieval runs locally with no network access.

---

## Topics Covered (Wikipedia)

Health, Environment, Technology, Economy, Entertainment, Sports, Politics, Education, Travel, Food

50,000 Wikipedia articles across all 10 topics, stored in `src/wikipedia_combined.json`.

---

## Setup

**Requirements**
- Python 3.10 or later
- [Ollama](https://ollama.com) installed and running
- A model pulled in Ollama (default: `mistral`)

**Steps**

```bash
# 1. Pull the local LLM (one-time, ~4.1 GB download)
ollama pull mistral

# 2. Clone the repo
git clone https://github.com/vikranthbandaru/Cognify.git
cd Cognify

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify .env (already configured)
cat .env
# LLM_MODEL=ollama/mistral
# OLLAMA_API_BASE=http://localhost:11434

# 5. Run
cd src
streamlit run app.py
```

Open **http://localhost:8501** in your browser. Keep Ollama running in the background.

**Switching models**

To use a different local model, pull it and update `.env`:
```
LLM_MODEL=ollama/llama3.2
```

To switch to a cloud model (e.g. Gemini Flash), update `.env`:
```
GEMINI_API_KEY=your-key-here
LLM_MODEL=gemini/gemini-2.0-flash
```

No code changes needed. LiteLLM handles the switch transparently.

---

## Features

### Wikipedia Q&A (Chat tab)
- Ask questions across 10 Wikipedia topic areas or let the system detect the topic automatically
- Filter queries to specific topics using the multiselect
- See exactly which Wikipedia articles sourced each answer (collapsible panel per message)
- Chat history persists across the session with context-aware query expansion

### My Documents (My Documents tab)
- Upload PDFs, parsed with heading-aware + page-fallback extraction (PyMuPDF)
- Add web pages, extracted with trafilatura for high-quality content with table support
- Text is chunked into overlapping 300-word sections so nothing is missed
- Independent chat panel queries only your documents, no Wikipedia involved
- Source sections shown in collapsible panels per answer
- Filter by specific documents or search all at once

### Analytics (Analytics tab)
- 9 charts:
  - Topic distribution pie chart
  - Query type donut (topic vs general)
  - Avg relevancy score per topic (horizontal bar)
  - Relevancy score over time (interactive line chart)
  - Query complexity vs relevancy scatter plot
  - Query length histogram
  - Response length trend (gradient area chart)
  - Activity in last 5 minutes (area chart)
  - Top queries table (sortable, interactive)
- 4 summary metric cards always visible above the tabs

### LLM Setup (LLM Setup tab)
- Check Ollama status
- View installed models
- Install new local models with one click (Qwen, Llama, Phi, Gemma, Mistral)
- Configure cloud LLMs (Gemini, OpenAI) with API key management

---

## Desktop App (Windows)

Don't want to clone and set up Python? Download the pre-built desktop app:

1. Go to the [Releases](https://github.com/vikranthbandaru/Cognify/releases) page
2. Download `Cognify-Windows-v1.0.zip`
3. Extract the zip
4. Install [Ollama](https://ollama.com) and run `ollama pull mistral`
5. Double-click `Cognify.exe` — the app opens in your browser automatically

---

## Demo

[Watch the demo video on Google Drive](https://drive.google.com/file/d/1BQER9_mGMLbQq6ZQiHBnUi1jEwZ04U7A/view)

---
