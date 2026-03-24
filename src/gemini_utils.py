import os
import traceback
from llm import chat as _chat


# ─────────────────────────────────────────────
# Query Classification
# ─────────────────────────────────────────────

def classify_query(query):
    """
    Classify the query as 'chit-chat' or 'topic-related'.
    Returns exactly one of those two strings.
    """
    prompt = """Classify the following query as either 'chit-chat' or 'topic-related'.
Respond with exactly one label, nothing else.

Examples:
"How's the weather today?" => chit-chat
"Can you tell me more about deep learning?" => topic-related
"What's your favorite movie?" => chit-chat
"Explain the benefits of federated learning." => topic-related
"Hey, how are you?" => chit-chat
"What causes climate change?" => topic-related

Query: "{query}"
Label:""".format(query=query)

    result = _chat([{"role": "user", "content": prompt}], max_tokens=20)
    return "topic-related" if "topic-related" in result.lower() else "chit-chat"


# ─────────────────────────────────────────────
# Topic Classification
# ─────────────────────────────────────────────

def classify_topics(query, topics):
    """
    Classify the query into one or more of the predefined topics.
    Returns a list of matching topic strings.
    """
    topics_numbered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(topics)])
    prompt = f"""Classify this query into one or more of the topics listed below.
Return ONLY the matching topic names separated by commas. No explanations.

Topics:
{topics_numbered}

Examples:
Query: "What are AI advancements in healthcare?" => Technology, Health
Query: "What is deforestation?" => Environment
Query: "How does the economy affect education?" => Economy, Education
Query: "Best travel destinations and local food?" => Travel, Food

Query: "{query}"
Topics:"""

    try:
        result = _chat([{"role": "user", "content": prompt}], max_tokens=100)
        classified = [t.strip() for t in result.split(",") if t.strip() in topics]
        return classified if classified else ["Uncategorized"]
    except Exception as e:
        return [f"Error: {e}"]


# ─────────────────────────────────────────────
# Chit-Chat
# ─────────────────────────────────────────────

def generate_chitchat_response(query):
    """Generate a friendly conversational response for casual queries."""
    prompt = f"""You are a friendly, conversational assistant. Respond naturally and engagingly to casual queries.
If you don't have enough information, politely decline.

Examples:
User: "What's your favorite food?" => "I don't eat, but if I could, pizza sounds amazing!"
User: "Tell me a joke." => "Why don't scientists trust atoms? Because they make up everything!"
User: "How are you?" => "I'm doing great, thanks for asking! How can I help you today?"

User: "{query}"
Assistant:"""

    try:
        return _chat([{"role": "user", "content": prompt}], temperature=0.7, max_tokens=200)
    except Exception as e:
        return f"Oops! Something went wrong: {e}"


# ─────────────────────────────────────────────
# Query Expansion (PageIndex: context-aware search)
# ─────────────────────────────────────────────

def query_completion(query, combined_history):
    """
    Expand or complete the query using prior conversation context.
    Resolves pronouns and ambiguity so retrieval is more accurate.
    """
    if not combined_history.strip():
        return query

    prompt = f"""Using the conversation history, rewrite the query to be fully self-contained.
Resolve pronouns (it, its, they, this) and fill in missing context.
Return ONLY the rewritten query, nothing else.

Examples:
History: "User: Tell me about France. Assistant: France is a country in Western Europe."
Query: "What is its capital?"
Rewritten: "What is the capital of France?"

History: "User: What causes diabetes? Assistant: Diabetes is caused by insulin issues."
Query: "What are its symptoms?"
Rewritten: "What are the symptoms of diabetes?"

History:
{combined_history}

Query: {query}
Rewritten:"""

    try:
        result = _chat([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=150)
        return result if result else query
    except Exception:
        return query


# ─────────────────────────────────────────────
# PageIndex-style Local Retrieval
# ─────────────────────────────────────────────

STOP_WORDS = {
    'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'who',
    'are', 'do', 'does', 'did', 'can', 'could', 'would', 'should', 'tell',
    'me', 'about', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'and',
    'or', 'but', 'not', 'some', 'any', 'its', 'it', 'this', 'that', 'was',
    'were', 'been', 'be', 'have', 'has', 'had', 'will', 'by', 'from', 'as'
}


def keyword_search(articles, query, top_n=50):
    """
    Fast keyword overlap search across a list of article dicts.
    Each article: {"title": str, "summary": str, ...}
    Returns top_n articles sorted by relevance score.
    """
    query_words = set(query.lower().split()) - STOP_WORDS
    if not query_words:
        return articles[:top_n]

    scored = []
    for article in articles:
        title = article.get("title", "").lower()
        summary = article.get("summary", "")[:400].lower()
        text_words = set((title + " " + summary).split())

        # Title matches weighted 3x, summary matches 1x
        title_words = set(title.split())
        title_score = len(query_words & title_words) * 3
        summary_score = len(query_words & text_words)
        score = title_score + summary_score

        if score > 0:
            scored.append((score, article))

    scored.sort(key=lambda x: -x[0])
    return [item[1] for item in scored[:top_n]]


def select_relevant_articles(query, candidates, top_k=8):
    """
    PageIndex core step: use Gemini to reason over candidate articles
    and select the most relevant ones. Falls back to top-k by score.
    """
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates

    candidate_text = "\n".join([
        f"[{i}] {a.get('title', 'Unknown')}: {a.get('summary', '')[:180]}..."
        for i, a in enumerate(candidates)
    ])

    prompt = f"""You are a retrieval assistant. Given the query below, select the {top_k} most relevant articles.
Return ONLY the article numbers separated by commas (e.g., "0, 3, 5"). No explanations.

Query: "{query}"

Articles:
{candidate_text}

Most relevant article numbers:"""

    try:
        result = _chat([{"role": "user", "content": prompt}], max_tokens=60)
        indices = [int(x.strip()) for x in result.split(",") if x.strip().isdigit()]
        indices = [i for i in indices if 0 <= i < len(candidates)][:top_k]
        return [candidates[i] for i in indices] if indices else candidates[:top_k]
    except Exception:
        return candidates[:top_k]


# ─────────────────────────────────────────────
# Answer Relevance Evaluation
# ─────────────────────────────────────────────

def evaluate_answer_relevance(question, answer):
    """Rate how relevant the answer is to the question. Returns int 1–10."""
    prompt = f"""Rate the relevance of this answer to the question on a scale of 1 to 10.
1 = completely irrelevant, 10 = perfectly answers the question.
Respond with a single digit only.

Question: {question}
Answer: {answer}

Score:"""

    try:
        result = _chat([{"role": "user", "content": prompt}], max_tokens=5)
        digits = ''.join(c for c in result if c.isdigit())
        return min(10, max(1, int(digits[:2]))) if digits else 5
    except Exception:
        return 5


# ─────────────────────────────────────────────
# Full RAG Pipeline
# ─────────────────────────────────────────────

def answer_question_with_rag(wiki_data, question, history, filter_topics):
    """
    PageIndex-style two-level hierarchical RAG pipeline:

    Level 1: Topic filtering - restrict search to relevant topic buckets.
    Level 2: Keyword pre-filter - fast local search narrows 50k articles to 50 candidates.
    Level 3: Gemini reasoning - LLM picks the 8 most relevant from 50 candidates.
    Level 4: Answer generation - Gemini answers using retrieved summaries as context.

    Args:
        wiki_data (dict): {topic: [article_dicts]} loaded from wikipedia_combined.json
        question  (str):  User's question
        history   (list): Chat history [{"role": ..., "content": ...}]
        filter_topics (list): Topics to search within

    Returns:
        str: Generated answer
    """
    try:
        # Build recent conversation history string (last 6 messages = 3 exchanges)
        combined_history = "\n".join([
            f"User: {msg['content']}" if msg["role"] == "user"
            else f"Assistant: {msg['content']}"
            for msg in history[-6:]
        ])

        # ── Level 1: Expand query with conversation context ──
        expanded_query = query_completion(question, combined_history)

        # ── Level 2: Keyword search within relevant topics ──
        valid_topics = [t for t in filter_topics if t in (wiki_data or {})]
        if not valid_topics and wiki_data:
            valid_topics = list(wiki_data.keys())

        candidates = []
        for topic in valid_topics:
            articles = wiki_data.get(topic, [])
            top_articles = keyword_search(articles, expanded_query, top_n=30)
            candidates.extend(top_articles)

        if not candidates:
            return "I couldn't find any relevant articles. Please try rephrasing your question."

        # Level 3: Gemini reasons over candidates, picks top 8
        selected = select_relevant_articles(expanded_query, candidates, top_k=8)

        # ── Level 4: Build context and generate answer ──
        context = "\n\n".join([
            f"**{a.get('title', 'Unknown')}**\n{a.get('summary', '')}"
            for a in selected
        ])

        prompt = (
            "You are a knowledgeable assistant answering questions from Wikipedia article summaries. "
            "Use the context below to give an accurate, concise answer in under 150 words. "
            "If the context doesn't contain enough information, say so politely.\n\n"
            f"Context:\n{context}\n\n"
            f"Conversation History:\n{combined_history}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        return _chat([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=400)

    except Exception as e:
        traceback.print_exc()
        return f"An error occurred during processing: {e}"
