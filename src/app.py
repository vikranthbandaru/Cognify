import streamlit as st
import time
import json
import os
import subprocess
import sys
import webbrowser
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import altair as alt
import traceback
import html as html_mod
import base64
from gemini_utils import (
    answer_question_with_rag,
    generate_chitchat_response,
    classify_query,
    classify_topics,
    evaluate_answer_relevance,
    keyword_search,
    select_relevant_articles,
)
from doc_utils import index_pdf, index_url, answer_from_docs, evaluate_doc_answer, source_id

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

TOPICS = [
    "Health", "Environment", "Technology", "Economy", "Entertainment",
    "Sports", "Politics", "Education", "Travel", "Food"
]

COLORS = {
    "primary":   "#818CF8",
    "secondary": "#38BDF8",
    "accent":    "#FB7185",
    "success":   "#34D399",
    "warning":   "#FBBF24",
    "muted":     "#94A3B8",
    "bg":        "#0F172A",
    "card":      "#1E293B",
    "text":      "#E2E8F0",
    "text_dim":  "#94A3B8",
}

TOPIC_PALETTE = [
    "#818CF8", "#38BDF8", "#FB7185", "#34D399", "#FBBF24",
    "#A78BFA", "#22D3EE", "#F87171", "#60A5FA", "#2DD4BF",
]

# Display-friendly classification labels
LABEL_MAP = {"chit-chat": "General", "topic-related": "Topic"}

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Cognify",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Load Wikipedia data (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_wiki_data():
    path = os.path.join(os.path.dirname(__file__), "wikipedia_combined.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

wiki_data = load_wiki_data()

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────

defaults = {
    "history": [],
    "query_data": pd.DataFrame(columns=[
        "timestamp", "query", "terms", "response_terms",
        "classification", "topics", "relevancy_score",
    ]),
    "relevancy_scores": [],
    "source_articles": [],
    "user_index": {},
    "doc_history": [],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────

# Load hero background image as base64
_hero_b64 = ""
try:
    _hero_path = os.path.join(os.path.dirname(__file__), "hero_bg.png")
    if os.path.exists(_hero_path):
        with open(_hero_path, "rb") as _f:
            _hero_b64 = base64.b64encode(_f.read()).decode()
except Exception:
    pass

# Load fonts
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">'
    '<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">',
    unsafe_allow_html=True,
)

st.markdown("""
<style>

/* ══════════════════════════════════════════════
   FIX: Clip broken Material Icons text
   If the font fails to load, icon names like
   "face", "smart_toy", "arrow_right" are clipped
   inside a small box instead of overlapping.
   ══════════════════════════════════════════════ */
.material-symbols-rounded {
    overflow: hidden !important;
    display: inline-block !important;
    width: 24px !important;
    height: 24px !important;
    line-height: 24px !important;
    vertical-align: middle !important;
    text-overflow: clip !important;
}

/* ══ Global Typography ══ */
html, body, [class*="css"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ══ Animated Gradient Header ══ */
.cognify-header {
    text-align: center;
    padding: 56px 20px 32px 20px;
    position: relative;
    background-size: cover;
    background-position: center;
    border-radius: 0 0 24px 24px;
    margin: -1rem -1rem 24px -1rem;
    overflow: hidden;
}
.cognify-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 30% 50%, rgba(99,102,241,0.15) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 50%, rgba(167,139,250,0.1) 0%, transparent 60%);
    pointer-events: none;
}
.cognify-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 60px;
    background: linear-gradient(to top, var(--background-color, #0F172A), transparent);
    pointer-events: none;
}
.cognify-header h1 {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 10px;
    font-family: 'Inter', sans-serif !important;
    position: relative;
    z-index: 1;
}
.cognify-header .accent {
    background: linear-gradient(135deg, #818CF8, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
@keyframes shimmer {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
.cognify-header p {
    color: var(--text-color, #94A3B8);
    opacity: 0.6;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
    margin-top: 0;
    font-weight: 500;
    position: relative;
    z-index: 1;
}
.cognify-header .divider {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #6366F1, #A78BFA);
    margin: 16px auto 0 auto;
    border-radius: 2px;
    position: relative;
    z-index: 1;
}

/* ══ Floating particles (CSS only) ══ */
.cognify-header .particle {
    position: absolute;
    border-radius: 50%;
    background: rgba(99,102,241,0.15);
    animation: float 6s ease-in-out infinite;
    pointer-events: none;
}
.cognify-header .particle:nth-child(1) { width:80px;height:80px;top:10%;left:10%;animation-delay:0s; }
.cognify-header .particle:nth-child(2) { width:50px;height:50px;top:60%;left:75%;animation-delay:2s;background:rgba(167,139,250,0.1); }
.cognify-header .particle:nth-child(3) { width:35px;height:35px;top:20%;left:80%;animation-delay:4s;background:rgba(236,72,153,0.08); }
@keyframes float {
    0%, 100% { transform: translateY(0) scale(1); opacity:0.5; }
    50% { transform: translateY(-20px) scale(1.1); opacity:1; }
}

/* ══ Metric Cards - Glassmorphism ══ */
div[data-testid="stMetric"] {
    background: var(--secondary-background-color) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 18px;
    padding: 24px 22px;
    box-shadow: 0 4px 20px rgba(99,102,241,0.08);
    position: relative;
    overflow: hidden;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
div[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366F1, #A78BFA, #EC4899);
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(99,102,241,0.18);
    border-color: rgba(99,102,241,0.3) !important;
}
[data-testid="stMetricValue"] {
    color: #6366F1 !important;
    font-weight: 800 !important;
    font-size: 2.2rem !important;
    letter-spacing: -0.5px;
}
[data-testid="stMetricLabel"] {
    color: var(--text-color) !important;
    opacity: 0.5;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700 !important;
}

/* ══ Buttons - Gradient with Glow ══ */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #6366F1, #818CF8) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 14px;
    padding: 12px 28px;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    white-space: nowrap;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 14px rgba(99,102,241,0.3);
}
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #818CF8, #A78BFA) !important;
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(99,102,241,0.4);
}
.stButton > button:active,
.stFormSubmitButton > button:active {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(99,102,241,0.25);
}

/* ══ Inputs - Clean with Glow Focus ══ */
.stTextInput > div > div > input {
    border-radius: 14px !important;
    padding: 14px 18px;
    border: 2px solid rgba(128,128,128,0.12) !important;
    font-size: 0.92rem;
    background: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
    transition: all 0.2s ease;
}
.stTextInput > div > div > input::placeholder {
    opacity: 0.4;
}
.stTextInput > div > div > input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 4px rgba(99,102,241,0.12), 0 4px 16px rgba(99,102,241,0.08) !important;
}

/* ══ Tabs - Pill Style ══ */
[data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: none !important;
    background: var(--secondary-background-color);
    border-radius: 16px;
    padding: 6px;
}
[data-baseweb="tab"] {
    border-radius: 12px !important;
    font-weight: 600;
    padding: 10px 22px;
    font-size: 0.88rem;
    transition: all 0.2s ease;
    border-bottom: none !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #6366F1 !important;
    background: var(--background-color) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

/* ══ Chat Messages - Modern Bubbles ══ */
[data-testid="stChatMessage"] {
    border-radius: 20px;
    padding: 20px 24px;
    margin-bottom: 10px;
    background: var(--secondary-background-color) !important;
    border: 1px solid rgba(128,128,128,0.08) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s ease;
}
[data-testid="stChatMessage"]:hover {
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

/* ══ Expander - Sleek ══ */
[data-testid="stExpander"] {
    border: 1px solid rgba(128,128,128,0.12) !important;
    border-radius: 14px;
    background: var(--secondary-background-color) !important;
    overflow: hidden;
    transition: all 0.2s ease;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(99,102,241,0.25) !important;
}

/* ══ DataFrame ══ */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(128,128,128,0.1);
}

/* ══ Charts ══ */
.vega-embed {
    border-radius: 14px;
    overflow: hidden;
}

/* ══ Multiselect Tags ══ */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: linear-gradient(135deg, #6366F1, #818CF8) !important;
    border-radius: 10px;
    color: #FFFFFF !important;
    font-weight: 600;
    font-size: 0.82rem;
    padding: 4px 12px;
}

/* ══ Selectbox ══ */
[data-baseweb="select"] > div {
    background: var(--secondary-background-color) !important;
    border: 1.5px solid rgba(128,128,128,0.15) !important;
    border-radius: 14px !important;
    transition: border-color 0.15s ease;
}
[data-baseweb="select"] > div:hover {
    border-color: rgba(99,102,241,0.3) !important;
}

/* ══ Code Blocks ══ */
[data-testid="stCode"] {
    border-radius: 14px;
}

/* ══ Dividers ══ */
hr {
    border: none !important;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(128,128,128,0.15), transparent) !important;
    margin: 20px 0;
}

/* ══ Form Container ══ */
[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}

/* ══ Alerts ══ */
[data-testid="stAlert"] {
    border-radius: 14px;
}

/* ══ Spinner ══ */
[data-testid="stSpinner"] {
    color: #6366F1 !important;
}

/* ══ Custom Scrollbar ══ */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(128,128,128,0.2);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(128,128,128,0.35);
}

/* ══ Footer ══ */
.footer {
    text-align: center;
    color: var(--text-color);
    opacity: 0.3;
    font-size: 0.74rem;
    padding: 36px 0 10px 0;
    margin-top: 48px;
    font-weight: 500;
    letter-spacing: 0.3px;
}
</style>
""", unsafe_allow_html=True)

# Build header with hero background
_hero_style = f'background-image:url(data:image/png;base64,{_hero_b64});' if _hero_b64 else ''
st.markdown(f"""
<div class="cognify-header" style="{_hero_style}">
  <div class="particle"></div>
  <div class="particle"></div>
  <div class="particle"></div>
  <h1><span class="accent">Cognify</span></h1>
  <p>Your Local Intelligence Engine &middot; Wikipedia + Documents &middot; Zero Cloud Dependencies</p>
  <div class="divider"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Metric cards (always visible)
# ─────────────────────────────────────────────

qd = st.session_state.query_data
total_q    = len(qd)
avg_rel    = round(qd["relevancy_score"].mean(), 1) if total_q else "N/A"
topic_q    = int((qd["classification"] == "topic-related").sum()) if total_q else 0
chitchat_q = int((qd["classification"] == "chit-chat").sum()) if total_q else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Queries", total_q)
c2.metric("Avg Relevancy Score", avg_rel)
c3.metric("Topic Queries", topic_q)
c4.metric("General Queries", chitchat_q)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

tab_chat, tab_mydocs, tab_analytics, tab_setup, tab_about = st.tabs([
    "Chat", "My Documents", "Analytics", "LLM Setup", "About"
])


# ══════════════════════════════════════════════
# TAB 1: CHAT
# ══════════════════════════════════════════════

with tab_chat:
    selected_categories = st.multiselect(
        "Filter by topic (optional, leave blank to search all Wikipedia topics):",
        TOPICS, key="topic_filter",
    )

    # Build lookup: msg_index -> articles
    source_map = {s["msg_index"]: s["articles"] for s in st.session_state.source_articles}

    for i, msg in enumerate(st.session_state.history):
        is_user = msg["role"] == "user"
        badge_bg = "linear-gradient(135deg,#6366F1,#818CF8)" if is_user else "linear-gradient(135deg,#10B981,#34D399)"
        badge_text = "You" if is_user else "C"
        safe_content = html_mod.escape(msg["content"]).replace("\n", "<br>")
        st.markdown(f"""
        <div style="display:flex;gap:14px;align-items:flex-start;margin:14px 0;">
            <div style="background:{badge_bg};color:#fff;border-radius:50%;min-width:38px;width:38px;height:38px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.78rem;flex-shrink:0;box-shadow:0 2px 8px rgba(99,102,241,0.2);">{badge_text}</div>
            <div style="flex:1;padding:14px 18px;border-radius:18px;background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.08);line-height:1.65;font-size:0.92rem;box-shadow:0 1px 6px rgba(0,0,0,0.04);">{safe_content}</div>
        </div>
        """, unsafe_allow_html=True)
        # Show source articles if this is an assistant message with sources
        if not is_user and i in source_map and source_map[i]:
            articles = source_map[i]
            links_html = "".join(
                f'<li style="padding:4px 0;"><a href="{html_mod.escape(a.get("url",""))}" target="_blank" style="color:#6366F1;text-decoration:none;font-weight:500;">{html_mod.escape(a.get("title","Untitled"))}</a></li>'
                for a in articles
            )
            st.markdown(f"""
            <details style="margin:0 0 12px 52px;border:1px solid rgba(128,128,128,0.1);border-radius:14px;padding:10px 16px;background:var(--secondary-background-color);font-size:0.85rem;">
                <summary style="cursor:pointer;font-weight:600;color:var(--text-color);opacity:0.7;">Sources ({len(articles)} Wikipedia articles)</summary>
                <ul style="margin-top:8px;padding-left:18px;list-style:disc;">{links_html}</ul>
            </details>
            """, unsafe_allow_html=True)

    # Input row
    with st.form(key="chat_form", clear_on_submit=True):
        in_col, btn_col = st.columns([5, 1])
        with in_col:
            question = st.text_input(
                "question", label_visibility="collapsed",
                placeholder="Ask anything about Health, Technology, Sports, or just say hi",
            )
        with btn_col:
            ask_clicked = st.form_submit_button("Ask", use_container_width=True)

    if ask_clicked:
        if question.strip():
            with st.spinner("Thinking..."):
                try:
                    classification = classify_query(question)
                    filter_topics  = []
                    sources        = []

                    if classification == "chit-chat":
                        response      = generate_chitchat_response(question)
                        filter_topics = ["chit-chat"]

                    else:
                        topics        = classify_topics(question, TOPICS)
                        filter_topics = list(set(topics) | set(selected_categories)) \
                                        if selected_categories else topics
                        response      = answer_question_with_rag(
                            wiki_data, question, st.session_state.history, filter_topics
                        )

                        valid      = [t for t in filter_topics if t in wiki_data] or list(wiki_data.keys())
                        candidates = []
                        for t in valid:
                            candidates.extend(keyword_search(wiki_data[t], question, top_n=30))
                        top_arts = candidates[:8]
                        sources  = [
                            {"title": a.get("title", ""), "url": a.get("URL", a.get("url", ""))}
                            for a in top_arts
                        ]

                    st.session_state.history.append({"role": "user",      "content": question})
                    st.session_state.history.append({"role": "assistant", "content": response})

                    if sources:
                        st.session_state.source_articles.append({
                            "msg_index": len(st.session_state.history) - 1,
                            "articles":  sources,
                        })

                    relevancy_score = evaluate_answer_relevance(question, response)
                    st.session_state.relevancy_scores.append(relevancy_score)

                    st.session_state.query_data = pd.concat([
                        st.session_state.query_data,
                        pd.DataFrame([{
                            "timestamp":      time.time(),
                            "query":          question,
                            "terms":          len(question.split()),
                            "response_terms": len(response.split()),
                            "classification": classification,
                            "topics":         filter_topics,
                            "relevancy_score": relevancy_score,
                        }])
                    ], ignore_index=True)

                    st.rerun()

                except Exception as e:
                    traceback.print_exc()
                    st.error(f"Error: {e}")
        else:
            st.warning("Please type a question first.")


# ══════════════════════════════════════════════
# TAB 2: ANALYTICS
# ══════════════════════════════════════════════

with tab_analytics:
    qd = st.session_state.query_data

    if qd.empty:
        st.info("No queries yet. Head to the Chat tab and ask something!")
    else:
        # Prepare display labels
        display_qd = qd.copy()
        display_qd["classification_label"] = display_qd["classification"].map(LABEL_MAP).fillna(display_qd["classification"])

        # Row 1: Topic pie | Query-type donut
        r1_left, r1_right = st.columns(2)

        with r1_left:
            st.markdown("#### Queries by Topic")
            topic_counts = display_qd["topics"].explode().value_counts()
            # Map chit-chat label in topics too
            topic_counts.index = topic_counts.index.map(lambda x: LABEL_MAP.get(x, x))
            colors_pie   = TOPIC_PALETTE[:len(topic_counts)]
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_alpha(0)
            ax.set_facecolor('none')
            wedges, texts, autotexts = ax.pie(
                topic_counts.values,
                labels=None,
                colors=colors_pie,
                autopct=lambda p: f"{p:.0f}%" if p > 5 else "",
                startangle=140,
                wedgeprops={"edgecolor": "#0F172A", "linewidth": 1.5},
            )
            for at in autotexts:
                at.set_fontsize(8)
                at.set_color("#E2E8F0")
            ax.legend(
                wedges, topic_counts.index,
                loc="center left", bbox_to_anchor=(1, 0.5),
                fontsize=8, frameon=False, labelcolor="#CBD5E1",
            )
            ax.set_title("Topic Distribution", fontsize=11, fontweight="bold", pad=8, color="#E2E8F0")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with r1_right:
            st.markdown("#### Query Type Split")
            type_counts = display_qd["classification_label"].value_counts()
            donut_colors = [COLORS["primary"] if t == "Topic" else COLORS["accent"]
                            for t in type_counts.index]
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_alpha(0)
            ax.set_facecolor('none')
            wedges, _, autotexts = ax.pie(
                type_counts.values,
                colors=donut_colors,
                autopct="%1.0f%%",
                startangle=90,
                wedgeprops={"width": 0.52, "edgecolor": "#0F172A", "linewidth": 2},
            )
            for at in autotexts:
                at.set_fontsize(9)
                at.set_color("#E2E8F0")
            ax.text(0, 0, str(total_q), ha="center", va="center",
                    fontsize=22, fontweight="bold", color="#E2E8F0")
            ax.text(0, -0.18, "total", ha="center", va="center",
                    fontsize=9, color=COLORS["muted"])
            patches = [
                mpatches.Patch(color=COLORS["primary"], label="Topic"),
                mpatches.Patch(color=COLORS["accent"],  label="General"),
            ]
            ax.legend(handles=patches, loc="lower center",
                      bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False, labelcolor="#CBD5E1")
            ax.set_title("Query Types", fontsize=11, fontweight="bold", pad=8, color="#E2E8F0")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")

        # Row 2: Avg relevancy by topic | Relevancy trend
        r2_left, r2_right = st.columns(2)

        with r2_left:
            st.markdown("#### Avg Relevancy by Topic")
            topic_rel = (
                display_qd.explode("topics")
                .groupby("topics")["relevancy_score"]
                .mean()
                .reset_index()
                .rename(columns={"topics": "Topic", "relevancy_score": "Avg Relevancy"})
                .sort_values("Avg Relevancy")
            )
            topic_rel["Topic"] = topic_rel["Topic"].map(lambda x: LABEL_MAP.get(x, x))
            chart = (
                alt.Chart(topic_rel)
                .mark_bar(
                    color=COLORS["primary"],
                    cornerRadiusTopRight=5,
                    cornerRadiusBottomRight=5,
                )
                .encode(
                    y=alt.Y("Topic:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=11)),
                    x=alt.X("Avg Relevancy:Q",
                             scale=alt.Scale(domain=[0, 10]),
                             title="Score (1-10)",
                             axis=alt.Axis(grid=True)),
                    tooltip=["Topic:N", alt.Tooltip("Avg Relevancy:Q", format=".1f")],
                )
                .properties(height=280)
                .configure_axis(labelColor="#94A3B8", titleColor="#94A3B8")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart, use_container_width=True)

        with r2_right:
            st.markdown("#### Relevancy Over Time")
            trend = display_qd[["timestamp", "relevancy_score"]].copy()
            trend["time"] = pd.to_datetime(trend["timestamp"], unit="s")
            chart = (
                alt.Chart(trend)
                .mark_line(point=alt.OverlayMarkDef(color=COLORS["primary"], size=60),
                            color=COLORS["primary"], strokeWidth=2)
                .encode(
                    x=alt.X("time:T", title="Time", axis=alt.Axis(format="%H:%M")),
                    y=alt.Y("relevancy_score:Q",
                             scale=alt.Scale(domain=[0, 10]),
                             title="Score",
                             axis=alt.Axis(grid=True)),
                    tooltip=[
                        alt.Tooltip("time:T", format="%H:%M:%S", title="Time"),
                        alt.Tooltip("relevancy_score:Q", title="Score"),
                    ],
                )
                .properties(height=280)
                .configure_axis(labelColor="#555", titleColor="#555")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart, use_container_width=True)

        st.markdown("---")

        # Row 3: Scatter plot | Query length histogram
        r3_left, r3_right = st.columns(2)

        with r3_left:
            st.markdown("#### Query Complexity vs Relevancy")
            scatter_data = display_qd[["terms", "relevancy_score", "classification_label", "query"]].copy()
            chart = (
                alt.Chart(scatter_data)
                .mark_circle(size=90, opacity=0.75)
                .encode(
                    x=alt.X("terms:Q", title="Query Word Count", axis=alt.Axis(grid=True)),
                    y=alt.Y("relevancy_score:Q",
                             title="Relevancy Score",
                             scale=alt.Scale(domain=[0, 10]),
                             axis=alt.Axis(grid=True)),
                    color=alt.Color(
                        "classification_label:N",
                        scale=alt.Scale(
                            domain=["Topic", "General"],
                            range=[COLORS["primary"], COLORS["accent"]],
                        ),
                        legend=alt.Legend(title="Query Type"),
                    ),
                    tooltip=[
                        alt.Tooltip("query:N",               title="Query"),
                        alt.Tooltip("terms:Q",               title="Word Count"),
                        alt.Tooltip("relevancy_score:Q",     title="Score"),
                        alt.Tooltip("classification_label:N", title="Type"),
                    ],
                )
                .properties(height=280)
                .configure_axis(labelColor="#555", titleColor="#555")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart, use_container_width=True)

        with r3_right:
            st.markdown("#### Query Length Distribution")
            fig, ax = plt.subplots(figsize=(5, 3.8))
            fig.patch.set_alpha(0)
            ax.set_facecolor('none')
            ax.hist(display_qd["terms"], bins=max(5, len(display_qd) // 3),
                    color=COLORS["primary"], edgecolor="#0F172A", rwidth=0.88)
            ax.set_xlabel("Word Count", fontsize=10, color="#94A3B8")
            ax.set_ylabel("Frequency",  fontsize=10, color="#94A3B8")
            ax.set_title("How long are your queries?", fontsize=10, color="#E2E8F0")
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['bottom', 'left']].set_color('#334155')
            ax.tick_params(colors="#94A3B8")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")

        # Row 4: Response length over time (full width)
        st.markdown("#### Response Length Over Time")
        resp_trend = display_qd[["timestamp", "response_terms"]].copy()
        resp_trend["time"] = pd.to_datetime(resp_trend["timestamp"], unit="s")
        chart = (
            alt.Chart(resp_trend)
            .mark_area(
                line={"color": COLORS["secondary"], "strokeWidth": 2},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color=COLORS["secondary"], offset=0),
                        alt.GradientStop(color="white", offset=1),
                    ],
                    x1=1, x2=1, y1=1, y2=0,
                ),
                opacity=0.6,
            )
            .encode(
                x=alt.X("time:T", title="Time", axis=alt.Axis(format="%H:%M")),
                y=alt.Y("response_terms:Q", title="Words in Response"),
                tooltip=[
                    alt.Tooltip("time:T", format="%H:%M:%S", title="Time"),
                    alt.Tooltip("response_terms:Q", title="Response Words"),
                ],
            )
            .properties(height=180)
            .configure_axis(labelColor="#555", titleColor="#555")
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart, use_container_width=True)

        # Row 5: Activity last 5 min (full width)
        st.markdown("#### Activity, Last 5 Minutes")
        now    = time.time()
        last_5 = display_qd[display_qd["timestamp"] > (now - 300)].copy()
        if not last_5.empty:
            last_5["bucket"] = (last_5["timestamp"] // 10 * 10)
            last_5["time"]   = pd.to_datetime(last_5["bucket"], unit="s")
            counts = last_5.groupby("time").size().reset_index(name="count")
            chart = (
                alt.Chart(counts)
                .mark_area(
                    line={"color": COLORS["success"], "strokeWidth": 2},
                    color=alt.Gradient(
                        gradient="linear",
                        stops=[
                            alt.GradientStop(color=COLORS["success"], offset=0),
                            alt.GradientStop(color="white", offset=1),
                        ],
                        x1=1, x2=1, y1=1, y2=0,
                    ),
                    point=alt.OverlayMarkDef(color=COLORS["success"], size=50),
                    opacity=0.65,
                )
                .encode(
                    x=alt.X("time:T", title="Time", axis=alt.Axis(format="%H:%M:%S")),
                    y=alt.Y("count:Q", title="Queries per 10 sec"),
                    tooltip=[
                        alt.Tooltip("time:T", format="%H:%M:%S", title="Time"),
                        alt.Tooltip("count:Q", title="Queries"),
                    ],
                )
                .properties(height=160)
                .configure_axis(labelColor="#555", titleColor="#555")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("No queries in the past 5 minutes.")

        st.markdown("---")

        # Row 6: Top queries table (full width)
        st.markdown("#### Top Queries")
        top_q = (
            display_qd.groupby("query")
            .agg(
                times_asked=("query", "count"),
                avg_score=("relevancy_score", "mean"),
                type=("classification_label", "first"),
            )
            .sort_values("times_asked", ascending=False)
            .head(10)
            .reset_index()
            .rename(columns={
                "query": "Query",
                "times_asked": "Times Asked",
                "avg_score": "Avg Score",
                "type": "Type",
            })
        )
        top_q["Avg Score"] = pd.to_numeric(top_q["Avg Score"], errors="coerce").round(1)
        st.dataframe(top_q, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 3: MY DOCUMENTS
# ══════════════════════════════════════════════

with tab_mydocs:
    left_col, right_col = st.columns([2, 3])

    # Left: Indexing panel
    with left_col:
        st.markdown("### Index Your Resources")
        st.caption("Build a personal search index from PDFs or web pages, completely separate from Wikipedia.")

        # Upload PDFs
        st.markdown("**PDFs**")
        uploaded_files = st.file_uploader(
            "Drop PDF files here", type=["pdf"],
            accept_multiple_files=True, key="pdf_uploader",
        )
        if uploaded_files:
            for f in uploaded_files:
                if f.name not in st.session_state.user_index:
                    with st.spinner(f"Building index for {f.name}..."):
                        entry, msg = index_pdf(f.read(), f.name)
                    if entry:
                        st.session_state.user_index[f.name] = entry
                        st.success(msg)
                    else:
                        st.error(msg)

        # Add URL
        st.markdown("**Web Pages**")
        uc, bc = st.columns([4, 1])
        with uc:
            url_input = st.text_input("url", label_visibility="collapsed",
                                      placeholder="https://example.com/article",
                                      key="url_input")
        with bc:
            add_url = st.button("Add", use_container_width=True, key="add_url_btn")

        if add_url and url_input.strip():
            if url_input in st.session_state.user_index:
                st.info("Already indexed.")
            else:
                with st.spinner("Fetching and indexing..."):
                    entry, msg = index_url(url_input.strip())
                if entry:
                    st.session_state.user_index[url_input] = entry
                    st.success(msg)
                else:
                    st.error(msg)

        # Indexed documents list
        st.markdown("---")
        st.markdown("**Indexed Documents**")
        if not st.session_state.user_index:
            st.info("No documents yet.")
        else:
            total_nodes = sum(e.get("node_count", 0) for e in st.session_state.user_index.values())
            st.caption(f"{len(st.session_state.user_index)} source(s), {total_nodes} total sections")
            for src, entry in list(st.session_state.user_index.items()):
                label = src if len(src) < 45 else src[:42] + "..."
                nc, dc = st.columns([3, 1])
                with nc:
                    kind = "[PDF]" if entry.get("type") == "pdf" else "[URL]"
                    st.markdown(f"**{kind}** {label}")
                    st.caption(f"{entry.get('node_count', 0)} sections")
                with dc:
                    if st.button("Remove", key=f"del_{source_id(src)}"):
                        del st.session_state.user_index[src]
                        st.rerun()

    # Right: Independent chat panel
    with right_col:
        st.markdown("### Chat with Your Documents")
        st.caption("This chat uses only your indexed documents, not Wikipedia.")

        if not st.session_state.user_index:
            st.info("Index at least one document on the left to start chatting.")
        else:
            # Source filter
            all_sources = list(st.session_state.user_index.keys())
            selected_srcs = st.multiselect(
                "Search in (leave blank for all):",
                all_sources,
                key="doc_source_filter",
            )

            # Chat history display
            for i, msg in enumerate(st.session_state.doc_history):
                is_user = msg["role"] == "user"
                badge_bg = "linear-gradient(135deg,#6366F1,#818CF8)" if is_user else "linear-gradient(135deg,#10B981,#34D399)"
                badge_text = "You" if is_user else "C"
                safe_content = html_mod.escape(msg["content"]).replace("\n", "<br>")
                st.markdown(f"""
                <div style="display:flex;gap:14px;align-items:flex-start;margin:14px 0;">
                    <div style="background:{badge_bg};color:#fff;border-radius:50%;min-width:38px;width:38px;height:38px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.78rem;flex-shrink:0;box-shadow:0 2px 8px rgba(99,102,241,0.2);">{badge_text}</div>
                    <div style="flex:1;padding:14px 18px;border-radius:18px;background:var(--secondary-background-color);border:1px solid rgba(128,128,128,0.08);line-height:1.65;font-size:0.92rem;box-shadow:0 1px 6px rgba(0,0,0,0.04);">{safe_content}</div>
                </div>
                """, unsafe_allow_html=True)
                if not is_user and msg.get("nodes"):
                    nodes = msg["nodes"]
                    nodes_html = "".join(
                        f'<div style="padding:6px 0;border-bottom:1px solid rgba(128,128,128,0.08);"><strong>{html_mod.escape(n["title"])}</strong><br><span style="opacity:0.7;font-size:0.82rem;">{html_mod.escape(n["summary"])}</span></div>'
                        for n in nodes
                    )
                    st.markdown(f"""
                    <details style="margin:0 0 12px 52px;border:1px solid rgba(128,128,128,0.1);border-radius:14px;padding:10px 16px;background:var(--secondary-background-color);font-size:0.85rem;">
                        <summary style="cursor:pointer;font-weight:600;color:var(--text-color);opacity:0.7;">Sources ({len(nodes)} sections)</summary>
                        <div style="margin-top:8px;">{nodes_html}</div>
                    </details>
                    """, unsafe_allow_html=True)

            # Input
            with st.form(key="doc_chat_form", clear_on_submit=True):
                qi, bi = st.columns([5, 1])
                with qi:
                    doc_question = st.text_input(
                        "doc_q", label_visibility="collapsed",
                        placeholder="Ask anything about your documents...",
                    )
                with bi:
                    doc_ask = st.form_submit_button("Ask", use_container_width=True)

            if doc_ask and doc_question.strip():
                with st.spinner("Searching your documents..."):
                    try:
                        srcs = selected_srcs if selected_srcs else None
                        answer, nodes = answer_from_docs(
                            st.session_state.user_index,
                            doc_question,
                            st.session_state.doc_history,
                            selected_sources=srcs,
                        )
                        score = evaluate_doc_answer(doc_question, answer)

                        st.session_state.doc_history.append(
                            {"role": "user", "content": doc_question, "nodes": []}
                        )
                        st.session_state.doc_history.append(
                            {"role": "assistant", "content": answer, "nodes": nodes}
                        )
                        # Track in main analytics
                        st.session_state.query_data = pd.concat([
                            st.session_state.query_data,
                            pd.DataFrame([{
                                "timestamp":       time.time(),
                                "query":           doc_question,
                                "terms":           len(doc_question.split()),
                                "response_terms":  len(answer.split()),
                                "classification":  "topic-related",
                                "topics":          ["My Documents"],
                                "relevancy_score": score,
                            }])
                        ], ignore_index=True)
                        st.session_state.relevancy_scores.append(score)
                        st.rerun()
                    except Exception as e:
                        traceback.print_exc()
                        st.error(f"Error: {e}")

            if st.session_state.doc_history:
                if st.button("Clear document chat history", key="clear_doc_history"):
                    st.session_state.doc_history = []
                    st.rerun()


# ══════════════════════════════════════════════
# TAB 4: MODEL SETUP
# ══════════════════════════════════════════════

AVAILABLE_MODELS = {
    "Qwen 2.5 3B (recommended, fast, 2 GB)":     "qwen2.5:3b",
    "Qwen 2.5 7B (smarter, 4.7 GB)":             "qwen2.5:7b",
    "Llama 3.2 3B (Meta, 2 GB)":                  "llama3.2",
    "Phi-3 Mini (Microsoft, 2.3 GB)":             "phi3:mini",
    "Gemma 3 4B (Google, 3.3 GB)":                "gemma3:4b",
    "Mistral 7B (great for reasoning, 4.1 GB)":   "mistral",
}

CLOUD_MODELS = {
    "Gemini 2.0 Flash (Google, free tier)":      "gemini/gemini-2.0-flash",
    "Gemini 2.0 Flash Lite (faster, free tier)":  "gemini/gemini-2.0-flash-lite",
    "GPT-3.5 Turbo (OpenAI)":                     "gpt-3.5-turbo",
}

with tab_setup:
    st.markdown("### LLM Setup")
    st.write("Choose and install a free local LLM, or connect a cloud API. Changes take effect after restarting the app.")

    # Ollama status
    st.markdown("#### Ollama Status")
    try:
        import requests as _req
        r = _req.get("http://localhost:11434", timeout=2)
        st.success("Ollama is running.")
        ollama_running = True
    except Exception:
        st.error("Ollama is not running. Start it with `ollama serve` or download it at **ollama.com**.")
        ollama_running = False
        if st.button("Open ollama.com"):
            webbrowser.open("https://ollama.com")

    if ollama_running:
        # Currently pulled models
        st.markdown("#### Installed Models")
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = [l for l in result.stdout.strip().splitlines() if l and "NAME" not in l]
                if lines:
                    for line in lines:
                        st.code(line, language=None)
                else:
                    st.info("No models installed yet.")
            else:
                st.warning("Could not list models.")
        except Exception:
            st.warning("Could not run `ollama list`.")

        # Install a free local model
        st.markdown("#### Install a Free Local LLM")
        st.caption("Models download once and run forever for free on your machine.")

        chosen_label = st.selectbox(
            "Pick a model:",
            list(AVAILABLE_MODELS.keys()),
            key="model_picker",
        )
        chosen_model = AVAILABLE_MODELS[chosen_label]

        if st.button(f"Install {chosen_model}", key="install_model_btn", type="primary"):
            env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
            with st.spinner(f"Downloading {chosen_model}... (this may take several minutes)"):
                try:
                    proc = subprocess.run(
                        ["ollama", "pull", chosen_model],
                        capture_output=True, text=True, timeout=600,
                    )
                    if proc.returncode == 0:
                        env_lines = []
                        if os.path.exists(env_path):
                            with open(env_path) as ef:
                                env_lines = ef.readlines()
                        new_lines = [l for l in env_lines if not l.startswith("LLM_MODEL")]
                        new_lines.append(f"LLM_MODEL=ollama/{chosen_model}\n")
                        with open(env_path, "w") as ef:
                            ef.writelines(new_lines)
                        st.success(f"{chosen_model} installed! Restart the app to use it.")
                    else:
                        st.error(f"Install failed: {proc.stderr}")
                except subprocess.TimeoutExpired:
                    st.error("Download timed out. Try again or pull manually: `ollama pull " + chosen_model + "`")

    # Cloud API option
    st.markdown("---")
    st.markdown("#### Use a Cloud LLM Instead")
    st.caption("No local model needed, but requires an internet connection and API key.")

    cloud_label = st.selectbox("Cloud model:", list(CLOUD_MODELS.keys()), key="cloud_picker")
    cloud_model = CLOUD_MODELS[cloud_label]
    api_key_input = st.text_input(
        "API Key:", type="password",
        placeholder="Paste your Gemini or OpenAI key here",
        key="api_key_input",
    )

    if st.button("Save Cloud Config", key="save_cloud_btn"):
        if api_key_input.strip():
            env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
            env_lines = []
            if os.path.exists(env_path):
                with open(env_path) as ef:
                    env_lines = ef.readlines()
            new_lines = [l for l in env_lines
                         if not l.startswith("LLM_MODEL")
                         and not l.startswith("GEMINI_API_KEY")
                         and not l.startswith("OPENAI_API_KEY")
                         and not l.startswith("OLLAMA_API_BASE")]
            new_lines.append(f"LLM_MODEL={cloud_model}\n")
            if "gemini" in cloud_model:
                new_lines.append(f"GEMINI_API_KEY={api_key_input.strip()}\n")
            else:
                new_lines.append(f"OPENAI_API_KEY={api_key_input.strip()}\n")
            with open(env_path, "w") as ef:
                ef.writelines(new_lines)
            st.success("Config saved. Restart the app to apply.")
        else:
            st.warning("Please enter an API key.")

    # Current config
    st.markdown("---")
    st.markdown("#### Current Config")
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as ef:
            config_text = ef.read()
        import re as _re
        masked = _re.sub(r"(KEY=)\S+", r"\1***", config_text)
        st.code(masked, language="bash")


# ══════════════════════════════════════════════
# TAB 5: ABOUT
# ══════════════════════════════════════════════

with tab_about:
    col_info, col_stack = st.columns([3, 2])

    with col_info:
        st.markdown("### Cognify")
        st.write(
            "Cognify is a locally-run Q&A system that answers questions from two knowledge "
            "sources: a built-in library of 50,000 Wikipedia articles across 10 topics, and "
            "any documents you upload yourself (PDFs and web pages). It runs entirely on your "
            "machine with a local LLM via Ollama. No API keys, no cloud services, no internet "
            "connection needed after setup."
        )

        st.markdown("### How It Works")
        st.markdown("""
**Wikipedia Q&A (Chat tab):**
1. **Query classification** - decides if the input is casual or topic-related.
2. **Topic classification** - maps the question to one or more topic buckets (Health, Technology, etc.).
3. **PageIndex retrieval** - a fast local keyword pass narrows 50,000 articles to 50 candidates; the LLM picks top 8.
4. **RAG answer generation** - the LLM writes a concise answer using the 8 retrieved article summaries as context.
5. **Relevancy scoring** - the LLM rates its own answer from 1 to 10 for quality tracking.

**My Documents (My Documents tab):**
1. **Indexing** - PDFs are parsed with heading detection (PyMuPDF); URLs are extracted via trafilatura.
2. **Text chunking** - long sections are split into overlapping 300-word chunks so no content is missed.
3. **PageIndex retrieval** - keyword pre-filter + LLM selects the 10 most relevant sections.
4. **RAG answer generation** - the LLM synthesizes across selected sections with source citations.
        """)

        st.markdown("### Data")
        total_articles = sum(len(v) for v in wiki_data.values())
        st.write(
            f"**{total_articles:,} Wikipedia articles** across **{len(wiki_data)} topics**: "
            + ", ".join(wiki_data.keys()) + "."
        )
        st.write(
            "**My Documents:** Upload any PDF or paste any URL to build a personal knowledge base. "
            "Documents are indexed in-session and fully searchable."
        )

    with col_stack:
        st.markdown("### Tech Stack")
        stack = {
            "LLM":              "Local via Ollama (Qwen, Llama, Mistral, Gemma, Phi)",
            "LLM bridge":       "LiteLLM (swap models via .env)",
            "Retrieval":        "PageIndex-style 2-level search (keyword + LLM reasoning)",
            "PDF parsing":      "PyMuPDF (heading detection + page fallback)",
            "Web extraction":   "trafilatura (content extraction, table support)",
            "Frontend":         "Streamlit 1.41",
            "Charts":           "Altair 5 + Matplotlib 3",
            "Data":             "Pandas 2.2",
        }
        for k, v in stack.items():
            st.markdown(f"**{k}:** {v}")

        st.markdown("### Architecture")
        st.code("""
Wikipedia RAG:
  User query > classify > topic buckets
    > keyword search (50K > 50) > LLM select (> 8)
    > LLM generate answer > relevancy score

My Documents RAG:
  PDF/URL > extract text > chunk (300w overlap)
    > build PageIndex tree > store in session
  Query > expand context > keyword filter (> 60)
    > LLM select (> 10) > LLM generate answer
        """, language="text")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.markdown("""
<div class="footer">
    Cognify &middot; 2025
</div>
""", unsafe_allow_html=True)
