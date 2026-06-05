"""
Chunking Strategy Visualizer — Bộ luật Lao động 45/2019/QH14
Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src import (
    Document, EmbeddingStore,
    FixedSizeChunker, RecursiveChunker, SectionChunker, SentenceChunker,
    _mock_embed,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants (module-level — evaluated once)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data/legal-doc")
CORPUS_META: dict = json.loads((DATA_DIR / "corpus_metadata.json").read_text(encoding="utf-8"))

BENCHMARK_QUERIES = [
    {"id": "Q1",
     "query": "Thời giờ làm việc bình thường tối đa là bao nhiêu giờ mỗi ngày và mỗi tuần?",
     "gold": "Không quá 08 giờ/ngày và 48 giờ/tuần (Điều 105)",
     "filter": {"topic": "thoi-gio-lam-viec"},
     "keywords": ["08 giờ", "48 giờ", "Điều 105", "bình thường"]},
    {"id": "Q2",
     "query": "Giờ làm việc ban đêm được tính từ mấy giờ đến mấy giờ?",
     "gold": "Từ 22 giờ đến 06 giờ sáng ngày hôm sau (Điều 106)",
     "filter": {"topic": "thoi-gio-lam-viec"},
     "keywords": ["22 giờ", "06 giờ", "ban đêm", "Điều 106"]},
    {"id": "Q3",
     "query": "Người sử dụng lao động có trách nhiệm gì về đào tạo bồi dưỡng kỹ năng nghề?",
     "gold": "Xây dựng kế hoạch hằng năm và dành kinh phí đào tạo (Điều 60)",
     "filter": {"topic": "giao-duc-nghe-nghiep"},
     "keywords": ["kế hoạch", "kinh phí", "đào tạo", "kỹ năng", "Điều 60"]},
    {"id": "Q4",
     "query": "Thanh tra lao động có những nội dung thanh tra gì?",
     "gold": "Thanh tra chấp hành pháp luật; điều tra tai nạn; hướng dẫn tiêu chuẩn; giải quyết khiếu nại; xử lý vi phạm (Điều 214)",
     "filter": {"topic": "thanh-tra"},
     "keywords": ["thanh tra", "tai nạn", "khiếu nại", "vi phạm", "Điều 214"]},
    {"id": "Q5",
     "query": "Người lao động nước ngoài làm việc tại Việt Nam có thuộc đối tượng áp dụng Bộ luật Lao động không?",
     "gold": "Có, Điều 2 khoản 3 quy định người lao động nước ngoài thuộc đối tượng áp dụng",
     "filter": {"topic": "quy-dinh-chung"},
     "keywords": ["nước ngoài", "Điều 2", "đối tượng áp dụng"]},
]

STRATEGY_LABELS = {"A": "A — SentenceChunker", "B": "B — FixedSizeChunker",
                   "C": "C — RecursiveChunker", "D": "D — SectionChunker"}
STRATEGY_COLORS = {"A": "#636EFA", "B": "#EF553B", "C": "#00CC96", "D": "#AB63FA"}


# ─────────────────────────────────────────────────────────────────────────────
# Pure helper functions (no Streamlit state — safe to cache)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_dieu(text: str) -> str:
    nums = re.findall(r"Điều\s+(\d+)", text)
    return ",".join(sorted(set(nums), key=int)) if nums else ""


def _load_file(fname: str) -> tuple[str, dict]:
    text = (DATA_DIR / fname).read_text(encoding="utf-8")
    return text, CORPUS_META[fname]


def _make_chunker(strategy_id: str, params: dict):
    if strategy_id == "A":
        return SentenceChunker(max_sentences_per_chunk=params.get("max_sentences", 3))
    if strategy_id == "B":
        return FixedSizeChunker(chunk_size=params.get("chunk_size", 400),
                                overlap=params.get("overlap", 100))
    if strategy_id == "C":
        return RecursiveChunker(separators=params.get("separators", ["\n\n", "\n", ". "]),
                                chunk_size=params.get("chunk_size", 500))
    if strategy_id == "D":
        return SectionChunker(max_size=params.get("max_size", 800),
                              min_size=params.get("min_size", 80))
    raise ValueError(strategy_id)


def _highlight(text: str, terms: list[str]) -> str:
    out = text
    for t in dict.fromkeys(terms):
        out = re.sub(f"({re.escape(t)})",
                     r'<mark style="background:#fff59d;border-radius:3px">\1</mark>',
                     out, flags=re.IGNORECASE)
    return out


def _is_relevant(content: str, keywords: list[str]) -> bool:
    low = content.lower()
    return sum(1 for kw in keywords if kw.lower() in low) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Cached compute functions (module-level — Streamlit can hash them safely)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_docs_cached(strategy_id: str, params_json: str, files_key: str) -> list[dict]:
    """Return list of serialisable dicts (not Document objects) for caching."""
    params = json.loads(params_json)
    chunker = _make_chunker(strategy_id, params)
    rows: list[dict] = []
    for fname in files_key.split("|"):
        text, fmeta = _load_file(fname)
        for i, chunk in enumerate(chunker.chunk(text)):
            rows.append({
                "id":       f"{strategy_id}_{fname}_{i}",
                "content":  chunk,
                "source":   fmeta["source"],
                "category": fmeta["category"],
                "date":     fmeta["date"],
                "language": fmeta["language"],
                "topic":    fmeta.get("topic", ""),
                "chapters": fmeta.get("chapters", ""),
                "strategy": strategy_id,
                "dieu_range": _extract_dieu(chunk),
                "chunk_len": len(chunk),
            })
    return rows


def rows_to_docs(rows: list[dict]) -> list[Document]:
    return [
        Document(id=r["id"], content=r["content"],
                 metadata={k: v for k, v in r.items() if k not in ("id", "content")})
        for r in rows
    ]


@st.cache_resource(show_spinner=False)
def build_store(strategy_id: str, params_json: str, files_key: str,
                embedder_name: str) -> EmbeddingStore:
    rows = build_docs_cached(strategy_id, params_json, files_key)
    docs = rows_to_docs(rows)

    if embedder_name == "local":
        from src import LocalEmbedder
        emb_fn = LocalEmbedder()
    else:
        emb_fn = _mock_embed

    store = EmbeddingStore(
        collection_name=f"{strategy_id}_{hash(params_json + files_key)}",
        embedding_fn=emb_fn,
    )
    store.add_documents(docs)
    return store


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chunking Strategy Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
mark { padding: 0 2px; }
.chunk-card { border:1px solid #e0e0e0; border-radius:8px; padding:12px 16px;
              margin-bottom:8px; background:#fafafa; font-size:0.9em; line-height:1.5; }
.chunk-card.s-A { border-left:4px solid #636EFA; }
.chunk-card.s-B { border-left:4px solid #EF553B; }
.chunk-card.s-C { border-left:4px solid #00CC96; }
.chunk-card.s-D { border-left:4px solid #AB63FA; }
.score-pill { display:inline-block; background:#e8f5e9; color:#2e7d32;
              border-radius:4px; padding:1px 7px; font-size:0.82em; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Cấu hình")

    st.subheader("📂 Tài liệu")
    all_files = list(CORPUS_META.keys())
    selected_files: list[str] = st.multiselect(
        "Chọn file:",
        options=all_files,
        default=all_files,
        format_func=lambda f: CORPUS_META[f]["chapters"] + " · " + CORPUS_META[f]["category"],
    )
    if not selected_files:
        st.warning("Chọn ít nhất 1 file.")
        st.stop()

    st.subheader("🔡 Embedder")
    embedder_name = "local" if st.radio(
        "Backend:",
        ["mock (tức thì, không semantic)", "local (all-MiniLM-L6-v2, ~30s)"],
        index=0,
    ).startswith("local") else "mock"

    st.divider()
    st.subheader("📐 Strategies")

    active: dict[str, dict] = {}

    with st.expander("A — SentenceChunker", expanded=True):
        if st.checkbox("Bật A", value=True, key="en_A"):
            active["A"] = {"max_sentences": st.slider("max_sentences", 1, 8, 3, key="A_ms")}

    with st.expander("B — FixedSizeChunker", expanded=True):
        if st.checkbox("Bật B", value=True, key="en_B"):
            cs = st.slider("chunk_size", 100, 1000, 400, step=50, key="B_cs")
            ov = st.slider("overlap", 0, 300, 100, step=10, key="B_ov")
            active["B"] = {"chunk_size": cs, "overlap": ov}

    with st.expander("C — RecursiveChunker", expanded=True):
        if st.checkbox("Bật C", value=True, key="en_C"):
            cs_c = st.slider("chunk_size ", 100, 1000, 500, step=50, key="C_cs")
            seps = st.multiselect("Separators:", ["\n\n", "\n", ". ", " "],
                                  default=["\n\n", "\n", ". "], key="C_sep")
            active["C"] = {"chunk_size": cs_c,
                           "separators": seps if seps else ["\n\n", ". "]}

    with st.expander("D — SectionChunker (Điều/Mục/Chương)", expanded=True):
        if st.checkbox("Bật D", value=True, key="en_D"):
            mx = st.slider("max_size", 200, 2000, 800, step=100, key="D_max")
            mn = st.slider("min_size", 10, 300, 80, step=10, key="D_min")
            active["D"] = {"max_size": mx, "min_size": mn}

    if not active:
        st.warning("Bật ít nhất 1 strategy.")
        st.stop()

    st.divider()
    st.caption(f"{len(active)} strategies · {len(selected_files)} files")


# ─────────────────────────────────────────────────────────────────────────────
# Pre-build chunk lists (no embedding — fast)
# ─────────────────────────────────────────────────────────────────────────────

files_key = "|".join(sorted(selected_files))
all_rows: dict[str, list[dict]] = {}

with st.spinner("Chunking corpus..."):
    for sid, params in active.items():
        all_rows[sid] = build_docs_cached(sid, json.dumps(params, sort_keys=True), files_key)


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

st.title("📊 So Sánh Chunking Strategy")
st.caption("Bộ luật Lao động 45/2019/QH14")

tab_stats, tab_dist, tab_search, tab_bench = st.tabs([
    "📈 Thống kê",
    "📉 Phân phối",
    "🔍 Tìm kiếm",
    "🎯 Benchmark",
])


# ───────────────────────────────────────────────────────────── TAB 1 ─────────
with tab_stats:
    st.header("Thống kê tổng quan")

    stat_rows = []
    for sid, rows in all_rows.items():
        lens = [r["chunk_len"] for r in rows]
        s = pd.Series(lens)
        stat_rows.append({
            "Strategy": STRATEGY_LABELS[sid],
            "Tổng chunks": len(rows),
            "Avg": round(s.mean()),
            "Median": round(s.median()),
            "Min": int(s.min()),
            "Max": int(s.max()),
            "Std": round(s.std()),
        })
    df_stat = pd.DataFrame(stat_rows)
    st.dataframe(df_stat, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    cmap = {STRATEGY_LABELS[s]: STRATEGY_COLORS[s] for s in active}
    with c1:
        fig = px.bar(df_stat, x="Strategy", y="Tổng chunks", color="Strategy",
                     color_discrete_map=cmap, text_auto=True,
                     title="Số lượng chunks")
        fig.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        df2 = df_stat.melt(id_vars="Strategy", value_vars=["Avg", "Min", "Max"],
                           var_name="Metric", value_name="Chars")
        fig2 = px.bar(df2, x="Strategy", y="Chars", color="Metric", barmode="group",
                      title="Kích thước chunk (chars)",
                      color_discrete_sequence=["#4FC3F7", "#81C784", "#FF8A65"])
        fig2.update_layout(height=340)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Metadata mẫu — Strategy D (5 chunks đầu)")
    sample_sid = "D" if "D" in all_rows else list(all_rows.keys())[0]
    preview = [
        {k: v for k, v in r.items() if k not in ("content",)} | {"content[:60]": r["content"][:60]}
        for r in all_rows[sample_sid][:5]
    ]
    st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)


# ───────────────────────────────────────────────────────────── TAB 2 ─────────
with tab_dist:
    st.header("Phân phối kích thước chunk")

    dist_rows = [
        {"Strategy": STRATEGY_LABELS[sid], "Chunk length": r["chunk_len"],
         "category": r["category"]}
        for sid, rows in all_rows.items() for r in rows
    ]
    df_dist = pd.DataFrame(dist_rows)

    chart_type = st.radio("Loại biểu đồ:", ["Violin", "Box", "Histogram"], horizontal=True)
    cmap2 = {STRATEGY_LABELS[s]: STRATEGY_COLORS[s] for s in active}

    if chart_type == "Violin":
        fig = px.violin(df_dist, x="Strategy", y="Chunk length", color="Strategy",
                        color_discrete_map=cmap2, box=True, points="outliers",
                        title="Phân phối kích thước (Violin)")
    elif chart_type == "Box":
        fig = px.box(df_dist, x="Strategy", y="Chunk length", color="Strategy",
                     color_discrete_map=cmap2, points="outliers",
                     title="Phân phối kích thước (Box plot)")
    else:
        fig = px.histogram(df_dist, x="Chunk length", color="Strategy",
                           color_discrete_map=cmap2, nbins=60,
                           barmode="overlay", opacity=0.65,
                           title="Histogram kích thước chunk")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Bucket table
    buckets = [(0, 50), (51, 150), (151, 400), (401, 800), (801, 99999)]
    bucket_labels = ["1–50", "51–150", "151–400", "401–800", "801+"]
    brows = []
    for sid, rows in all_rows.items():
        lens = [r["chunk_len"] for r in rows]
        total = max(len(lens), 1)
        row = {"Strategy": STRATEGY_LABELS[sid]}
        for label, (lo, hi) in zip(bucket_labels, buckets):
            cnt = sum(1 for l in lens if lo <= l <= hi)
            row[label] = f"{cnt} ({cnt*100//total}%)"
        brows.append(row)
    st.subheader("Phân nhóm kích thước")
    st.dataframe(pd.DataFrame(brows), use_container_width=True, hide_index=True)


# ───────────────────────────────────────────────────────────── TAB 3 ─────────
with tab_search:
    st.header("🔍 Tìm kiếm trực tiếp")

    query = st.text_input(
        "Câu hỏi:",
        value="Giờ làm việc ban đêm được tính từ mấy giờ đến mấy giờ?",
    )
    c_flt, c_k = st.columns([3, 1])
    with c_flt:
        topic_options = ["— không filter —"] + sorted(
            {CORPUS_META[f]["topic"] for f in selected_files}
        )
        chosen_topic = st.selectbox("Filter topic:", topic_options)
    with c_k:
        top_k = st.slider("Top-K:", 1, 10, 3)

    meta_filter = None if chosen_topic.startswith("—") else {"topic": chosen_topic}
    highlight_terms = [w for w in query.split() if len(w) >= 3]

    if st.button("🔎 Tìm kiếm", type="primary"):
        if not query.strip():
            st.warning("Nhập câu hỏi trước.")
        else:
            spinner_msg = st.empty()
            spinner_msg.info("Đang build stores... (lần đầu có thể mất vài giây)")
            cols = st.columns(len(active))
            for col_obj, (sid, params) in zip(cols, active.items()):
                store = build_store(sid, json.dumps(params, sort_keys=True),
                                    files_key, embedder_name)
                if meta_filter:
                    results = store.search_with_filter(query, top_k=top_k,
                                                        metadata_filter=meta_filter)
                else:
                    results = store.search(query, top_k=top_k)

                with col_obj:
                    color = STRATEGY_COLORS[sid]
                    st.markdown(f"<h4 style='color:{color}'>{STRATEGY_LABELS[sid]}</h4>",
                                unsafe_allow_html=True)
                    for rank, r in enumerate(results, 1):
                        score = r.get("score", 0.0)
                        content = r.get("content", "")
                        meta = r.get("metadata", {})
                        body = _highlight(content[:400], highlight_terms)
                        if len(content) > 400:
                            body += "…"
                        st.markdown(
                            f'<div class="chunk-card s-{sid}">'
                            f'<b>#{rank}</b> <span class="score-pill">{score:.4f}</span> '
                            f'<span style="color:#888;font-size:0.8em"> '
                            f'{meta.get("source","")[:25]} | Điều {meta.get("dieu_range","—")} | {meta.get("chunk_len",0)} chars'
                            f'</span><br><br>{body}</div>',
                            unsafe_allow_html=True,
                        )
            spinner_msg.empty()
    else:
        st.info("Nhập câu hỏi và nhấn **Tìm kiếm**.")


# ───────────────────────────────────────────────────────────── TAB 4 ─────────
with tab_bench:
    st.header("🎯 Benchmark — 5 Queries chuẩn")
    st.markdown("Precision@3 = top-3 kết quả chứa ít nhất 2 keyword của gold answer.")

    if st.button("▶ Chạy Benchmark", type="primary"):
        progress = st.progress(0, text="Building stores…")

        stores: dict[str, EmbeddingStore] = {}
        for i, (sid, params) in enumerate(active.items()):
            progress.progress((i + 1) / len(active), text=f"Building store {sid}…")
            stores[sid] = build_store(sid, json.dumps(params, sort_keys=True),
                                      files_key, embedder_name)
        progress.empty()

        # ── Run queries ──────────────────────────────────────────────────────
        prec: dict[str, list[int]] = {s: [] for s in stores}
        scores: dict[str, list[float]] = {s: [] for s in stores}
        detail: list[dict] = []

        for bq in BENCHMARK_QUERIES:
            for sid, store in stores.items():
                top3 = store.search_with_filter(bq["query"], top_k=3,
                                                 metadata_filter=bq["filter"])
                score = top3[0].get("score", 0.0) if top3 else 0.0
                hit = any(_is_relevant(r.get("content", ""), bq["keywords"]) for r in top3)
                prec[sid].append(1 if hit else 0)
                scores[sid].append(score)
                detail.append({
                    "Query": bq["id"],
                    "Strategy": STRATEGY_LABELS[sid],
                    "Score": round(score, 4),
                    "✅?": "✅" if hit else "❌",
                    "Top-1 chunk (100 chars)": (top3[0].get("content", "")[:100] + "…") if top3 else "—",
                })

        # ── Precision table ──────────────────────────────────────────────────
        st.subheader("Precision@3")
        prec_rows = []
        for sid in stores:
            row = {"Strategy": STRATEGY_LABELS[sid]}
            for i, bq in enumerate(BENCHMARK_QUERIES):
                row[bq["id"]] = "✅" if prec[sid][i] else "❌"
            row["Tổng"] = f"{sum(prec[sid])}/5"
            prec_rows.append(row)
        st.dataframe(pd.DataFrame(prec_rows), use_container_width=True, hide_index=True)

        # ── Score heatmap ────────────────────────────────────────────────────
        st.subheader("Score heatmap (filtered, top-1)")
        score_df = pd.DataFrame(
            {STRATEGY_LABELS[s]: scores[s] for s in stores},
            index=[bq["id"] for bq in BENCHMARK_QUERIES],
        )
        fig_heat = px.imshow(
            score_df.T,
            color_continuous_scale="RdYlGn",
            zmin=0.4, zmax=1.0,
            title="Similarity Score — xanh lá = cao hơn",
            aspect="auto",
        )
        fig_heat.update_traces(text=score_df.T.round(3).values, texttemplate="%{text}")
        fig_heat.update_layout(height=280)
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── Avg score bar ────────────────────────────────────────────────────
        avg_rows = [{"Strategy": STRATEGY_LABELS[s],
                     "Avg Score": round(sum(scores[s]) / 5, 4)} for s in stores]
        fig_avg = px.bar(avg_rows, x="Strategy", y="Avg Score",
                         color="Strategy",
                         color_discrete_map={STRATEGY_LABELS[s]: STRATEGY_COLORS[s] for s in active},
                         text_auto=True, title="Avg Similarity Score (filtered top-1)")
        fig_avg.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_avg, use_container_width=True)

        # ── Detail table ─────────────────────────────────────────────────────
        st.subheader("Chi tiết từng query × strategy")
        st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)

        with st.expander("📋 Gold Answers"):
            for bq in BENCHMARK_QUERIES:
                st.markdown(f"**{bq['id']}:** {bq['query']}")
                st.success(bq["gold"])

    else:
        st.info("Nhấn **▶ Chạy Benchmark** để bắt đầu.")


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Lab 7 — Data Foundations | "
    f"Embedder: **{embedder_name}** | "
    f"Strategies: **{', '.join(active.keys())}** | "
    f"Files: **{len(selected_files)}** | "
    f"Chunks built: **{sum(len(r) for r in all_rows.values())}**"
)
