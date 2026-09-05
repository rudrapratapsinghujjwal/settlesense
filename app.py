"""
SettleSense — Streamlit Dashboard
====================================
Four views:
  1. Controller Overview  — batch summary, KPIs
  2. Exception Explorer   — per-exception detail, evidence, human actions
  3. Evaluation           — saved metrics, per-category breakdown
  4. Audit Trail          — immutable action log

Design: Premium dark theme, Inter font, high-contrast text, readable everywhere.
"""

import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from settlesense.config import config, load_config
from settlesense.database import (
    get_connection,
    initialize_database,
    insert_human_override,
)
from settlesense.pipeline import ensure_pipeline_ready, run_evaluation, run_full_pipeline

logger = logging.getLogger(__name__)

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SettleSense — AI Finance Controller",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset ─────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: #06080f !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #e2e8f0 !important;
}

/* ── Sidebar ─────────────────────────── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2433 !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebarUserContent"] { padding: 20px 16px !important; }

/* ── Main content ────────────────────── */
.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1400px !important;
}

/* ── Typography ──────────────────────── */
h1 { font-size: 2rem !important; font-weight: 800 !important; color: #f1f5f9 !important; letter-spacing: -0.02em !important; margin-bottom: 0.4rem !important; }
h2 { font-size: 1.25rem !important; font-weight: 700 !important; color: #f1f5f9 !important; }
h3 { font-size: 1rem !important; font-weight: 600 !important; color: #e2e8f0 !important; }
p, li { color: #cbd5e1 !important; }

/* ── KPI Card ────────────────────────── */
.kpi-card {
    background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
    border: 1px solid #1e2d45;
    border-radius: 14px;
    padding: 20px 18px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.15s;
}
.kpi-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(99,102,241,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.kpi-card:hover { border-color: #3b4b6b; transform: translateY(-2px); }
.kpi-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    margin-bottom: 10px;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -0.03em;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 6px;
    font-weight: 400;
}
.kv-blue   { color: #60a5fa; }
.kv-green  { color: #34d399; }
.kv-amber  { color: #fbbf24; }
.kv-red    { color: #f87171; }
.kv-indigo { color: #818cf8; }
.kv-slate  { color: #94a3b8; }

/* ── Section header ──────────────────── */
.sec-head {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #475569;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2433;
    margin: 24px 0 14px 0;
}

/* ── Status badge ────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.b-green  { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
.b-amber  { background: rgba(251,191,36,0.12);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
.b-red    { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.25); }
.b-blue   { background: rgba(96,165,250,0.12);  color: #60a5fa; border: 1px solid rgba(96,165,250,0.25); }
.b-indigo { background: rgba(129,140,248,0.12); color: #818cf8; border: 1px solid rgba(129,140,248,0.25); }
.b-mock   { background: rgba(251,146,60,0.08);  color: #fb923c; border: 1px dashed rgba(251,146,60,0.35); }
.b-slate  { background: rgba(100,116,139,0.12); color: #94a3b8; border: 1px solid rgba(100,116,139,0.25); }

/* ── Alert box ───────────────────────── */
.alert-warn {
    background: rgba(251,146,60,0.07);
    border: 1px solid rgba(251,146,60,0.25);
    border-left: 3px solid #fb923c;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 0.82rem;
    color: #fcd5ac;
    line-height: 1.6;
}
.alert-info {
    background: rgba(96,165,250,0.06);
    border: 1px solid rgba(96,165,250,0.2);
    border-left: 3px solid #60a5fa;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 0.82rem;
    color: #bfdbfe;
    line-height: 1.6;
}
.alert-err {
    background: rgba(248,113,113,0.07);
    border: 1px solid rgba(248,113,113,0.22);
    border-left: 3px solid #f87171;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 0.82rem;
    color: #fecaca;
    line-height: 1.6;
}

/* ── Confidence bar ──────────────────── */
.cbar-wrap { margin: 8px 0; }
.cbar-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.78rem;
    color: #94a3b8;
    margin-bottom: 5px;
    font-weight: 500;
}
.cbar-val { font-weight: 700; }
.cbar-bg {
    background: #1e2433;
    border-radius: 100px;
    height: 6px;
    overflow: hidden;
}
.cbar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.5s ease;
}

/* ── Detail card ─────────────────────── */
.detail-card {
    background: #0d1117;
    border: 1px solid #1e2433;
    border-radius: 12px;
    padding: 20px;
}

/* ── Evidence item ───────────────────── */
.ev-item {
    background: #0d1117;
    border-left: 3px solid #4f46e5;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 7px 0;
}
.ev-field { font-size: 0.72rem; font-weight: 700; color: #818cf8; text-transform: uppercase; letter-spacing: 0.06em; }
.ev-value { font-size: 0.88rem; color: #e2e8f0; font-family: 'Menlo', 'Monaco', monospace; margin: 3px 0; word-break: break-all; }
.ev-rel   { font-size: 0.75rem; color: #64748b; font-style: italic; }
.ev-untrusted { border-left-color: #d97706; }

/* ── Reasoning box ───────────────────── */
.reasoning-box {
    background: #0d1117;
    border: 1px solid #1e2433;
    border-radius: 10px;
    padding: 16px;
    font-size: 0.87rem;
    color: #cbd5e1;
    line-height: 1.7;
}
.rec-box {
    background: rgba(79,70,229,0.07);
    border: 1px solid rgba(79,70,229,0.2);
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 0.87rem;
    color: #c7d2fe;
    line-height: 1.65;
}

/* ── Record ID pill ──────────────────── */
.rec-id {
    font-family: 'Menlo', 'Monaco', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: #818cf8;
    background: rgba(129,140,248,0.08);
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid rgba(129,140,248,0.18);
    margin: 6px 0 10px 0;
    letter-spacing: 0.02em;
}

/* ── Recent-decision row ─────────────── */
.rd-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #12182a;
    font-size: 0.8rem;
    gap: 8px;
}
.rd-conf { font-weight: 700; color: #94a3b8; font-variant-numeric: tabular-nums; }

/* ── Metric small ────────────────────── */
.mini-metric {
    background: #0d1117;
    border: 1px solid #1e2433;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.mini-val { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; }
.mini-lbl { font-size: 0.66rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #475569; margin-top: 5px; }

/* ── Streamlit-specific ──────────────── */
#MainMenu, footer, .stDeployButton { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }

/* Select boxes + inputs */
.stSelectbox > div > div,
.stTextInput > div > div {
    background: #0d1117 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.stSelectbox label, .stTextInput label { color: #94a3b8 !important; font-size: 0.78rem !important; font-weight: 600 !important; }

/* Buttons */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
    border: 1px solid #1e2d45 !important;
    background: #111827 !important;
    color: #e2e8f0 !important;
    transition: all 0.15s !important;
    padding: 8px 16px !important;
}
.stButton > button:hover {
    background: #1e293b !important;
    border-color: #4f46e5 !important;
    color: #c7d2fe !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border-color: transparent !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4338ca, #6d28d9) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid #1e2433 !important;
}
.stDataFrame thead th {
    background: #0d1117 !important;
    color: #64748b !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 700 !important;
    border-bottom: 1px solid #1e2433 !important;
}
.stDataFrame tbody td {
    color: #cbd5e1 !important;
    font-size: 0.83rem !important;
    border-bottom: 1px solid #0f1623 !important;
}
.stDataFrame tbody tr:hover td { background: #0f172a !important; }

/* Radio */
.stRadio > div { gap: 6px !important; }
.stRadio label { font-size: 0.85rem !important; color: #94a3b8 !important; font-weight: 500 !important; }

/* Spinner */
.stSpinner { color: #818cf8 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2d3f5e; }

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: #111827 !important;
    border-color: #1e2d45 !important;
    color: #94a3b8 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _badge(label: str, cls: str) -> str:
    return f'<span class="badge {cls}">{label}</span>'

STATUS_BADGE = {
    "auto_resolved":  lambda: _badge("✓ Auto-Resolved",  "b-green"),
    "human_review":   lambda: _badge("⚑ Needs Review",   "b-amber"),
    "human_approved": lambda: _badge("✓ Approved",       "b-blue"),
    "human_rejected": lambda: _badge("✗ Rejected",       "b-red"),
    "human_overridden":lambda: _badge("↺ Overridden",    "b-indigo"),
    "escalated":      lambda: _badge("⬆ Escalated",      "b-red"),
}

CAT_BADGE = {
    "split_settlement":       lambda: _badge("Split Settlement",  "b-blue"),
    "refund_misattribution":  lambda: _badge("Refund Mismatch",   "b-indigo"),
    "fee_tier":               lambda: _badge("Fee Tier",          "b-amber"),
    "near_duplicate":         lambda: _badge("Near Duplicate",    "b-red"),
    "unresolved":             lambda: _badge("Unresolved",        "b-slate"),
    "clean":                  lambda: _badge("Clean",             "b-green"),
}

def _sbadge(status: str) -> str:
    return STATUS_BADGE.get(status, lambda: _badge(status, "b-slate"))()

def _cbadge(cat: str) -> str:
    return CAT_BADGE.get(cat, lambda: _badge(cat, "b-slate"))()

def _conf_color(v: float) -> str:
    if v >= 0.75: return "#34d399"
    if v >= 0.50: return "#fbbf24"
    return "#f87171"

def _kv_class(v: float, good_high: bool = True) -> str:
    if good_high:
        if v >= 0.75: return "kv-green"
        if v >= 0.40: return "kv-amber"
        return "kv-red"
    else:
        if v <= 0.05: return "kv-green"
        if v <= 0.10: return "kv-amber"
        return "kv-red"

def _pct(v) -> str:
    try: return f"{float(v):.1%}"
    except: return "—"

def _is_mock(reasoning: str) -> bool:
    return "[MOCK" in (reasoning or "") or not (reasoning or "").strip()

def kpi(col, value: str, label: str, cls: str, sub: str = ""):
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {cls}">{value}</div>'
        f'{"<div class=kpi-sub>" + sub + "</div>" if sub else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )

def conf_bar(label: str, value: float, color: str = None):
    c = color or _conf_color(value)
    st.markdown(
        f'<div class="cbar-wrap">'
        f'<div class="cbar-row"><span>{label}</span>'
        f'<span class="cbar-val" style="color:{c}">{value:.0%}</span></div>'
        f'<div class="cbar-bg"><div class="cbar-fill" style="width:{value*100:.1f}%;background:{c};"></div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─── Cache / Init ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _startup():
    return ensure_pipeline_ready(config)

@st.cache_resource(show_spinner=False)
def _conn():
    initialize_database(config.db_path)
    return get_connection(config.db_path)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def sidebar() -> str:
    sb = st.sidebar

    sb.markdown("""
    <div style="padding:6px 0 22px 0">
      <div style="font-size:1.5rem;font-weight:900;background:linear-gradient(135deg,#818cf8,#a78bfa);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-0.02em">
        ⚡ SettleSense
      </div>
      <div style="font-size:0.72rem;color:#475569;margin-top:2px;font-weight:500">
        AI Finance Controller
      </div>
      <div style="font-size:0.63rem;color:#334155;margin-top:5px">
        Razorpay AI Buildathon · Track 04
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = sb.radio("", ["Controller Overview", "Exception Explorer", "Evaluation", "Audit Trail"],
                    label_visibility="collapsed")

    sb.markdown('<hr style="border-color:#1e2433;margin:16px 0">', unsafe_allow_html=True)

    provider_color = "#fb923c" if config.llm.provider == "mock" else "#34d399"
    # Display friendly name for Google AI / Gemini
    _key = config.llm.openai_api_key or ""
    _is_google = _key.startswith("AQ.") or _key.startswith("AI")
    provider_label = (
        "GEMINI" if (config.llm.provider == "openai" and _is_google)
        else config.llm.provider.upper()
    )
    sb.markdown(f"""
    <div style="font-size:0.72rem;line-height:2;color:#64748b">
      <span style="color:#475569;font-weight:600">Provider</span><br>
      <span style="color:{provider_color};font-weight:700">{provider_label}</span>
      {'<span style="color:#64748b"> · ' + config.llm.model + '</span>' if config.llm.provider != "mock" else ''}<br>
      <span style="color:#475569;font-weight:600">Threshold</span>
      <span style="color:#818cf8;font-weight:700"> {config.confidence_threshold:.0%}</span><br>
      <span style="color:#475569;font-weight:600">Seed</span>
      <span style="color:#60a5fa"> {config.random_seed}</span>
    </div>
    """, unsafe_allow_html=True)

    if config.llm.provider == "mock":
        sb.markdown("""
        <div class="alert-warn" style="margin-top:14px;font-size:0.73rem">
          <b>⚠ Mock Mode Active</b><br>
          No valid LLM key found.<br>
          AI outputs are deterministic.<br>
          Set ANTHROPIC_API_KEY or<br>
          OPENAI_API_KEY for real AI.
        </div>
        """, unsafe_allow_html=True)

    return page


# ─── VIEW 1: Controller Overview ──────────────────────────────────────────────

def view_overview(conn):
    st.markdown("# Controller Overview")
    st.markdown(
        '<p style="color:#64748b;margin-bottom:28px;font-size:0.9rem;font-style:italic">'
        '"Automate what is certain. Explain what is ambiguous. Escalate what is uncertain."'
        '</p>', unsafe_allow_html=True)

    decisions = conn.execute(
        """SELECT pd.record_id, pd.category, pd.status,
                  COALESCE(cs.calibrated_confidence, 0) as conf
           FROM pipeline_decisions pd
           LEFT JOIN confidence_signals cs ON pd.record_id = cs.record_id"""
    ).fetchall()

    if not decisions:
        st.markdown('<div class="alert-info">🚀 No pipeline data yet. Click <b>Run Pipeline</b> to begin.</div>', unsafe_allow_html=True)
        if st.button("▶  Run Pipeline", type="primary"):
            with st.spinner("Running full pipeline…"):
                run_full_pipeline(config, split="tune", save_to_db=True)
                st.success("Done!")
                st.rerun()
        return

    total = len(decisions)
    auto = sum(1 for d in decisions if d["status"] == "auto_resolved")
    human = sum(1 for d in decisions if d["status"] in ("human_review", "escalated"))
    auto_rate = auto / total if total else 0.0

    # Pull run summary from audit
    audit_row = conn.execute(
        "SELECT detail FROM audit_log WHERE action='run_complete' ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    total_processed, clean_count, throughput, elapsed = 0, 0, 0.0, 0.0
    if audit_row:
        try:
            d = json.loads(audit_row["detail"])
            total_processed = d.get("total_records", total)
            clean_count = d.get("clean_records", 0)
            throughput = d.get("throughput_rps", 0.0)
            elapsed = d.get("total_time_ms", 0.0)
        except Exception:
            pass

    # Evaluation metrics
    eval_row = conn.execute(
        "SELECT false_auto_resolve_rate, overall_accuracy FROM evaluation_runs ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    far = eval_row["false_auto_resolve_rate"] if eval_row else None
    acc = eval_row["overall_accuracy"] if eval_row else None

    # ── Row 1: primary KPIs ──
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpi(c1, str(total_processed), "Total Processed",  "kv-blue",   f"{elapsed:.0f} ms")
    kpi(c2, str(clean_count),     "Clean Matched",    "kv-green",  f"{clean_count/total_processed:.0%}" if total_processed else "")
    kpi(c3, str(total),           "Exceptions",       "kv-amber",  f"{total/total_processed:.0%} of total" if total_processed else "")
    kpi(c4, str(auto),            "AI Auto-Resolved", "kv-green",  "✓ confident")
    kpi(c5, str(human),           "Human Review",     "kv-amber",  "⚑ queued")
    kpi(c6, _pct(auto_rate),      "Automation Rate",  "kv-indigo", f"{throughput:.0f} rec/s" if throughput else "")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: trust KPIs ──
    c7,c8,c9,c10 = st.columns(4)
    if far is not None:
        kpi(c7, _pct(far), "False Auto-Resolve ↓", _kv_class(far, good_high=False), "Target: < 5%")
    else:
        kpi(c7, "—", "False Auto-Resolve ↓", "kv-slate", "Run evaluation")
    if acc is not None:
        kpi(c8, _pct(acc), "Classification Acc", _kv_class(acc, good_high=True))
    else:
        kpi(c8, "—", "Classification Acc", "kv-slate", "Run evaluation")
    kpi(c9, _pct(config.confidence_threshold), "Confidence Gate", "kv-indigo", "Logistic regression")
    _pkey = config.llm.openai_api_key or ""
    _plabel = "GEMINI" if (config.llm.provider == "openai" and (_pkey.startswith("AQ.") or _pkey.startswith("AI"))) else config.llm.provider.upper()
    kpi(c10, _plabel, "LLM Provider", "kv-blue" if config.llm.provider != "mock" else "kv-amber")


    st.markdown("<br>", unsafe_allow_html=True)

    # ── Category breakdown + recent ──
    st.markdown('<div class="sec-head">Exception Category Breakdown</div>', unsafe_allow_html=True)

    from collections import Counter
    cats = Counter(d["category"] for d in decisions)

    col_tbl, col_rec = st.columns([3, 2])

    with col_tbl:
        CAT_LABELS = {
            "split_settlement": "Split Settlement",
            "refund_misattribution": "Refund Misattribution",
            "fee_tier": "Fee Tier",
            "near_duplicate": "Near Duplicate",
            "unresolved": "Unresolved",
        }
        rows = []
        for cat, label in CAT_LABELS.items():
            n = cats.get(cat, 0)
            if n:
                rows.append({"Category": label, "Count": n, "Share": _pct(n/total)})
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No exceptions classified yet.")

    with col_rec:
        st.markdown('<div style="font-size:0.75rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px">Recent Decisions</div>', unsafe_allow_html=True)
        for d in reversed(decisions[-8:]):
            conf = float(d["conf"])
            st.markdown(
                f'<div class="rd-row">'
                f'{_cbadge(d["category"])} {_sbadge(d["status"])}'
                f'<span class="rd-conf" style="color:{_conf_color(conf)}">{conf:.0%}</span>'
                f'</div>', unsafe_allow_html=True)

    # ── Controls ──
    st.markdown('<div class="sec-head" style="margin-top:28px">Pipeline Controls</div>', unsafe_allow_html=True)
    btn1, btn2, btn3 = st.columns([1, 1, 4])
    with btn1:
        if st.button("🔄  Re-run Pipeline"):
            with st.spinner("Running…"):
                run_full_pipeline(config, split="tune", save_to_db=True)
                st.rerun()
    with btn2:
        if st.button("📊  Evaluate", type="primary"):
            with st.spinner("Evaluating…"):
                r = run_evaluation(config, split="tune")
                if "error" in r:
                    st.error(r["error"])
                else:
                    st.success(f"Accuracy: {r.get('overall_accuracy',0):.0%}")
                    st.rerun()


# ─── VIEW 2: Exception Explorer ───────────────────────────────────────────────

def view_exceptions(conn):
    st.markdown("# Exception Explorer")
    st.markdown(
        '<p style="color:#64748b;margin-bottom:24px;font-size:0.88rem">'
        'Inspect evidence, AI reasoning, calibrated confidence, and take human actions.'
        '</p>', unsafe_allow_html=True)

    rows = conn.execute("""
        SELECT pd.decision_id, pd.record_id, pd.category, pd.status,
               COALESCE(cs.calibrated_confidence, 0) as conf,
               pd.threshold_used,
               ad.proposed_linked_ids, ad.evidence_used,
               ad.raw_model_signal, ad.recommended_action,
               ad.reasoning_summary, ad.is_valid,
               ad.validation_error, ad.hallucinated_evidence,
               COALESCE(cs.candidate_margin, 0)      as margin,
               COALESCE(cs.rule_agreement, 0)        as rule_agree,
               COALESCE(cs.evidence_completeness, 0) as ev_complete
        FROM pipeline_decisions pd
        LEFT JOIN ai_decisions ad ON pd.decision_id = ad.decision_id
        LEFT JOIN confidence_signals cs ON pd.record_id = cs.record_id
        ORDER BY pd.timestamp DESC
    """).fetchall()

    if not rows:
        st.markdown('<div class="alert-info">No decisions yet. Run the pipeline from Controller Overview.</div>', unsafe_allow_html=True)
        return

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        cat_f = st.selectbox("Category", ["All","split_settlement","refund_misattribution","fee_tier","near_duplicate","unresolved"])
    with f2:
        sta_f = st.selectbox("Status", ["All","human_review","auto_resolved","human_approved","human_rejected","escalated"])
    with f3:
        sort = st.selectbox("Sort", ["Confidence ↑ (lowest first)","Confidence ↓","Newest first"])

    filtered = list(rows)
    if cat_f != "All":  filtered = [r for r in filtered if r["category"] == cat_f]
    if sta_f != "All":  filtered = [r for r in filtered if r["status"] == sta_f]
    if sort == "Confidence ↑ (lowest first)": filtered.sort(key=lambda r: r["conf"])
    elif sort == "Confidence ↓":              filtered.sort(key=lambda r: -r["conf"])

    st.markdown(f'<div style="font-size:0.78rem;color:#475569;margin-bottom:10px">Showing <b style="color:#94a3b8">{len(filtered)}</b> of <b style="color:#94a3b8">{len(rows)}</b> exceptions</div>', unsafe_allow_html=True)

    if not filtered:
        st.info("No exceptions match the selected filters.")
        return

    opts = [f"{i+1}. {r['record_id']} — {r['category'].replace('_',' ').title()} ({r['conf']:.0%})" for i, r in enumerate(filtered)]
    idx = st.selectbox("Select Exception", range(len(opts)), format_func=lambda i: opts[i])
    row = filtered[idx]

    st.markdown("---")

    is_mock = _is_mock(row["reasoning_summary"] or "")
    if is_mock:
        st.markdown('<div class="alert-warn"><b>⚠ Mock AI Result</b> — Set a valid LLM API key for real AI classification. All signals and gate logic are real; only the model response is synthetic.</div>', unsafe_allow_html=True)

    if row["hallucinated_evidence"]:
        st.markdown('<div class="alert-err"><b>🚨 Hallucination Detected</b> — Model cited evidence absent from source data. Automatically sent to human review.</div>', unsafe_allow_html=True)

    if row["validation_error"]:
        st.markdown(f'<div class="alert-err"><b>Schema Error:</b> {row["validation_error"]}</div>', unsafe_allow_html=True)

    # ── Main two-column layout ──
    left, right = st.columns([3, 2])

    with left:
        st.markdown(
            f'<div class="detail-card">'
            f'<div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#475569">Record ID</div>'
            f'<div class="rec-id">{row["record_id"]}</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:7px;margin-bottom:4px">'
            f'{_cbadge(row["category"])} {_sbadge(row["status"])}'
            f'{" " + _badge("MOCK", "b-mock") if is_mock else ""}'
            f'</div>'
            f'</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-head">AI Reasoning</div>', unsafe_allow_html=True)
        reasoning = row["reasoning_summary"] or "<em style='color:#475569'>No reasoning available (mock mode or invalid output)</em>"
        st.markdown(f'<div class="reasoning-box">{reasoning}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-head">Recommended Action</div>', unsafe_allow_html=True)
        rec = row["recommended_action"] or "<em style='color:#475569'>No recommendation available</em>"
        st.markdown(f'<div class="rec-box">💡 {rec}</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="detail-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:#475569;margin-bottom:14px">Confidence Signals</div>', unsafe_allow_html=True)
        conf = float(row["conf"])
        conf_bar("Calibrated Confidence",    conf,               _conf_color(conf))
        conf_bar("Candidate Margin",         float(row["margin"]),     "#60a5fa")
        conf_bar("Rule Agreement",           float(row["rule_agree"]), "#34d399")
        conf_bar("Evidence Completeness",    float(row["ev_complete"]),"#818cf8")

        st.markdown(
            f'<div style="margin-top:14px;font-size:0.73rem;color:#475569;line-height:1.9">'
            f'Gate: <b style="color:#94a3b8">{row["threshold_used"]:.0%}</b> threshold &nbsp;·&nbsp; '
            f'Decision: <b style="color:{"#34d399" if row["status"]=="auto_resolved" else "#fbbf24"}">'
            f'{"✓ Auto-Resolved" if row["status"]=="auto_resolved" else "⚑ Human Review"}'
            f'</b></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Evidence ──
    st.markdown('<div class="sec-head">Evidence Used</div>', unsafe_allow_html=True)
    try:
        ev_items = json.loads(row["evidence_used"] or "[]")
    except Exception:
        ev_items = []

    if ev_items:
        for ev in ev_items:
            untrusted = "TEXT_ONLY" in ev.get("field", "")
            cls = "ev-item ev-untrusted" if untrusted else "ev-item"
            tag = ' <span style="font-size:0.65rem;color:#d97706;font-style:normal">[untrusted text]</span>' if untrusted else ""
            st.markdown(
                f'<div class="{cls}">'
                f'<div class="ev-field">{ev.get("field","")}{tag}</div>'
                f'<div class="ev-value">{ev.get("value","")}</div>'
                f'<div class="ev-rel">{ev.get("relevance","")}</div>'
                f'</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#475569;font-size:0.85rem;padding:10px 0">No evidence items captured.</div>', unsafe_allow_html=True)

    # ── Human actions ──
    st.markdown('<div class="sec-head">Human Review Actions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.77rem;color:#475569;margin-bottom:14px">'
        'Actions are written to the SQLite audit trail. On Hugging Face free tier, '
        'they do not persist across container restarts (ephemeral storage limitation).'
        '</div>', unsafe_allow_html=True)

    b1, b2, b3, b4 = st.columns(4)
    rec_id = row["record_id"]
    did = row["decision_id"]

    with b1:
        if st.button("✅  Approve", key=f"a_{rec_id}", type="primary"):
            insert_human_override(conn, did, rec_id, "approved")
            st.success("Approved.")
            st.rerun()
    with b2:
        if st.button("❌  Reject", key=f"r_{rec_id}"):
            insert_human_override(conn, did, rec_id, "rejected")
            st.warning("Rejected.")
            st.rerun()
    with b3:
        ov_cat = st.selectbox("", ["split_settlement","refund_misattribution","fee_tier","near_duplicate"],
                               key=f"oc_{rec_id}", label_visibility="collapsed")
        if st.button("↺  Override", key=f"o_{rec_id}"):
            insert_human_override(conn, did, rec_id, "overridden", ov_cat)
            st.info(f"Overridden → {ov_cat}")
            st.rerun()
    with b4:
        if st.button("⬆  Escalate", key=f"e_{rec_id}"):
            insert_human_override(conn, did, rec_id, "escalated")
            st.error("Escalated.")
            st.rerun()


# ─── VIEW 3: Evaluation ───────────────────────────────────────────────────────

def view_evaluation(conn):
    st.markdown("# Evaluation")
    st.markdown(
        '<p style="color:#64748b;margin-bottom:24px;font-size:0.88rem">'
        'Metrics loaded from the database — never recomputed on page load. Holdout is sealed until submission.'
        '</p>', unsafe_allow_html=True)

    split = st.selectbox("Dataset Split", ["tune", "validation", "holdout"])
    latest = conn.execute(
        "SELECT * FROM evaluation_runs WHERE dataset_split=? ORDER BY timestamp DESC LIMIT 1", (split,)
    ).fetchone()

    if not latest:
        st.markdown(f'<div class="alert-info">No evaluation for <b>{split}</b> split yet.</div>', unsafe_allow_html=True)
        if st.button(f"📊  Run Evaluation ({split})", type="primary"):
            with st.spinner("Evaluating…"):
                r = run_evaluation(config, split=split)
                if "error" in r: st.error(r["error"])
                else: st.success("Done!"); st.rerun()
        return

    # ── KPIs ──
    c1,c2,c3,c4,c5 = st.columns(5)
    acc   = float(latest["overall_accuracy"])
    arate = float(latest["automation_rate"])
    far   = float(latest["false_auto_resolve_rate"])
    n_auto = int(latest["auto_resolved"])
    n_hum  = int(latest["human_review"])
    total  = int(latest["total_records"])

    def mini(col, val, lbl, cls):
        col.markdown(f'<div class="mini-metric"><div class="mini-val {cls}">{val}</div><div class="mini-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    mini(c1, _pct(acc),   "Overall Accuracy",      _kv_class(acc))
    mini(c2, _pct(arate), "Automation Rate",        "kv-blue")
    mini(c3, _pct(far),   "False Auto-Resolve ↓",  _kv_class(far, good_high=False))
    mini(c4, f"{n_auto}/{total}", "Auto-Resolved",  "kv-green")
    mini(c5, f"{n_hum}/{total}",  "Human Review",   "kv-amber")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-category ──
    st.markdown('<div class="sec-head">Per-Category Breakdown</div>', unsafe_allow_html=True)
    try:
        per_cat = json.loads(latest["per_category_json"])
        if per_cat:
            name_map = {
                "split_settlement": "Split Settlement",
                "refund_misattribution": "Refund Misattribution",
                "fee_tier": "Fee Tier",
                "near_duplicate": "Near Duplicate",
                "unresolved": "Unresolved",
            }
            df = pd.DataFrame(per_cat)
            df["Category"] = df["category"].map(name_map).fillna(df["category"])
            df = df[df["support"] > 0].copy()
            df = df[["Category","precision","recall","f1","support","tp","fp","fn"]]
            df.columns = ["Category","Precision","Recall","F1","Support","TP","FP","FN"]
            for col in ["Precision","Recall","F1"]:
                df[col] = df[col].map(lambda v: f"{v:.2f}")
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.caption(f"Could not render per-category: {e}")

    # ── Confusion matrix ──
    st.markdown('<div class="sec-head">Confusion Matrix (True vs Predicted)</div>', unsafe_allow_html=True)
    try:
        conf_mx = json.loads(latest["confusion_matrix_json"])
        labels  = json.loads(latest["confusion_labels_json"])
        label_short = {
            "split_settlement": "Split",
            "refund_misattribution": "Refund",
            "fee_tier": "Fee",
            "near_duplicate": "Dup",
            "unresolved": "?",
        }
        rows_data = []
        for true in labels:
            row_d = {"True \\ Predicted": label_short.get(true, true)}
            for pred in labels:
                row_d[label_short.get(pred, pred)] = conf_mx.get(true, {}).get(pred, 0)
            rows_data.append(row_d)
        if rows_data:
            cm_df = pd.DataFrame(rows_data).set_index("True \\ Predicted")
            st.dataframe(cm_df, use_container_width=True)
    except Exception:
        st.caption("Confusion matrix not available.")

    # ── Run metadata ──
    st.markdown('<div class="sec-head">Run Metadata</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="detail-card" style="font-size:0.8rem;line-height:2;color:#64748b">
      <b style="color:#94a3b8">Run ID</b> &nbsp; <code style="color:#818cf8">{latest['run_id']}</code><br>
      <b style="color:#94a3b8">Timestamp</b> &nbsp; {latest['timestamp']}<br>
      <b style="color:#94a3b8">Model</b> &nbsp; {latest['model_name']}<br>
      <b style="color:#94a3b8">Prompt Version</b> &nbsp; {latest['prompt_version']}<br>
      <b style="color:#94a3b8">Threshold</b> &nbsp; {latest['threshold']}<br>
      <b style="color:#94a3b8">Seed</b> &nbsp; {latest['random_seed']}<br>
      <b style="color:#94a3b8">Notes</b> &nbsp; {latest['notes'] or '—'}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button(f"🔁  Re-run Evaluation ({split})"):
        with st.spinner("Evaluating…"):
            r = run_evaluation(config, split=split)
            if "error" in r: st.error(r["error"])
            else: st.success("Done!"); st.rerun()


# ─── VIEW 4: Audit Trail ──────────────────────────────────────────────────────

def view_audit(conn):
    st.markdown("# Audit Trail")
    st.markdown(
        '<p style="color:#64748b;margin-bottom:24px;font-size:0.88rem">'
        'Immutable log of all pipeline actions and human decisions. Download for compliance.'
        '</p>', unsafe_allow_html=True)

    # Human overrides
    overrides = conn.execute("""
        SELECT ho.override_id, ho.record_id, ho.action,
               ho.override_category, ho.notes, ho.timestamp,
               pd.category, COALESCE(cs.calibrated_confidence,0) as conf
        FROM human_overrides ho
        LEFT JOIN pipeline_decisions pd ON ho.decision_id = pd.decision_id
        LEFT JOIN confidence_signals cs ON ho.record_id = cs.record_id
        ORDER BY ho.timestamp DESC
    """).fetchall()

    if overrides:
        st.markdown('<div class="sec-head">Human Review Actions</div>', unsafe_allow_html=True)
        ovr_data = []
        for o in overrides:
            ovr_data.append({
                "Timestamp":       o["timestamp"],
                "Record ID":       o["record_id"],
                "AI Category":     (o["category"] or "").replace("_"," ").title(),
                "Confidence":      _pct(o["conf"]),
                "Human Action":    o["action"],
                "Override Cat.":   (o["override_category"] or "—").replace("_"," ").title(),
                "Notes":           o["notes"] or "—",
            })
        st.dataframe(pd.DataFrame(ovr_data), use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="alert-info">No human actions recorded yet. Use Exception Explorer to approve, reject, or override decisions.</div>', unsafe_allow_html=True)

    # Pipeline log
    st.markdown('<div class="sec-head">Pipeline Audit Log (latest 50)</div>', unsafe_allow_html=True)
    log_rows = conn.execute("""
        SELECT timestamp, record_id, stage, action, detail, level
        FROM audit_log ORDER BY timestamp DESC LIMIT 50
    """).fetchall()

    if log_rows:
        log_data = []
        for r in log_rows:
            try:
                det = json.dumps(json.loads(r["detail"]), separators=(",",":"))[:100]
            except Exception:
                det = str(r["detail"])[:100]
            log_data.append({
                "Timestamp": r["timestamp"],
                "Record":    r["record_id"] or "—",
                "Stage":     r["stage"],
                "Action":    r["action"],
                "Detail":    det,
                "Level":     r["level"],
            })
        st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)

        all_log = conn.execute("SELECT * FROM audit_log ORDER BY timestamp").fetchall()
        log_json = json.dumps([dict(r) for r in all_log], indent=2, default=str)
        st.download_button(
            "⬇  Download Full Audit Log (JSON)",
            log_json,
            "settlesense_audit_log.json",
            "application/json",
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # with st.spinner("Initialising SettleSense pipeline…"):
    #     _startup()
    conn = _conn()
    page = sidebar()

    if   page == "Controller Overview":  view_overview(conn)
    elif page == "Exception Explorer":   view_exceptions(conn)
    elif page == "Evaluation":           view_evaluation(conn)
    elif page == "Audit Trail":          view_audit(conn)


if __name__ == "__main__":
    main()
