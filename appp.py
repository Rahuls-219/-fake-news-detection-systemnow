"""
VeriLens AI — app.py
Premium Streamlit frontend for the hybrid AI fake-news detection system.
Run: streamlit run app.py
"""

import streamlit as st
import time
import os
from pathlib import Path

# ── Page config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="VeriLens AI · Truth at a Glance",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load backend (lazy so Streamlit can boot fast) ──────────────────────────
from utils import load_model, predict
from verifier import verify_with_search
from llm_engine import llm_analyze, build_hybrid_verdict

# ── Cache model ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    try:
        return load_model()
    except FileNotFoundError:
        return None, None

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root palette ── */
:root {
  --bg:        #0a0c10;
  --surface:   #13161c;
  --border:    #1e232d;
  --accent:    #4fffb0;
  --accent2:   #ff4f82;
  --text:      #e8eaf0;
  --muted:     #6b7280;
  --fake-clr:  #ff4f82;
  --real-clr:  #4fffb0;
  --warn-clr:  #ffd166;
  --radius:    14px;
}

/* ── Base reset ── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 900px 600px at 10% 0%, rgba(79,255,176,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 700px 500px at 90% 100%, rgba(255,79,130,0.06) 0%, transparent 60%),
    var(--bg) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Headings ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* ── Streamlit text area ── */
textarea {
  background: var(--surface) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 15px !important;
  caret-color: var(--accent) !important;
  transition: border-color .25s;
}
textarea:focus { border-color: var(--accent) !important; outline: none !important; box-shadow: 0 0 0 3px rgba(79,255,176,0.08) !important; }
textarea::placeholder { color: var(--muted) !important; }

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #4fffb0 0%, #00d68f 100%) !important;
  color: #0a0c10 !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 15px !important;
  border: none !important;
  border-radius: 50px !important;
  padding: 0.75rem 2.5rem !important;
  cursor: pointer !important;
  transition: opacity .2s, transform .15s !important;
  width: 100%;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1rem 1.25rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}

/* ── Spinner override ── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div { background: var(--accent) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# COMPONENT HELPERS
# ════════════════════════════════════════════════════════════════════════════

def hero_header():
    st.markdown("""
    <div style="text-align:center; padding: 3.5rem 0 1.5rem;">
      <div style="font-family:'Syne',sans-serif; font-size:11px; font-weight:700;
                  letter-spacing:4px; color:#4fffb0; text-transform:uppercase;
                  margin-bottom:1rem;">
        Hybrid AI · LLM + ML · Real-time Fact-Checking
      </div>
      <h1 style="font-family:'Syne',sans-serif; font-size:clamp(2.5rem,5vw,4rem);
                 font-weight:800; margin:0; letter-spacing:-1.5px; line-height:1.05;">
        Veri<span style="color:#4fffb0;">Lens</span>
        <span style="font-size:0.55em; font-weight:400; color:#6b7280;">AI</span>
      </h1>
      <p style="color:#6b7280; font-size:1.1rem; margin:0.75rem 0 0;
                font-family:'DM Sans',sans-serif; font-style:italic;">
        See through misinformation — instantly.
      </p>
    </div>
    """, unsafe_allow_html=True)


def verdict_card(label: str, confidence: float, fake_p: float, real_p: float):
    color = "#4fffb0" if label == "REAL" else "#ff4f82"
    icon  = "✅" if label == "REAL" else "⚠️"
    label_text = "REAL NEWS" if label == "REAL" else "FAKE NEWS"

    gauge_pct = int(confidence)
    gauge_color = color

    st.markdown(f"""
    <div style="
      background: linear-gradient(135deg, #13161c 0%, #0f1116 100%);
      border: 1.5px solid {color}33;
      border-radius: 20px;
      padding: 2.5rem 2rem;
      text-align: center;
      box-shadow: 0 0 60px {color}18;
      margin-bottom: 1.5rem;
    ">
      <div style="font-size: 3rem; margin-bottom: 0.5rem;">{icon}</div>
      <div style="
        font-family: 'Syne', sans-serif;
        font-size: 2rem; font-weight: 800;
        color: {color}; letter-spacing: 2px;
        margin-bottom: 0.25rem;
      ">{label_text}</div>
      <div style="color: #6b7280; font-size: 0.9rem; margin-bottom: 2rem;">
        Hybrid AI Verdict
      </div>

      <!-- Confidence Gauge Ring -->
      <div style="position:relative; display:inline-block; margin-bottom: 1.5rem;">
        <svg width="140" height="140" viewBox="0 0 140 140">
          <circle cx="70" cy="70" r="58" fill="none" stroke="#1e232d" stroke-width="10"/>
          <circle cx="70" cy="70" r="58" fill="none" stroke="{gauge_color}" stroke-width="10"
            stroke-dasharray="{int(gauge_pct * 3.645)} 364.5"
            stroke-dashoffset="91.125"
            stroke-linecap="round"
            style="transition: stroke-dasharray 1s ease;"/>
          <text x="70" y="65" text-anchor="middle"
            font-family="Syne, sans-serif" font-weight="800"
            font-size="26" fill="{gauge_color}">{gauge_pct}%</text>
          <text x="70" y="84" text-anchor="middle"
            font-family="DM Sans, sans-serif" font-size="11" fill="#6b7280">Confidence</text>
        </svg>
      </div>

      <!-- Probability bars -->
      <div style="display:flex; gap:1.5rem; justify-content:center; text-align:left;">
        <div style="min-width:120px;">
          <div style="font-size:11px; color:#6b7280; margin-bottom:4px; letter-spacing:1px;">REAL PROBABILITY</div>
          <div style="background:#1e232d; border-radius:4px; height:6px; overflow:hidden;">
            <div style="width:{real_p}%; height:100%; background:#4fffb0; border-radius:4px;"></div>
          </div>
          <div style="font-size:14px; font-weight:600; color:#4fffb0; margin-top:4px;">{real_p:.1f}%</div>
        </div>
        <div style="min-width:120px;">
          <div style="font-size:11px; color:#6b7280; margin-bottom:4px; letter-spacing:1px;">FAKE PROBABILITY</div>
          <div style="background:#1e232d; border-radius:4px; height:6px; overflow:hidden;">
            <div style="width:{fake_p}%; height:100%; background:#ff4f82; border-radius:4px;"></div>
          </div>
          <div style="font-size:14px; font-weight:600; color:#ff4f82; margin-top:4px;">{fake_p:.1f}%</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def score_breakdown_card(ml_score: float, llm_score: float, evidence_score: float, final_score: float):
    def bar(val, color):
        pct = int(val * 100)
        return f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">\
<div style="background:#1e232d;border-radius:4px;height:8px;flex:1;overflow:hidden;">\
<div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div>\
</div>\
<div style="font-size:13px;font-weight:600;color:{color};min-width:40px;text-align:right;">{pct}%</div>\
</div>'

    st.markdown(f"""
    <div style="background:#13161c; border:1px solid #1e232d; border-radius:14px; padding:1.5rem; margin-bottom:1rem;">
      <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:14px;
                  letter-spacing:2px; text-transform:uppercase; color:#6b7280; margin-bottom:1.25rem;">
        Hybrid Score Breakdown
      </div>

      <div style="margin-bottom:6px; font-size:12px; color:#6b7280;">ML Model (40%)</div>
      {bar(ml_score, '#818cf8')}

      <div style="margin-bottom:6px; font-size:12px; color:#6b7280;">LLM Reasoning (40%)</div>
      {bar(llm_score, '#fb923c')}

      <div style="margin-bottom:6px; font-size:12px; color:#6b7280;">Evidence Check (20%)</div>
      {bar(evidence_score, '#34d399')}

      <hr style="border-color:#1e232d; margin: 1rem 0;">

      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:14px; color:#e8eaf0;">
          FINAL SCORE
        </div>
        <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:1.4rem; color:#4fffb0;">
          {int(final_score * 100)}%
        </div>
      </div>
      <div style="font-size:11px; color:#6b7280; text-align:right; margin-top:2px;">
        Probability of being REAL
      </div>
    </div>
    """, unsafe_allow_html=True)


def keyword_pills(keywords: list, label: str):
    color = "#ff4f82" if label == "FAKE" else "#4fffb0"
    pills = "".join([
        f'<span style="background:{color}18; color:{color}; border:1px solid {color}33; '
        f'border-radius:50px; padding:4px 14px; font-size:13px; margin:4px; display:inline-block;">'
        f'#{k}</span>'
        for k in keywords
    ])
    st.markdown(f"""
    <div style="background:#13161c; border:1px solid #1e232d; border-radius:14px; padding:1.25rem; margin-bottom:1rem;">
      <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:12px;
                  letter-spacing:2px; text-transform:uppercase; color:#6b7280; margin-bottom:0.75rem;">
        Signal Keywords
      </div>
      <div>{pills}</div>
    </div>
    """, unsafe_allow_html=True)


def example_cards():
    examples = [
        ("🔬 Scientific", "NASA's James Webb Telescope captures images of galaxies 13 billion light years away, revealing early universe formation patterns."),
        ("⚠️ Sensational", "BREAKING: Scientists discover miracle cure hidden by Big Pharma for 50 years — share before deleted!"),
        ("📰 Political", "Senate passes bipartisan infrastructure bill after months of negotiations, allocating $1.2 trillion for roads and broadband."),
    ]
    cols = st.columns(3)
    for col, (tag, text) in zip(cols, examples):
        with col:
            col.markdown(f"""
            <div style="background:#13161c; border:1px solid #1e232d; border-radius:12px;
                        padding:1rem; cursor:pointer; transition:border-color .2s;"
                 onmouseover="this.style.borderColor='#4fffb0'"
                 onmouseout="this.style.borderColor='#1e232d'">
              <div style="font-size:11px; color:#6b7280; letter-spacing:1px; margin-bottom:6px;">{tag}</div>
              <div style="font-size:13px; color:#e8eaf0; line-height:1.55;">{text}</div>
            </div>
            """, unsafe_allow_html=True)
            if col.button("Use this →", key=f"ex_{tag}"):
                st.session_state["input_text"] = text
                st.rerun()


def loading_animation(step: str):
    steps = [
        "🔍 Extracting claims...",
        "🤖 Running ML classifier...",
        "🧠 LLM reasoning layer...",
        "🌐 Fact-checking sources...",
        "⚖️ Building hybrid verdict...",
    ]
    for i, s in enumerate(steps):
        active = s == step
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; padding: 6px 0;
                    opacity: {'1' if active else '0.35'}; transition:opacity .3s;">
          <div style="width:8px; height:8px; border-radius:50%;
                      background: {'#4fffb0' if active else '#1e232d'};
                      {'box-shadow: 0 0 8px #4fffb0;' if active else ''}"></div>
          <span style="font-size:13px; color: {'#4fffb0' if active else '#6b7280'};">{s}</span>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════════
def main():
    hero_header()

    model, vectorizer = get_model()

    # ── API key warning ──────────────────────────────────────────────────────
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        st.markdown("""
        <div style="background:#ffd16618; border:1px solid #ffd16644; border-radius:10px;
                    padding:0.75rem 1.25rem; color:#ffd166; font-size:13px; margin-bottom:1.5rem; text-align:center;">
          ⚡ <strong>OPENAI_API_KEY not set.</strong> LLM reasoning will run in demo mode.
          Set the env var for full hybrid AI power.
        </div>
        """, unsafe_allow_html=True)

    # ── Input panel ──────────────────────────────────────────────────────────
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown("""
        <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:13px;
                    letter-spacing:2px; text-transform:uppercase; color:#6b7280; margin-bottom:0.75rem;">
          Paste Article or Headline
        </div>
        """, unsafe_allow_html=True)

        default_text = st.session_state.get("input_text", "")
        input_text = st.text_area(
            label="news_input",
            value=default_text,
            placeholder="""Paste a news headline, article, or claim here...
            
        """,
            height=220,
            label_visibility="collapsed",
        )

        col_btn, col_clear = st.columns([3, 1])
        with col_btn:
            analyze_clicked = st.button("🔬 Analyze with VeriLens AI", use_container_width=True)
        with col_clear:
            if st.button("Clear", use_container_width=True):
                st.session_state.pop("input_text", None)
                st.session_state.pop("result", None)
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Syne',sans-serif; font-size:11px; font-weight:700;
                    letter-spacing:2px; text-transform:uppercase; color:#6b7280; margin-bottom:0.75rem;">
          Try an Example
        </div>
        """, unsafe_allow_html=True)
        example_cards()

    # ── Analysis ─────────────────────────────────────────────────────────────
    with right:
        if analyze_clicked and input_text.strip():
            result_placeholder = st.empty()

            with result_placeholder.container():
                st.markdown("""
                <div style="background:#13161c; border:1px solid #1e232d; border-radius:14px; padding:1.5rem;">
                  <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:13px;
                              letter-spacing:2px; text-transform:uppercase; color:#4fffb0; margin-bottom:1rem;">
                    Analyzing...
                  </div>
                """, unsafe_allow_html=True)

                pipeline_steps = [
                    "🔍 Extracting claims...",
                    "🤖 Running ML classifier...",
                    "🧠 LLM reasoning layer...",
                    "🌐 Fact-checking sources...",
                    "⚖️ Building hybrid verdict...",
                ]

                step_slots = [st.empty() for _ in pipeline_steps]
                for i, step in enumerate(pipeline_steps):
                    for j, slot in enumerate(step_slots):
                        active = j == i
                        slot.markdown(f"""
                        <div style="display:flex; align-items:center; gap:10px; padding:5px 0;
                                    opacity: {'1' if active else ('0.6' if j < i else '0.25')};">
                          <div style="width:8px; height:8px; border-radius:50%;
                                      background: {'#4fffb0' if j <= i else '#1e232d'};
                                      {'box-shadow:0 0 8px #4fffb0;' if active else ''}"></div>
                          <span style="font-size:13px; color: {'#4fffb0' if active else ('#e8eaf0' if j < i else '#6b7280')};">
                            {step}
                          </span>
                        </div>
                        """, unsafe_allow_html=True)
                    time.sleep(0.4)

                st.markdown("</div>", unsafe_allow_html=True)

            # ── Run pipeline ────────────────────────────────────────────────
            ml_result   = predict(input_text, model, vectorizer) if model else None
            llm_result  = llm_analyze(input_text)
            ev_result   = verify_with_search(input_text, ml_result)
            final       = build_hybrid_verdict(ml_result, llm_result, ev_result)

            st.session_state["result"] = {
                "input":    input_text,
                "ml":       ml_result,
                "llm":      llm_result,
                "evidence": ev_result,
                "final":    final,
            }
            result_placeholder.empty()

        # ── Display stored result ────────────────────────────────────────────
        if "result" in st.session_state:
            r = st.session_state["result"]
            f = r["final"]

            verdict_card(
                label=f["label"],
                confidence=f["confidence"],
                fake_p=f["fake_prob"],
                real_p=f["real_prob"],
            )

            score_breakdown_card(
                ml_score=f["ml_score"],
                llm_score=f["llm_score"],
                evidence_score=f["evidence_score"],
                final_score=f["final_score"],
            )

            if r["ml"] and r["ml"].get("top_keywords"):
                keyword_pills(r["ml"]["top_keywords"], f["label"])

        elif not analyze_clicked:
            # ── Idle state ───────────────────────────────────────────────────
            st.markdown("""
            <div style="background:#13161c; border:1px solid #1e232d; border-radius:14px;
                        padding:3rem 2rem; text-align:center; color:#6b7280;">
              <div style="font-size:3rem; margin-bottom:1rem; opacity:0.4;">🔬</div>
              <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:600; color:#2a2f3a; margin-bottom:0.5rem;">
                Awaiting Analysis
              </div>
              <div style="font-size:13px; max-width:280px; margin:0 auto; line-height:1.6;">
                Paste a headline or article on the left, then click Analyze.
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Full-width detail panels (only when result exists) ───────────────────
    if "result" in st.session_state:
        r = st.session_state["result"]
        f = r["final"]
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🧠 LLM Reasoning", "🌐 Evidence Panel", "⚙️ Technical Details"])

        with tab1:
            llm = r.get("llm", {})
            st.markdown(f"""
            <div style="background:#13161c; border-radius:14px; padding:1.5rem; border:1px solid #1e232d;">
              <div style="font-family:'Syne',sans-serif; font-size:13px; font-weight:700;
                          letter-spacing:2px; text-transform:uppercase; color:#fb923c; margin-bottom:1rem;">
                GPT Reasoning Summary
              </div>
              <div style="line-height:1.8; color:#c9cdd6; font-size:14px;">
                {llm.get("reasoning", "LLM reasoning not available — set OPENAI_API_KEY.")}
              </div>
            </div>
            """, unsafe_allow_html=True)

            if llm.get("red_flags"):
                st.markdown("<br>", unsafe_allow_html=True)
                flags = llm["red_flags"]
                flag_html = "".join([
                    f'<div style="display:flex;gap:10px;align-items:flex-start;margin-bottom:8px;">'
                    f'<span style="color:#ff4f82;margin-top:2px;">⚑</span>'
                    f'<span style="font-size:13px;color:#c9cdd6;">{flag}</span></div>'
                    for flag in flags
                ])
                st.markdown(f"""
                <div style="background:#ff4f8210; border:1px solid #ff4f8230; border-radius:12px; padding:1.25rem;">
                  <div style="font-family:'Syne',sans-serif; font-size:12px; font-weight:700;
                              letter-spacing:2px; text-transform:uppercase; color:#ff4f82; margin-bottom:0.75rem;">
                    Red Flags Detected
                  </div>
                  {flag_html}
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            ev = r.get("evidence", {})
            ev_label = ev.get("label", "UNCERTAIN")
            ev_color = {"REAL": "#4fffb0", "FAKE": "#ff4f82", "UNCERTAIN": "#ffd166"}.get(ev_label, "#6b7280")

            st.markdown(f"""
            <div style="background:#13161c; border-radius:14px; padding:1.5rem; border:1px solid #1e232d; margin-bottom:1rem;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                <div style="font-family:'Syne',sans-serif; font-size:13px; font-weight:700;
                            letter-spacing:2px; text-transform:uppercase; color:#6b7280;">
                  Evidence Verdict
                </div>
                <div style="background:{ev_color}22; color:{ev_color}; border:1px solid {ev_color}44;
                            border-radius:50px; padding:4px 14px; font-size:12px; font-weight:700;">
                  {ev_label}
                </div>
              </div>
              <div style="font-size:14px; color:#c9cdd6; line-height:1.7; margin-bottom:1rem;">
                {ev.get("reason", "No reason available.")}
              </div>
              <div style="background:#0a0c10; border-radius:8px; padding:1rem; font-size:12px;
                          color:#6b7280; line-height:1.7; font-family:monospace;">
                <strong style="color:#4b5563;">Sources consulted:</strong><br>
                {ev.get("source", "No source data.")[:600]}
              </div>
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            ml = r.get("ml") or {}
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("ML Confidence", f"{ml.get('confidence', 0):.1f}%")
            col2.metric("ML Label", ml.get("label", "N/A"))
            col3.metric("LLM Score", f"{int(f['llm_score']*100)}%")
            col4.metric("Final Score", f"{int(f['final_score']*100)}%")

            with st.expander("Raw ML Output"):
                st.json(ml)
            with st.expander("Raw LLM Output"):
                st.json(r.get("llm", {}))
            with st.expander("Raw Evidence Output"):
                st.json(r.get("evidence", {}))

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:3rem 0 1.5rem; color:#2a2f3a; font-size:12px;">
      <span style="font-family:'Syne',sans-serif; font-weight:700; color:#1e232d;">VeriLens AI</span>
      &nbsp;·&nbsp; Hybrid ML + LLM Fact-Checking
      &nbsp;·&nbsp; Built for accuracy, built for truth.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
