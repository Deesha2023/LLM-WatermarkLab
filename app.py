"""
app.py
------
The LLM-WatermarkLab dashboard. Run with:

    streamlit run app.py

Four tabs:
  1. Generate & Detect      - pick any watermark method, generate, see the detector's verdict.
  2. Attack Lab             - attack any text, compare detection before / after.
  3. Benchmark Results      - every method x every attack: tables, charts, trade-offs.
  4. Special Features Lab   - Semantic-Anchor recovery, Adaptive Strength, Error-Correcting payload.

Design note: this file is a visual layer on top of the watermarklab package. All generation,
detection, attack and benchmark logic is unchanged; only presentation (theme, layout, cards,
chart styling) lives here.
"""

import html
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from watermarklab import config
from watermarklab.methods import METHODS, PLAIN_LABEL, get_method

st.set_page_config(page_title="LLM-WatermarkLab", page_icon="💧", layout="wide",
                   initial_sidebar_state="collapsed")


def _fit_kwargs():
    """Full-width kwargs that work on old (use_container_width) and new (width='stretch') Streamlit."""
    import inspect
    try:
        ann = str(inspect.signature(st.plotly_chart).parameters["width"].annotation)
        if "stretch" in ann:
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


FIT = _fit_kwargs()

# ===========================================================================
# DESIGN SYSTEM
# ===========================================================================
C = {
    "bg": "#0a0f1e", "panel": "#111a33", "panel2": "#162247", "line": "#26335f",
    "text": "#e9eefc", "muted": "#97a6cf", "cyan": "#22d3ee", "violet": "#8b5cf6",
    "pink": "#f472b6", "green": "#34d399", "red": "#fb7185", "amber": "#fbbf24", "blue": "#60a5fa",
}
METHOD_COLORS = dict(zip([m.label for m in METHODS.values()],
                         [C["blue"], C["violet"], C["cyan"], C["amber"], C["pink"], C["green"]]))
COLORWAY = list(METHOD_COLORS.values())

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
:root{--bg:#0a0f1e;--panel:#111a33;--panel2:#162247;--line:#26335f;--text:#e9eefc;--muted:#97a6cf;
--cyan:#22d3ee;--violet:#8b5cf6;--pink:#f472b6;--green:#34d399;--red:#fb7185;--amber:#fbbf24;}
html,body,[class*="css"],.stApp{font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,sans-serif;}
.stApp{background:radial-gradient(1200px 600px at 8% -10%,rgba(139,92,246,.22),transparent 60%),
radial-gradient(900px 500px at 100% 0%,rgba(34,211,238,.16),transparent 55%),var(--bg);color:var(--text);}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1280px;}
h1,h2,h3,h4{color:var(--text);letter-spacing:-.01em;}
p,li,label,.stMarkdown{color:var(--text);}
[data-testid="stCaptionContainer"],.stCaption{color:var(--muted)!important;}
code{font-family:'JetBrains Mono',monospace;background:rgba(34,211,238,.1);color:var(--cyan);padding:1px 6px;border-radius:6px;}

/* hero */
.wl-hero{position:relative;overflow:hidden;border-radius:22px;padding:30px 34px;margin-bottom:18px;
background:linear-gradient(120deg,rgba(139,92,246,.35),rgba(34,211,238,.20) 55%,rgba(244,114,182,.22));
border:1px solid rgba(255,255,255,.14);box-shadow:0 20px 60px -20px rgba(139,92,246,.55);}
.wl-hero:after{content:"";position:absolute;right:-60px;top:-60px;width:260px;height:260px;border-radius:50%;
background:radial-gradient(circle,rgba(34,211,238,.35),transparent 65%);filter:blur(4px);}
.wl-badge{display:inline-block;font-size:.72rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
color:#0a0f1e;background:linear-gradient(90deg,var(--cyan),var(--green));padding:4px 12px;border-radius:99px;}
.wl-title{font-size:2.55rem;font-weight:800;line-height:1.1;margin:12px 0 6px;
background:linear-gradient(90deg,#fff,#c7d2fe 45%,var(--cyan));-webkit-background-clip:text;background-clip:text;
-webkit-text-fill-color:transparent;}
.wl-sub{color:#cdd6f4;font-size:1.02rem;max-width:760px;}
.wl-chips{margin-top:16px;display:flex;gap:10px;flex-wrap:wrap;position:relative;z-index:1;}
.wl-chip{padding:6px 14px;border-radius:99px;font-size:.82rem;font-weight:600;color:var(--text);
background:rgba(10,15,30,.55);border:1px solid rgba(255,255,255,.16);backdrop-filter:blur(6px);}
.wl-chip b{color:var(--cyan);}

/* tabs (selectors cover both older baseweb tabs and newer role=tab markup) */
.stTabs [role="tablist"],.stTabs [data-baseweb="tab-list"]{gap:6px;background:rgba(17,26,51,.7);padding:6px;border-radius:14px;
border:1px solid var(--line);}
.stTabs [role="tab"],.stTabs [data-baseweb="tab"]{height:44px;border-radius:10px;padding:0 22px!important;width:auto!important;
flex:0 0 auto;color:var(--muted);font-weight:700;background:transparent;white-space:nowrap;display:flex;align-items:center;}
.stTabs [role="tab"] p,.stTabs [data-baseweb="tab"] p{white-space:nowrap;margin:0;font-weight:700;}
.stTabs [role="tab"]:hover,.stTabs [data-baseweb="tab"]:hover{color:var(--text);background:rgba(255,255,255,.05);}
.stTabs [role="tab"][aria-selected="true"],.stTabs [data-baseweb="tab"][aria-selected="true"]{
background:linear-gradient(90deg,var(--cyan),#7dd3fc)!important;color:#06101f!important;}
.stTabs [role="tab"][aria-selected="true"] p,.stTabs [data-baseweb="tab"][aria-selected="true"] p{color:#06101f!important;}
.stTabs [role="tab"][aria-selected="true"]:hover{background:linear-gradient(90deg,var(--cyan),#7dd3fc)!important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"],.stTabs .react-aria-SelectionIndicator{display:none!important;}
.stTabs [role="tablist"]{border-bottom:1px solid var(--line);}

/* section headings */
.wl-h{display:flex;align-items:center;gap:12px;margin:22px 0 4px;}
.wl-h .dot{width:10px;height:28px;border-radius:6px;background:linear-gradient(var(--cyan),var(--violet));}
.wl-h h3{margin:0;font-size:1.35rem;font-weight:800;}
.wl-lead{color:var(--muted);margin:0 0 14px 22px;font-size:.95rem;}

/* panels & cards */
.wl-panel{background:linear-gradient(180deg,rgba(22,34,71,.85),rgba(17,26,51,.85));border:1px solid var(--line);
border-radius:16px;padding:16px 18px;margin:8px 0;}
.wl-info{background:rgba(34,211,238,.08);border:1px solid rgba(34,211,238,.35);border-left:4px solid var(--cyan);
border-radius:12px;padding:12px 16px;color:#d6f6ff;font-size:.93rem;margin:6px 0 12px;}
.wl-idea{background:linear-gradient(120deg,rgba(139,92,246,.18),rgba(34,211,238,.08));border:1px solid rgba(139,92,246,.4);
border-radius:14px;padding:14px 18px;margin:8px 0 14px;font-size:.95rem;line-height:1.6;}
.wl-idea b{color:var(--cyan);}
.wl-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:10px 0;}
.wl-card{position:relative;background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);
border-radius:14px;padding:14px 16px;overflow:hidden;}
.wl-card:before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--accent,#22d3ee);}
.wl-card-label{font-size:.72rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);}
.wl-card-value{font-size:1.5rem;font-weight:800;margin-top:4px;color:var(--text);font-family:'JetBrains Mono',monospace;overflow-wrap:anywhere;line-height:1.2;}
.wl-card-sub{font-size:.78rem;color:var(--muted);margin-top:2px;}
.wl-verdict{display:flex;align-items:center;gap:14px;border-radius:16px;padding:16px 20px;margin:6px 0 10px;border:1px solid;}
.wl-verdict.yes{background:linear-gradient(120deg,rgba(52,211,153,.22),rgba(52,211,153,.06));border-color:rgba(52,211,153,.55);}
.wl-verdict.no{background:linear-gradient(120deg,rgba(251,113,133,.22),rgba(251,113,133,.06));border-color:rgba(251,113,133,.55);}
.wl-verdict .ico{font-size:1.9rem;line-height:1;}
.wl-verdict .t{font-weight:800;font-size:1.2rem;letter-spacing:.01em;}
.wl-verdict .s{color:var(--muted);font-size:.85rem;}
.wl-pill{display:inline-block;padding:3px 11px;border-radius:99px;font-size:.78rem;font-weight:700;margin:2px 6px 2px 0;
border:1px solid var(--line);background:rgba(255,255,255,.05);color:var(--text);}
.wl-pill.ok{background:rgba(52,211,153,.16);border-color:rgba(52,211,153,.5);color:#a7f3d0;}
.wl-pill.bad{background:rgba(251,113,133,.16);border-color:rgba(251,113,133,.5);color:#fecdd3;}
.wl-pill.info{background:rgba(34,211,238,.14);border-color:rgba(34,211,238,.45);color:#a5f3fc;}
.wl-bits{display:flex;gap:4px;flex-wrap:wrap;margin:4px 0 8px;}
.wl-bit{width:30px;height:34px;border-radius:8px;display:flex;align-items:center;justify-content:center;
font-family:'JetBrains Mono',monospace;font-weight:700;border:1px solid var(--line);background:rgba(255,255,255,.05);}
.wl-bit.ok{background:rgba(52,211,153,.2);border-color:rgba(52,211,153,.6);color:#a7f3d0;}
.wl-bit.bad{background:rgba(251,113,133,.2);border-color:rgba(251,113,133,.6);color:#fecdd3;}
.wl-label{font-size:.78rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-top:6px;}
.wl-tokens{line-height:2.15;font-size:.95rem;background:rgba(10,15,30,.55);border:1px solid var(--line);
border-radius:14px;padding:12px 16px;}
.wl-legend span{margin-right:14px;font-size:.8rem;color:var(--muted);}
.wl-ba{display:inline-block;padding:4px 14px;border-radius:99px;font-weight:800;font-size:.8rem;letter-spacing:.1em;text-transform:uppercase;}
.wl-ba.before{background:rgba(96,165,250,.2);color:#bfdbfe;border:1px solid rgba(96,165,250,.5);}
.wl-ba.after{background:rgba(244,114,182,.2);color:#fbcfe8;border:1px solid rgba(244,114,182,.5);}
.wl-footer{text-align:center;color:var(--muted);font-size:.8rem;margin-top:34px;padding-top:14px;border-top:1px solid var(--line);}

/* streamlit widgets */
.stTextArea textarea,.stTextInput input,.stNumberInput input{background:#0c1530!important;color:var(--text)!important;
border:1px solid var(--line)!important;border-radius:12px!important;}
.stTextArea textarea:focus,.stTextInput input:focus,.stNumberInput input:focus{border-color:var(--cyan)!important;
box-shadow:0 0 0 2px rgba(34,211,238,.25)!important;}
div[data-baseweb="select"]>div{background:#0c1530!important;border:1px solid var(--line)!important;border-radius:12px!important;}
.stButton>button,.stDownloadButton>button{border-radius:12px;font-weight:700;border:1px solid var(--line);
background:var(--panel2);color:var(--text);transition:all .15s ease;}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-1px);border-color:var(--cyan);color:#fff;}
.stButton>button[kind="primary"]{background:linear-gradient(90deg,var(--violet),var(--cyan));color:#06101f;border:none;
box-shadow:0 10px 28px -10px rgba(34,211,238,.7);}
.stButton>button[kind="primary"]:hover{filter:brightness(1.1);color:#06101f;}
[data-testid="stMetric"]{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);
border-radius:14px;padding:12px 16px;}
[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace;font-weight:800;}
[data-testid="stExpander"]{background:rgba(17,26,51,.6);border:1px solid var(--line);border-radius:14px;}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden;}
.stAlert{border-radius:14px;}
hr{border-color:var(--line);}
</style>
"""


def _compact(s: str) -> str:
    """Remove blank lines/indent so Markdown never mistakes our HTML for a code block."""
    return "\n".join(line.strip() for line in s.splitlines() if line.strip())


def H(markup: str):
    st.markdown(_compact(markup), unsafe_allow_html=True)


def section(title: str, lead: str = ""):
    H(f"<div class='wl-h'><span class='dot'></span><h3>{html.escape(title)}</h3></div>"
      + (f"<p class='wl-lead'>{lead}</p>" if lead else ""))


def card(label, value, sub="", accent=C["cyan"]):
    return (f"<div class='wl-card' style='--accent:{accent}'><div class='wl-card-label'>{label}</div>"
            f"<div class='wl-card-value'>{value}</div><div class='wl-card-sub'>{sub}</div></div>")


def cards(*items):
    H("<div class='wl-grid'>" + "".join(items) + "</div>")


def style_fig(fig, height=420, legend_bottom=True):
    fig.update_layout(
        template="plotly_dark", height=height, colorway=COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,26,51,.55)",
        font=dict(family="Inter, system-ui, sans-serif", color=C["text"], size=13),
        title=dict(font=dict(size=16, color=C["text"]), x=0.01),
        margin=dict(l=24, r=18, t=56, b=24),
        legend=dict(orientation="h", y=-0.22, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=12)) if legend_bottom else {},
        hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["line"], font=dict(color=C["text"])),
    )
    fig.update_xaxes(gridcolor="rgba(151,166,207,.12)", zerolinecolor="rgba(151,166,207,.25)")
    fig.update_yaxes(gridcolor="rgba(151,166,207,.12)", zerolinecolor="rgba(151,166,207,.25)")
    return fig


_chart_counter = [0]


def show_chart(fig):
    """Plotly chart with a guaranteed-unique key (the same figure can legitimately appear twice)."""
    _chart_counter[0] += 1
    st.plotly_chart(fig, key=f"wl_chart_{_chart_counter[0]}", **FIT)


def gauge(z: float, title="z-score"):
    thr = config.Z_SCORE_THRESHOLD
    lo, hi = min(-2.0, z - 1.0), max(thr + 6.0, z + 1.0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(z), number=dict(font=dict(size=40, color=C["text"]), valueformat=".2f"),
        title=dict(text=title, font=dict(size=13, color=C["muted"])),
        gauge=dict(
            axis=dict(range=[lo, hi], tickcolor=C["muted"], tickfont=dict(color=C["muted"], size=10)),
            bar=dict(color=C["green"] if z >= thr else C["red"], thickness=0.32),
            bgcolor="rgba(255,255,255,.04)", borderwidth=0,
            steps=[dict(range=[lo, thr], color="rgba(251,113,133,.16)"),
                   dict(range=[thr, hi], color="rgba(52,211,153,.16)")],
            threshold=dict(line=dict(color=C["amber"], width=4), thickness=0.85, value=thr)),
    ))
    fig.update_layout(height=215, margin=dict(l=22, r=22, t=44, b=6), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, system-ui, sans-serif", color=C["text"]))
    return fig


def bit_chips(bits: str, ref: str = None):
    out = []
    for i, b in enumerate(bits):
        cls = "" if ref is None else ("ok" if ref[i] == b else "bad")
        out.append(f"<div class='wl-bit {cls}'>{b}</div>")
    return "<div class='wl-bits'>" + "".join(out) + "</div>"


@st.cache_resource(show_spinner="Loading language model (first run downloads ~330 MB)...")
def load_backend():
    from watermarklab.backend import get_backend
    return get_backend()


def show_detection(res: dict, key_prefix: str = ""):
    thr = config.Z_SCORE_THRESHOLD
    yes = bool(res["is_watermarked"])
    left, right = st.columns([1, 1.45])
    with left:
        show_chart(gauge(res["z_score"]))
    with right:
        H(f"""<div class='wl-verdict {'yes' if yes else 'no'}'>
        <div class='ico'>{'✅' if yes else '🚫'}</div>
        <div><div class='t'>{'WATERMARK DETECTED' if yes else 'NO WATERMARK FOUND'}</div>
        <div class='s'>z = {res['z_score']:.2f} {'≥' if yes else '<'} threshold {thr:.1f}</div></div></div>""")
        cards(card("Detected?", "YES" if yes else "NO", "z ≥ threshold" if yes else "below threshold",
                   C["green"] if yes else C["red"]),
              card("Confidence", f"{res['confidence']*100:.1f}%", "50% = decision boundary", C["cyan"]),
              card("z-score", f"{res['z_score']:.2f}", f"threshold {thr:.1f}", C["violet"]),
              card("p-value", f"{res.get('p_value', float('nan')):.1e}", "one-sided tail", C["amber"]))
    if "payload" in res:
        crc = bool(res["crc_ok"])
        dec_bits = format(res["payload"], "08b")
        H(f"""<div class='wl-label'>Decoded hidden payload</div>{bit_chips(dec_bits)}
        <span class='wl-pill info'>value = {res['payload']}</span>
        <span class='wl-pill {'ok' if crc else 'bad'}'>hard-decision CRC {'passed' if crc else 'failed'}</span>
        <span class='wl-pill info'>bit errors corrected: {res['corrected_bits']}</span>""")
    if "z_anchor_layer" in res:
        z_ctx = res.get("z_context_layer", res.get("z_payload_layer", 0.0))
        used = bool(res.get("recovery_used"))
        H(f"""<div class='wl-label'>Layer evidence</div>
        <span class='wl-pill info'>context / payload layer z = {z_ctx:.2f}</span>
        <span class='wl-pill info'>anchor-word layer z = {res['z_anchor_layer']:.2f}
        ({res.get('anchors_scored', 0)} anchors scored)</span>
        <span class='wl-pill {'ok' if used else ''}'>recovery from anchors: {'USED' if used else 'not needed'}</span>""")
    with st.expander("Raw detector output"):
        st.json({k: v for k, v in res.items() if k != "slot_votes"})


def render_tokens(trace, mode: str):
    """Colour the generated tokens. mode: 'target' (in-target = green) or 'anchor' (anchors underlined)."""
    spans = []
    for t in trace:
        tok = html.escape(t["token"]) or "&nbsp;"
        style = ""
        if mode == "target":
            style = ("background:rgba(52,211,153,.24);border-bottom:2px solid #34d399;" if t["in_target"]
                     else "background:rgba(251,113,133,.24);border-bottom:2px solid #fb7185;")
        if t.get("is_anchor"):
            style += "font-weight:800;color:#67e8f9;text-decoration:underline;text-decoration-color:#22d3ee;"
        spans.append(f"<span style='{style}padding:2px 1px;border-radius:4px'>{tok}</span>")
    legend = ("<div class='wl-legend'><span><b style='color:#34d399'>■</b> in green list</span>"
              "<span><b style='color:#fb7185'>■</b> not in green list</span>"
              "<span><b style='color:#67e8f9'>underlined</b> = anchor word</span></div>" if mode == "target" else
              "<div class='wl-legend'><span><b style='color:#67e8f9'>underlined bold</b> = anchor word "
              "(meaning-bearing seed)</span></div>")
    H(legend + "<div class='wl-tokens'>" + "".join(spans) + "</div>")


# ===========================================================================
# PAGE HEADER
# ===========================================================================
st.markdown(_compact(CSS), unsafe_allow_html=True)
H(f"""<div class='wl-hero'>
<span class='wl-badge'>Generative AI Security</span>
<div class='wl-title'>💧 LLM-WatermarkLab</div>
<div class='wl-sub'>A Robustness-Aware Framework for Watermarking and Detection of AI-Generated Text.
Plant a hidden statistical signal, attack it, and measure what survives.</div>
<div class='wl-chips'>
<span class='wl-chip'><b>{len(METHODS)}</b> watermark methods</span>
<span class='wl-chip'><b>7</b> attacks</span>
<span class='wl-chip'><b>3</b> special features</span>
<span class='wl-chip'>Statistical + <b>ML</b> detection</span>
</div></div>""")

backend_slot = st.empty()

tab1, tab2, tab3, tab4 = st.tabs(["⚡ Generate & Detect", "🛡️ Attack Lab", "📊 Benchmark Results",
                                  "✨ Special Features Lab"])

# ---------------------------------------------------------------------------
# TAB 1: Generate & Detect
# ---------------------------------------------------------------------------
with tab1:
    section("Generate watermarked text and detect it live",
            "Choose a method, write a prompt, and watch the detector decide.")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        prompt = st.text_area("Prompt", value="Explain what artificial intelligence is to a beginner.", height=100)
    with col_b:
        method_label = st.radio("Watermark method", [m.label for m in METHODS.values()] + [PLAIN_LABEL])
        max_tokens = st.slider("Max new tokens", 40, 300, 120, step=10)
        seed = st.number_input("Random seed", value=config.SEED, step=1)
    spec = METHODS.get(next((k for k, m in METHODS.items() if m.label == method_label), ""), None)
    if spec is not None:
        H(f"<div class='wl-info'>💡 <b>{html.escape(spec.label)}</b> — {html.escape(spec.description)}</div>")
    payload = config.ECC_DEFAULT_PAYLOAD
    if spec is not None and spec.ecc:
        payload = int(st.number_input(f"Hidden payload (0-{(1 << config.ECC_PAYLOAD_BITS) - 1})",
                                      min_value=0, max_value=(1 << config.ECC_PAYLOAD_BITS) - 1,
                                      value=config.ECC_DEFAULT_PAYLOAD, step=1))
        if max_tokens < config.ECC_MIN_TOKENS:
            st.warning(f"Payload decoding needs roughly {config.ECC_MIN_TOKENS}+ tokens to be reliable.")

    if st.button("Generate", type="primary"):
        be = load_backend()
        from watermarklab.detector import detect_method
        from watermarklab.generator import generate_plain, generate_watermarked
        with st.spinner("Generating..."):
            if spec is None:
                text, trace = generate_plain(prompt, max_tokens, seed=int(seed)), []
                detect_key = "basic"
            else:
                g = generate_watermarked(prompt, spec, max_tokens, payload=payload if spec.ecc else None,
                                         seed=int(seed))
                text, trace, detect_key = g.text, g.trace, spec.key
            st.session_state.update(last_generated_text=text, last_method=detect_key,
                                    last_payload=payload, last_trace=trace)
        section("Generated text")
        st.text_area("Generated text", value=text, height=200, label_visibility="collapsed")
        section("Detector verdict")
        show_detection(detect_method(text, detect_key))
        if spec is None:
            st.caption("Plain text is checked with the basic detector: it should NOT be flagged (false-positive control).")

# ---------------------------------------------------------------------------
# TAB 2: Attack Lab
# ---------------------------------------------------------------------------
with tab2:
    section("Attack a piece of watermarked text and see if it still detects",
            "Paraphrase, translate, delete or restructure the text, then re-run the detector.")
    from watermarklab.attacks import ATTACKS

    default_text = st.session_state.get("last_generated_text", "")
    text_to_attack = st.text_area("Text to attack (defaults to your last generated text from Tab 1)",
                                  value=default_text, height=150)
    keys = list(METHODS)
    default_idx = keys.index(st.session_state.get("last_method", "basic")) if st.session_state.get("last_method", "basic") in keys else 0
    detect_key = st.selectbox("Detector to use", keys, index=default_idx, format_func=lambda k: METHODS[k].label)
    attack_choice = st.selectbox("Attack type", list(ATTACKS))
    use_model = st.checkbox("Use real NLP models for this attack (slower, needs download). "
                            "Uncheck to use the fast offline fallback.", value=False)

    if st.button("Run Attack", type="primary"):
        if not text_to_attack.strip():
            st.warning("Enter or generate some text first.")
        else:
            load_backend()
            with st.spinner("Running attack..."):
                from watermarklab.detector import detect_method
                from watermarklab.utils import text_quality_score
                attack_fn = ATTACKS[attack_choice]
                try:
                    attacked_text = attack_fn(text_to_attack, use_model=use_model)
                except TypeError:
                    attacked_text = attack_fn(text_to_attack)
                before = detect_method(text_to_attack, detect_key)
                after = detect_method(attacked_text, detect_key)
                quality = text_quality_score(text_to_attack, attacked_text)

            dz = after["z_score"] - before["z_score"]
            survived = bool(after["is_watermarked"])
            cards(card("Attack", html.escape(attack_choice), "applied to the text", C["pink"]),
                  card("z-score change", f"{dz:+.2f}", f"{before['z_score']:.2f} → {after['z_score']:.2f}",
                       C["green"] if dz >= -0.5 else C["red"]),
                  card("Watermark survived?", "YES" if survived else "NO",
                       "still above threshold" if survived else "washed out", C["green"] if survived else C["red"]),
                  card("Text quality", f"{quality*100:.1f}%", "similarity to original", C["cyan"]))

            col1, col2 = st.columns(2)
            for col, title, cls, txt, res in ((col1, "Before attack", "before", text_to_attack, before),
                                              (col2, "After attack", "after", attacked_text, after)):
                with col:
                    H(f"<span class='wl-ba {cls}'>{title}</span>")
                    st.text_area(title, value=txt, height=180, key=f"{title}_box", label_visibility="collapsed")
                    yes = bool(res["is_watermarked"])
                    H(f"""<div class='wl-verdict {'yes' if yes else 'no'}'><div class='ico'>{'✅' if yes else '🚫'}</div>
                    <div><div class='t'>{'DETECTED' if yes else 'NOT DETECTED'}</div>
                    <div class='s'>confidence {res['confidence']*100:.1f}% · z = {res['z_score']:.2f}</div></div></div>""")
                    show_chart(gauge(res["z_score"]))
                    if "payload" in res:
                        dec_bits = format(res["payload"], "08b")
                        H(f"<div class='wl-label'>Decoded payload</div>{bit_chips(dec_bits)}")
            if "payload" in before:
                expected = st.session_state.get("last_payload")
                ok = after["payload"] == before["payload"]
                H(f"<span class='wl-pill {'ok' if ok else 'bad'}'>Payload survived the attack: {ok}</span>"
                  + (f"<span class='wl-pill info'>embedded payload was {format(expected, '08b')}</span>"
                     if expected is not None else ""))

# ---------------------------------------------------------------------------
# TAB 3: Benchmark Results
# ---------------------------------------------------------------------------
with tab3:
    section("Full benchmark: every method x every attack",
            "Detection accuracy, false positives, fluency cost and payload recovery in one place.")
    from watermarklab.evaluate import (generation_quality_summary, load_prompts, robustness_table,
                                       run_full_benchmark, summarize_benchmark, tradeoff_table)

    benchmark_path = os.path.join(config.RESULTS_DIR, "benchmark.csv")
    c1, c2 = st.columns([1, 3])
    with c1:
        n_prompts = st.slider("Prompts", 2, len(load_prompts()), min(6, len(load_prompts())))
        n_samples = st.slider("Samples per prompt", 1, 3, 2)
        run_new = st.button("Run full benchmark now", type="primary")
        st.caption("Uses fast offline attack fallbacks. More prompts x samples = steadier numbers. "
                   "For real-model attacks use `python run_demo.py --real-attacks`.")

    if run_new:
        load_backend()
        bar = st.progress(0.0, text="Running benchmark...")
        df = run_full_benchmark(prompts=load_prompts()[:n_prompts], n_samples=n_samples,
                                progress=lambda f: bar.progress(min(f, 1.0), text="Running benchmark..."))
        bar.empty()
        with st.spinner("Measuring fluency cost..."):
            quality = generation_quality_summary(prompts=load_prompts()[:n_prompts], n_samples=n_samples)
        quality.to_csv(os.path.join(config.RESULTS_DIR, "quality.csv"), index=False)
        st.success("Benchmark complete.")
    elif os.path.exists(benchmark_path):
        df = pd.read_csv(benchmark_path)
        qp = os.path.join(config.RESULTS_DIR, "quality.csv")
        quality = pd.read_csv(qp) if os.path.exists(qp) else None
    else:
        df, quality = None, None

    if df is not None:
        summary = summarize_benchmark(df)
        rt = robustness_table(df)

        best = rt["Mean under attack"].idxmax()
        fp_total = float(rt["Control: no watermark"].max()) if "Control: no watermark" in rt else 0.0
        best_txt = html.escape(str(best))
        mean_best = rt["Mean under attack"].max()
        cards(card("Most robust method", best_txt, f"{mean_best:.1f}% mean detection under attack", C["green"]),
              card("Worst false-positive rate", f"{fp_total:.1f}%", "plain text wrongly flagged (lower = better)",
                   C["green"] if fp_total == 0 else C["red"]),
              card("Methods compared", f"{rt.shape[0]}", "rows in the headline table", C["violet"]),
              card("Samples", f"{len(df):,}", "generated + attacked + detected", C["cyan"]))

        section("Robustness heatmap",
                "Detection accuracy (%) after each attack. Greener = the watermark survives better.")
        heat_cols = [c for c in rt.columns if c != "Control: no watermark"]
        hz = rt[heat_cols]
        fig_h = go.Figure(go.Heatmap(
            z=hz.values, x=heat_cols, y=list(hz.index), zmin=0, zmax=100,
            colorscale=[[0, "#7f1d3a"], [0.35, "#fb7185"], [0.6, "#fbbf24"], [1, "#34d399"]],
            text=[[f"{v:.0f}" for v in row] for row in hz.values], texttemplate="%{text}",
            textfont=dict(size=13, color="#06101f"), hoverongaps=False,
            colorbar=dict(title="%", thickness=12, outlinewidth=0),
            hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>", xgap=3, ygap=3))
        fig_h.update_yaxes(autorange="reversed")
        style_fig(fig_h, height=380, legend_bottom=False)
        fig_h.update_xaxes(tickangle=-25)
        show_chart(fig_h)

        st.markdown("**Headline table: detection accuracy (%) by method and attack** "
                    "(the *Control* column is the false-positive rate - lower is better)")
        st.dataframe(rt, **FIT)

        section("Robustness vs. fluency cost",
                "A method is only 'more robust' if it isn't simply using a stronger, more fluency-damaging bias.")
        tt = tradeoff_table(df, quality)
        st.dataframe(tt, **FIT)
        if "perplexity_vs_plain" in tt:
            figt = px.scatter(tt, x="perplexity_vs_plain", y="mean_detection_under_attack", color="method",
                              text="method", title="Detection under attack vs. fluency cost (up and left is better)",
                              color_discrete_map=METHOD_COLORS,
                              labels={"perplexity_vs_plain": "Perplexity vs plain text (x)",
                                      "mean_detection_under_attack": "Mean detection under attack (%)"})
            figt.update_traces(textposition="top center", marker=dict(size=16, line=dict(width=2, color="#0a0f1e")))
            style_fig(figt, height=460)
            show_chart(figt)

        st.markdown("**Summary: detection accuracy (%), mean z, text quality (%), payload recovery (%)**")
        st.dataframe(summary, **FIT)

        plot_df = summary[summary["attack"] != "Control: no watermark"]
        fig = px.bar(plot_df, x="attack", y="detection_accuracy", color="method", barmode="group",
                     title="Detection Accuracy After Each Attack", color_discrete_map=METHOD_COLORS,
                     labels={"detection_accuracy": "Detection Accuracy (%)", "attack": "Attack"})
        fig.update_traces(marker_line_width=0)
        style_fig(fig, height=440)
        show_chart(fig)
        fig2 = px.bar(plot_df[plot_df["attack"] != "No attack"], x="attack", y="avg_text_quality", color="method",
                      barmode="group", title="Text Quality After Each Attack", color_discrete_map=METHOD_COLORS,
                      labels={"avg_text_quality": "Text Quality (%)", "attack": "Attack"})
        fig2.update_traces(marker_line_width=0)
        style_fig(fig2, height=440)
        show_chart(fig2)

        pr = summary[summary["payload_recovery"].notna()]
        if len(pr):
            fig3 = px.bar(pr, x="attack", y="payload_recovery", color="method", barmode="group",
                          title="Hidden payload recovered correctly (%) - error-correcting methods",
                          color_discrete_map=METHOD_COLORS,
                          labels={"payload_recovery": "Payload recovery (%)", "attack": "Attack"})
            fig3.update_traces(marker_line_width=0)
            style_fig(fig3, height=440)
            show_chart(fig3)

        with st.expander("Raw per-prompt results"):
            st.dataframe(df, **FIT)
        st.download_button("Download full results (CSV)", data=df.to_csv(index=False),
                           file_name="benchmark_results.csv", mime="text/csv")
    else:
        st.info("No results yet - click 'Run full benchmark now' to generate them.")

# ---------------------------------------------------------------------------
# TAB 4: Special Features Lab
# ---------------------------------------------------------------------------
with tab4:
    section("The three special features, one by one",
            "Each demo generates text, attacks it where relevant, and shows exactly what the new idea does.")
    feat = st.radio("Feature", ["1. Semantic-Anchor Watermark Recovery", "2. Adaptive Watermark Strength",
                                "3. Error-Correcting Watermark"], horizontal=True)
    fprompt = st.text_input("Prompt", value="Explain what artificial intelligence is to a beginner.", key="fprompt")
    fseed = int(st.number_input("Random seed", value=config.SEED, step=1, key="fseed"))
    ftokens = st.slider("Tokens", 60, 300, 140, step=10, key="ftokens")

    from watermarklab.attacks import ATTACKS
    from watermarklab.detector import detect_method
    from watermarklab.generator import generate_watermarked

    if feat.startswith("1"):
        H("<div class='wl-idea'>⚓ <b>Idea.</b> Paraphrasing wrecks watermarks that depend on the <i>exact previous words</i>. "
          "This method keys the green list on the latest <b>anchor</b> (a meaning-bearing word such as "
          "<i>computer</i>, <i>learn</i>), judges words by their <b>synonym cluster</b>, and adds a second, "
          "context-free layer on the anchor words themselves. If the context layer is destroyed, "
          "the <b>anchor layer recovers the watermark</b>; the detector reports the best of "
          "context / anchor / combined evidence.</div>")
        atk = st.selectbox("Attack to apply", ["Paraphrasing", "Back-Translation", "Synonym Replacement",
                                               "Random Word Deletion", "LLM Rewriting"], key="atk1")
        if st.button("Run anchor demo", type="primary", key="b1"):
            load_backend()
            g = generate_watermarked(fprompt, "anchor", ftokens, seed=fseed)
            attacked = ATTACKS[atk](g.text)
            section("Generated text", "Anchor words (the meaning-bearing seeds) are highlighted.")
            render_tokens(g.trace, "none")
            d0, d1 = detect_method(g.text, "anchor"), detect_method(attacked, "anchor")
            cA, cB = st.columns(2)
            with cA:
                H("<span class='wl-ba before'>Before attack</span>")
                st.text_area("Original text", g.text, height=140, key="att1_before")
                show_detection(d0)
            with cB:
                H(f"<span class='wl-ba after'>After {html.escape(atk)}</span>")
                st.text_area("Attacked text", attacked, height=140, key="att1")
                show_detection(d1)
            chart = pd.DataFrame({
                "stage": ["Before", "Before", "After", "After"],
                "layer": ["Context layer", "Anchor layer"] * 2,
                "z": [d0["z_context_layer"], d0["z_anchor_layer"], d1["z_context_layer"], d1["z_anchor_layer"]],
            })
            fig = px.bar(chart, x="stage", y="z", color="layer", barmode="group",
                         title="z-score per layer (threshold = %.1f)" % config.Z_SCORE_THRESHOLD,
                         color_discrete_map={"Context layer": C["violet"], "Anchor layer": C["cyan"]})
            fig.add_hline(y=config.Z_SCORE_THRESHOLD, line_dash="dash", line_color=C["amber"])
            fig.update_traces(marker_line_width=0)
            style_fig(fig, height=400)
            show_chart(fig)

    elif feat.startswith("2"):
        H("<div class='wl-idea'>📈 <b>Idea.</b> A fixed bias hurts quality where the next word is obvious and is wasted there anyway. "
          "Adaptive strength measures the model's <i>uncertainty</i> (entropy) at every step and applies a "
          "strong bias only where many words fit, gentle elsewhere — and nudges the bias up if the "
          "watermark falls behind. The detector down-weights predictable tokens accordingly.</div>")
        if st.button("Run adaptive demo", type="primary", key="b2"):
            load_backend()
            g = generate_watermarked(fprompt, "adaptive", ftokens, seed=fseed)
            tr = pd.DataFrame(g.trace)
            tr["position"] = np.arange(len(tr))
            section("Generated text", "Green = the token landed in the green list, red = it did not.")
            render_tokens(g.trace, "target")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tr["position"], y=tr["entropy"], name="Model uncertainty (entropy)",
                                     mode="lines", line=dict(color=C["amber"], width=2.5)))
            fig.add_trace(go.Scatter(x=tr["position"], y=tr["delta"], name="Bias applied (adaptive)",
                                     mode="lines", line=dict(color=C["cyan"], width=2.5),
                                     fill="tozeroy", fillcolor="rgba(34,211,238,.10)"))
            fig.add_hline(y=config.WATERMARK_STRENGTH, line_dash="dot", line_color=C["pink"],
                          annotation_text="fixed bias of the basic method", annotation_font_color=C["pink"])
            fig.update_layout(title="Bias follows uncertainty", xaxis_title="Token position", yaxis_title="value")
            style_fig(fig, height=400)
            show_chart(fig)
            section("Detector verdict")
            show_detection(detect_method(g.text, "adaptive"))
            st.caption(f"Mean bias used: {tr['delta'].mean():.2f} (basic method: {config.WATERMARK_STRENGTH:.2f}).")

    else:
        H(f"<div class='wl-idea'>🛡️ <b>Idea.</b> Hide a short message (e.g. an author or model ID) inside the text. The payload "
          f"({config.ECC_PAYLOAD_BITS} bits) gets a CRC checksum and a <b>Hamming(7,4)</b> code, then each token "
          "votes for one codeword bit. The decoder picks the most likely valid payload from all votes "
          "(soft maximum-likelihood decoding), so altered tokens or empty slots do not destroy the message.</div>")
        c1, c2 = st.columns(2)
        with c1:
            pl = int(st.number_input("Payload", min_value=0, max_value=(1 << config.ECC_PAYLOAD_BITS) - 1,
                                     value=config.ECC_DEFAULT_PAYLOAD, key="pl3"))
        with c2:
            atk = st.selectbox("Attack to apply", ["(none)"] + list(ATTACKS), index=2, key="atk3")
        if st.button("Run error-correcting demo", type="primary", key="b3"):
            load_backend()
            g = generate_watermarked(fprompt, "ecc", ftokens, payload=pl, seed=fseed)
            text = g.text if atk == "(none)" else ATTACKS[atk](g.text)
            st.text_area("Text that is decoded", text, height=140, key="txt3")
            d = detect_method(text, "ecc")
            ok = d["payload"] == pl
            sent_bits, dec_bits = format(pl, "08b"), format(d["payload"], "08b")
            H(f"""<div class='wl-verdict {'yes' if ok else 'no'}'><div class='ico'>{'🔓' if ok else '🔒'}</div>
            <div><div class='t'>{'PAYLOAD RECOVERED' if ok else 'PAYLOAD NOT RECOVERED'}</div>
            <div class='s'>embedded vs decoded, bit by bit (green = match)</div></div></div>
            <div class='wl-label'>Embedded</div>{bit_chips(sent_bits)}
            <div class='wl-label'>Decoded</div>{bit_chips(dec_bits, sent_bits)}""")
            show_detection(d)
            votes = np.array(d["slot_votes"])
            sent = g.codeword
            vdf = pd.DataFrame({"slot": np.tile(np.arange(len(votes)), 2),
                                "votes": np.concatenate([votes[:, 0], votes[:, 1]]),
                                "bit": ["votes for 0"] * len(votes) + ["votes for 1"] * len(votes)})
            fig = px.bar(vdf, x="slot", y="votes", color="bit", barmode="group",
                         title="Token votes per codeword slot (true codeword bit shown below)",
                         color_discrete_map={"votes for 0": C["blue"], "votes for 1": C["pink"]})
            fig.update_xaxes(tickmode="array", tickvals=list(range(len(sent))),
                             ticktext=[f"{i}<br>[{b}]" for i, b in enumerate(sent)])
            fig.update_traces(marker_line_width=0)
            style_fig(fig, height=420)
            show_chart(fig)

H("<div class='wl-footer'>LLM-WatermarkLab · statistical green-list watermarking · robustness evaluation platform</div>")

# backend banner (shown once the model has been loaded by any button)
try:
    from watermarklab import backend as _b
    if _b._BACKEND is not None:
        if _b._BACKEND.is_fallback or _b._BACKEND.name == "toy-ngram":
            backend_slot.warning("Running on the OFFLINE TOY language model (no Hugging Face model could be "
                                 "loaded). Everything works, but numbers are for testing only. Install "
                                 "torch + transformers and connect to the internet once to use distilgpt2.")
        else:
            backend_slot.caption(f"Language model: {_b._BACKEND.name}")
except Exception:
    pass
