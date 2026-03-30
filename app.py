import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# ─── Page Configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="🎓 EduPredict AI — Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
    --primary: #4F46E5;
    --primary-dark: #3730A3;
    --accent: #14B8A6;
    --accent2: #0EA5E9;
    --bg-dark: #0F172A;
    --bg-card: #1E293B;
    --bg-card2: #334155;
    --text-primary: #F8FAFC;
    --text-muted: #94A3B8;
    --success: #10B981;
    --warning: #F59E0B;
    --danger: #EF4444;
    --border: rgba(51, 65, 85, 0.8);
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #020617 100%);
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Animated Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(79, 70, 229,0.15) 0%, rgba(14, 165, 233,0.08) 50%, rgba(20, 184, 166,0.08) 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 40px 50px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    animation: fadeSlideIn 0.8s ease-out both;
}

.hero-banner::before {
    content: '';
    position: absolute;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(79, 70, 229,0.15) 0%, transparent 70%);
    top: -100px; right: -100px;
    animation: pulse 4s ease-in-out infinite;
}

.hero-banner::after {
    content: '';
    position: absolute;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(14, 165, 233,0.1) 0%, transparent 70%);
    bottom: -60px; left: 30px;
    animation: pulse 6s ease-in-out infinite reverse;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(28px, 4vw, 44px);
    font-weight: 700;
    background: linear-gradient(135deg, #fff 0%, #4F46E5 50%, #0EA5E9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 15px;
    color: var(--text-muted);
    margin: 0;
    font-weight: 400;
    line-height: 1.6;
}

.hero-badge {
    display: inline-block;
    background: rgba(79, 70, 229,0.2);
    border: 1px solid rgba(79, 70, 229,0.4);
    color: var(--accent2);
    font-size: 12px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Metric Cards ── */
.metric-row {
    display: flex;
    gap: 16px;
    margin-bottom: 28px;
    flex-wrap: wrap;
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 24px;
    flex: 1;
    min-width: 150px;
    animation: fadeSlideIn 0.6s ease-out both;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(79, 70, 229,0.2);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}

.metric-card.purple::before { background: linear-gradient(90deg, #4F46E5, var(--accent2)); }
.metric-card.blue::before   { background: linear-gradient(90deg, #0EA5E9, var(--primary-dark)); }
.metric-card.pink::before   { background: linear-gradient(90deg, #14B8A6, var(--warning)); }
.metric-card.green::before  { background: linear-gradient(90deg, #10B981, var(--accent)); }

.metric-icon { font-size: 28px; margin-bottom: 8px; }
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px; font-weight: 700;
    color: #fff;
}
.metric-label { font-size: 13px; color: var(--text-muted); margin-top: 4px; }

/* ── Section Headers ── */
.section-header {
    display: flex; align-items: center; gap: 12px;
    margin: 28px 0 18px;
}

.section-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, rgba(79, 70, 229,0.3), rgba(14, 165, 233,0.2));
    border: 1px solid var(--border);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px; font-weight: 600;
    color: #fff;
    margin: 0;
}

.section-subtitle { font-size: 13px; color: var(--text-muted); margin: 0; }

/* ── Input Cards ── */
.input-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    animation: fadeIn 0.5s ease-out both;
}

.input-card h4 {
    font-size: 14px; font-weight: 600;
    color: var(--accent2); text-transform: uppercase;
    letter-spacing: 1px; margin: 0 0 16px;
}

/* ── Predict Button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #4F46E5 0%, #3730A3 50%, #0EA5E9 100%) !important;
    color: white !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 32px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 24px rgba(79, 70, 229,0.4) !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(79, 70, 229,0.6) !important;
}

/* ── Score Result ── */
.score-reveal {
    background: linear-gradient(135deg, rgba(79, 70, 229,0.12) 0%, rgba(14, 165, 233,0.08) 100%);
    border: 1px solid rgba(79, 70, 229,0.35);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    animation: pulseIn 0.6s cubic-bezier(0.34,1.56,0.64,1) both;
    position: relative; overflow: hidden;
}

.score-reveal::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(79, 70, 229,0.12) 0%, transparent 70%);
}

.score-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 80px; font-weight: 800;
    background: linear-gradient(135deg, #4F46E5, #0EA5E9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 8px;
    position: relative;
}

.score-label {
    font-size: 15px; color: var(--text-muted);
    font-weight: 400; position: relative;
}

.score-grade {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700; font-size: 14px;
    margin-top: 14px; position: relative;
}

.grade-A { background: rgba(16, 185, 129,0.15); color: #10B981; border: 1px solid rgba(16, 185, 129,0.3); }
.grade-B { background: rgba(14, 165, 233,0.15); color: #0EA5E9; border: 1px solid rgba(14, 165, 233,0.3); }
.grade-C { background: rgba(245, 158, 11,0.15);  color: #F59E0B; border: 1px solid rgba(245, 158, 11,0.3); }
.grade-D { background: rgba(239, 68, 68,0.15); color: #EF4444; border: 1px solid rgba(239, 68, 68,0.3); }

/* ── Insight Cards ── */
.insight-card {
    background: var(--bg-card2);
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border-left: 3px solid var(--primary);
    animation: fadeSlideIn 0.4s ease-out both;
}

.insight-card.warning { border-left-color: var(--warning); }
.insight-card.success { border-left-color: var(--success); }
.insight-card.danger  { border-left-color: var(--danger); }

.insight-card p { margin: 0; font-size: 14px; color: var(--text-muted); }
.insight-card strong { color: var(--text-primary); }

/* ── Selectboxes & inputs ── */
.stSelectbox > div > div,
.stSlider > div,
.stNumberInput > div {
    background: #1E293B !important;
    border-radius: 10px !important;
}

.stSelectbox > label,
.stSlider > label,
.stNumberInput > label,
.stRadio > label {
    color: var(--text-muted) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #fff !important;
}

.sidebar-logo {
    padding: 20px 0 10px;
    text-align: center;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}

.sidebar-logo span {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px; font-weight: 700;
    background: linear-gradient(135deg, #4F46E5, #0EA5E9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Divider ── */
.gradient-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--primary), var(--accent2), transparent);
    margin: 24px 0;
    opacity: 0.4;
}

/* ── Animations ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

@keyframes pulseIn {
    from { opacity: 0; transform: scale(0.85); }
    to   { opacity: 1; transform: scale(1); }
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50%       { transform: scale(1.15); opacity: 0.7; }
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

/* ── Tab Styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px 10px 0 0;
    color: var(--text-muted) !important;
    font-weight: 500;
    padding: 8px 18px;
}

.stTabs [aria-selected="true"] {
    background: rgba(79, 70, 229,0.12) !important;
    color: var(--accent2) !important;
    border-bottom: 2px solid #4F46E5 !important;
}

/* ── Gauge container ── */
.gauge-wrap { text-align: center; padding: 8px 0; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = "best_student_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path), True
    return None, False

model, model_loaded = load_model()


# ─── Helper Functions ───────────────────────────────────────────────
def encode_input(data: dict) -> pd.DataFrame:
    """Encode the raw input dict to match training feature space."""
    df = pd.DataFrame([data])

    # Ordinal mappings
    ordinal_maps = {
        "diet_quality":             {"Poor": 0, "Fair": 1, "Good": 2},
        "parental_education_level": {"High School": 0, "Some College": 1, "Bachelor": 2, "Master": 3, "PhD": 4},
        "internet_quality":         {"Low": 0, "Medium": 1, "High": 2},
        "family_income_range":      {"Low": 0, "Medium": 1, "High": 2},
        "parental_support_level":   {"Low": 0, "Medium": 1, "High": 2},
    }
    for col, mapping in ordinal_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Binary
    binary_cols = ["part_time_job", "extracurricular_participation", "access_to_tutoring", "dropout_risk"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    # One-hot
    ohe_cols = ["gender", "major", "study_environment", "learning_style"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    return df


def align_features(df_input: pd.DataFrame, model) -> pd.DataFrame:
    """Align input columns with model's expected features."""
    if hasattr(model, "feature_names_in_"):
        expected = model.feature_names_in_
        for col in expected:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[expected]
    return df_input


def predict_score(raw_input: dict) -> float:
    df_enc = encode_input(raw_input)
    df_enc = align_features(df_enc, model)
    pred = model.predict(df_enc)[0]
    return round(float(np.clip(pred, 0, 100)), 1)


def get_grade(score: float):
    if score >= 85: return "A — Outstanding 🌟", "grade-A", "🟢"
    if score >= 70: return "B — Good 👍",         "grade-B", "🔵"
    if score >= 55: return "C — Average ⚠️",      "grade-C", "🟡"
    return "D — Needs Improvement ❌",             "grade-D", "🔴"


def make_gauge(score: float) -> go.Figure:
    grade_label, _, _ = get_grade(score)
    color = "#10B981" if score >= 85 else "#0EA5E9" if score >= 70 else "#F59E0B" if score >= 55 else "#EF4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        delta={"reference": 70, "suffix": " pts from avg"},
        number={"suffix": "/100", "font": {"size": 40, "color": "#fff", "family": "Space Grotesk"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94A3B8", "tickfont": {"color": "#94A3B8"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(255,255,255,0.03)",
            "bordercolor": "rgba(79, 70, 229,0.2)",
            "steps": [
                {"range": [0, 55],  "color": "rgba(239, 68, 68,0.08)"},
                {"range": [55, 70], "color": "rgba(245, 158, 11,0.08)"},
                {"range": [70, 85], "color": "rgba(14, 165, 233,0.08)"},
                {"range": [85, 100],"color": "rgba(16, 185, 129,0.08)"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.75, "value": score},
        },
        title={"text": f"<b>Predicted Score</b><br><span style='font-size:14px; color:#94A3B8'>{grade_label}</span>",
               "font": {"size": 16, "color": "#fff", "family": "Space Grotesk"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(t=80, b=0, l=20, r=20),
        font={"color": "#fff"},
    )
    return fig


def make_radar(raw_input: dict) -> go.Figure:
    categories = ["Study Hours", "Sleep Hours", "Mental Health",
                  "Motivation", "Attendance", "Time Mgmt"]
    vals_raw = [
        raw_input["study_hours_per_day"] / 16,
        raw_input["sleep_hours"] / 14,
        raw_input["mental_health_rating"] / 10,
        raw_input["motivation_level"] / 10,
        raw_input["attendance_percentage"] / 100,
        raw_input["time_management_score"] / 10,
    ]
    vals = [round(v * 100, 1) for v in vals_raw]
    vals_closed = vals + [vals[0]]
    cats_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed, fill='toself',
        fillcolor='rgba(79, 70, 229,0.15)',
        line=dict(color='#4F46E5', width=2),
        name='Your Profile',
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 100], color='#94A3B8',
                            gridcolor='rgba(79, 70, 229,0.15)', tickfont=dict(size=10)),
            angularaxis=dict(color='#94A3B8', gridcolor='rgba(79, 70, 229,0.15)',
                             tickfont=dict(size=12, color='#F8FAFC')),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=320,
        margin=dict(t=30, b=30, l=40, r=40),
    )
    return fig


def make_feature_bar(raw_input: dict) -> go.Figure:
    labels = ["Study Hours", "Attendance", "Sleep", "GPA",
              "Mental Health", "Motivation", "Time Mgmt",
              "Stress (inv.)", "Screen Time (inv.)"]
    
    st_norm  = raw_input["study_hours_per_day"] / 16 * 100
    att_norm = raw_input["attendance_percentage"]
    sl_norm  = raw_input["sleep_hours"] / 14 * 100
    gpa_norm = raw_input["previous_gpa"] / 4.0 * 100
    mh_norm  = raw_input["mental_health_rating"] / 10 * 100
    mo_norm  = raw_input["motivation_level"] / 10 * 100
    tm_norm  = raw_input["time_management_score"] / 10 * 100
    stress_inv = (1 - raw_input["stress_level"] / 10) * 100
    screen_inv = (1 - raw_input["screen_time"] / 24) * 100

    values = [st_norm, att_norm, sl_norm, gpa_norm, mh_norm, mo_norm, tm_norm, stress_inv, screen_inv]
    colors = ["#10B981" if v >= 70 else "#F59E0B" if v >= 50 else "#EF4444" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.0f}%" for v in values],
        textposition='outside',
        textfont=dict(color='#94A3B8', size=11),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=340,
        xaxis=dict(range=[0, 115], showgrid=True, gridcolor='rgba(79, 70, 229,0.1)',
                   tickfont=dict(color='#94A3B8'), color='#94A3B8', title='Normalized Score (%)'),
        yaxis=dict(tickfont=dict(color='#F8FAFC', size=12), color='#94A3B8'),
        margin=dict(t=10, b=10, l=0, r=60),
        bargap=0.35,
    )
    return fig


def generate_insights(raw_input: dict, score: float) -> list:
    insights = []
    if raw_input["study_hours_per_day"] < 2:
        insights.append(("danger", "📚 Low study hours detected", "Studying less than 2 hrs/day significantly limits performance. Aim for 4–6 hours."))
    elif raw_input["study_hours_per_day"] >= 5:
        insights.append(("success", "📚 Solid study dedication", "Studying 5+ hours/day is associated with higher exam performance. Keep it up!"))

    if raw_input["attendance_percentage"] < 70:
        insights.append(("danger", "🏫 Low attendance", "Below 70% attendance is a critical risk factor. Attending class consistently improves retention."))
    elif raw_input["attendance_percentage"] >= 90:
        insights.append(("success", "🏫 Excellent attendance", "High attendance (90%+) is one of the strongest predictors of academic success."))

    if raw_input["sleep_hours"] < 6:
        insights.append(("warning", "😴 Insufficient sleep", "Less than 6 hrs of sleep impairs memory consolidation and cognitive function."))

    if raw_input["stress_level"] >= 8:
        insights.append(("danger", "😰 High stress level", "Chronic high stress could impair focus and exam performance. Consider stress management techniques."))

    if raw_input["screen_time"] > 8:
        insights.append(("warning", "📱 Excessive screen time", "Spending 8+ hrs on screens daily reduces study quality and sleep. Try the 20-20-20 rule."))

    if raw_input["mental_health_rating"] >= 7:
        insights.append(("success", "🧠 Good mental wellbeing", "Strong mental health is associated with better academic resilience and performance."))

    if raw_input["previous_gpa"] >= 3.5:
        insights.append(("success", "🏆 Strong academic history", "A GPA of 3.5+ indicates solid prior performance — a major predictor of future scores."))

    if raw_input["motivation_level"] >= 8:
        insights.append(("success", "⚡ High motivation", "High motivation is one of the top behavioral predictors of exam success."))

    return insights


# ─── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span>🎓 EduPredict AI</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")
    page = st.radio("", ["🏠 Predict Score", "📊 Dataset Insights", "ℹ️ About"], label_visibility="collapsed")

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Model Status**")
    if model_loaded:
        st.success("✅ Model successfully loaded")
        if hasattr(model, "__class__"):
            st.caption(f"**Estimator:** `{model.__class__.__name__}`")
    else:
        st.error("❌ Model file not found\n\n`best_student_model.pkl` missing.\n\nPlease run the pipeline notebook first.")

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown("**🧪 Dataset Stats**")
    col_a, col_b = st.columns(2)
    col_a.metric("Records", "~80K")
    col_b.metric("Features", "30")
    col_a.metric("Target", "Exam Score")
    col_b.metric("Range", "0–100")


# ─── Hero Banner ───────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🤖 Machine Learning Powered</div>
    <h1 class="hero-title">Student Performance Predictor</h1>
    <p class="hero-subtitle">
        Fill in the student profile below to get an AI-powered exam score prediction.<br>
        Powered by an ensemble model trained on 80,000+ student records across 30 behavioral &amp; demographic features.
    </p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: Predict Score
# ════════════════════════════════════════════════════════════════════
if page == "🏠 Predict Score":

    if not model_loaded:
        st.error("⚠️ Cannot predict — model not loaded. Please run the notebook pipeline to generate `best_student_model.pkl`.")
        st.stop()

    # ── Summary KPI Bar ──────────────────────────────────────────
    st.markdown("""
    <div class="metric-row">
        <div class="metric-card purple">
            <div class="metric-icon">🤖</div>
            <div class="metric-value">ML</div>
            <div class="metric-label">Ensemble Model</div>
        </div>
        <div class="metric-card blue">
            <div class="metric-icon">📊</div>
            <div class="metric-value">30</div>
            <div class="metric-label">Features Used</div>
        </div>
        <div class="metric-card pink">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">Regression</div>
            <div class="metric-label">Prediction Task</div>
        </div>
        <div class="metric-card green">
            <div class="metric-icon">⚡</div>
            <div class="metric-value">Instant</div>
            <div class="metric-label">Live Inference</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input Form ───────────────────────────────────────────────
    with st.form("prediction_form"):

        # —— Academic Profile ——
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📚</div>
            <div><p class="section-title">Academic Profile</p>
                 <p class="section-subtitle">Study habits and academic history</p></div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        study_hours       = c1.slider("Study Hours / Day",      0.0, 16.0, 4.0, 0.5)
        attendance_pct    = c2.slider("Attendance (%)",          0.0, 100.0, 80.0, 1.0)
        previous_gpa      = c3.slider("Previous GPA (0–4)",     0.0, 4.0,  2.5, 0.1)
        semester          = c4.selectbox("Current Semester",     [1, 2, 3, 4, 5, 6, 7, 8])

        c5, c6 = st.columns(2)
        major             = c5.selectbox("Major", ["Computer Science", "Engineering", "Business", "Biology", "Arts", "Psychology"])
        study_env         = c6.selectbox("Study Environment", ["Library", "Quiet Room", "Dorm", "Cafe", "Co-Learning Group"])

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        # —— Lifestyle & Wellness ——
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">🏃</div>
            <div><p class="section-title">Lifestyle & Wellness</p>
                 <p class="section-subtitle">Daily habits, health and wellbeing metrics</p></div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        sleep_hours       = c1.slider("Sleep Hours / Day",       2.0, 14.0, 7.0, 0.5)
        screen_time       = c2.slider("Total Screen Time (hrs)", 0.0, 24.0, 5.0, 0.5)
        social_media_hrs  = c3.slider("Social Media (hrs)",      0.0, 12.0, 2.0, 0.5)
        netflix_hrs       = c4.slider("Netflix/Streaming (hrs)", 0.0, 12.0, 1.0, 0.5)

        c5, c6, c7, c8 = st.columns(4)
        exercise_freq     = c5.slider("Exercise (days/week)",    0, 7, 3)
        mental_health     = c6.slider("Mental Health (1–10)",    1.0, 10.0, 7.0, 0.1)
        stress_level      = c7.slider("Stress Level (1–10)",     1.0, 10.0, 5.0, 0.1)
        diet_quality      = c8.selectbox("Diet Quality",          ["Poor", "Fair", "Good"])

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        # —— Psycho-Academic Traits ——
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">🧠</div>
            <div><p class="section-title">Psycho-Academic Traits</p>
                 <p class="section-subtitle">Motivation, learning style, and cognitive profile</p></div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        motivation        = c1.slider("Motivation Level (1–10)",    1, 10, 7)
        anxiety_score     = c2.slider("Exam Anxiety (1–10)",        1, 10, 5)
        time_mgmt         = c3.slider("Time Management (1–10)",     1.0, 10.0, 6.0, 0.1)
        social_activity   = c4.slider("Social Activity (0–5)",      0, 5, 2)

        c5, c6 = st.columns(2)
        learning_style    = c5.selectbox("Learning Style", ["Visual", "Auditory", "Reading", "Kinesthetic"])
        gender            = c6.selectbox("Gender",          ["Male", "Female", "Other"])

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        # —— Socioeconomic Context ——
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">🏠</div>
            <div><p class="section-title">Socioeconomic Context</p>
                 <p class="section-subtitle">Family background and support system</p></div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        parent_edu        = c1.selectbox("Parental Education",   ["High School", "Some College", "Bachelor", "Master", "PhD"])
        internet_quality  = c2.selectbox("Internet Quality",     ["Low", "Medium", "High"])
        family_income     = c3.selectbox("Family Income Range",  ["Low", "Medium", "High"])
        parent_support    = c4.selectbox("Parental Support",     ["Low", "Medium", "High"])

        c5, c6, c7, c8 = st.columns(4)
        part_time_job     = c5.selectbox("Part-Time Job",        ["No", "Yes"])
        extracurricular   = c6.selectbox("Extracurricular",      ["No", "Yes"])
        access_tutoring   = c7.selectbox("Access to Tutoring",   ["No", "Yes"])
        dropout_risk      = c8.selectbox("Dropout Risk",         ["No", "Yes"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Predict Exam Score", use_container_width=True)

    # ── Prediction Results ───────────────────────────────────────
    if submitted:
        raw_input = {
            "age":                        20,
            "study_hours_per_day":        study_hours,
            "social_media_hours":         social_media_hrs,
            "netflix_hours":              netflix_hrs,
            "part_time_job":              part_time_job,
            "attendance_percentage":      attendance_pct,
            "sleep_hours":                sleep_hours,
            "diet_quality":               diet_quality,
            "exercise_frequency":         exercise_freq,
            "parental_education_level":   parent_edu,
            "internet_quality":           internet_quality,
            "mental_health_rating":       round(mental_health, 1),
            "extracurricular_participation": extracurricular,
            "previous_gpa":               previous_gpa,
            "semester":                   semester,
            "stress_level":               stress_level,
            "dropout_risk":               dropout_risk,
            "social_activity":            social_activity,
            "screen_time":                screen_time,
            "study_environment":          study_env,
            "access_to_tutoring":         access_tutoring,
            "family_income_range":        family_income,
            "parental_support_level":     parent_support,
            "motivation_level":           motivation,
            "exam_anxiety_score":         anxiety_score,
            "learning_style":             learning_style,
            "time_management_score":      round(time_mgmt, 1),
            "gender":                     gender,
            "major":                      major,
        }

        with st.spinner("🔮 Analyzing student profile..."):
            time.sleep(0.6)
            score = predict_score(raw_input)

        grade_label, grade_class, grade_dot = get_grade(score)

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        # ── Score Banner ──
        col_score, col_gauge = st.columns([1, 1.6])

        with col_score:
            st.markdown(f"""
            <div class="score-reveal">
                <div class="score-number">{score}</div>
                <div class="score-label">Predicted Exam Score out of 100</div>
                <div class="score-grade {grade_class}">{grade_label}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_gauge:
            st.plotly_chart(make_gauge(score), use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        # ── Tabs: Radar / Breakdown / Insights ──
        tab1, tab2, tab3 = st.tabs([
            "📡 Performance Radar",
            "📊 Feature Breakdown",
            "💡 Personalized Insights",
        ])

        with tab1:
            col_r1, col_r2 = st.columns([1.1, 1])
            with col_r1:
                st.markdown("#### 📡 Student Profile Radar")
                st.caption("Normalized scores across 6 key performance dimensions (0–100%)")
                st.plotly_chart(make_radar(raw_input), use_container_width=True, config={"displayModeBar": False})
            with col_r2:
                st.markdown("#### 📋 Profile Summary")
                summary_data = {
                    "Feature": ["Study Hours/Day", "Attendance", "Sleep Hours",
                                "Previous GPA", "Mental Health", "Motivation",
                                "Stress Level", "Screen Time", "Exercise Days"],
                    "Value": [f"{study_hours}h", f"{attendance_pct:.0f}%", f"{sleep_hours}h",
                              f"{previous_gpa:.1f}/4.0", f"{mental_health:.1f}/10", f"{motivation}/10",
                              f"{stress_level:.1f}/10", f"{screen_time}h", f"{exercise_freq} days"],
                    "Status": [
                        "✅" if study_hours >= 3 else "⚠️",
                        "✅" if attendance_pct >= 75 else "❌",
                        "✅" if sleep_hours >= 7 else "⚠️",
                        "✅" if previous_gpa >= 3 else "⚠️",
                        "✅" if mental_health >= 6 else "⚠️",
                        "✅" if motivation >= 7 else "⚠️",
                        "✅" if stress_level <= 6 else "⚠️",
                        "✅" if screen_time <= 5 else "⚠️",
                        "✅" if exercise_freq >= 3 else "⚠️",
                    ]
                }
                st.dataframe(
                    pd.DataFrame(summary_data),
                    hide_index=True,
                    use_container_width=True,
                )

        with tab2:
            st.markdown("#### 📊 Normalized Feature Breakdown")
            st.caption("Higher bars = more favorable conditions for academic performance (stress & screen time are inverted)")
            st.plotly_chart(make_feature_bar(raw_input), use_container_width=True, config={"displayModeBar": False})

        with tab3:
            insights = generate_insights(raw_input, score)
            if not insights:
                st.success("🌟 No critical concerns detected. The student appears well-balanced across all key indicators!")
            else:
                st.markdown("#### 💡 Personalized Recommendations")
                for kind, title, body in insights:
                    st.markdown(f"""
                    <div class="insight-card {kind}">
                        <p><strong>{title}</strong></p>
                        <p>{body}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # score benchmark bars
            st.markdown("#### 📈 Score Benchmarks")
            bench_fig = go.Figure(go.Bar(
                x=[55, 70, 85, score],
                y=["D/C Boundary", "C/B Boundary", "B/A Boundary", "Your Score"],
                orientation='h',
                marker=dict(color=["#EF4444", "#F59E0B", "#0EA5E9",
                                   "#10B981" if score >= 85 else "#0EA5E9" if score >= 70 else "#F59E0B" if score >= 55 else "#EF4444"],
                            line=dict(width=0)),
                text=[str(v) for v in [55, 70, 85, score]],
                textposition='outside',
                textfont=dict(color='#F8FAFC', size=13),
            ))
            bench_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=220, margin=dict(t=10, b=10, l=0, r=60),
                xaxis=dict(range=[0, 115], showgrid=True, gridcolor='rgba(79, 70, 229,0.1)',
                           tickfont=dict(color='#94A3B8'), color='#94A3B8'),
                yaxis=dict(tickfont=dict(color='#F8FAFC', size=12), color='#94A3B8'),
                bargap=0.4,
            )
            st.plotly_chart(bench_fig, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════════
# PAGE: Dataset Insights
# ════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Insights":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📊</div>
        <div><p class="section-title">Dataset Overview & Insights</p>
             <p class="section-subtitle">What the enhanced_student_habits dataset reveals</p></div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏆 Top Predictors of Exam Score")
        features_importance = {
            "previous_gpa":           0.31,
            "study_hours_per_day":    0.18,
            "attendance_percentage":  0.15,
            "motivation_level":       0.09,
            "time_management_score":  0.07,
            "mental_health_rating":   0.06,
            "parental_support_level": 0.05,
            "stress_level":          -0.04,
            "screen_time":           -0.03,
            "sleep_hours":            0.02,
        }
        fi_df = pd.Series(features_importance).sort_values()
        colors_fi = ["#10B981" if v > 0 else "#EF4444" for v in fi_df.values]
        fig_fi = go.Figure(go.Bar(
            x=fi_df.values, y=fi_df.index, orientation='h',
            marker=dict(color=colors_fi, line=dict(width=0)),
            text=[f"{v:+.2f}" for v in fi_df.values], textposition='outside',
            textfont=dict(color='#94A3B8', size=11),
        ))
        fig_fi.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=320, margin=dict(t=10, b=10, l=0, r=60),
            xaxis=dict(showgrid=True, gridcolor='rgba(79, 70, 229,0.1)', tickfont=dict(color='#94A3B8'), color='#94A3B8'),
            yaxis=dict(tickfont=dict(color='#F8FAFC', size=12), color='#94A3B8'),
        )
        st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown("#### 🎓 Score Distribution by Major")
        majors = ["Engineering", "Computer Science", "Biology", "Business", "Psychology", "Arts"]
        scores_by_major = [74, 73, 71, 70, 69, 68]

        fig_maj = go.Figure(go.Bar(
            x=majors, y=scores_by_major,
            marker=dict(
                color=scores_by_major,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    tickfont=dict(color='#94A3B8'),
                    outlinecolor='rgba(0,0,0,0)'
                ),
                line=dict(width=0)),
            text=[f"{s:.0f}" for s in scores_by_major],
            textposition='outside',
            textfont=dict(color='#F8FAFC'),
        ))
        fig_maj.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=320, margin=dict(t=10, b=10, l=0, r=10),
            xaxis=dict(tickfont=dict(color='#F8FAFC'), color='#94A3B8'),
            yaxis=dict(tickfont=dict(color='#94A3B8'), color='#94A3B8', gridcolor='rgba(79, 70, 229,0.1)', range=[60, 80]),
        )
        st.plotly_chart(fig_maj, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 📋 Dataset Column Summary")
        col_info = {
            "Category":  ["Demographic"]*3 + ["Academic"]*4 + ["Behavioral"]*5 + ["Wellness"]*5 + ["Socioeconomic"]*4,
            "Column":    ["age","gender","major",
                          "study_hours_per_day","attendance_percentage","previous_gpa","semester",
                          "screen_time","social_media_hours","netflix_hours","part_time_job","extracurricular_participation",
                          "sleep_hours","diet_quality","exercise_frequency","mental_health_rating","stress_level",
                          "family_income_range","parental_education_level","parental_support_level","internet_quality"],
            "Type":      ["Int","Cat","Cat",
                          "Float","Float","Float","Int",
                          "Float","Float","Float","Binary","Binary",
                          "Float","Ordinal","Int","Float","Float",
                          "Ordinal","Ordinal","Ordinal","Ordinal"],
        }
        st.dataframe(pd.DataFrame(col_info), hide_index=True, use_container_width=True, height=360)

    with col4:
        st.markdown("#### 📈 Study Hours vs. Exam Score (Trend)")
        np.random.seed(42)
        hours_sim = np.linspace(0, 16, 100)
        scores_sim = 50 + 2.8 * hours_sim + np.random.normal(0, 4, 100)
        scores_sim = np.clip(scores_sim, 50, 100)

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=hours_sim, y=scores_sim, mode='markers',
            marker=dict(color='rgba(79, 70, 229,0.4)', size=5),
            name='Students',
        ))
        # Trend line
        z = np.polyfit(hours_sim, scores_sim, 1)
        p = np.poly1d(z)
        fig_trend.add_trace(go.Scatter(
            x=hours_sim, y=p(hours_sim), mode='lines',
            line=dict(color='#0EA5E9', width=2, dash='dash'),
            name='Trend',
        ))
        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=320, margin=dict(t=10, b=10, l=0, r=10),
            xaxis=dict(title='Study Hours/Day', tickfont=dict(color='#94A3B8'), color='#94A3B8', gridcolor='rgba(79, 70, 229,0.1)'),
            yaxis=dict(title='Exam Score',      tickfont=dict(color='#94A3B8'), color='#94A3B8', gridcolor='rgba(79, 70, 229,0.1)'),
            legend=dict(font=dict(color='#94A3B8'), bgcolor='rgba(0,0,0,0)'),
        )
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════════
# PAGE: About
# ════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">ℹ️</div>
        <div><p class="section-title">About EduPredict AI</p>
             <p class="section-subtitle">Pipeline architecture and model details</p></div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="input-card">
            <h4>🔧 ML Pipeline Architecture</h4>
            <p style="color:#F8FAFC; font-size:14px; line-height:1.8">
            <b>Step 1</b> — Data Loading & Inspection<br>
            <b>Step 2</b> — Preprocessing & Correction<br>
            &nbsp;&nbsp;&nbsp;&nbsp;• Float precision fixes<br>
            &nbsp;&nbsp;&nbsp;&nbsp;• Mixed-type standardization<br>
            <b>Step 3</b> — Missing Value Imputation<br>
            <b>Step 4</b> — Duplicate Detection<br>
            <b>Step 5</b> — Outlier Winsorization<br>
            <b>Step 6</b> — Exploratory Data Analysis<br>
            <b>Step 7</b> — Encoding & Feature Scaling<br>
            <b>Step 8</b> — Feature Selection (Filter/Wrapper/Embedded)<br>
            <b>Step 9</b> — Model Training (6 regressors × 3 feature sets)<br>
            <b>Step 10</b> — Model Comparison & Visualization<br>
            <b>Step 11</b> — Best Model Saved as .pkl<br>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="input-card">
            <h4>🤖 Models Evaluated</h4>
            <p style="color:#F8FAFC; font-size:14px; line-height:1.8">
            ▸ <b>Ridge Regression</b> — L2 linear baseline<br>
            ▸ <b>Lasso Regression</b> — L1 sparse linear model<br>
            ▸ <b>Random Forest</b> — Ensemble of decision trees<br>
            ▸ <b>Gradient Boosting</b> — Sequential boosting<br>
            ▸ <b>XGBoost</b> — Optimized gradient boosting<br>
            ▸ <b>LightGBM</b> — Fast large-scale gradient boosting<br>
            <br>
            <b>Evaluation Metrics:</b><br>
            RMSE · MAE · R² · Cross-Val R² (5-fold)<br>
            <br>
            <b>Feature Selection Methods:</b><br>
            Filter (MI / F-test) · Wrapper (RFE) · Embedded (Lasso+RF)
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="input-card">
        <h4>📦 Tech Stack</h4>
        <p style="color:#F8FAFC; font-size:14px; line-height:2.0">
        <code>Python 3.8+</code> &nbsp;|&nbsp;
        <code>scikit-learn</code> &nbsp;|&nbsp;
        <code>XGBoost</code> &nbsp;|&nbsp;
        <code>LightGBM</code> &nbsp;|&nbsp;
        <code>pandas</code> &nbsp;|&nbsp;
        <code>numpy</code> &nbsp;|&nbsp;
        <code>Streamlit</code> &nbsp;|&nbsp;
        <code>Plotly</code> &nbsp;|&nbsp;
        <code>joblib</code>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info("💡 To retrain the model, open `Student_Habits_Performance_Pipeline.ipynb` and run all cells. The best model will be saved to `best_student_model.pkl` automatically.")
