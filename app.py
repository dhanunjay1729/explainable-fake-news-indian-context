"""
Interactive Web Application for Explainable Fake News Detection in Indian Context.
Final-Year NLP Academic Demonstration App built with Streamlit and LIME.

Usage:
    streamlit run app.py
"""

import os
import sys
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability import LIMEExplainer
from src.preprocessing import StandardPreprocessor, IndianContextPreprocessor

# Streamlit Page Config
st.set_page_config(
    page_title="Explainable Fake News Detection (Indian Context)",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e3799;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b6584;
        margin-bottom: 1.5rem;
    }
    .badge-bar {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    .badge {
        background-color: #f1f2f6;
        border: 1px solid #ced6e0;
        padding: 0.25rem 0.65rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #2f3542;
    }
    .fake-card {
        background: linear-gradient(135deg, #ff7675 0%, #d63031 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(214, 48, 49, 0.3);
        margin-bottom: 1.2rem;
    }
    .real-card {
        background: linear-gradient(135deg, #2ed573 0%, #10ac84 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(16, 172, 132, 0.3);
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    """Caches and loads the serialized models from models/ directory."""
    models_dir = PROJECT_ROOT / 'models'
    models = {}
    try:
        models['Proposed: Logistic Regression'] = joblib.load(models_dir / 'proposed_logistic_regression_pipeline.joblib')
        models['Proposed: Linear SVM'] = joblib.load(models_dir / 'proposed_linear_svm_pipeline.joblib')
        models['Base: Multinomial Naive Bayes'] = joblib.load(models_dir / 'base_naive_bayes_pipeline.joblib')
    except Exception as e:
        st.error(f"Error loading model pipelines from {models_dir}: {e}")
    return models


# Load models
models_dict = load_all_models()

# Preset Indian News Examples
PRESET_EXAMPLES = {
    "Select an example to auto-fill...": "",
    "🚨 Viral WhatsApp GPS Nano-Chip Note (Fake)": (
        "BREAKING: Reserve Bank of India announces new Rs 1000 currency note equipped with GPS nano-chip "
        "capable of tracking black money stored in underground lockers via satellite. Share with all WhatsApp groups!"
    ),
    "🏛️ RBI Monetary Policy Announcement (Real)": (
        "The Reserve Bank of India Monetary Policy Committee decided unanimously to keep the policy repo rate "
        "unchanged at 6.50 percent, Governor Shaktikanta Das announced at the press briefing in Mumbai on Tuesday."
    ),
    "🚨 Free 5000 Recharge Viral Hoax (Fake)": (
        "VIRAL ALERT: Prime Minister Modi ji approves free Rs 5000 recharge for all Jio and Airtel prepaid users "
        "to celebrate Digital India milestone. Click the link to claim before midnight today. Forward to 5 groups."
    ),
    "🏛️ Supreme Court Constitutional Bench (Real)": (
        "A Constitution Bench of the Supreme Court of India agreed to hear petitions challenging the validity of "
        "the Electoral Bonds scheme, directing the registry to list the matter for priority hearing."
    ),
    "🚨 NASA Diwali Satellite Image Claim (Fake)": (
        "Big revelation: NASA satellite image released on Twitter shows entire India completely illuminated in gold "
        "light on Diwali night while neighbouring countries remain completely in the dark. Proud moment for India!"
    )
}

# Header Section
st.markdown('<div class="main-title">Explainable Fake News Detection in Indian Context</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">NLP Semester 7 Final-Year Academic Project | Comparative Preprocessing, TF-IDF Optimization & Local Model Interpretability (LIME)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="badge-bar">
    <span class="badge">Dataset: IFND (Indian Fake News Dataset)</span>
    <span class="badge">Base Model: Multinomial Naive Bayes</span>
    <span class="badge">Proposed Model: Logistic Regression + Indian NLP</span>
    <span class="badge">XAI: LIME Explainer</span>
    <span class="badge">Features: Sparse TF-IDF (1,2-grams)</span>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/news.png", width=64)
st.sidebar.title("Configuration & Controls")

selected_model_name = st.sidebar.selectbox(
    "Active Classification Model:",
    options=list(models_dict.keys()),
    index=0,
    help="Select between our proposed models and the base reference model."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Test Prompts")
selected_preset_label = st.sidebar.selectbox(
    "Pre-loaded Indian Test Cases:",
    options=list(PRESET_EXAMPLES.keys()),
    index=0
)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Live Detection & LIME Explainability",
    "📊 Empirical Research Results & Figures",
    "📚 Academic Project Guide & Presentation"
])

# TAB 1: Live Detection & LIME
with tab1:
    col_input, col_pred = st.columns([1.1, 0.9])

    with col_input:
        st.subheader("Input News Article / Social Media Claim")

        default_text = PRESET_EXAMPLES.get(selected_preset_label, "")
        user_input = st.text_area(
            "Paste Indian news headline, WhatsApp forward, or social media post:",
            value=default_text,
            height=180,
            placeholder="e.g. VIRAL ALERT: Central government to cancel all voter ID cards not linked with Aadhaar..."
        )

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            analyze_button = st.button("🚀 Analyze & Explain", type="primary", use_container_width=True)
        with col_btn2:
            compare_button = st.button("⚖️ Compare Base vs. Proposed", use_container_width=True)

    with col_pred:
        st.subheader("Prediction & Confidence")
        if not analyze_button and not compare_button and not user_input:
            st.info("👈 Enter a news statement or select an example from the sidebar and click **Analyze & Explain**.")
        else:
            if not user_input.strip():
                st.warning("Please provide a text statement to analyze.")
            else:
                active_pipeline = models_dict.get(selected_model_name)
                if active_pipeline is not None:
                    # Get prediction and probabilities
                    proba = active_pipeline.predict_proba([user_input])[0]
                    p_fake = proba[0]
                    p_real = proba[1]
                    pred_class = 0 if p_fake >= 0.5 else 1

                    if pred_class == 0:
                        st.markdown(f"""
                        <div class="fake-card">
                            <h2 style='margin:0; font-size:1.6rem;'>🚨 FAKE / MISINFORMATION</h2>
                            <p style='margin: 0.3rem 0 0 0; font-size: 1rem;'>Confidence: <b>{p_fake*100:.1f}%</b> (P(Real): {p_real*100:.1f}%)</p>
                            <small>Pattern: Sensationalist framing / Unverified assertion</small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="real-card">
                            <h2 style='margin:0; font-size:1.6rem;'>✅ REAL / CREDIBLE NEWS</h2>
                            <p style='margin: 0.3rem 0 0 0; font-size: 1rem;'>Confidence: <b>{p_real*100:.1f}%</b> (P(Fake): {p_fake*100:.1f}%)</p>
                            <small>Pattern: Verified institutional reporting structure</small>
                        </div>
                        """, unsafe_allow_html=True)

                    # Gauge / Bar of probability
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=p_fake * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Misinformation Probability Score (%)", 'font': {'size': 14}},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#d63031" if p_fake >= 0.5 else "#10ac84"},
                            'steps': [
                                {'range': [0, 40], 'color': "rgba(46, 213, 115, 0.2)"},
                                {'range': [40, 60], 'color': "rgba(255, 165, 2, 0.2)"},
                                {'range': [60, 100], 'color': "rgba(255, 71, 87, 0.2)"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 3},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

    # LIME Explainability Section
    if user_input.strip() and (analyze_button or compare_button):
        st.markdown("---")
        st.subheader("🧠 Local Interpretable Model-agnostic Explanations (LIME)")
        st.markdown(
            "LIME builds a local surrogate linear model around this specific input to explain **which words drive the decision towards Fake or Real**."
        )

        with st.spinner("Computing LIME word attributions..."):
            active_pipeline = models_dict.get(selected_model_name)
            explainer = LIMEExplainer(pipeline=active_pipeline, class_names=['Fake', 'Real'])
            lime_res = explainer.explain_instance(user_input, num_features=12)
            features = lime_res.get('features', [])

        if features:
            col_chart, col_text = st.columns([1.1, 0.9])

            with col_chart:
                words = [f[0] for f in features]
                weights = [f[1] for f in features]
                colors = ['#2ed573' if w > 0 else '#ff4757' for w in weights]
                labels = ['Supports REAL' if w > 0 else 'Supports FAKE' for w in weights]

                fig_bar = go.Figure(go.Bar(
                    x=weights,
                    y=words,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{w:+.3f}" for w in weights],
                    textposition='outside'
                ))
                fig_bar.update_layout(
                    title="LIME Word-Level Feature Contributions",
                    xaxis_title="Attribution Weight (Negative = Fake, Positive = Real)",
                    yaxis=dict(autorange="reversed"),
                    height=380,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_text:
                st.markdown("#### In-Text Lexical Highlighting")
                st.markdown("Words that influenced the classifier are highlighted in the passage:")
                
                # Create HTML highlighted text
                highlighted_html = user_input
                feature_dict = {f[0].lower(): f[1] for f in features}
                
                # Split tokens and highlight
                tokens = user_input.split()
                html_tokens = []
                for tok in tokens:
                    clean_tok = "".join(c for c in tok.lower() if c.isalnum())
                    if clean_tok in feature_dict:
                        w = feature_dict[clean_tok]
                        if w < 0:
                            # Contributed to Fake (Red)
                            bg = f"rgba(255, 71, 87, {min(abs(w)*3, 0.8):.2f})"
                            html_tokens.append(f"<span style='background-color:{bg}; color:black; padding:2px 4px; border-radius:3px; font-weight:bold;'>{tok}</span>")
                        else:
                            # Contributed to Real (Green)
                            bg = f"rgba(46, 213, 115, {min(abs(w)*3, 0.8):.2f})"
                            html_tokens.append(f"<span style='background-color:{bg}; color:black; padding:2px 4px; border-radius:3px; font-weight:bold;'>{tok}</span>")
                    else:
                        html_tokens.append(tok)
                
                rendered_text = " ".join(html_tokens)
                st.markdown(
                    f"<div style='border:1px solid #dfe4ea; background-color:#f8f9fa; padding:15px; border-radius:8px; line-height:1.8; font-size:1.05rem;'>{rendered_text}</div>",
                    unsafe_allow_html=True
                )
                st.markdown("""
                <div style='margin-top:10px; font-size:0.85rem; color:#57606f;'>
                    <span style='background-color:rgba(255, 71, 87, 0.5); padding:2px 6px; border-radius:3px;'>Red tokens</span> = Pushes towards FAKE &nbsp;|&nbsp; 
                    <span style='background-color:rgba(46, 213, 115, 0.5); padding:2px 6px; border-radius:3px;'>Green tokens</span> = Pushes towards REAL
                </div>
                """, unsafe_allow_html=True)

    # Base vs Proposed Side-by-Side Comparison
    if compare_button and user_input.strip():
        st.markdown("---")
        st.subheader("⚖️ Base Model vs. Proposed Model Comparison on Given Input")

        col_base, col_prop = st.columns(2)

        base_pipe = models_dict.get('Base: Multinomial Naive Bayes')
        prop_pipe = models_dict.get('Proposed: Logistic Regression')

        with col_base:
            st.markdown("### Base Pipeline: Naive Bayes + Standard Preprocessing")
            if base_pipe:
                b_prob = base_pipe.predict_proba([user_input])[0]
                b_pred = "FAKE" if b_prob[0] >= 0.5 else "REAL"
                st.metric("Base Model Prediction", b_pred, delta=f"Conf: {max(b_prob)*100:.1f}%")
                st.write(f"P(Fake): `{b_prob[0]:.4f}` | P(Real): `{b_prob[1]:.4f}`")
                st.caption("Standard preprocessing removes regional stopwords and misses Indian-specific entities.")

        with col_prop:
            st.markdown("### Proposed Pipeline: Logistic Regression + Indian Context")
            if prop_pipe:
                p_prob = prop_pipe.predict_proba([user_input])[0]
                p_pred = "FAKE" if p_prob[0] >= 0.5 else "REAL"
                st.metric("Proposed Model Prediction", p_pred, delta=f"Conf: {max(p_prob)*100:.1f}%")
                st.write(f"P(Fake): `{p_prob[0]:.4f}` | P(Real): `{p_prob[1]:.4f}`")
                st.caption("Preserves Hindi honorifics, transliterated financial numbers, and regional named entities.")


# TAB 2: Empirical Research Results & Figures
with tab2:
    st.subheader("Empirical Findings & Visual Benchmarks")
    st.markdown("All visualizations below were generated directly from our 5 reproducible experimental test suites.")

    results_dir = PROJECT_ROOT / 'results'

    res_tab1, res_tab2, res_tab3, res_tab4, res_tab5 = st.tabs([
        "1. Exploratory Data Analysis",
        "2. Preprocessing Benchmark",
        "3. Feature Representations & Latency",
        "4. Classifier Performance & ROC",
        "5. LIME Diagnostics Summary"
    ])

    with res_tab1:
        st.markdown("#### Dataset Lexical Profiles & Class Distributions")
        c1, c2 = st.columns(2)
        with c1:
            if (results_dir / 'class_distribution.png').exists():
                st.image(str(results_dir / 'class_distribution.png'), caption="Balanced IFND Class Distribution (Fake vs Real)")
            if (results_dir / 'top_unigrams.png').exists():
                st.image(str(results_dir / 'top_unigrams.png'), caption="Top 20 Unigram Frequencies")
        with c2:
            if (results_dir / 'article_length_hist.png').exists():
                st.image(str(results_dir / 'article_length_hist.png'), caption="Article Character Length Distribution")
            if (results_dir / 'word_count_boxplot.png').exists():
                st.image(str(results_dir / 'word_count_boxplot.png'), caption="Word Count Boxplot by News Category")
        
        if (results_dir / 'wordclouds_combined.png').exists():
            st.image(str(results_dir / 'wordclouds_combined.png'), caption="Comparative Word Clouds: Fake News (Left) vs Real News (Right)")

    with res_tab2:
        st.markdown("#### Preprocessing Pipeline Comparison: Standard vs. Indian Context")
        if (results_dir / 'preprocessing_comparison.png').exists():
            st.image(str(results_dir / 'preprocessing_comparison.png'), caption="F1-Macro Impact of Indian-Context Preprocessing across Classifiers")
        
        if (results_dir / 'preprocessing_results.json').exists():
            with open(results_dir / 'preprocessing_results.json') as f:
                pre_data = json.load(f)
            st.json(pre_data)

    with res_tab3:
        st.markdown("#### Sparse TF-IDF vs. Dense CBOW Word2Vec: Performance & Latency")
        c1, c2 = st.columns(2)
        with c1:
            if (results_dir / 'feature_comparison_bar.png').exists():
                st.image(str(results_dir / 'feature_comparison_bar.png'), caption="Macro F1-Score across Feature Representations")
        with c2:
            if (results_dir / 'feature_comparison_scatter.png').exists():
                st.image(str(results_dir / 'feature_comparison_scatter.png'), caption="Computational Efficiency vs Classification Performance Trade-off")

    with res_tab4:
        st.markdown("#### Base vs Proposed Model Benchmarking (5-Fold Stratified Cross-Validation)")
        c1, c2 = st.columns(2)
        with c1:
            if (results_dir / 'roc_curves.png').exists():
                st.image(str(results_dir / 'roc_curves.png'), caption="Receiver Operating Characteristic (ROC) Comparison")
            if (results_dir / 'confusion_matrices.png').exists():
                st.image(str(results_dir / 'confusion_matrices.png'), caption="Confusion Matrices Across Tested Models")
        with c2:
            if (results_dir / 'model_comparison.png').exists():
                st.image(str(results_dir / 'model_comparison.png'), caption="Performance Across Accuracy, Precision, Recall, and F1")
            if (results_dir / 'lr_top_features.png').exists():
                st.image(str(results_dir / 'lr_top_features.png'), caption="Most Discriminative Learned Coefficients (Logistic Regression)")

    with res_tab5:
        st.markdown("#### LIME Diagnostic Matrix (True Positives, True Negatives & Error Studies)")
        if (results_dir / 'lime_summary_2x2.png').exists():
            st.image(str(results_dir / 'lime_summary_2x2.png'), caption="2x2 LIME Attribution Grid for Factuality and Misinformation Diagnostics")


# TAB 3: Academic Project Guide & Presentation Outline
with tab3:
    st.subheader("Academic Project Structure & Presentation Guide")
    st.markdown(
        "Below is the complete academic summary directly from `PROJECT_GUIDE.md` for your report and presentation defense:"
    )

    with st.expander("📌 Problem Statement (Indian Context)", expanded=True):
        st.markdown("""
        The proliferation of misinformation across Indian digital ecosystems presents unique challenges:
        - **Linguistic diversity and code-mixing**: Multilingual social media posts combine English with Hindi/regional terms.
        - **Cultural & political nuances**: Indian news contains unique named entities (e.g., 'Aadhaar', 'PIB', 'Modi ji', 'crore').
        - **Black-box distrust**: Opaque deep learning architectures do not provide journalists and fact-checkers with interpretable evidence.
        - **Low latency deployment**: TF-IDF models achieve competitive performance with near-instantaneous inference times compared to large transformer embeddings.
        """)

    with st.expander("🎯 5 Formal Project Objectives"):
        st.markdown("""
        1. **Corpus Characterization**: Analyze lexical, syntactic, and n-gram distributions of Indian fake and real news in the IFND dataset.
        2. **Specialized Preprocessing**: Develop and empirically benchmark an Indian-context preprocessing pipeline against standard NLP workflows.
        3. **Empirical Feature Justification**: Benchmark sparse TF-IDF (unigram, bigram, n-grams) against dense representations (Word2Vec) on performance and latency.
        4. **Comparative Classification**: Implement and benchmark the baseline Multinomial Naive Bayes against proposed regularized Logistic Regression and Linear SVM classifiers using 5-fold cross-validation.
        5. **Transparent Explainability**: Integrate LIME to produce faithful local instance-level attributions and conduct error diagnostics.
        """)

    with st.expander("📊 Suggested 18-Slide Presentation Structure"):
        st.markdown("""
        - **Slide 1**: Title, Author, Guide & Institution
        - **Slide 2**: Introduction & Motivation (Indian Misinformation Landscape)
        - **Slide 3**: Problem Statement & Challenges
        - **Slide 4**: Research Objectives
        - **Slide 5**: Literature Review & Research Gaps
        - **Slide 6**: Proposed System Architecture
        - **Slide 7**: Dataset Description (IFND)
        - **Slide 8**: Experiment 1 - Exploratory Data Analysis & Lexical Diversity
        - **Slide 9**: Experiment 2 - Preprocessing Comparison (Standard vs. Indian Context)
        - **Slide 10**: Experiment 3 - Feature Representation Comparison (TF-IDF vs Embeddings)
        - **Slide 11**: Experiment 4 - Base vs Proposed Classifier Evaluation
        - **Slide 12**: Experiment 4 - ROC & Confusion Matrix Analysis
        - **Slide 13**: Experiment 5 - Explainable AI via LIME (True Classifications)
        - **Slide 14**: Experiment 5 - LIME Error Diagnostics (False Alarms & Misses)
        - **Slide 15**: Interactive Demonstration Overview (Streamlit UI)
        - **Slide 16**: Critical Discussion & Limitations
        - **Slide 17**: Future Work (Multilingual Indic-BERT extension)
        - **Slide 18**: Conclusion & References
        """)
