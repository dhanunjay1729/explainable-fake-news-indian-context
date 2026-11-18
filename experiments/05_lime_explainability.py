"""
Experiment 5: LIME Explainability & Qualitative Error Analysis.
Generates local, instance-level explanations using LIME for four key evaluation case studies:
1. True Fake (Correctly classified fake news)
2. True Real (Correctly classified real news)
3. False Positive (Real news misclassified as fake)
4. False Negative (Fake news misclassified as real)
"""

import sys
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.dataset_loader import load_ifnd_dataset
from src.preprocessing import IndianContextPreprocessor
from src.explainability import LIMEExplainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_lime_analysis():
    logger.info("=== Starting Experiment 5: LIME Explainability & Error Analysis ===")
    df = load_ifnd_dataset(data_dir=str(PROJECT_ROOT / 'data'))
    raw_texts = df['text'].values
    labels = df['label'].values

    # Train/test split on raw texts so LIME can explain the original headline text!
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        raw_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Build end-to-end pipeline with IndianContextPreprocessor custom transformer or vectorizer
    preprocessor = IndianContextPreprocessor()
    
    # We create a pipeline where TF-IDF uses preprocessor's clean function
    vectorizer = TfidfVectorizer(
        preprocessor=preprocessor.preprocess,
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True
    )
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)

    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', clf)
    ])

    logger.info("Training Proposed Logistic Regression Pipeline...")
    pipeline.fit(X_train_raw, y_train)

    y_pred = pipeline.predict(X_test_raw)
    y_proba = pipeline.predict_proba(X_test_raw)

    explainer = LIMEExplainer(pipeline=pipeline, class_names=['Fake', 'Real'])

    # Find candidate indices for the 4 case studies
    # Case 1: True Positive (Fake classified as Fake, highest confidence)
    tp_indices = [i for i in range(len(y_test)) if y_test[i] == 0 and y_pred[i] == 0]
    case1_idx = tp_indices[np.argmax([y_proba[i][0] for i in tp_indices])] if tp_indices else 0

    # Case 2: True Negative (Real classified as Real, highest confidence)
    tn_indices = [i for i in range(len(y_test)) if y_test[i] == 1 and y_pred[i] == 1]
    case2_idx = tn_indices[np.argmax([y_proba[i][1] for i in tn_indices])] if tn_indices else 1

    # Case 3: False Positive (Real classified as Fake - Type I error)
    fp_indices = [i for i in range(len(y_test)) if y_test[i] == 1 and y_pred[i] == 0]
    case3_idx = fp_indices[0] if fp_indices else (1 if case2_idx != 1 else 2)

    # Case 4: False Negative (Fake classified as Real - Type II error)
    fn_indices = [i for i in range(len(y_test)) if y_test[i] == 0 and y_pred[i] == 1]
    case4_idx = fn_indices[0] if fn_indices else (0 if case1_idx != 0 else 3)

    case_studies = [
        {
            'case_num': 1,
            'name': 'Case 1: Correct Fake (True Negative for Misinformation)',
            'idx': case1_idx,
            'expected': 'Fake',
            'desc': 'Identifies emotional, sensationalist, or conspiratorial lexical cues.'
        },
        {
            'case_num': 2,
            'name': 'Case 2: Correct Real (True Positive for Factuality)',
            'idx': case2_idx,
            'expected': 'Real',
            'desc': 'Identifies institutional proper nouns and verified reporting verbiage.'
        },
        {
            'case_num': 3,
            'name': 'Case 3: Error Analysis - False Alarm (Real predicted as Fake)',
            'idx': case3_idx,
            'expected': 'Real',
            'desc': 'Diagnostic evaluation: Sensationalist tokens in legitimate news confusing the classifier.'
        },
        {
            'case_num': 4,
            'name': 'Case 4: Error Analysis - Missed Misinformation (Fake predicted as Real)',
            'idx': case4_idx,
            'expected': 'Fake',
            'desc': 'Diagnostic evaluation: Deceptive mimicry of official administrative syntax.'
        }
    ]

    saved_cases = []

    # Individual analysis and HTML/PNG export
    for c in case_studies:
        text = str(X_test_raw[c['idx']])
        true_label = 'Real' if y_test[c['idx']] == 1 else 'Fake'
        pred_label = 'Real' if y_pred[c['idx']] == 1 else 'Fake'
        probs = y_proba[c['idx']]

        logger.info(f"\n--- {c['name']} ---")
        logger.info(f"Headline: {text[:140]}...")
        logger.info(f"Ground Truth: {true_label} | Predicted: {pred_label} (P(Fake)={probs[0]:.3f}, P(Real)={probs[1]:.3f})")

        html_file = RESULTS_DIR / f"lime_case_{c['case_num']}.html"
        png_file = RESULTS_DIR / f"lime_case_{c['case_num']}.png"

        try:
            explainer.save_html(text, str(html_file), num_features=10)
        except Exception as e:
            logger.warning(f"Could not save HTML for case {c['case_num']}: {e}")

        explainer.plot_instance(
            text, str(png_file),
            title=f"{c['name']}\nTrue: {true_label} | Pred: {pred_label} (Conf: {max(probs)*100:.1f}%)",
            num_features=10
        )

        res = explainer.explain_instance(text, num_features=10)
        saved_cases.append({
            'case_id': c['case_num'],
            'description': c['desc'],
            'text': text,
            'ground_truth': true_label,
            'predicted': pred_label,
            'probability_fake': round(float(probs[0]), 4),
            'probability_real': round(float(probs[1]), 4),
            'top_lime_features': res.get('features', [])
        })

    # Summary 2x2 Grid Figure for Paper/Report
    logger.info("Generating 2x2 LIME Summary Grid Plot...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    for i, c in enumerate(case_studies):
        text = str(X_test_raw[c['idx']])
        true_label = 'Real' if y_test[c['idx']] == 1 else 'Fake'
        pred_label = 'Real' if y_pred[c['idx']] == 1 else 'Fake'
        probs = y_proba[c['idx']]

        res = explainer.explain_instance(text, num_features=8)
        features = res.get('features', [])
        words = [f[0] for f in features]
        weights = [f[1] for f in features]
        colors = ['#2ed573' if w >= 0 else '#ff4757' for w in weights]

        y_pos = np.arange(len(words))
        axes[i].barh(y_pos, weights, color=colors, edgecolor='black', linewidth=0.5)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(words, fontsize=9)
        axes[i].invert_yaxis()
        axes[i].axvline(0, color='black', linestyle='--', linewidth=0.7)
        axes[i].set_title(f"{c['name']}\nTrue: {true_label} | Predicted: {pred_label} ({max(probs)*100:.1f}%)",
                          fontsize=10, fontweight='bold')
        axes[i].set_xlabel('LIME Attribution Weight', fontsize=9)

    plt.suptitle("LIME Local Text Attribution Analysis (Indian News Context)", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'lime_summary_2x2.png', dpi=300)
    plt.close()

    # Save summary JSON
    with open(RESULTS_DIR / 'lime_results.json', 'w') as f:
        json.dump(saved_cases, f, indent=2)
    logger.info(f"Experiment 5 complete. Summary saved to: {RESULTS_DIR / 'lime_results.json'}")


if __name__ == '__main__':
    run_lime_analysis()
