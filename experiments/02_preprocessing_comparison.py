"""
Experiment 2: Preprocessing Comparison.
Empirically benchmarks Standard English Preprocessing vs. Specialized Indian-Context Preprocessing
across Multinomial Naive Bayes, Logistic Regression, and Linear SVM classifiers.
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.dataset_loader import load_ifnd_dataset
from src.preprocessing import StandardPreprocessor, IndianContextPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_models_on_text(X_train, X_test, y_train, y_test):
    """Vectorizes with TF-IDF (1,2-grams) and evaluates 3 models."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
    X_tr_tfidf = vectorizer.fit_transform(X_train)
    X_te_tfidf = vectorizer.transform(X_test)

    models = {
        'Multinomial NB': MultinomialNB(alpha=1.0),
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        'Linear SVM': CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, random_state=42), cv=3)
    }

    metrics = {}
    for name, clf in models.items():
        clf.fit(X_tr_tfidf, y_train)
        y_pred = clf.predict(X_te_tfidf)
        y_proba = clf.predict_proba(X_te_tfidf)[:, 1] if hasattr(clf, 'predict_proba') else None

        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
        metrics[name] = {
            'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
            'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            'recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            'f1_macro': round(float(f1_score(y_test, y_pred, average='macro')), 4),
            'f1_weighted': round(float(f1_score(y_test, y_pred, average='weighted')), 4),
            'roc_auc': round(float(auc), 4)
        }
    return metrics


def run_preprocessing_comparison():
    logger.info("=== Starting Experiment 2: Preprocessing Comparison ===")
    df = load_ifnd_dataset(data_dir=str(PROJECT_ROOT / 'data'))
    raw_texts = df['text'].tolist()
    labels = df['label'].values

    logger.info("Running Standard Preprocessing Pipeline...")
    std_prep = StandardPreprocessor()
    std_texts = std_prep.preprocess_batch(raw_texts)

    logger.info("Running Indian-Context Preprocessing Pipeline...")
    ind_prep = IndianContextPreprocessor()
    ind_texts = ind_prep.preprocess_batch(raw_texts)

    # Consistent split
    X_tr_std, X_te_std, y_train, y_test = train_test_split(
        std_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_tr_ind, X_te_ind, _, _ = train_test_split(
        ind_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    logger.info("Evaluating models on Standard Preprocessed data...")
    std_results = evaluate_models_on_text(X_tr_std, X_te_std, y_train, y_test)

    logger.info("Evaluating models on Indian-Context Preprocessed data...")
    ind_results = evaluate_models_on_text(X_tr_ind, X_te_ind, y_train, y_test)

    # Compile comparison
    comparison = {}
    print("\n" + "="*80)
    print(f"{'Model':<22} | {'Standard F1':<12} | {'Indian-Context F1':<18} | {'Delta (Δ)':<10}")
    print("="*80)

    plot_data = []
    for model_name in std_results.keys():
        f1_std = std_results[model_name]['f1_macro']
        f1_ind = ind_results[model_name]['f1_macro']
        delta = round(f1_ind - f1_std, 4)
        delta_pct = f"{'+' if delta >= 0 else ''}{delta*100:.2f}%"

        comparison[model_name] = {
            'standard': std_results[model_name],
            'indian_context': ind_results[model_name],
            'delta_f1_macro': delta,
            'delta_pct': delta_pct
        }
        print(f"{model_name:<22} | {f1_std:<12.4f} | {f1_ind:<18.4f} | {delta_pct:<10}")

        plot_data.append({'Model': model_name, 'Pipeline': 'Standard Preprocessing', 'F1 Score': f1_std})
        plot_data.append({'Model': model_name, 'Pipeline': 'Indian-Context Preprocessing', 'F1 Score': f1_ind})
    print("="*80 + "\n")

    # Grouped Bar Chart
    logger.info("Generating Preprocessing Comparison Bar Chart...")
    plot_df = pd.DataFrame(plot_data)
    plt.figure(figsize=(9, 5))
    palette = {'Standard Preprocessing': '#747d8c', 'Indian-Context Preprocessing': '#2ed573'}
    ax = sns.barplot(x='Model', y='F1 Score', hue='Pipeline', data=plot_df, palette=palette, edgecolor='black', linewidth=0.6)
    plt.title('Impact of Indian-Context Preprocessing on Macro F1 Score', fontsize=13, fontweight='bold', pad=15)
    plt.ylim(0.0, 1.05)
    plt.ylabel('Macro F1 Score', fontsize=11)
    plt.xlabel('Classification Model', fontsize=11)
    plt.legend(title='Preprocessing Pipeline', frameon=True)

    # Annotate bar values
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                        textcoords='offset points')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'preprocessing_comparison.png', dpi=300)
    plt.close()

    # Save JSON results
    with open(RESULTS_DIR / 'preprocessing_results.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Experiment 2 complete. Saved to: {RESULTS_DIR / 'preprocessing_results.json'}")


if __name__ == '__main__':
    run_preprocessing_comparison()
