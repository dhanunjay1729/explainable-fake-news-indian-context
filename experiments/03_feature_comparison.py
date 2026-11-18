"""
Experiment 3: Feature Representation Comparison.
Empirically justifies TF-IDF sparse features against dense semantic embeddings (Word2Vec, Sentence-BERT)
by benchmarking Classification Performance (F1) vs. Computational Latency.
"""

import sys
import json
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.dataset_loader import load_ifnd_dataset
from src.preprocessing import IndianContextPreprocessor
from src.feature_extraction import extract_tfidf_features, extract_word2vec_features, extract_sbert_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_feature_comparison():
    logger.info("=== Starting Experiment 3: Feature Representation Comparison ===")
    df = load_ifnd_dataset(data_dir=str(PROJECT_ROOT / 'data'))
    
    preprocessor = IndianContextPreprocessor()
    processed_texts = preprocessor.preprocess_batch(df['text'].tolist())
    labels = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    feature_methods = {}

    # 1. TF-IDF Unigrams
    logger.info("Extracting TF-IDF (Unigrams)...")
    X_tr, X_te, _, dur = extract_tfidf_features(X_train, X_test, ngram_range=(1, 1), max_features=10000)
    feature_methods['TF-IDF (Unigram)'] = {'train': X_tr, 'test': X_te, 'feat_time': dur, 'dim': 10000}

    # 2. TF-IDF Bigrams
    logger.info("Extracting TF-IDF (Bigrams)...")
    X_tr, X_te, _, dur = extract_tfidf_features(X_train, X_test, ngram_range=(2, 2), max_features=15000)
    feature_methods['TF-IDF (Bigram)'] = {'train': X_tr, 'test': X_te, 'feat_time': dur, 'dim': 15000}

    # 3. TF-IDF Combined (1,2-grams)
    logger.info("Extracting TF-IDF (1,2-grams)...")
    X_tr, X_te, _, dur = extract_tfidf_features(X_train, X_test, ngram_range=(1, 2), max_features=20000)
    feature_methods['TF-IDF (1-2 gram)'] = {'train': X_tr, 'test': X_te, 'feat_time': dur, 'dim': 20000}

    # 4. Dense Word2Vec (100d)
    logger.info("Extracting Word2Vec Document Embeddings (100d)...")
    X_tr, X_te, _, dur = extract_word2vec_features(X_train, X_test, vector_size=100)
    feature_methods['Word2Vec (Avg 100d)'] = {'train': X_tr, 'test': X_te, 'feat_time': dur, 'dim': 100}

    # 5. Sentence-BERT (Optional / Fallback)
    logger.info("Attempting Sentence-BERT Encoding...")
    X_tr, X_te, _, dur = extract_sbert_features(X_train, X_test)
    if X_tr is not None:
        feature_methods['Sentence-BERT (MiniLM)'] = {'train': X_tr, 'test': X_te, 'feat_time': dur, 'dim': 384}

    # Train Logistic Regression on each representation
    results = {}
    print("\n" + "="*85)
    print(f"{'Feature Representation':<26} | {'Dim':<7} | {'Accuracy':<9} | {'F1 Macro':<9} | {'Total Time (s)':<14}")
    print("="*85)

    for name, data in feature_methods.items():
        start_train = time.time()
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(data['train'], y_train)
        train_dur = time.time() - start_train

        y_pred = clf.predict(data['test'])
        y_proba = clf.predict_proba(data['test'])[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, y_proba)
        total_time = data['feat_time'] + train_dur

        results[name] = {
            'dimensionality': data['dim'],
            'accuracy': round(float(acc), 4),
            'f1_macro': round(float(f1), 4),
            'roc_auc': round(float(auc), 4),
            'feature_extraction_time': round(float(data['feat_time']), 4),
            'model_training_time': round(float(train_dur), 4),
            'total_time_seconds': round(float(total_time), 4)
        }

        print(f"{name:<26} | {data['dim']:<7} | {acc:<9.4f} | {f1:<9.4f} | {total_time:<14.2f}")
    print("="*85 + "\n")

    # 1. Bar Chart: F1 Macro Comparison
    logger.info("Plotting Feature Comparison Bar Chart...")
    plt.figure(figsize=(9, 5))
    names = list(results.keys())
    f1_scores = [results[n]['f1_macro'] for n in names]
    colors = ['#54a0ff' if 'TF-IDF' in n else '#ff9f43' for n in names]

    bars = plt.bar(names, f1_scores, color=colors, edgecolor='black', linewidth=0.6, width=0.5)
    plt.title('Macro F1 Score across Feature Representations (Logistic Regression)', fontsize=12, fontweight='bold', pad=15)
    plt.ylabel('Macro F1 Score', fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.xticks(rotation=15, ha='right', fontsize=10)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.4f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'feature_comparison_bar.png', dpi=300)
    plt.close()

    # 2. Scatter Plot: F1 Score vs Training Latency (Trade-off demonstration)
    logger.info("Plotting Efficiency-Accuracy Trade-off Scatter Plot...")
    plt.figure(figsize=(9, 6))
    times = [results[n]['total_time_seconds'] for n in names]
    
    plt.scatter(times, f1_scores, s=150, c=['#1dd1a1' if 'TF-IDF' in n else '#ff6b6b' for n in names],
                edgecolor='black', linewidth=1.2, zorder=5)

    for i, txt in enumerate(names):
        offset_x = 0.05 * (max(times) - min(times) + 0.1)
        plt.annotate(txt, (times[i] + offset_x, f1_scores[i]), fontsize=10, fontweight='bold')

    plt.title('Empirical Justification of TF-IDF: F1 Score vs. Computational Latency', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Total Pipeline Time (Feature Extraction + Training) [Seconds]', fontsize=11)
    plt.ylabel('Macro F1 Score', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'feature_comparison_scatter.png', dpi=300)
    plt.close()

    # Save JSON results
    with open(RESULTS_DIR / 'feature_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Experiment 3 complete. Saved to: {RESULTS_DIR / 'feature_results.json'}")


if __name__ == '__main__':
    run_feature_comparison()
