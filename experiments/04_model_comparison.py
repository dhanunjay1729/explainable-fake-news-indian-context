"""
Experiment 4: Base Model vs. Proposed Model Comparison.
Evaluates Multinomial Naive Bayes (Baseline) against Logistic Regression and Linear SVM (Proposed),
including 5-fold Stratified Cross-Validation, ROC curves, PR curves, Confusion Matrices, and coefficient analysis.
"""

import sys
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)

from src.dataset_loader import load_ifnd_dataset
from src.preprocessing import IndianContextPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_model_comparison():
    logger.info("=== Starting Experiment 4: Base Model vs. Proposed Model Comparison ===")
    df = load_ifnd_dataset(data_dir=str(PROJECT_ROOT / 'data'))
    
    preprocessor = IndianContextPreprocessor()
    raw_texts = df['text'].tolist()
    labels = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        raw_texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    models = {
        'Multinomial Naive Bayes (Baseline)': MultinomialNB(alpha=1.0),
        'Logistic Regression (Proposed 1)': LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42),
        'Linear SVM (Proposed 2)': CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, class_weight='balanced', random_state=42), cv=3
        )
    }

    results = {}
    fitted_pipelines = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in models.items():
        logger.info(f"Training and Evaluating: {name}...")
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                preprocessor=preprocessor.preprocess,
                ngram_range=(1, 2),
                max_features=20000,
                sublinear_tf=True
            )),
            ('classifier', clf)
        ])

        # 5-Fold Stratified Cross-Validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)

        # Fit on full training set
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predictions on test set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1_mac = f1_score(y_test, y_pred, average='macro')
        f1_wt = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_macro': round(float(f1_mac), 4),
            'f1_weighted': round(float(f1_wt), 4),
            'roc_auc': round(float(auc), 4),
            'cv_mean': round(float(np.mean(cv_scores)), 4),
            'cv_std': round(float(np.std(cv_scores)), 4),
            'confusion_matrix': cm,
            'y_pred': y_pred.tolist(),
            'y_proba': y_proba.tolist() if y_proba is not None else []
        }

    # Print summary table
    print("\n" + "="*95)
    print(f"{'Model':<35} | {'Acc':<7} | {'Prec':<7} | {'Recall':<7} | {'F1-Mac':<7} | {'AUC':<7} | {'5-Fold CV F1':<12}")
    print("="*95)
    for name, m in results.items():
        print(f"{name:<35} | {m['accuracy']:<7.4f} | {m['precision']:<7.4f} | {m['recall']:<7.4f} | "
              f"{m['f1_macro']:<7.4f} | {m['roc_auc']:<7.4f} | {m['cv_mean']:.4f}±{m['cv_std']:.4f}")
    print("="*95 + "\n")

    # 1. Confusion Matrices (3 side-by-side heatmaps)
    logger.info("Plotting Confusion Matrices...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, (name, m) in enumerate(results.items()):
        cm = np.array(m['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i],
                    xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'],
                    annot_kws={'size': 13, 'weight': 'bold'})
        short_name = name.split(' (')[0]
        axes[i].set_title(f"{short_name}\nAcc: {m['accuracy']*100:.1f}%", fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Predicted Label', fontsize=10)
        axes[i].set_ylabel('True Label' if i == 0 else '', fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'confusion_matrices.png', dpi=300)
    plt.close()

    # 2. ROC Curves (Combined)
    logger.info("Plotting ROC Curves...")
    plt.figure(figsize=(8, 6))
    colors = ['#ff7675', '#0984e3', '#00b894']
    for i, (name, m) in enumerate(results.items()):
        if m['y_proba']:
            fpr, tpr, _ = roc_curve(y_test, m['y_proba'])
            plt.plot(fpr, tpr, color=colors[i], lw=2.2, label=f"{name} (AUC = {m['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], color='#636e72', linestyle='--', lw=1.2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'roc_curves.png', dpi=300)
    plt.close()

    # 3. Precision-Recall Curves
    logger.info("Plotting Precision-Recall Curves...")
    plt.figure(figsize=(8, 6))
    for i, (name, m) in enumerate(results.items()):
        if m['y_proba']:
            precision, recall, _ = precision_recall_curve(y_test, m['y_proba'])
            plt.plot(recall, precision, color=colors[i], lw=2.2, label=f"{name}")
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.title('Precision-Recall Curves Comparison', fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'pr_curves.png', dpi=300)
    plt.close()

    # 4. Metrics Comparison Grouped Bar Chart
    logger.info("Plotting Overall Metrics Comparison Bar Chart...")
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_macro', 'roc_auc']
    plot_rows = []
    for name, m in results.items():
        short_name = name.split(' (')[0]
        for met in metrics_to_plot:
            plot_rows.append({'Model': short_name, 'Metric': met.replace('_', ' ').title(), 'Score': m[met]})
    plot_df = pd.DataFrame(plot_rows)

    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(x='Metric', y='Score', hue='Model', data=plot_df, palette=['#ff7675', '#0984e3', '#00b894'],
                     edgecolor='black', linewidth=0.5)
    plt.title('Classification Performance across Evaluation Metrics', fontsize=13, fontweight='bold', pad=15)
    plt.ylim(0.0, 1.1)
    plt.ylabel('Score (0.0 - 1.0)', fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'model_comparison.png', dpi=300)
    plt.close()

    # 5. Top Logistic Regression Feature Coefficients
    logger.info("Plotting Top LR Feature Coefficients...")
    lr_pipe = fitted_pipelines['Logistic Regression (Proposed 1)']
    vec = lr_pipe.named_steps['vectorizer']
    clf = lr_pipe.named_steps['classifier']
    feature_names = vec.get_feature_names_out()
    coef = clf.coef_[0]

    top_real_idx = np.argsort(coef)[-15:]
    top_fake_idx = np.argsort(coef)[:15]

    top_features = pd.concat([
        pd.DataFrame({'word': feature_names[top_fake_idx], 'coefficient': coef[top_fake_idx], 'class': 'Predicts Fake'}),
        pd.DataFrame({'word': feature_names[top_real_idx], 'coefficient': coef[top_real_idx], 'class': 'Predicts Real'})
    ])

    plt.figure(figsize=(9, 7))
    colors_dict = {'Predicts Fake': '#ff6b6b', 'Predicts Real': '#1dd1a1'}
    sns.barplot(x='coefficient', y='word', hue='class', data=top_features, palette=colors_dict, edgecolor='black', linewidth=0.5)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title('Most Discriminative TF-IDF N-grams in Indian News (Logistic Regression)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Learned Log-Odds Coefficient', fontsize=11)
    plt.ylabel('N-gram Feature', fontsize=11)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'lr_top_features.png', dpi=300)
    plt.close()

    # Clean predictions arrays before saving JSON to keep it compact
    save_results = {}
    for k, v in results.items():
        save_results[k] = {ik: iv for ik, iv in v.items() if ik not in ['y_pred', 'y_proba']}
    with open(RESULTS_DIR / 'model_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    # Save trained pipelines to models/ directory
    models_dir = PROJECT_ROOT / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_pipelines['Multinomial Naive Bayes (Baseline)'], models_dir / 'base_naive_bayes_pipeline.joblib')
    joblib.dump(fitted_pipelines['Logistic Regression (Proposed 1)'], models_dir / 'proposed_logistic_regression_pipeline.joblib')
    joblib.dump(fitted_pipelines['Linear SVM (Proposed 2)'], models_dir / 'proposed_linear_svm_pipeline.joblib')
    logger.info(f"Model pipelines successfully saved to: {models_dir}")

    logger.info(f"Experiment 4 complete. Saved to: {RESULTS_DIR / 'model_results.json'}")


if __name__ == '__main__':
    run_model_comparison()
