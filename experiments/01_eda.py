"""
Experiment 1: Exploratory Data Analysis (EDA) on Indian Fake News Dataset (IFND).
Analyzes class balance, length distribution, lexical diversity, n-gram frequencies, and word clouds.
"""

import sys
import re
import json
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

from src.dataset_loader import load_ifnd_dataset

# Setup logging & styling
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set_theme(style='whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_eda():
    logger.info("=== Starting Experiment 1: Exploratory Data Analysis ===")
    df = load_ifnd_dataset(data_dir=str(PROJECT_ROOT / 'data'))
    
    # Text length, word counts, sentence counts, and paragraph counts
    def count_sentences(t):
        s = [x for x in re.split(r'[.!?]+', str(t)) if x.strip()]
        return max(len(s), 1)

    def count_paragraphs(t):
        p = [x for x in str(t).split('\n') if x.strip()]
        return max(len(p), 1)

    df['char_length'] = df['text'].astype(str).str.len()
    df['word_count'] = df['text'].astype(str).str.split().str.len()
    df['sentence_count'] = df['text'].apply(count_sentences)
    df['paragraph_count'] = df['text'].apply(count_paragraphs)

    fake_df = df[df['label'] == 0]
    real_df = df[df['label'] == 1]

    stats = {
        'total_samples': int(len(df)),
        'fake_samples': int(len(fake_df)),
        'real_samples': int(len(real_df)),
        'fake_ratio': round(float(len(fake_df) / len(df)), 4),
        'char_length_fake': {
            'mean': round(float(fake_df['char_length'].mean()), 2),
            'median': round(float(fake_df['char_length'].median()), 2),
            'std': round(float(fake_df['char_length'].std()), 2)
        },
        'char_length_real': {
            'mean': round(float(real_df['char_length'].mean()), 2),
            'median': round(float(real_df['char_length'].median()), 2),
            'std': round(float(real_df['char_length'].std()), 2)
        },
        'word_count_fake': {
            'mean': round(float(fake_df['word_count'].mean()), 2),
            'median': round(float(fake_df['word_count'].median()), 2),
            'std': round(float(fake_df['word_count'].std()), 2)
        },
        'word_count_real': {
            'mean': round(float(real_df['word_count'].mean()), 2),
            'median': round(float(real_df['word_count'].median()), 2),
            'std': round(float(real_df['word_count'].std()), 2)
        },
        'sentence_count_fake': {
            'mean': round(float(fake_df['sentence_count'].mean()), 2),
            'median': round(float(fake_df['sentence_count'].median()), 2),
            'std': round(float(fake_df['sentence_count'].std()), 2)
        },
        'sentence_count_real': {
            'mean': round(float(real_df['sentence_count'].mean()), 2),
            'median': round(float(real_df['sentence_count'].median()), 2),
            'std': round(float(real_df['sentence_count'].std()), 2)
        },
        'paragraph_count_fake': {
            'mean': round(float(fake_df['paragraph_count'].mean()), 2),
            'median': round(float(fake_df['paragraph_count'].median()), 2),
            'std': round(float(fake_df['paragraph_count'].std()), 2)
        },
        'paragraph_count_real': {
            'mean': round(float(real_df['paragraph_count'].mean()), 2),
            'median': round(float(real_df['paragraph_count'].median()), 2),
            'std': round(float(real_df['paragraph_count'].std()), 2)
        }
    }

    logger.info(f"Summary Statistics: {json.dumps(stats, indent=2)}")

    # 1. Class Distribution Pie Chart
    logger.info("Plotting Class Distribution...")
    plt.figure(figsize=(6, 5))
    labels = ['Fake News', 'Real News']
    sizes = [len(fake_df), len(real_df)]
    colors = ['#ff6b6b', '#1dd1a1']
    explode = (0.05, 0)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=140, shadow=True, textprops={'fontsize': 11, 'weight': 'bold'})
    plt.title('IFND Dataset Class Distribution', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'class_distribution.png', dpi=300)
    plt.close()

    # 2. Article Length Distribution Histogram
    logger.info("Plotting Article Length Histogram...")
    plt.figure(figsize=(9, 5))
    sns.histplot(fake_df['char_length'], color='#ff6b6b', label='Fake News', kde=True, bins=30, alpha=0.6)
    sns.histplot(real_df['char_length'], color='#10ac84', label='Real News', kde=True, bins=30, alpha=0.6)
    plt.title('Character Length Distribution: Fake vs Real News', fontsize=13, fontweight='bold')
    plt.xlabel('Character Count', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'article_length_hist.png', dpi=300)
    plt.close()

    # 3. Word Count Boxplot
    logger.info("Plotting Word Count Boxplot...")
    plt.figure(figsize=(7, 5))
    df_plot = df.copy()
    df_plot['Category'] = df_plot['label'].map({0: 'Fake News', 1: 'Real News'})
    sns.boxplot(x='Category', y='word_count', data=df_plot, palette=['#ff6b6b', '#1dd1a1'], width=0.4)
    plt.title('Word Count Comparison by Class', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Words', fontsize=11)
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'word_count_boxplot.png', dpi=300)
    plt.close()

    # 4. Top-20 Unigrams Bar Chart
    logger.info("Extracting and Plotting Top Unigrams...")
    vec = CountVectorizer(stop_words='english', max_features=20)
    
    vec_fake = CountVectorizer(stop_words='english', max_features=15).fit(fake_df['text'])
    fake_words = pd.DataFrame({
        'word': vec_fake.get_feature_names_out(),
        'count': vec_fake.transform(fake_df['text']).toarray().sum(axis=0)
    }).sort_values('count', ascending=True)

    vec_real = CountVectorizer(stop_words='english', max_features=15).fit(real_df['text'])
    real_words = pd.DataFrame({
        'word': vec_real.get_feature_names_out(),
        'count': vec_real.transform(real_df['text']).toarray().sum(axis=0)
    }).sort_values('count', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(fake_words['word'], fake_words['count'], color='#ff6b6b', edgecolor='black', linewidth=0.5)
    axes[0].set_title('Top Words in Fake News (Indian Context)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Frequency', fontsize=10)

    axes[1].barh(real_words['word'], real_words['count'], color='#10ac84', edgecolor='black', linewidth=0.5)
    axes[1].set_title('Top Words in Real News (Indian Context)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Frequency', fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'top_unigrams.png', dpi=300)
    plt.close()

    # 5. Top-20 Bigrams Bar Chart
    logger.info("Extracting and Plotting Top Bigrams...")
    vec_fake_bi = CountVectorizer(stop_words='english', ngram_range=(2, 2), max_features=15).fit(fake_df['text'])
    fake_bi = pd.DataFrame({
        'bigram': vec_fake_bi.get_feature_names_out(),
        'count': vec_fake_bi.transform(fake_df['text']).toarray().sum(axis=0)
    }).sort_values('count', ascending=True)

    vec_real_bi = CountVectorizer(stop_words='english', ngram_range=(2, 2), max_features=15).fit(real_df['text'])
    real_bi = pd.DataFrame({
        'bigram': vec_real_bi.get_feature_names_out(),
        'count': vec_real_bi.transform(real_df['text']).toarray().sum(axis=0)
    }).sort_values('count', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].barh(fake_bi['bigram'], fake_bi['count'], color='#ee5253', edgecolor='black', linewidth=0.5)
    axes[0].set_title('Top Bigrams in Fake News', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Frequency', fontsize=10)

    axes[1].barh(real_bi['bigram'], real_bi['count'], color='#01a3a4', edgecolor='black', linewidth=0.5)
    axes[1].set_title('Top Bigrams in Real News', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Frequency', fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'top_bigrams.png', dpi=300)
    plt.close()

    # 6. Word Clouds
    try:
        from wordcloud import WordCloud
        logger.info("Generating Word Clouds...")
        wc_fake = WordCloud(width=800, height=450, background_color='white', colormap='Reds').generate(' '.join(fake_df['text']))
        wc_real = WordCloud(width=800, height=450, background_color='white', colormap='Greens').generate(' '.join(real_df['text']))

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes[0].imshow(wc_fake, interpolation='bilinear')
        axes[0].axis('off')
        axes[0].set_title('Fake News Lexical Cloud (Indian Context)', fontsize=14, fontweight='bold', pad=10)

        axes[1].imshow(wc_real, interpolation='bilinear')
        axes[1].axis('off')
        axes[1].set_title('Real News Lexical Cloud (Indian Context)', fontsize=14, fontweight='bold', pad=10)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'wordclouds_combined.png', dpi=300)
        plt.close()
    except ImportError:
        logger.warning("WordCloud library not installed; skipping word cloud graphic.")

    # Save summary stats
    with open(RESULTS_DIR / 'eda_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Experiment 1 complete. Plots and stats saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    run_eda()
