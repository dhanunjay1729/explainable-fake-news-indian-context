"""
Explainable Fake News Detection in Indian Context
Source package initialization.
"""

from .preprocessing import StandardPreprocessor, IndianContextPreprocessor
from .feature_extraction import extract_tfidf_features, extract_word2vec_features, extract_sbert_features
from .explainability import LIMEExplainer
from .dataset_loader import load_ifnd_dataset

__all__ = [
    'StandardPreprocessor',
    'IndianContextPreprocessor',
    'extract_tfidf_features',
    'extract_word2vec_features',
    'extract_sbert_features',
    'LIMEExplainer',
    'load_ifnd_dataset',
]
