"""
Feature Extraction Module for Fake News Detection.
Implements:
1. TF-IDF feature representations (Unigram, Bigram, Combined n-grams)
2. Dense Word2Vec document embeddings (Gensim averaged vectors)
3. Dense Sentence-BERT contextual embeddings (sentence-transformers)
"""

import time
import logging
from typing import Tuple, Optional, Any, Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def extract_tfidf_features(
    X_train: List[str],
    X_test: List[str],
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 20000,
    sublinear_tf: bool = True,
    min_df: int = 2,
    max_df: float = 0.95
) -> Tuple[Any, Any, TfidfVectorizer, float]:
    """
    Extracts sparse TF-IDF n-gram feature matrices.

    Returns:
        X_train_tfidf: Sparse matrix of training features
        X_test_tfidf: Sparse matrix of test features
        vectorizer: Fitted TfidfVectorizer instance
        duration: Time taken in seconds for fit and transform
    """
    start_time = time.time()
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        max_df=max_df
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    duration = time.time() - start_time

    return X_train_tfidf, X_test_tfidf, vectorizer, duration


def extract_word2vec_features(
    X_train: List[str],
    X_test: List[str],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 15
) -> Tuple[np.ndarray, np.ndarray, Optional[Any], float]:
    """
    Extracts dense document vectors by averaging Word2Vec word embeddings.

    Returns:
        X_train_w2v: (N_train, vector_size) numpy array
        X_test_w2v: (N_test, vector_size) numpy array
        model: Fitted Gensim Word2Vec model (or None if gensim unavailable)
        duration: Elapsed time in seconds
    """
    start_time = time.time()
    try:
        from gensim.models import Word2Vec
    except ImportError:
        logger.warning("Gensim is not installed. Falling back to zeros for Word2Vec.")
        dummy_train = np.zeros((len(X_train), vector_size))
        dummy_test = np.zeros((len(X_test), vector_size))
        return dummy_train, dummy_test, None, 0.0

    # Tokenize corpus for Gensim
    tokenized_train = [doc.split() for doc in X_train]
    tokenized_test = [doc.split() for doc in X_test]

    # Train Word2Vec CBOW model
    w2v_model = Word2Vec(
        sentences=tokenized_train,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        seed=42
    )

    def doc_to_vector(tokens: List[str]) -> np.ndarray:
        vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if not vectors:
            return np.zeros(vector_size)
        return np.mean(vectors, axis=0)

    X_train_w2v = np.array([doc_to_vector(doc) for doc in tokenized_train])
    X_test_w2v = np.array([doc_to_vector(doc) for doc in tokenized_test])
    duration = time.time() - start_time

    return X_train_w2v, X_test_w2v, w2v_model, duration


def extract_sbert_features(
    X_train: List[str],
    X_test: List[str],
    model_name: str = 'paraphrase-MiniLM-L6-v2',
    batch_size: int = 32
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Any], float]:
    """
    Extracts dense contextual sentence representations using Sentence-BERT.

    Returns:
        X_train_sbert: (N_train, 384) numpy array
        X_test_sbert: (N_test, 384) numpy array
        sbert_model: Loaded SentenceTransformer model
        duration: Elapsed time in seconds
    """
    start_time = time.time()
    try:
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer(model_name)
        X_train_sbert = sbert.encode(X_train, batch_size=batch_size, show_progress_bar=False)
        X_test_sbert = sbert.encode(X_test, batch_size=batch_size, show_progress_bar=False)
        duration = time.time() - start_time
        return np.array(X_train_sbert), np.array(X_test_sbert), sbert, duration
    except Exception as e:
        logger.info(f"Sentence-Transformers not available or skipped ({e}). Proceeding without SBERT.")
        return None, None, None, 0.0
