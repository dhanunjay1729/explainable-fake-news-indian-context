"""
Text Preprocessing Module for Fake News Detection in the Indian Context.
Contains:
1. StandardPreprocessor - Traditional English NLP pipeline
2. IndianContextPreprocessor - Domain-aware Indian English/Hinglish pipeline
"""

import re
import string
import logging
from typing import List, Dict, Optional, Set

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Configure logging
logger = logging.getLogger(__name__)

# Ensure NLTK data dependencies are loaded silently
def _ensure_nltk_resources():
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except Exception as e:
            logger.debug(f"NLTK download notice for {r}: {e}")

_ensure_nltk_resources()


class StandardPreprocessor:
    """
    Standard generic English text preprocessor.
    Used as the comparative baseline for preprocessing evaluation.
    
    Steps:
    1. Lowercasing
    2. HTML tag and URL removal
    3. Punctuation and numeric character removal
    4. Tokenization (NLTK word_tokenize)
    5. Standard NLTK English stopword removal
    6. WordNet Lemmatization
    """

    def __init__(self, min_token_len: int = 2):
        self.min_token_len = min_token_len
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words: Set[str] = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """Removes URLs, HTML tags, punctuation, and non-alphabetic characters."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Lowercase
        text = text.lower()
        # Keep only alphabetic characters and whitespace
        text = re.sub(r'[^a-z\s]', ' ', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess(self, text: str) -> str:
        """Processes a single text instance."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return ""

        try:
            tokens = word_tokenize(cleaned)
        except Exception:
            tokens = cleaned.split()

        filtered_tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if len(t) >= self.min_token_len and t not in self.stop_words
        ]
        return " ".join(filtered_tokens)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Processes a list of text instances."""
        return [self.preprocess(t) for t in texts]


class IndianContextPreprocessor:
    """
    Domain-Specific Indian Context Preprocessor.
    Tailored for Indian English, Hinglish transliterations, news jargon,
    and regional socio-political entities.

    Enhancements over Standard Preprocessing:
    1. Socio-political and Indian English abbreviation expansion.
    2. Indian filler and transliterated Hinglish stopword removal (e.g. ji, sahab, hai, ki, ke, lakh, crore).
    3. Removal of viral social media forward noise and clickbait hooks.
    4. Entity-safe lemmatization preventing corruption of regional proper nouns.
    """

    # Socio-political expansions & common transliterated contractions
    INDIAN_NORMALIZATIONS: Dict[str, str] = {
        'govt': 'government',
        'pm': 'prime minister',
        'cm': 'chief minister',
        'bjp': 'bharatiya janata party',
        'cong': 'congress',
        'inc': 'indian national congress',
        'aap': 'aam aadmi party',
        'tmc': 'trinamool congress',
        'rss': 'rashtriya swayamsevak sangh',
        'cbi': 'central bureau investigation',
        'ed': 'enforcement directorate',
        'sc': 'supreme court',
        'hc': 'high court',
        'caa': 'citizenship amendment act',
        'nrc': 'national register citizens',
        'rti': 'right to information',
        'mla': 'member legislative assembly',
        'mp': 'member parliament',
        'delhi': 'delhi',
        'mumbai': 'mumbai',
        'modi': 'narendra modi',
        'rahul': 'rahul gandhi',
        'doesnt': 'does not',
        'dont': 'do not',
        'cant': 'cannot',
        'wont': 'will not',
        'isnt': 'is not',
        'wasnt': 'was not',
        'shouldnt': 'should not',
        'couldnt': 'could not',
        'didnt': 'did not',
    }

    # Transliterated Hindi stopwords and grammatical fillers
    INDIAN_STOPWORDS: Set[str] = {
        # Honorifics & relational fillers
        'ji', 'sahab', 'sahib', 'shri', 'smt', 'wala', 'wali', 'wale',
        # Hindi transliterated connectives / pronouns
        'hai', 'hain', 'tha', 'thi', 'the', 'nahi', 'nahin', 'mat',
        'aur', 'bhi', 'toh', 'yeh', 'woh', 'kya', 'kab', 'kaise', 'kyun',
        'abhi', 'par', 'mein', 'hum', 'aap', 'unke', 'unki', 'iske', 'uske',
        'sab', 'kuch', 'hoga', 'hogi', 'raha', 'rahi', 'rahe',
        # Generic broadcast filler (neutral reporting verbs)
        'said', 'says', 'told', 'reported', 'according'
    }

    # Known Indian political, economic & geographical proper nouns to preserve without aggressive stemming
    PRESERVED_ENTITIES: Set[str] = {
        'modi', 'gandhi', 'kejriwal', 'shah', 'yogi', 'mamata', 'pawar',
        'delhi', 'kashmir', 'punjab', 'bengal', 'kerala', 'up', 'bihar',
        'gujarat', 'maharashtra', 'ayodhya', 'hindu', 'muslim', 'dalit',
        'bharat', 'india', 'pakistan', 'china',
        'lakh', 'crore', 'aadhaar', 'rbi', 'isro', 'aiims', 'upi'
    }

    def __init__(self, min_token_len: int = 2):
        self.min_token_len = min_token_len
        self.lemmatizer = WordNetLemmatizer()

        # Combine standard English stopwords with specialized Indian stopwords
        try:
            standard_stops = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            standard_stops = set(stopwords.words('english'))

        self.all_stopwords: Set[str] = standard_stops.union(self.INDIAN_STOPWORDS)

    def normalize_abbreviations(self, text: str) -> str:
        """Expands Indian socio-political abbreviations and contractions using word boundaries."""
        words = text.split()
        normalized_words = [self.INDIAN_NORMALIZATIONS.get(w, w) for w in words]
        return " ".join(normalized_words)

    def clean_text(self, text: str) -> str:
        """Applies domain-aware sanitization."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # Remove URLs and web references
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        # Remove HTML artifacts
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove WhatsApp forward indicators (e.g. *Forwarded as received*, [01/01, 10:00])
        text = re.sub(r'forwarded\s+as\s+received', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\d{1,2}/\d{1,2}.*?\]', ' ', text)
        # Lowercase
        text = text.lower()
        # Normalization of abbreviations
        text = self.normalize_abbreviations(text)
        # Keep only alphabetic characters and single spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess(self, text: str) -> str:
        """Full Indian-context preprocessing pipeline."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return ""

        try:
            tokens = word_tokenize(cleaned)
        except Exception:
            tokens = cleaned.split()

        processed_tokens: List[str] = []
        for token in tokens:
            if len(token) < self.min_token_len:
                continue
            if token in self.all_stopwords:
                continue
            # Preserve critical entities directly without lemmatizing
            if token in self.PRESERVED_ENTITIES:
                processed_tokens.append(token)
            else:
                processed_tokens.append(self.lemmatizer.lemmatize(token))

        return " ".join(processed_tokens)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Batch processing helper."""
        return [self.preprocess(t) for t in texts]

    def get_reduction_stats(self, original_text: str, processed_text: str) -> Dict[str, float]:
        """Computes compression/filtering metrics for evaluation."""
        orig_words = len(original_text.split())
        proc_words = len(processed_text.split())
        reduction = (1.0 - (proc_words / max(orig_words, 1))) * 100.0
        return {
            'original_word_count': orig_words,
            'processed_word_count': proc_words,
            'reduction_percentage': round(reduction, 2)
        }
