# Explainable Fake News Detection in Indian Context

## Overview

This project implements an **Explainable Fake News Detection** system tailored for the **Indian context**. It builds upon a base fake news classifier by introducing Indian-context-aware preprocessing (handling transliterated words, Indian named entities, and culturally specific patterns), comparing multiple feature representation strategies (TF-IDF, Word2Vec, and optionally Sentence-BERT), and providing model-agnostic explanations using **LIME** (Local Interpretable Model-agnostic Explanations).

The project is structured as a series of five reproducible experiments, each producing quantitative metrics and visualizations saved to the `results/` directory.

---

## Project Structure

```
project/
├── data/                    # Dataset directory (IFND_Dataset.csv)
├── src/                     # Source modules
│   ├── __init__.py
│   ├── preprocessing.py     # Standard + Indian-context preprocessors
│   ├── feature_extraction.py # TF-IDF, Word2Vec, SBERT extractors
│   ├── explainability.py    # LIME explainability module
│   └── dataset_loader.py    # Auto dataset loader & synthesizer
├── experiments/             # 5 Reproducible Experiment Suites
│   ├── 01_eda.py           # Exploratory Data Analysis & Word Clouds
│   ├── 02_preprocessing_comparison.py # Standard vs Indian-Context Preprocessing
│   ├── 03_feature_comparison.py       # TF-IDF vs Word2Vec vs Latency
│   ├── 04_model_comparison.py         # Base (NB) vs Proposed (LR, SVM) ROC/AUC
│   └── 05_lime_explainability.py      # LIME local attributions & error cases
├── models/                  # Serialized trained pipelines (.joblib)
├── results/                 # Publication plots, charts, and metric logs
├── presentation/            # Academic presentation slides (.pptx, .pdf)
├── app.py                   # Interactive Streamlit Web Demonstration
├── PROJECT_DOCUMENTATION.md # Comprehensive Academic Project Report
├── requirements.txt         # Package dependencies
├── run_all_experiments.py  # Master runner script
└── README.md
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK Resources

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### 3. (Optional) Install Sentence-BERT

If you want to include Sentence-BERT in the feature comparison experiment:

```bash
pip install sentence-transformers>=2.2.0
```

---

## Dataset

- Place the **IFND (Indian Fake News Dataset)** CSV file at `data/IFND_Dataset.csv`.
- The code auto-detects the text/statement column and label column from the dataset.
- Expected format: a CSV with at least one text column containing news statements and one label column indicating real/fake.

---

## Running Experiments

### Run All Experiments

```bash
python run_all_experiments.py
```

### Run Specific Experiments

```bash
python run_all_experiments.py --exp 1       # Only EDA
python run_all_experiments.py --exp 2 3     # Preprocessing + Feature comparison
python run_all_experiments.py --exp 4 5     # Model comparison + LIME
```

---

## Interactive Streamlit Web Application

To launch the live interactive demonstration app for your project presentation:

```bash
streamlit run app.py
```

Features included in the web app:
- **Interactive Detection**: Paste any Indian news article or WhatsApp forward to predict Fake vs. Real with probability gauges.
- **LIME Word-Level Attribution**: Real-time bar charts and in-text highlighting showing exactly which words drove the decision.
- **Base vs. Proposed Side-by-Side Comparison**: Directly compare predictions between Base Naive Bayes and Proposed Logistic Regression.
- **Preloaded Indian Case Studies**: 1-click test buttons for viral scams (GPS nano-chip notes, free recharge hoaxes) vs. official press releases (RBI, Supreme Court).
- **Academic Results Gallery**: Full visual display of all 5 experimental benchmark charts.

---

## Experiments

### Experiment 1: Exploratory Data Analysis (`01_eda.py`)
Generates descriptive statistics, class distribution plots, text length distributions, and word clouds for the dataset. Provides a foundational understanding of the data before modeling.

### Experiment 2: Preprocessing Comparison (`02_preprocessing_comparison.py`)
Compares **standard NLP preprocessing** (lowercasing, stopword removal, lemmatization) against an **Indian-context-enhanced** pipeline that additionally handles transliterated terms, Indian named entities, and culturally specific stop-patterns. Evaluates impact on downstream classification.

### Experiment 3: Feature Representation Comparison (`03_feature_comparison.py`)
Benchmarks three feature extraction strategies — **TF-IDF**, **Word2Vec (averaged)**, and optionally **Sentence-BERT** — measuring their effect on classification accuracy, precision, recall, and F1-score.

### Experiment 4: Model Comparison (`04_model_comparison.py`)
Compares the **base model** (standard preprocessing + TF-IDF + Logistic Regression) against the **proposed model** (Indian-context preprocessing + best features + tuned classifier). Reports full classification metrics and generates comparison visualizations.

### Experiment 5: LIME Explainability (`05_lime_explainability.py`)
Applies **LIME** to explain individual predictions from the proposed model. Generates per-instance explanation plots, global feature importance summaries, and highlights which words drive fake vs. real classifications.

---

## Results

After running experiments, check the `results/` folder for:

- **Plots**: PNG visualizations (class distributions, confusion matrices, LIME explanations, etc.)
- **Metrics**: JSON files containing accuracy, precision, recall, F1-score, and other metrics.
- **Reports**: Summary comparisons across experiments.

---

## References

1. Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). *Fake News Detection on Social Media: A Data Mining Perspective.* ACM SIGKDD Explorations Newsletter, 19(1), 22–36.

2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1135–1144.

3. Sharma, K., Qian, F., Jiang, H., Ruchansky, N., Zhang, M., & Liu, Y. (2019). *Combating Fake News: A Survey on Identification and Mitigation Techniques.* ACM Transactions on Intelligent Systems and Technology, 10(3), 1–42.

4. Patwa, P., Sharma, S., Pykl, S., Guptha, V., Kumari, G., et al. (2021). *Fighting an Infodemic: COVID-19 Fake News Dataset.* Proceedings of the CONSTRAINT Shared Task at AAAI 2021.

5. Kar, D., Bhardwaj, M., Samanta, S., & Azad, A. P. (2021). *No Rumours Please! A Multi-Indic-Lingual Approach for COVID Fake-Tweet Detection.* Proceedings of the European Conference on Information Retrieval (ECIR 2021).

6. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* Proceedings of NAACL-HLT 2019.

---

## License

This project is developed for academic purposes as part of a Natural Language Processing course (Semester 7).
