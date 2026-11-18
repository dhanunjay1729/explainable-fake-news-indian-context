"""
Explainability Module using LIME (Local Interpretable Model-agnostic Explanations).
Generates both local prediction attributions and global model feature importances.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """
    Explainability engine for Fake News Detection using LIME.
    Interprets individual text predictions by learning an interpretable local linear surrogate.
    """

    def __init__(self, pipeline: Any, class_names: Optional[List[str]] = None):
        """
        Args:
            pipeline: Scikit-learn Pipeline containing vectorizer and classifier with predict_proba.
            class_names: Names of classes, defaults to ['Fake', 'Real'] (where 0=Fake, 1=Real).
        """
        self.pipeline = pipeline
        self.class_names = class_names or ['Fake', 'Real']
        self._lime_explainer = None

    def _get_explainer(self):
        if self._lime_explainer is None:
            try:
                from lime.lime_text import LimeTextExplainer
                self._lime_explainer = LimeTextExplainer(
                    class_names=self.class_names,
                    random_state=42
                )
            except ImportError:
                logger.error("LIME package is not installed. Install via: pip install lime")
                return None
        return self._lime_explainer

    def explain_instance(
        self,
        text: str,
        num_features: int = 15,
        num_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Explains a single news headline/statement.

        Returns:
            Dictionary containing:
            - features: list of (word, weight) tuples
            - fake_features: words contributing towards FAKE
            - real_features: words contributing towards REAL
            - prediction_proba: array of probabilities [P(Fake), P(Real)]
            - predicted_label: 'Fake' or 'Real'
            - raw_exp: LIME explanation object
        """
        explainer = self._get_explainer()
        if explainer is None:
            return {'error': 'LIME not installed'}

        # Ensure probability prediction function
        def predict_fn(texts):
            if hasattr(self.pipeline, 'predict_proba'):
                return self.pipeline.predict_proba(texts)
            else:
                preds = self.pipeline.predict(texts)
                proba = np.zeros((len(preds), 2))
                proba[preds == 0, 0] = 1.0
                proba[preds == 1, 1] = 1.0
                return proba

        exp = explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )

        probs = predict_fn([text])[0]
        pred_label_idx = int(np.argmax(probs))
        predicted_label = self.class_names[pred_label_idx]

        feature_list = exp.as_list()
        # In binary classification, positive weights point towards class 1 ('Real'), negative towards 0 ('Fake')
        fake_features = [(w, abs(wt)) for w, wt in feature_list if wt < 0]
        real_features = [(w, wt) for w, wt in feature_list if wt >= 0]

        return {
            'features': feature_list,
            'fake_features': sorted(fake_features, key=lambda x: x[1], reverse=True),
            'real_features': sorted(real_features, key=lambda x: x[1], reverse=True),
            'prediction_proba': probs.tolist(),
            'predicted_label': predicted_label,
            'confidence': float(probs[pred_label_idx]),
            'raw_exp': exp
        }

    def save_html(self, text: str, output_path: str, num_features: int = 15) -> bool:
        """Saves an interactive HTML visualization of the LIME explanation."""
        res = self.explain_instance(text, num_features=num_features)
        if 'raw_exp' in res and res['raw_exp'] is not None:
            res['raw_exp'].save_to_file(output_path)
            return True
        return False

    def plot_instance(
        self,
        text: str,
        output_path: str,
        title: Optional[str] = None,
        num_features: int = 12
    ):
        """
        Generates and saves a publication-quality Matplotlib horizontal bar chart
        of the word attribution weights.
        """
        res = self.explain_instance(text, num_features=num_features)
        if 'features' not in res:
            return

        features = res['features']
        words = [f[0] for f in features]
        weights = [f[1] for f in features]
        colors = ['#2ed573' if w >= 0 else '#ff4757' for w in weights]

        plt.figure(figsize=(8, 5))
        y_pos = np.arange(len(words))
        plt.barh(y_pos, weights, color=colors, align='center', edgecolor='black', linewidth=0.5)
        plt.yticks(y_pos, words, fontsize=10)
        plt.gca().invert_yaxis()  # Top feature at the top
        plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
        plt.xlabel('LIME Feature Weight (Green: Toward Real, Red: Toward Fake)', fontsize=10)
        plot_title = title or f"LIME Explanation: Predicted {res['predicted_label']} ({res['confidence']*100:.1f}%)"
        plt.title(plot_title, fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def get_global_feature_importance(
        pipeline: Any,
        top_n: int = 20
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extracts global model feature coefficients from Logistic Regression or Linear SVM.
        """
        vectorizer = pipeline.named_steps['vectorizer']
        classifier = pipeline.named_steps['classifier']
        feature_names = vectorizer.get_feature_names_out()

        if hasattr(classifier, 'coef_'):
            coef = classifier.coef_[0] if classifier.coef_.ndim > 1 else classifier.coef_
        elif hasattr(classifier, 'estimator') and hasattr(classifier.estimator, 'coef_'):
            coef = classifier.estimator.coef_[0]
        else:
            return {'fake_features': [], 'real_features': []}

        top_real_idx = np.argsort(coef)[-top_n:][::-1]
        top_fake_idx = np.argsort(coef)[:top_n]

        real_features = [(feature_names[i], float(coef[i])) for i in top_real_idx]
        fake_features = [(feature_names[i], float(abs(coef[i]))) for i in top_fake_idx]

        return {
            'real_features': real_features,
            'fake_features': fake_features
        }
