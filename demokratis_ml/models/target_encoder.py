"""Wrapper to handle string target encoding for classifiers that need numeric labels."""

from typing import Any, Self

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


class TargetEncoderClassifier(ClassifierMixin, BaseEstimator):
    """Wrapper to handle string target encoding for classifiers that need numeric labels."""

    def __init__(self, classifier: Any) -> None:  # noqa: D107
        super().__init__()
        self.classifier = classifier
        self.label_encoder = LabelEncoder()

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Self:
        """Fit the classifier on the encoded target labels."""
        # Encode the target variable
        y_encoded = self.label_encoder.fit_transform(y)
        # Fit the classifier with encoded labels
        self.classifier.fit(X, y_encoded)
        return self

    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        """Predict the original labels - encoding on input and decoding on output."""
        check_is_fitted(self.classifier)
        # Get numeric predictions
        y_pred_encoded = self.classifier.predict(X)
        # Decode back to original labels
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X: npt.ArrayLike) -> np.ndarray:
        """Predict class probabilities for the original labels."""
        check_is_fitted(self.classifier)
        return self.classifier.predict_proba(X)

    def score(self, X: npt.ArrayLike, y: npt.ArrayLike) -> float:
        """Score the classifier using encoded target labels."""
        check_is_fitted(self.classifier)
        y_encoded = self.label_encoder.transform(y)
        return self.classifier.score(X, y_encoded)

    @property
    def classes_(self) -> np.ndarray:
        """Get the original class labels."""
        check_is_fitted(self.classifier)
        return self.label_encoder.classes_

    def __sklearn_is_fitted__(self) -> bool:
        """Proxy for the wrapped classifier's fitted state."""
        try:
            check_is_fitted(self.classifier)
        except NotFittedError:
            return False
        return True
