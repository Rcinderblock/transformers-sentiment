import unittest
from unittest.mock import Mock

import numpy as np
from pydantic import ValidationError

from app import PredictionRequest
from inference import predict_baseline


class AppTests(unittest.TestCase):
    def test_request_rejects_empty_and_overlong_text(self):
        for text in ["", "   ", "a" * 20001]:
            with self.subTest(length=len(text)), self.assertRaises(ValidationError):
                PredictionRequest(text=text)

    def test_baseline_probability_names_follow_classifier_class_order(self):
        model, vectorizer = Mock(), Mock()
        model.classes_ = np.array([1, 0])
        model.predict.return_value = np.array([1])
        model.predict_proba.return_value = np.array([[0.8, 0.2]])
        result = predict_baseline("good movie", model, vectorizer)[0]
        self.assertEqual(result["label"], "Positive")
        self.assertEqual(result["probabilities"], {"Positive": 0.8, "Negative": 0.2})
