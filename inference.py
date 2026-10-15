"""Предсказания обеих моделей с одинаковым соответствием классов."""

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import LABELS, MAX_LENGTH, MODEL_DIR, ROOT
from encoder import choose_device, normalize_texts, tokenize_texts


def load_fine_tuned(device="auto"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, local_files_only=True
        )
        .to(choose_device(device))
        .eval()
    )
    if model.config.num_labels != 2 or model.config.id2label != LABELS:
        raise ValueError("Метки сохранённой модели не соответствуют задаче")
    return model, tokenizer


def predict_fine_tuned(texts, model, tokenizer, batch_size=32):
    texts = normalize_texts(texts)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Размер батча должен быть положительным")
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenize_texts(batch, tokenizer, MAX_LENGTH).to(
                next(model.parameters()).device
            )
            probabilities = model(**inputs).logits.softmax(dim=-1).cpu().tolist()
            for text, values in zip(batch, probabilities, strict=True):
                prediction = max(range(len(values)), key=values.__getitem__)
                result.append(
                    {
                        "text": text,
                        "prediction": prediction,
                        "label": LABELS[prediction],
                        "probabilities": {
                            LABELS[i]: value for i, value in enumerate(values)
                        },
                    }
                )
    return result


def load_baseline():
    return joblib.load(ROOT / "baseline_model.pkl"), joblib.load(
        ROOT / "vectorizer.pkl"
    )


def predict_baseline(texts, model, vectorizer):
    texts = normalize_texts(texts)
    if not texts:
        return []
    features = vectorizer.transform(texts)
    probabilities = model.predict_proba(features)
    predictions = model.predict(features)
    if set(model.classes_) != set(LABELS):
        raise ValueError("Классы baseline не соответствуют задаче")
    return [
        {
            "text": text,
            "prediction": int(prediction),
            "label": LABELS[int(prediction)],
            "probabilities": {
                LABELS[int(label)]: float(value)
                for label, value in zip(model.classes_, values, strict=True)
            },
        }
        for text, prediction, values in zip(
            texts, predictions, probabilities, strict=True
        )
    ]
