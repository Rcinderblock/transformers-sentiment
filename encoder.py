"""Токенизация, скрытые состояния и неизменяемые CLS-признаки."""

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from config import MAX_LENGTH, MODEL_NAME, MODEL_REVISION


def choose_device(name="auto"):
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Ускоритель MPS недоступен")
    return device


def load_encoder(device="auto", attentions=False):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        attn_implementation="eager" if attentions else "sdpa",
    )
    model.to(choose_device(device)).eval()
    return tokenizer, model


def normalize_texts(texts):
    texts = [texts] if isinstance(texts, str) else list(texts)
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Тексты должны быть непустыми строками")
    return texts


def tokenize_texts(texts, tokenizer, max_length=MAX_LENGTH):
    texts = normalize_texts(texts)
    if (
        not texts
        or type(max_length) is not int
        or not 2 <= max_length <= tokenizer.model_max_length
    ):
        raise ValueError("Нужны тексты и допустимая длина последовательности")
    return tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )


def explain_tokenization(text, tokenizer):
    text = normalize_texts(text)[0]
    tokens = tokenizer.tokenize(text)
    result = {
        "text": text,
        "tokens": tokens,
        "ids": tokenizer.convert_tokens_to_ids(tokens),
        "count": len(tokens),
        "with_special_tokens": tokenizer(text)["input_ids"],
    }
    print(result)
    return result


def get_embeddings(texts, tokenizer, model, batch_size=32):
    texts = normalize_texts(texts)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Размер батча должен быть положительным целым числом")
    if not texts:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    model.eval()
    device = next(model.parameters()).device
    values = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            tokens = tokenize_texts(texts[start : start + batch_size], tokenizer).to(
                device
            )
            values.append(model(**tokens).last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(values)


def similarity(text1, text2, tokenizer, model):
    emb = get_embeddings([text1, text2], tokenizer, model)
    return float(cosine_similarity(emb[:1], emb[1:])[0, 0])


class CLSVectorizer:
    """Сохранённый преобразователь текста в CLS-признаки исходного трансформера."""

    def __init__(self, device="auto"):
        self.device = device
        self.model_name = MODEL_NAME
        self.revision = MODEL_REVISION
        self.max_length = MAX_LENGTH
        self._encoder = None

    def transform(self, texts):
        if (self.model_name, self.revision, self.max_length) != (
            MODEL_NAME,
            MODEL_REVISION,
            MAX_LENGTH,
        ):
            raise ValueError("Параметры исходной модели изменились")
        if self._encoder is None:
            self._encoder = load_encoder(self.device)
        return get_embeddings(texts, *self._encoder)

    def __getstate__(self):
        return {**self.__dict__, "_encoder": None, "device": "auto"}
