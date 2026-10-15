"""Загрузка IMDB и фиксированное разделение отзывов без пересечений."""

import hashlib
import html
import re

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from config import DATA, DATASET_REVISION, MAX_LENGTH, SEED, load_json, save_json


def clean_text(text):
    return " ".join(
        html.unescape(re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)).split()
    )


def text_hash(text):
    return hashlib.sha256(clean_text(text).casefold().encode()).hexdigest()


def validate_frame(frame):
    if not {"text", "label", "id"} <= set(frame.columns) or frame.empty:
        raise ValueError("Нужны непустые данные с колонками id, text, label")
    if (
        frame["text"].isna().any()
        or not frame["text"].map(lambda x: isinstance(x, str) and bool(x.strip())).all()
    ):
        raise ValueError("Тексты должны быть непустыми строками")
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Ожидаются оба класса: 0 — Negative, 1 — Positive")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("Идентификаторы должны быть уникальными")


def assert_disjoint(frames):
    seen_ids, seen_texts = set(), set()
    for frame in frames:
        validate_frame(frame)
        ids, texts = set(frame["id"]), set(frame["text"].map(text_hash))
        if seen_ids & ids or seen_texts & texts or len(texts) != len(frame):
            raise ValueError("Между выборками или внутри них повторяются отзывы")
        seen_ids |= ids
        seen_texts |= texts


def _source(split):
    path = hf_hub_download(
        "stanfordnlp/imdb",
        f"plain_text/{split}-00000-of-00001.parquet",
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    frame = pd.read_parquet(path)
    frame["id"] = [f"imdb-{split}-{i}" for i in range(len(frame))]
    frame["text"] = frame["text"].map(clean_text)
    frame["hash"] = frame["text"].map(text_hash)
    # Убираем и точные дубли, и противоречащие друг другу метки одного текста.
    conflicts = frame.groupby("hash")["label"].nunique()
    frame = frame[~frame["hash"].isin(conflicts[conflicts > 1].index)]
    return frame.drop_duplicates("hash")


def _balanced(frame, size):
    if type(size) is not int or size < 20 or size % 10:
        raise ValueError("Размер должен быть кратен 10 и не меньше 20")
    return (
        pd.concat(
            [
                frame[frame.label == label].sample(n=size // 2, random_state=SEED)
                for label in [0, 1]
            ]
        )
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )


def prepare_data(size=2000):
    if type(size) is not int or size < 100 or size % 50:
        raise ValueError(
            "Размер основного набора должен быть кратен 50 и не меньше 100"
        )
    main = _balanced(_source("train"), size)
    train, test = train_test_split(
        main, test_size=0.2, random_state=SEED, stratify=main.label
    )
    # Отдельный пул позволяет обучать обе модели на тех же 80% main;
    # выбор эпохи делается по validation, итоговый test не используется в обучении.
    pool = _source("test")
    validation = _balanced(pool[~pool.hash.isin(set(main.hash))], size // 5)
    frames = [train, validation, test]
    assert_disjoint(frames)
    DATA.mkdir(exist_ok=True)
    metadata = {
        "source": "stanfordnlp/imdb",
        "revision": DATASET_REVISION,
        "seed": SEED,
        "main_size": size,
        "test_fraction": 0.2,
        "validation_source": "independent official test pool",
        "splits": {},
    }
    for name, frame in zip(["train", "validation", "test"], frames, strict=True):
        path = DATA / f"{name}.csv"
        frame[["id", "text", "label"]].to_csv(path, index=False)
        metadata["splits"][name] = {
            "rows": len(frame),
            "labels": {str(k): int(v) for k, v in frame.label.value_counts().items()},
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    main[["id", "text", "label"]].to_csv(DATA / "sentiment.csv", index=False)
    save_json(DATA / "manifest.json", metadata)
    print(
        {
            name: len(frame)
            for name, frame in zip(["train", "validation", "test"], frames, strict=True)
        },
        flush=True,
    )
    return frames


def load_splits():
    manifest = load_json(DATA / "manifest.json")
    frames = []
    for name in ["train", "validation", "test"]:
        path = DATA / f"{name}.csv"
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != manifest["splits"][name]["sha256"]
        ):
            raise ValueError(
                f"Выборка {name} изменилась; заново выполните подготовку и обучение"
            )
        frames.append(pd.read_csv(path))
    assert_disjoint(frames)
    return frames


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        if len(texts) != len(labels) or not texts:
            raise ValueError("Нужны одинаковое число текстов и меток и непустой набор")
        self.encodings = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            **{name: values[index] for name, values in self.encodings.items()},
            "labels": self.labels[index],
        }
