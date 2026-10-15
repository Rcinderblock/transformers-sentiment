"""Обучение классификатора на фиксированных признаках и дообучение DistilBERT."""

import inspect
import json
import logging
import time
from pathlib import Path

import joblib
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from config import (
    DATA,
    LABELS,
    MAX_LENGTH,
    MODEL_DIR,
    MODEL_NAME,
    MODEL_REVISION,
    REPORTS,
    ROOT,
    SEED,
    load_json,
    save_json,
)
from data import SentimentDataset, load_splits
from encoder import CLSVectorizer, choose_device

logger = logging.getLogger(__name__)


def metrics(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels, predictions, labels=[0, 1], average="macro", zero_division=0
            )
        ),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=list(LABELS.values()),
            zero_division=0,
            output_dict=True,
        ),
    }


def write_result(path, title, result):
    Path(path).write_text(
        f"{title}\nAccuracy: {result['accuracy']:.6f}\nMacro F1: {result['macro_f1']:.6f}\n\n"
        + json.dumps(result["classification_report"], indent=2)
        + "\n",
        encoding="utf-8",
    )


def train_baseline(device="auto"):
    train, _, test = load_splits()
    vectorizer = CLSVectorizer(device)
    features = vectorizer.transform(train.text.tolist())
    test_features = vectorizer.transform(test.text.tolist())
    kwargs = {"max_iter": 1000, "random_state": SEED}
    if "n_jobs" in inspect.signature(LogisticRegression).parameters:
        kwargs["n_jobs"] = -1
    model = LogisticRegression(**kwargs).fit(features, train.label)
    result = metrics(test.label, model.predict(test_features))
    result["split"] = "test"
    result["data_manifest"] = load_json(DATA / "manifest.json")
    joblib.dump(model, ROOT / "baseline_model.pkl")
    joblib.dump(vectorizer, ROOT / "vectorizer.pkl")
    save_json(REPORTS / "baseline.json", result)
    write_result(ROOT / "baseline_results.txt", "Baseline: test", result)
    print(f"Baseline test: {result['macro_f1']:.4f}", flush=True)
    return result


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    loss_sum, count = 0.0, 0
    for step, batch in enumerate(dataloader, 1):
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Обучение остановлено: loss не является конечным числом")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        size = batch["labels"].shape[0]
        loss_sum += float(loss.detach().cpu()) * size
        count += size
        if step % 20 == 0:
            logger.info(
                "Батч %d/%d; loss %.4f", step, len(dataloader), loss_sum / count
            )
    return loss_sum / count


def evaluate(model, dataloader, device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            labels.extend(batch["labels"].tolist())
            inputs = {
                name: tensor.to(device)
                for name, tensor in batch.items()
                if name != "labels"
            }
            predictions.extend(model(**inputs).logits.argmax(dim=-1).cpu().tolist())
    result = metrics(labels, predictions)
    return result["accuracy"], result["macro_f1"]


def fine_tune(device="auto", epochs=3):
    if type(epochs) is not int or not 3 <= epochs <= 5:
        raise ValueError("Требуется от 3 до 5 эпох")
    set_seed(SEED)
    device = choose_device(device)
    train, validation, _ = load_splits()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        num_labels=2,
        id2label=LABELS,
        label2id={v: k for k, v in LABELS.items()},
    ).to(device)
    train_data = SentimentDataset(train.text.tolist(), train.label.tolist(), tokenizer)
    val_data = SentimentDataset(
        validation.text.tolist(), validation.label.tolist(), tokenizer
    )
    train_loader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(val_data, batch_size=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    history, best_f1 = [], -1.0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        accuracy, f1 = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "loss": loss,
            "validation_accuracy": accuracy,
            "validation_macro_f1": f1,
        }
        history.append(row)
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(MODEL_DIR, safe_serialization=True)
            tokenizer.save_pretrained(MODEL_DIR)
            best_epoch = epoch
        save_json(
            REPORTS / "training.json",
            {
                "model": MODEL_NAME,
                "revision": MODEL_REVISION,
                "max_length": MAX_LENGTH,
                "seed": SEED,
                "device": str(device),
                "epochs": epochs,
                "best_epoch": best_epoch,
                "best_validation_macro_f1": best_f1,
                "seconds": time.perf_counter() - started,
                "history": history,
                "data_manifest": load_json(DATA / "manifest.json"),
            },
        )
        print(row, flush=True)
    best = next(row for row in history if row["epoch"] == best_epoch)
    (ROOT / "fine_tuned_results.txt").write_text(
        f"Best epoch: {best_epoch}\nValidation Macro F1: {best_f1:.6f}\nValidation Accuracy: {best['validation_accuracy']:.6f}\n",
        encoding="utf-8",
    )
    return history
