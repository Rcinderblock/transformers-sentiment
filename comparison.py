"""Одинаковый test для двух моделей и сохранение измеренных результатов."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import DATA, LABELS, REPORTS, ROOT, load_json, save_json
from data import load_splits
from inference import (
    load_baseline,
    load_fine_tuned,
    predict_baseline,
    predict_fine_tuned,
)
from training import metrics, write_result

DEMO_TEXTS = [
    "This movie was absolutely fantastic!",
    "Terrible, waste of my time.",
    "It was okay, nothing special.",
    "Best film I've seen this year!",
    "Boring and too long.",
]


def compare(device="auto"):
    _, _, test = load_splits()
    manifest = load_json(DATA / "manifest.json")
    if (
        load_json(REPORTS / "training.json")["data_manifest"] != manifest
        or load_json(REPORTS / "baseline.json")["data_manifest"] != manifest
    ):
        raise ValueError("Модели обучались на других данных; повторите обучение")
    model, tokenizer = load_fine_tuned(device)
    ft = predict_fine_tuned(test.text.tolist(), model, tokenizer)
    ft_examples = predict_fine_tuned(DEMO_TEXTS, model, tokenizer)
    baseline, vectorizer = load_baseline()
    vectorizer.device = device
    base = predict_baseline(test.text.tolist(), baseline, vectorizer)
    base_examples = predict_baseline(DEMO_TEXTS, baseline, vectorizer)
    table = test.copy()
    for name, predictions in [("fine_tuned", ft), ("baseline", base)]:
        table[f"{name}_prediction"] = [item["prediction"] for item in predictions]
        table[f"{name}_positive_probability"] = [
            item["probabilities"]["Positive"] for item in predictions
        ]
    results = {}
    for name in ["fine_tuned", "baseline"]:
        results[name] = metrics(test.label, table[f"{name}_prediction"])
        matrix = confusion_matrix(
            test.label, table[f"{name}_prediction"], labels=[0, 1]
        )
        figure, axes = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(LABELS.values()),
            yticklabels=list(LABELS.values()),
            ax=axes,
        )
        axes.set(
            title=f"Confusion matrix: {name}",
            xlabel="Predicted label",
            ylabel="True label",
        )
        figure.tight_layout()
        figure.savefig(REPORTS / f"confusion_matrix_{name}.png", dpi=150)
        plt.close(figure)
        results[name]["confusion_matrix"] = matrix.tolist()
    results["split"] = "test"
    results["test_size"] = len(test)
    results["test_sha256"] = manifest["splits"]["test"]["sha256"]
    results["macro_f1_difference"] = (
        results["fine_tuned"]["macro_f1"] - results["baseline"]["macro_f1"]
    )
    results["relative_f1_improvement_percent"] = (
        100 * results["macro_f1_difference"] / results["baseline"]["macro_f1"]
        if results["baseline"]["macro_f1"]
        else None
    )
    table.to_csv(REPORTS / "test_predictions.csv", index=False)
    save_json(REPORTS / "comparison.json", results)
    save_json(
        REPORTS / "demo_predictions.json",
        {"fine_tuned": ft_examples, "baseline": base_examples},
    )
    write_result(ROOT / "baseline_results.txt", "Baseline: test", results["baseline"])
    write_result(
        ROOT / "comparison_results.txt", "Fine-tuned: test", results["fine_tuned"]
    )
    with (ROOT / "comparison_results.txt").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\nBaseline Macro F1: {results['baseline']['macro_f1']:.6f}\nBaseline Accuracy: {results['baseline']['accuracy']:.6f}\nF1 difference: {results['macro_f1_difference']:.6f}\nRelative F1 improvement (%): {results['relative_f1_improvement_percent']}\n"
        )
    print(
        {
            key: results[key]
            for key in [
                "test_size",
                "macro_f1_difference",
                "relative_f1_improvement_percent",
            ]
        },
        flush=True,
    )
    return table


def analyze_errors():
    frame = pd.read_csv(REPORTS / "test_predictions.csv")
    _, _, test = load_splits()
    if (
        frame.id.tolist() != test.id.tolist()
        or frame.label.tolist() != test.label.tolist()
        or frame.text.tolist() != test.text.tolist()
    ):
        raise ValueError("Предсказания не соответствуют текущей тестовой выборке")
    errors = frame[frame.label != frame.fine_tuned_prediction]
    fp = errors[(errors.label == 0) & (errors.fine_tuned_prediction == 1)]
    fn = errors[(errors.label == 1) & (errors.fine_tuned_prediction == 0)]
    lines = [
        "Анализ ошибок на test",
        f"Всего: {len(frame)}; ошибок: {len(errors)}; False Positive: {len(fp)}; False Negative: {len(fn)}",
        f"Средняя длина всех отзывов: {frame.text.str.len().mean():.1f} символов",
        f"Средняя длина ошибочных отзывов: {errors.text.str.len().mean():.1f} символов",
    ]
    for title, subset in [("FALSE POSITIVES", fp), ("FALSE NEGATIVES", fn)]:
        lines.append(f"\n{title}")
        for row in subset.head(5).itertuples():
            lines.append(
                f"\nID: {row.id}; истинный: {row.label}; предсказанный: {row.fine_tuned_prediction}; P(Positive): {row.fine_tuned_positive_probability:.4f}\n{row.text}"
            )
    lines.append(
        "\nМодель читает первые 128 токенов, включая служебные. Длинные отзывы обрезаются. Наблюдения по содержанию примеров сохранены в reports/error_observations.txt."
    )
    (ROOT / "error_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(
        REPORTS / "error_summary.json",
        {
            "test_size": len(frame),
            "errors": len(errors),
            "false_positives": len(fp),
            "false_negatives": len(fn),
            "mean_text_length": float(frame.text.str.len().mean()),
            "mean_error_length": float(errors.text.str.len().mean())
            if len(errors)
            else None,
            "fp_ids": fp.head(5).id.tolist(),
            "fn_ids": fn.head(5).id.tolist(),
        },
    )
    print(lines[:4], flush=True)
