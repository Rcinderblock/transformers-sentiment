"""Демонстрации токенизации, скрытых состояний и внимания."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from config import REPORTS, save_json
from encoder import (
    explain_tokenization,
    get_embeddings,
    load_encoder,
    similarity,
    tokenize_texts,
)

EXAMPLES = [
    "This movie was absolutely amazing!",
    "Terrible movie, waste of time.",
    "Pretty good, I liked it.",
    "Boring and too long.",
]


def day1():
    from transformers import AutoTokenizer

    from config import MODEL_NAME, MODEL_REVISION

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    batch = tokenize_texts(EXAMPLES[:2], tokenizer)
    single = tokenizer(EXAMPLES[0])
    result = {
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": tokenizer.model_max_length,
        "input_ids": single["input_ids"],
        "token_count": len(single["input_ids"]),
        "decoded": tokenizer.decode(single["input_ids"]),
        "batch_shape": list(batch["input_ids"].shape),
        "attention_mask": batch["attention_mask"].tolist(),
        "special_tokens": {
            name: {
                "token": getattr(tokenizer, f"{name}_token"),
                "id": getattr(tokenizer, f"{name}_token_id"),
            }
            for name in ["cls", "sep", "pad"]
        },
        "explanation": explain_tokenization("Transformers are amazing!", tokenizer),
    }
    save_json(REPORTS / "tokenization.json", result)
    print(result, flush=True)


def day2(device="auto"):
    tokenizer, model = load_encoder(device)
    tokens = tokenize_texts([EXAMPLES[0]], tokenizer).to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    embeddings = get_embeddings(EXAMPLES, tokenizer, model)
    result = {
        "architecture": str(model),
        "output_type": type(outputs).__name__,
        "last_hidden_state_shape": list(outputs.last_hidden_state.shape),
        "cls_shape": list(outputs.last_hidden_state[:, 0, :].shape),
        "first_five_values": embeddings[0, :5].tolist(),
        "embeddings_shape": list(embeddings.shape),
        "hidden_state_shapes": [list(value.shape) for value in outputs.hidden_states],
        "similarity": {
            "same_sentiment": similarity(
                "Great movie!", "Amazing film!", tokenizer, model
            ),
            "different_sentiment": similarity(
                "Great movie!", "Terrible film!", tokenizer, model
            ),
        },
    }
    save_json(REPORTS / "hidden_states.json", result)
    print(
        {key: value for key, value in result.items() if key != "architecture"},
        flush=True,
    )


def visualize_attention(
    tokens, attention, tokenizer, layer=0, head=0, prefix="attention"
):
    if not 0 <= layer < len(attention) or not 0 <= head < attention[layer].shape[1]:
        raise ValueError("Некорректный номер слоя или головы")
    labels = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0].tolist())
    matrix = attention[layer][0, head].detach().cpu().numpy()
    figure, axes = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        ax=axes,
        vmin=0,
        vmax=1,
    )
    axes.set(
        title=f"Attention: layer {layer}, head {head}",
        xlabel="Keys (attended tokens)",
        ylabel="Queries (current tokens)",
    )
    axes.tick_params(axis="x", labelrotation=65, labelsize=8)
    axes.tick_params(axis="y", labelrotation=0, labelsize=8)
    figure.tight_layout()
    path = REPORTS / "attention" / f"{prefix}_layer{layer}_head{head}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return str(path.relative_to(REPORTS))


def day3(device="auto"):
    tokenizer, model = load_encoder(device, attentions=True)
    tokens = tokenize_texts(["The amazing movie won many awards"], tokenizer).to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        attention = model(**tokens, output_attentions=True).attentions
    plots = [
        visualize_attention(tokens, attention, tokenizer, layer=layer)
        for layer in [0, len(attention) // 2, len(attention) - 1]
    ]
    plots += [
        visualize_attention(tokens, attention, tokenizer, head=head)
        for head in range(model.config.n_heads)
    ]
    statistics = []
    for layer, weights in enumerate(attention):
        values = weights[0].detach().cpu().numpy()
        statistics.append(
            {
                "layer": layer,
                "mean_max_weight": float(values.max(axis=-1).mean()),
                "mean_entropy": float(
                    -(values * np.log(np.clip(values, 1e-12, 1))).sum(axis=-1).mean()
                ),
            }
        )
    negative = tokenize_texts(
        ["This movie was absolutely terrible and I hated it"], tokenizer
    ).to(next(model.parameters()).device)
    with torch.no_grad():
        negative_attention = model(**negative, output_attentions=True).attentions
    plots.append(
        visualize_attention(
            negative,
            negative_attention,
            tokenizer,
            layer=len(attention) - 1,
            prefix="negative",
        )
    )
    negative_labels = tokenizer.convert_ids_to_tokens(negative["input_ids"][0].tolist())
    index = negative_labels.index("terrible")
    weights = negative_attention[-1][0, 0, index].detach().cpu().numpy()
    focus = [
        {"token": negative_labels[i], "weight": float(weights[i])}
        for i in np.argsort(weights)[::-1][:5]
    ]
    result = {
        "layers": len(attention),
        "heads": model.config.n_heads,
        "shape": list(attention[0].shape),
        "plots": list(dict.fromkeys(plots)),
        "layer_statistics": statistics,
        "first_layer_head_statistics": [
            {
                "head": head,
                "mean_max_weight": float(
                    attention[0][0, head].detach().cpu().max(dim=-1).values.mean()
                ),
                "cls_top_token": tokenizer.convert_ids_to_tokens(
                    int(tokens["input_ids"][0, attention[0][0, head, 0].argmax()])
                ),
            }
            for head in range(model.config.n_heads)
        ],
        "terrible_query_last_layer_head0": focus,
    }
    save_json(REPORTS / "attention_analysis.json", result)
    print(result, flush=True)
