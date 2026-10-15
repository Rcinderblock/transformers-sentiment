import tempfile
from pathlib import Path

import torch
from transformers import DistilBertConfig, DistilBertTokenizer


def tokenizer(test):
    directory = tempfile.TemporaryDirectory()
    test.addCleanup(directory.cleanup)
    path = Path(directory.name) / "vocab.txt"
    words = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "this",
        "movie",
        "was",
        "good",
        "bad",
        "!",
        "great",
        "terrible",
    ]
    path.write_text("\n".join(words), encoding="utf-8")
    return DistilBertTokenizer(vocab_file=str(path), model_max_length=512)


def tiny_config():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    return DistilBertConfig(
        vocab_size=13,
        dim=24,
        n_heads=4,
        n_layers=2,
        hidden_dim=48,
        dropout=0.0,
        attention_dropout=0.0,
        seq_classif_dropout=0.0,
        num_labels=2,
    )
