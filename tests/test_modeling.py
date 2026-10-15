import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertModel

from data import SentimentDataset
from encoder import get_embeddings, tokenize_texts
from inference import predict_fine_tuned
from tests.helpers import tiny_config, tokenizer
from training import evaluate, metrics, train_epoch


class ModelingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tokenizer(self)

    def test_padding_masks_and_truncation_preserve_special_tokens(self):
        tokens = tokenize_texts(
            ["good", "this movie was good !"], self.tokenizer, max_length=5
        )
        self.assertEqual(list(tokens.input_ids.shape), [2, 5])
        self.assertEqual(tokens.attention_mask[0].tolist(), [1, 1, 1, 0, 0])
        self.assertEqual(tokens.input_ids[1, -1].item(), self.tokenizer.sep_token_id)

    def test_cls_embeddings_are_batch_invariant_and_inference_has_no_gradients(self):
        model = DistilBertModel(tiny_config()).eval()
        texts = ["good", "this movie was bad !"]
        single = get_embeddings(texts, self.tokenizer, model, batch_size=1)
        batched = get_embeddings(texts, self.tokenizer, model, batch_size=2)
        self.assertEqual(batched.shape, (2, 24))
        np.testing.assert_allclose(single, batched, atol=1e-5)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(get_embeddings([], self.tokenizer, model).shape, (0, 24))

    def test_attention_matrices_have_layer_head_query_key_axes(self):
        config = tiny_config()
        config._attn_implementation = "eager"
        model = DistilBertModel(config).eval()
        tokens = tokenize_texts(["this movie was good"], self.tokenizer)
        with torch.no_grad():
            attention = model(**tokens, output_attentions=True).attentions
        self.assertEqual(len(attention), 2)
        self.assertEqual(tuple(attention[0].shape), (1, 4, 6, 6))
        torch.testing.assert_close(attention[0].sum(-1), torch.ones(1, 4, 6))

    def test_training_updates_transformer_weights_not_only_classifier(self):
        model = DistilBertForSequenceClassification(tiny_config())
        dataset = SentimentDataset(
            ["good movie", "bad movie", "great !", "terrible !"],
            [1, 0, 1, 0],
            self.tokenizer,
            max_length=8,
        )
        self.assertEqual(dataset[0]["labels"].dtype, torch.long)
        before = model.distilbert.embeddings.word_embeddings.weight.detach().clone()
        loss = train_epoch(
            model,
            DataLoader(dataset, batch_size=2),
            torch.optim.AdamW(model.parameters(), lr=0.01),
            torch.device("cpu"),
        )
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(
            torch.equal(before, model.distilbert.embeddings.word_embeddings.weight)
        )
        accuracy, f1 = evaluate(
            model, DataLoader(dataset, batch_size=2), torch.device("cpu")
        )
        self.assertTrue(0 <= accuracy <= 1 and 0 <= f1 <= 1)

    def test_predictions_are_batched_probabilities_with_binary_labels(self):
        model = DistilBertForSequenceClassification(tiny_config()).eval()
        texts = ["good movie", "terrible !"]
        single = predict_fine_tuned(texts, model, self.tokenizer, batch_size=1)
        batched = predict_fine_tuned(texts, model, self.tokenizer, batch_size=2)
        self.assertEqual(
            [x["prediction"] for x in single], [x["prediction"] for x in batched]
        )
        for item in batched:
            self.assertEqual(set(item["probabilities"]), {"Negative", "Positive"})
            self.assertAlmostEqual(sum(item["probabilities"].values()), 1.0, places=5)
        self.assertEqual(predict_fine_tuned([], model, self.tokenizer), [])

    def test_macro_f1_is_not_confused_with_accuracy(self):
        result = metrics([0, 0, 0, 1], [0, 0, 0, 0])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], 3 / 7)

    def test_blank_inputs_are_rejected(self):
        for texts in ["", ["  "], [1]]:
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                tokenize_texts(texts, self.tokenizer)
