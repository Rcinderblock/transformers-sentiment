import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import data


class DataTests(unittest.TestCase):
    def frame(self, prefix, size=100):
        frame = pd.DataFrame(
            {
                "id": [f"{prefix}-{i}" for i in range(size)],
                "text": [f"{prefix} unique review {i}" for i in range(size)],
                "label": [i % 2 for i in range(size)],
            }
        )
        frame["hash"] = frame.text.map(data.text_hash)
        return frame

    def test_html_cleanup_and_normalized_duplicate_detection(self):
        self.assertEqual(data.clean_text(" A<br /><br> B &amp; C  "), "A B & C")
        self.assertEqual(
            data.text_hash("GREAT<br>movie"), data.text_hash("great movie")
        )
        first, second = self.frame("first"), self.frame("second")
        second.loc[0, "text"] = first.loc[0, "text"].upper()
        with self.assertRaises(ValueError):
            data.assert_disjoint([first, second])

    def test_duplicate_ids_are_rejected_even_when_texts_differ(self):
        first, second = self.frame("a"), self.frame("b")
        second.loc[0, "id"] = first.loc[0, "id"]
        with self.assertRaises(ValueError):
            data.assert_disjoint([first, second])

    def test_preparation_is_reproducible_stratified_and_test_is_separate(self):
        with (
            tempfile.TemporaryDirectory() as name,
            patch.object(data, "DATA", Path(name)),
            patch.object(data, "_source", side_effect=lambda split: self.frame(split)),
        ):
            first = data.prepare_data(100)
            second = data.prepare_data(100)
            self.assertEqual([len(frame) for frame in first], [80, 20, 20])
            for a, b in zip(first, second, strict=True):
                self.assertEqual(a.id.tolist(), b.id.tolist())
                self.assertEqual(a.label.value_counts().nunique(), 1)
            data.load_splits()
            with (Path(name) / "test.csv").open("a") as handle:
                handle.write("\nchanged")
            with self.assertRaisesRegex(ValueError, "изменилась"):
                data.load_splits()

    def test_invalid_sizes_are_rejected_before_download(self):
        with patch.object(data, "_source") as source:
            for size in [0, 20, 101, True]:
                with self.subTest(size=size), self.assertRaises(ValueError):
                    data.prepare_data(size)
            source.assert_not_called()

    def test_invalid_texts_and_missing_classes_are_rejected(self):
        frame = self.frame("a")
        frame.loc[0, "text"] = "  "
        with self.assertRaises(ValueError):
            data.validate_frame(frame)
        frame = self.frame("a")
        frame["label"] = 0
        with self.assertRaises(ValueError):
            data.validate_frame(frame)
