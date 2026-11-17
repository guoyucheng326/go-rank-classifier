from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from go_rank_classifier.dataset import (
    DeterministicBatchSampler,
    SGFGameDataset,
    SplitData,
    load_split,
)
from go_rank_classifier.features import PASS_MOVE, board_features, transform_moves
from go_rank_classifier.sgf import iter_sgf_bytes, parse_sgf_bytes


class SGFParserTests(unittest.TestCase):
    def test_variation_is_ignored_and_parentheses_in_comment_are_safe(self) -> None:
        raw = (
            b"(;GM[1]SZ[19]C[main (line)];B[pd];W[dp]"
            b"(;B[dd];W[pp]);B[qp])"
        )
        game = parse_sgf_bytes(raw)
        self.assertEqual(game.board_size, 19)
        self.assertEqual(game.moves, (3 * 19 + 15, 15 * 19 + 3, 15 * 19 + 16))

    def test_pass_is_counted(self) -> None:
        game = parse_sgf_bytes(b"(;SZ[19];B[];W[pd])")
        self.assertEqual(game.moves, (PASS_MOVE, 3 * 19 + 15))

    def test_stream_splitter(self) -> None:
        stream = b"(;SZ[19];B[aa])(;SZ[19];W[bb])"
        games = list(iter_sgf_bytes([stream[:10], stream[10:]]))
        self.assertEqual(len(games), 2)
        self.assertTrue(all(game.startswith(b"(") and game.endswith(b")") for game in games))


class FeatureTests(unittest.TestCase):
    def test_all_eight_symmetries_preserve_move_set(self) -> None:
        moves = np.arange(361, dtype=np.int16)
        expected = set(moves.tolist())
        for symmetry in range(8):
            transformed = transform_moves(moves, symmetry)
            self.assertEqual(set(transformed.tolist()), expected)

    def test_feature_shape_and_color_planes(self) -> None:
        game = parse_sgf_bytes(b"(;SZ[19];B[pd];W[dp])")
        features = board_features(list(game.moves), 2)
        self.assertEqual(features.shape, (22, 19, 19))
        self.assertEqual(features[0, 3, 15], 1.0)
        self.assertEqual(features[1, 15, 3], 1.0)
        self.assertEqual(features.dtype, np.float32)


class DatasetTests(unittest.TestCase):
    def test_augmentation_changes_between_epochs(self) -> None:
        moves = np.tile(np.arange(250, dtype=np.int16), (2, 1))
        lengths = np.array([250, 180], dtype=np.int16)
        data = SplitData(moves=moves, lengths=lengths, labels=np.array([1, 2], np.int16))
        dataset = SGFGameDataset(data, training=True, seed=7)
        choices = set()
        for epoch in range(20):
            dataset.set_epoch(epoch)
            choices.add(dataset.augmentation_for(0))
        self.assertGreater(len(choices), 1)

        features, label = dataset[0]
        self.assertEqual(tuple(features.shape), (22, 19, 19))
        self.assertIsInstance(label, torch.Tensor)

    def test_load_memmapped_split_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "train"
            np.save(prefix.with_name("train_moves.npy"), np.zeros((2, 300), np.int16))
            np.save(prefix.with_name("train_lengths.npy"), np.array([100, 120], np.int16))
            np.save(prefix.with_name("train_labels.npy"), np.array([0, 27], np.int16))
            split = load_split(prefix)
            self.assertEqual(len(split), 2)
            self.assertEqual(int(split.labels[1]), 27)

    def test_batch_sampler_resumes_without_reordering(self) -> None:
        full = list(DeterministicBatchSampler(20, 4, 3, seed=11, drop_last=True))
        resumed = list(
            DeterministicBatchSampler(
                20, 4, 3, seed=11, start_batch=2, drop_last=True
            )
        )
        self.assertEqual(resumed, full[2:])


if __name__ == "__main__":
    unittest.main()

