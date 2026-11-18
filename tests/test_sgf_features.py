from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import torch

from go_rank_classifier.audit import audit
from go_rank_classifier.dataset import (
    DeterministicBatchSampler,
    KataGoFeatureData,
    PreShuffledBatchSampler,
    SGFGameDataset,
    SplitData,
    load_split,
)
from go_rank_classifier.features import PASS_MOVE, board_features, transform_moves
from go_rank_classifier.katago_features import (
    FEATURE_NAMES,
    aggregate_player,
    generate_resumable,
    make_query,
    parse_move_sample,
    select_positions,
)
from go_rank_classifier.metrics import (
    classification_metrics,
    decode_probabilities,
    expected_strength,
)
from go_rank_classifier.model import DualGoRankCNN
from go_rank_classifier.ranks import rank_id_from_sgf_label
from go_rank_classifier.sgf import iter_sgf_bytes, parse_sgf_bytes
from go_rank_classifier.shuffle_data import shuffle_split


def save_dual_prefix(root: Path, split: str, labels: list[tuple[int, int]]) -> None:
    count = len(labels)
    moves = np.full((count, 300), -1, dtype=np.int16)
    moves[:, :100] = np.arange(100, dtype=np.int16)
    np.save(root / f"{split}_moves.npy", moves)
    np.save(root / f"{split}_lengths.npy", np.full(count, 100, np.int16))
    np.save(root / f"{split}_black_labels.npy", np.array([x[0] for x in labels], np.int16))
    np.save(root / f"{split}_white_labels.npy", np.array([x[1] for x in labels], np.int16))
    np.save(root / f"{split}_hashes.npy", np.arange(count, dtype=np.uint64))


class SGFParserTests(unittest.TestCase):
    def test_variation_is_ignored_and_parentheses_in_comment_are_safe(self) -> None:
        raw = (
            b"(;GM[1]SZ[19]BR[9d]WR[P4]HA[0]C[main (line)]"
            b";B[pd];W[dp](;B[dd];W[pp]);B[qp])"
        )
        game = parse_sgf_bytes(raw)
        self.assertEqual(game.board_size, 19)
        self.assertEqual(game.moves, (3 * 19 + 15, 15 * 19 + 3, 15 * 19 + 16))
        self.assertEqual(game.black_rank, 1)
        self.assertEqual(game.white_rank, 0)
        self.assertFalse(game.has_setup_stones)

    def test_both_labels_and_zero_handicap_are_required(self) -> None:
        moves = b";B[pd];W[dp]" * 50
        valid = b"(;SZ[19]BR[2d]WR[3k]HA[0]" + moves + b")"
        invalid = b"(;SZ[19]BR[2d]WR[3k]HA[2]AB[pd]" + moves + b")"
        self.assertTrue(parse_sgf_bytes(valid).is_trainable())
        self.assertFalse(parse_sgf_bytes(invalid).is_trainable())

    def test_rank_label_normalization(self) -> None:
        self.assertEqual(rank_id_from_sgf_label("P4段"), 0)
        self.assertEqual(rank_id_from_sgf_label("10d"), 0)
        self.assertEqual(rank_id_from_sgf_label("9段"), 1)
        self.assertEqual(rank_id_from_sgf_label("18级"), 27)
        self.assertIsNone(rank_id_from_sgf_label("unknown"))

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
        game = parse_sgf_bytes(b"(;SZ[19]BR[1d]WR[1k];B[pd];W[dp])")
        features = board_features(list(game.moves), 2)
        self.assertEqual(features.shape, (22, 19, 19))
        self.assertEqual(features[0, 3, 15], 1.0)
        self.assertEqual(features[1, 15, 3], 1.0)
        self.assertEqual(features.dtype, np.float32)


class DatasetTests(unittest.TestCase):
    def test_augmentation_changes_between_epochs(self) -> None:
        moves = np.tile(np.arange(250, dtype=np.int16), (2, 1))
        lengths = np.array([250, 180], dtype=np.int16)
        data = SplitData(
            moves=moves,
            lengths=lengths,
            black_labels=np.array([1, 2], np.int16),
            white_labels=np.array([2, 3], np.int16),
        )
        dataset = SGFGameDataset(data, training=True, seed=7)
        choices = set()
        for epoch in range(20):
            dataset.set_epoch(epoch)
            choices.add(dataset.augmentation_for(0))
        self.assertGreater(len(choices), 1)

        features, black, white = dataset[0]
        self.assertEqual(tuple(features.shape), (22, 19, 19))
        self.assertIsInstance(black, torch.Tensor)
        self.assertIsInstance(white, torch.Tensor)

    def test_load_memmapped_split_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_dual_prefix(root, "train", [(0, 27), (1, 2)])
            split = load_split(root / "train")
            self.assertEqual(len(split), 2)
            self.assertEqual(int(split.black_labels[1]), 1)
            self.assertEqual(int(split.white_labels[0]), 27)
            self.assertFalse(split.shuffled)

    def test_batch_sampler_resumes_without_reordering(self) -> None:
        full = list(DeterministicBatchSampler(20, 4, 3, seed=11, drop_last=True))
        resumed = list(
            DeterministicBatchSampler(
                20, 4, 3, seed=11, start_batch=2, drop_last=True
            )
        )
        self.assertEqual(resumed, full[2:])

    def test_preshuffled_sampler_rotates_but_keeps_all_items(self) -> None:
        batches = list(
            PreShuffledBatchSampler(20, 4, epoch=2, seed=5, drop_last=True)
        )
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(20)))

    def test_shuffle_split_creates_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_dual_prefix(root, "train", [(0, 1), (2, 3), (4, 5)])
            shuffle_split(root / "train", seed=9, block_size=2)
            self.assertTrue((root / "train.shuffled").is_file())
            self.assertTrue(load_split(root / "train").shuffled)

    def test_dual_kata_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_dual_prefix(root, "train", [(0, 1), (2, 3)])
            kata_black = np.ones((2, 2, len(FEATURE_NAMES)), np.float32)
            kata_white = np.zeros((2, 2, len(FEATURE_NAMES)), np.float32)
            np.save(root / "train_katago_black.npy", kata_black)
            np.save(root / "train_katago_white.npy", kata_white)
            np.save(root / "train_katago_mask.npy", np.array([True, False]))
            np.save(root / "train_katago_hashes.npy", np.array([0, 1], np.uint64))
            (root / "train_katago_cutoffs.json").write_text("[100, 300]")
            split = load_split(root / "train")
            kata = KataGoFeatureData.load(root / "train")
            dataset = SGFGameDataset(split, training=False, katago=kata)
            self.assertEqual(len(dataset), 1)
            _, tabular, black, white = dataset[0]
            self.assertEqual(tabular.shape[0], len(FEATURE_NAMES) * 3)
            self.assertEqual(int(black), 0)
            self.assertEqual(int(white), 1)

    def test_duplicate_audit_detects_cross_split_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_dual_prefix(root, "train", [(0, 1), (1, 2)])
            save_dual_prefix(root, "val", [(0, 2)])
            save_dual_prefix(root, "test", [(2, 3)])
            train_moves = np.load(root / "train_moves.npy")
            test_moves = np.load(root / "test_moves.npy")
            test_moves[0] = train_moves[0]
            np.save(root / "test_moves.npy", test_moves)
            report = audit(root, block_size=1)
            self.assertEqual(report["cross_split_duplicates"]["train__test"], 1)


class KataGoFeatureTests(unittest.TestCase):
    def test_position_selection_matches_player_parity(self) -> None:
        self.assertTrue(all(index % 2 == 0 for index in select_positions(40, "B", 5)))
        self.assertTrue(all(index % 2 == 1 for index in select_positions(40, "W", 5)))

    def test_empty_player_aggregation_has_fixed_width(self) -> None:
        values = aggregate_player([])
        self.assertEqual(values.shape, (len(FEATURE_NAMES),))

    def test_query_and_response_metrics(self) -> None:
        moves = np.array([15 * 19 + 3], dtype=np.int16)
        query = make_query(
            "train:0",
            moves,
            [0],
            rules="chinese",
            komi=7.5,
            max_visits=500,
        )
        self.assertEqual(query["moves"], [["B", "D4"]])
        sample = parse_move_sample(
            {
                "turnNumber": 0,
                "rootInfo": {"winrate": 0.55, "scoreLead": 1},
                "_color": "B",
                "moveInfos": [
                    {"move": "Q16", "visits": 400, "winrate": 0.6, "scoreLead": 2},
                    {"move": "D4", "visits": 100, "winrate": 0.55, "scoreLead": 1},
                ],
            },
            played_move=int(moves[0]),
            max_move=100,
        )
        self.assertEqual(sample["played_rank"], 1)
        self.assertTrue(sample["top3"])
        self.assertAlmostEqual(sample["visit_ratio"], 0.25)
        self.assertAlmostEqual(sample["winrate_loss"], 0.05)
        self.assertAlmostEqual(sample["score_loss"], 1.0)

    def test_resumable_generation_from_cached_responses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            output_dir = root / "features"
            data_dir.mkdir()
            save_dual_prefix(data_dir, "train", [(1, 10)])
            (data_dir / "manifest.json").write_text(json.dumps({"fingerprint": "abc"}))
            responses = root / "responses.jsonl"
            responses.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "train:0",
                                "turnNumber": 0,
                                "rootInfo": {"winrate": 0.5, "scoreLead": 0},
                                "moveInfos": [
                                    {"move": "A19", "visits": 4, "winrate": 0.5, "scoreLead": 0}
                                ],
                            }
                        ),
                        json.dumps(
                            {
                                "id": "train:0",
                                "turnNumber": 1,
                                "rootInfo": {"winrate": 0.5, "scoreLead": 0},
                                "moveInfos": [
                                    {"move": "B19", "visits": 4, "winrate": 0.5, "scoreLead": 0}
                                ],
                            }
                        ),
                    ]
                )
                + "\n"
            )
            arguments = dict(
                katago_bin=None,
                model=None,
                config=None,
                rules="chinese",
                komi=7.5,
                max_visits=5,
                positions_per_player=1,
                games_per_class=0,
                job_size=1,
                seed=2,
                cutoffs=(100, 300),
                responses=responses,
                keep_queries=False,
            )
            manifest = generate_resumable(
                data_dir, "train", output_dir, **arguments
            )
            self.assertEqual(manifest["complete_games"], 1)
            self.assertEqual(
                np.load(output_dir / "train_katago_black.npy").shape,
                (1, 2, len(FEATURE_NAMES)),
            )
            resumed = generate_resumable(
                data_dir, "train", output_dir, **arguments
            )
            self.assertEqual(resumed["completed_jobs"], 1)


class MetricAndModelTests(unittest.TestCase):
    def test_per_class_and_continuous_metrics(self) -> None:
        result = classification_metrics(
            np.array([0, 0, 1, 2]),
            np.array([0, 1, 1, 2]),
            ("A", "B", "C"),
        )
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["within_1_0"], 1.0)

    def test_pro_maps_to_ten_dan(self) -> None:
        probabilities = np.zeros((1, 28), dtype=np.float64)
        probabilities[0, 0] = 0.5
        probabilities[0, 1] = 0.5
        self.assertAlmostEqual(float(expected_strength(probabilities)[0]), 9.5)
        self.assertIn(int(decode_probabilities(probabilities, "expected")[0]), (0, 1))

    def test_dual_model_outputs_two_heads(self) -> None:
        model = DualGoRankCNN(tabular_features=6)
        output = model(torch.zeros(2, 22, 19, 19), torch.zeros(2, 6))
        self.assertEqual(tuple(output[0].shape), (2, 28))
        self.assertEqual(tuple(output[1].shape), (2, 28))


if __name__ == "__main__":
    unittest.main()
