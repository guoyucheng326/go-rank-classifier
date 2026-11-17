from __future__ import annotations

import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()

