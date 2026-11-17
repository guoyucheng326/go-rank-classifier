from __future__ import annotations

from collections.abc import Sequence

import numpy as np

BOARD_SIZE = 19
PASS_MOVE = 361
FEATURE_CHANNELS = 22


def transform_moves(moves: Sequence[int], symmetry: int) -> np.ndarray:
    """Apply one of the eight dihedral board symmetries to move coordinates."""
    if symmetry not in range(8):
        raise ValueError("symmetry must be in [0, 7]")
    result = np.asarray(moves, dtype=np.int16).copy()
    active = result < PASS_MOVE
    if not active.any():
        return result

    row = result[active] // BOARD_SIZE
    column = result[active] % BOARD_SIZE
    if symmetry >= 4:
        column = (BOARD_SIZE - 1) - column
    rotation = symmetry % 4
    if rotation == 1:
        row, column = column, (BOARD_SIZE - 1) - row
    elif rotation == 2:
        row = (BOARD_SIZE - 1) - row
        column = (BOARD_SIZE - 1) - column
    elif rotation == 3:
        row, column = (BOARD_SIZE - 1) - column, row
    result[active] = row.astype(np.int16) * BOARD_SIZE + column.astype(np.int16)
    return result


def board_features(moves: Sequence[int], cutoff: int) -> np.ndarray:
    """Create a compact spatial tensor containing patterns and recent history."""
    if not 1 <= cutoff <= len(moves):
        raise ValueError(f"cutoff={cutoff} must be within [1, {len(moves)}]")

    selected = np.asarray(moves[:cutoff], dtype=np.int16)
    channels = np.zeros((FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    def add_range(
        start_channel: int,
        start_move: int,
        end_move: int,
        normalize: float,
    ) -> None:
        values = selected[start_move:end_move]
        if values.size == 0:
            return
        original_indices = np.arange(start_move, end_move)
        playable = values < PASS_MOVE
        values = values[playable]
        original_indices = original_indices[playable]
        rows, columns = np.divmod(values, BOARD_SIZE)
        is_black = original_indices % 2 == 0
        np.add.at(
            channels[start_channel],
            (rows[is_black], columns[is_black]),
            1.0,
        )
        np.add.at(
            channels[start_channel + 1],
            (rows[~is_black], columns[~is_black]),
            1.0,
        )
        channels[start_channel : start_channel + 2] /= max(normalize, 1.0)

    add_range(0, 0, selected.size, selected.size / 2.0)
    if selected.size > 100:
        add_range(2, 100, selected.size, (selected.size - 100) / 2.0)
    add_range(4, max(0, selected.size - 32), selected.size, min(32, selected.size) / 2.0)

    for lag in range(8):
        channel = 6 + 2 * lag
        if lag >= selected.size:
            break
        move = int(selected[-1 - lag])
        if move == PASS_MOVE:
            continue
        row, column = divmod(move, BOARD_SIZE)
        color_channel = channel if selected.size - 1 - lag >= 0 else channel
        channels[color_channel, row, column] = (
            1.0 if (selected.size - 1 - lag) % 2 == 0 else 0.0
        )
        channels[color_channel + 1, row, column] = (
            1.0 if (selected.size - 1 - lag) % 2 == 1 else 0.0
        )
    return channels

