from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .ranks import rank_id_from_sgf_label

PASS_MOVE = 361
MIN_MOVES = 100
MAX_VALID_MOVES = 350
TRUNCATE_MOVES = 300

_MOVE_PATTERN = re.compile(rb";\s*([BW])\s*\[\s*([a-zA-Z]{0,2})\s*\]")
_SIZE_PATTERN = re.compile(rb"(?:^|[;(\s])SZ\s*\[\s*(\d+)\s*\]")
_PROPERTY_PATTERN = re.compile(rb"([A-Z]+)((?:\[(?:\\.|[^\]])*\])+)")
_VALUE_PATTERN = re.compile(rb"\[((?:\\.|[^\]])*)\]")


@dataclass(frozen=True)
class ParsedSGF:
    moves: tuple[int, ...]
    board_size: int
    black_rank: int | None = None
    white_rank: int | None = None
    handicap: int = 0
    rules: str = ""
    komi: float | None = None
    has_setup_stones: bool = False

    @property
    def move_count(self) -> int:
        return len(self.moves)

    def is_trainable(self) -> bool:
        return (
            self.board_size == 19
            and MIN_MOVES <= self.move_count <= MAX_VALID_MOVES
            and self.black_rank is not None
            and self.white_rank is not None
            and self.handicap == 0
            and not self.has_setup_stones
        )

    def truncated_moves(self) -> tuple[int, ...]:
        return self.moves[:TRUNCATE_MOVES]


def _main_line_bytes(game: bytes) -> bytes:
    """Keep only nodes in the root game sequence and drop variations."""
    output = bytearray()
    depth = 0
    in_bracket = False
    escaped = False
    for value in game:
        if in_bracket:
            if escaped:
                escaped = False
            elif value == 92:  # backslash
                escaped = True
            elif value == 93:  # closing bracket
                in_bracket = False
            if depth == 1:
                output.append(value)
            continue

        if value == 91:  # opening bracket
            in_bracket = True
            if depth == 1:
                output.append(value)
        elif value == 40:  # opening parenthesis
            depth += 1
            if depth == 1:
                output.append(value)
        elif value == 41:  # closing parenthesis
            if depth == 1:
                output.append(value)
            depth -= 1
        elif depth == 1:
            output.append(value)
    return bytes(output)


def parse_sgf_bytes(raw: bytes) -> ParsedSGF:
    """Parse a single SGF game using the first variation at every branch."""
    main = _main_line_bytes(raw)
    size_match = _SIZE_PATTERN.search(main)
    board_size = int(size_match.group(1)) if size_match else 19
    properties: dict[str, list[str]] = {}
    for match in _PROPERTY_PATTERN.finditer(main):
        key = match.group(1).decode("ascii", errors="ignore")
        values = [
            value.decode("utf-8", errors="replace")
            for value in _VALUE_PATTERN.findall(match.group(2))
        ]
        properties.setdefault(key, []).extend(values)

    def first(key: str) -> str | None:
        values = properties.get(key)
        return values[0] if values else None

    handicap_text = first("HA")
    try:
        handicap = int(handicap_text) if handicap_text not in (None, "") else 0
    except ValueError:
        handicap = -1
    komi_text = first("KM")
    try:
        komi = float(komi_text) if komi_text not in (None, "") else None
    except ValueError:
        komi = None

    moves: list[int] = []
    for match in _MOVE_PATTERN.finditer(main):
        coordinate = match.group(2).lower()
        if not coordinate:
            moves.append(PASS_MOVE)
            continue
        if len(coordinate) != 2:
            continue
        column = coordinate[0] - ord("a")
        row = coordinate[1] - ord("a")
        if 0 <= row < 19 and 0 <= column < 19:
            moves.append(row * 19 + column)
    return ParsedSGF(
        moves=tuple(moves),
        board_size=board_size,
        black_rank=rank_id_from_sgf_label(first("BR")),
        white_rank=rank_id_from_sgf_label(first("WR")),
        handicap=handicap,
        rules=(first("RU") or "").strip(),
        komi=komi,
        has_setup_stones=bool(properties.get("AB") or properties.get("AW")),
    )


def parse_sgf_text(text: str) -> ParsedSGF:
    return parse_sgf_bytes(text.encode("utf-8", errors="ignore"))


def parse_sgf_file(path: str | Path) -> ParsedSGF:
    return parse_sgf_bytes(Path(path).read_bytes())


def iter_sgf_bytes(chunks: Iterable[bytes]) -> Iterator[bytes]:
    """Split a concatenated stream of SGF files into individual game bytes."""
    game = bytearray()
    depth = 0
    in_bracket = False
    escaped = False

    for chunk in chunks:
        for value in chunk:
            if not game:
                if value == 40:
                    game.append(value)
                    depth = 1
                continue

            game.append(value)
            if in_bracket:
                if escaped:
                    escaped = False
                elif value == 92:
                    escaped = True
                elif value == 93:
                    in_bracket = False
                continue

            if value == 91:
                in_bracket = True
            elif value == 40:
                depth += 1
            elif value == 41:
                depth -= 1
                if depth == 0:
                    yield bytes(game)
                    game.clear()

