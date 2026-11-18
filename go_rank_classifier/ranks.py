from __future__ import annotations

import re

RANKS: tuple[str, ...] = (
    ("Pro",)
    + tuple(f"{value}d" for value in range(9, 0, -1))
    + tuple(f"{value}k" for value in range(1, 19))
)

RANK_TO_ID: dict[str, int] = {rank: index for index, rank in enumerate(RANKS)}
ID_TO_RANK: dict[int, str] = {index: rank for rank, index in RANK_TO_ID.items()}
STRENGTHS: tuple[float, ...] = tuple(10.0 - index for index in range(len(RANKS)))
ID_TO_STRENGTH: dict[int, float] = dict(enumerate(STRENGTHS))

_DAN_PATTERN = re.compile(r"^(?:P|PRO|D)?\s*(\d{1,2})\s*(?:D|段)?$", re.IGNORECASE)
_KYU_PATTERN = re.compile(r"^(\d{1,2})\s*(?:K|级)$", re.IGNORECASE)


def rank_id_from_sgf_label(value: str | None) -> int | None:
    """Normalize an SGF BR/WR value to the 28-class amateur/Pro mapping."""
    if value is None:
        return None
    cleaned = value.strip().upper().replace(" ", "")
    if not cleaned:
        return None
    if cleaned in {"PRO", "P", "10D"}:
        return RANK_TO_ID["Pro"]
    if "P" in cleaned or cleaned.startswith("PRO"):
        p_match = re.search(r"(\d{1,2})", cleaned)
        if p_match and 1 <= int(p_match.group(1)) <= 9:
            return RANK_TO_ID["Pro"]

    dan = _DAN_PATTERN.match(cleaned)
    if dan:
        number = int(dan.group(1))
        if 1 <= number <= 9 and ("D" in cleaned or "段" in cleaned):
            return RANK_TO_ID[f"{number}d"]
        if 1 <= number <= 9 and "P" in cleaned:
            return RANK_TO_ID["Pro"]
        if number == 10 and ("D" in cleaned or cleaned == "10"):
            return RANK_TO_ID["Pro"]

    kyu = _KYU_PATTERN.match(cleaned)
    if kyu:
        number = int(kyu.group(1))
        if 1 <= number <= 18:
            return RANK_TO_ID[f"{number}k"]
    return None


def rank_from_directory(directory_name: str) -> str:
    """Return the canonical rank name for an archive directory."""
    if directory_name == "Pro":
        return "Pro"
    if directory_name in RANK_TO_ID:
        return directory_name
    raise ValueError(f"Unknown rank directory: {directory_name}")

