from __future__ import annotations

RANKS: tuple[str, ...] = (
    ("Pro",)
    + tuple(f"{value}d" for value in range(9, 0, -1))
    + tuple(f"{value}k" for value in range(1, 19))
)

RANK_TO_ID: dict[str, int] = {rank: index for index, rank in enumerate(RANKS)}
ID_TO_RANK: dict[int, str] = {index: rank for rank, index in RANK_TO_ID.items()}


def rank_from_directory(directory_name: str) -> str:
    """Return the canonical rank name for an archive directory."""
    if directory_name == "Pro":
        return "Pro"
    if directory_name in RANK_TO_ID:
        return directory_name
    raise ValueError(f"Unknown rank directory: {directory_name}")

