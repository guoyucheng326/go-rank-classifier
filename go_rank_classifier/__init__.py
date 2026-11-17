"""Fox Go rank classification from SGF move records."""

from .ranks import RANKS, RANK_TO_ID, rank_from_directory

__all__ = ["RANKS", "RANK_TO_ID", "rank_from_directory"]
__version__ = "0.1.0"

