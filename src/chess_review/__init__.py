"""chess-review: coach-facing chess game analysis."""
from .analysis import analyze_game
from .engine import Engine, resolve_engine_path
from .metrics import build_player_report
from .models import GameAnalysis, MoveAnalysis
from .opening_book import OpeningBook

__version__ = "0.1.0"

__all__ = [
    "analyze_game",
    "Engine",
    "resolve_engine_path",
    "build_player_report",
    "GameAnalysis",
    "MoveAnalysis",
    "OpeningBook",
]
