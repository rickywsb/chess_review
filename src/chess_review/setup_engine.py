"""One-time setup: verify/locate Stockfish and download the opening book."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import urllib.request

from .engine import resolve_engine_path

_LICHESS_OPENINGS_BASE = (
    "https://raw.githubusercontent.com/lichess-org/chess-openings/master"
)
_OPENING_FILES = ["a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv"]


def data_dir() -> str:
    d = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(d, exist_ok=True)
    return d


def opening_book_path() -> str:
    return os.path.join(data_dir(), "openings.tsv")


def download_opening_book(force: bool = False) -> str:
    """Download the lichess chess-openings TSVs and merge them into one file."""
    dest = opening_book_path()
    if os.path.exists(dest) and not force:
        print(f"Opening book already present: {dest}")
        return dest

    rows: list[str] = []
    header_written = False
    for fname in _OPENING_FILES:
        url = f"{_LICHESS_OPENINGS_BASE}/{fname}"
        print(f"Downloading {url} ...")
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (trusted host)
            text = resp.read().decode("utf-8")
        lines = text.splitlines()
        if not lines:
            continue
        if not header_written:
            rows.append(lines[0])
            header_written = True
        rows.extend(lines[1:])

    with open(dest, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")

    print(f"Opening book written: {dest} ({len(rows) - 1} lines)")
    return dest


def _try_brew_install_stockfish() -> bool:
    if platform.system() != "Darwin":
        return False
    if not shutil.which("brew"):
        return False
    print("Installing Stockfish via Homebrew ...")
    try:
        subprocess.run(["brew", "install", "stockfish"], check=True)
        return True
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"Homebrew install failed: {exc}")
        return False


def ensure_engine(explicit: str | None = None) -> str | None:
    """Locate Stockfish, attempting a platform-appropriate install if missing."""
    try:
        path = resolve_engine_path(explicit)
        print(f"Stockfish found: {path}")
        return path
    except FileNotFoundError:
        pass

    print("Stockfish not found on this system.")
    if _try_brew_install_stockfish():
        try:
            path = resolve_engine_path()
            print(f"Stockfish installed: {path}")
            return path
        except FileNotFoundError:
            pass

    system = platform.system()
    hint = {
        "Darwin": "brew install stockfish",
        "Linux": "sudo apt-get install stockfish  (or download from stockfishchess.org)",
        "Windows": "Download from https://stockfishchess.org/download/ and add to PATH",
    }.get(system, "Install Stockfish from https://stockfishchess.org/download/")
    print(f"Please install Stockfish manually: {hint}")
    print("Then re-run, or pass --engine /path/to/stockfish.")
    return None


def run_setup(explicit_engine: str | None = None, skip_book: bool = False,
              force_book: bool = False) -> None:
    ensure_engine(explicit_engine)
    if not skip_book:
        try:
            download_opening_book(force=force_book)
        except Exception as exc:  # network issues shouldn't abort setup
            print(f"Could not download opening book: {exc}")
            print("Opening-deviation detection will be skipped until the book is available.")
