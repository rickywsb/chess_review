"""Versioned, provider-neutral training dataset records and JSONL export."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Iterable, Optional

import chess.pgn

from . import coach_llm
from .opening_book import OpeningBook
from .polyglot_book import get_default_book
from .render import DETECTOR_VERSION

SCHEMA_VERSION = "1.0"
SCHEMA_ID = "chess-review://schemas/training-record/1.0"
SPLIT_VERSION = "game-sha256-80-10-10-v1"
VALID_SPLITS = {"train", "dev", "test"}


def _canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _sha256_value(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_identity(path: Optional[str]) -> Optional[dict]:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "filename": os.path.basename(path),
        "sha256": digest.hexdigest(),
    }


def game_id_for(game: chess.pgn.Game) -> str:
    """Return a stable, identity-free hash that groups all moves in a game."""
    board = game.board()
    payload = {
        "initial_fen": board.fen(),
        "moves": [move.uci() for move in game.mainline_moves()],
    }
    return _sha256_value(payload)


def split_for_game(game_id: str) -> str:
    """Assign one whole game to a deterministic 80/10/10 split."""
    bucket = int(game_id[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def build_provenance(engine_metadata: dict,
                     opening_book: Optional[OpeningBook] = None) -> dict:
    """Capture every implementation input that can change generated labels."""
    engine = dict(engine_metadata)
    engine_path = engine.pop("path", None)
    engine["binary"] = _file_identity(engine_path)
    detector_path = os.path.join(os.path.dirname(__file__), "render.py")
    polyglot = get_default_book()
    try:
        package_version = version("chess-review")
    except PackageNotFoundError:
        package_version = "unknown"
    return {
        "generator": {
            "package": "chess-review",
            "package_version": package_version,
            "schema_version": SCHEMA_VERSION,
        },
        "engine": engine,
        "prompt": coach_llm.prompt_metadata(),
        "detector": {
            "version": DETECTOR_VERSION,
            "source": _file_identity(detector_path),
        },
        "opening_books": {
            "named": _file_identity(opening_book.path) if opening_book else None,
            "polyglot": _file_identity(polyglot.path),
        },
        "split": {"version": SPLIT_VERSION},
    }


def build_record(*, game_id: str, split: str, ply: int, context: dict,
                 generation: dict, provenance: dict,
                 source: Optional[dict] = None) -> dict:
    """Build and revalidate one canonical training example."""
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid dataset split: {split}")
    judge = coach_llm._validate_judge(context, generation.get("judge"))
    target = coach_llm._validate_writer(context, generation.get("target"))
    verification = generation.get("verification")
    if judge is None or target is None or not isinstance(verification, dict):
        raise ValueError("Training generation failed contract validation")
    if not all(verification.get(key) is True for key in
               ("judge_contract", "writer_contract", "fail_closed")):
        raise ValueError("Training generation is not verified fail-closed")

    core = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "game_id": game_id,
        "split": split,
        "source": {
            "ply": ply,
            "move_number": context.get("move_number"),
            "side": context.get("side"),
            "played": context.get("played"),
            **(source or {}),
        },
        "provenance": provenance,
        "input": context,
        "judge": judge,
        "target": target,
        "verification": {
            **verification,
            "schema_contract": True,
        },
    }
    return {"sample_id": _sha256_value(core), **core}


def validate_record(record: dict) -> bool:
    """Recheck a serialized record, including its deterministic sample ID."""
    if not isinstance(record, dict) or record.get("schema_id") != SCHEMA_ID or \
            record.get("schema_version") != SCHEMA_VERSION:
        return False
    if record.get("split") not in VALID_SPLITS:
        return False
    context = record.get("input")
    if not isinstance(context, dict):
        return False
    if coach_llm._validate_judge(context, record.get("judge")) is None:
        return False
    if coach_llm._validate_writer(context, record.get("target")) is None:
        return False
    verification = record.get("verification")
    if not isinstance(verification, dict) or not all(
            verification.get(key) is True for key in
            ("judge_contract", "writer_contract", "fail_closed", "schema_contract")):
        return False
    sample_id = record.get("sample_id")
    core = {key: value for key, value in record.items() if key != "sample_id"}
    return isinstance(sample_id, str) and sample_id == _sha256_value(core)


def write_jsonl(path: str, records: Iterable[dict], *, provenance: dict,
                stats: Optional[dict] = None) -> str:
    """Atomically write validated records and a sidecar manifest."""
    records = list(records)
    if any(not validate_record(record) for record in records):
        raise ValueError("Refusing to write an invalid training record")
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(_canonical_json(record) + "\n")
    os.replace(temp_path, path)

    root, _ext = os.path.splitext(path)
    manifest_path = root + ".manifest.json"
    split_counts = Counter(record["split"] for record in records)
    manifest = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "filename": os.path.basename(path),
            "sha256": _file_identity(path)["sha256"],
            "records": len(records),
            "splits": {name: split_counts.get(name, 0)
                       for name in ("train", "dev", "test")},
        },
        "provenance": provenance,
        "stats": stats or {},
    }
    manifest_temp = manifest_path + ".tmp"
    with open(manifest_temp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, sort_keys=True, indent=2)
        fh.write("\n")
    os.replace(manifest_temp, manifest_path)
    return manifest_path