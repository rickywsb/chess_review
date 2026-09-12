"""Tests for the versioned training dataset contract."""
import io
import json
from argparse import Namespace
from types import SimpleNamespace

import chess
import chess.pgn

from chess_review.dataset import (
    SCHEMA_ID,
    SCHEMA_VERSION,
    VALID_SPLITS,
    build_provenance,
    build_record,
    game_id_for,
    split_for_game,
    validate_record,
    write_jsonl,
)
from chess_review.opening_book import OpeningBook


def _game(white="Alice", black="Bob"):
    text = (f'[White "{white}"]\n[Black "{black}"]\n[Result "*"]\n\n'
            "1. e4 e5 2. Nf3 Nc6 *")
    return chess.pgn.read_game(io.StringIO(text))


def _context():
    return {
        "phase": "中局",
        "side": "白方",
        "move_number": 12,
        "played": "g4",
        "best": "Nf3",
        "pv": "Nf3 Nf6",
        "refutation": "Qh4+",
        "eval_before": "+0.20",
        "eval_after": "-1.40",
        "state_before": "均势",
        "state_after": "略差",
        "cp_loss": 160,
        "reason_tag": "threw_game",
        "category": "big_error",
        "category_zh": "严重失误",
        "resulting_state": "已处于下风",
        "framing": "不要夸大。",
        "subtle": False,
        "best_is_check": False,
        "best_is_capture": False,
        "best_leads_to_mate_in": None,
        "facts": ["【位置】这一步削弱了王翼。"],
    }


def _generation():
    return {
        "judge": {
            "primary": "王翼变弱",
            "use_facts": ["【位置】这一步削弱了王翼。"],
            "honest_state": "已处于下风",
            "avoid": "不要说成强制杀",
        },
        "target": {
            "why": "这步削弱了王翼。",
            "consequence": "对手可以借将军取得主动。",
            "what_to_do": "先检查对手的强制回应。",
        },
        "verification": {
            "judge_contract": True,
            "writer_contract": True,
            "fail_closed": True,
        },
    }


def _provenance(tmp_path):
    engine_path = tmp_path / "stockfish"
    engine_path.write_bytes(b"test-engine")
    opening_path = tmp_path / "openings.tsv"
    opening_path.write_text("eco\tname\tpgn\n", encoding="utf-8")
    book = OpeningBook()
    book.path = str(opening_path)
    return build_provenance({
        "name": "Stockfish Test",
        "author": "Test",
        "path": str(engine_path),
        "depth": 18,
        "movetime_ms": None,
        "threads": 1,
        "hash_mb": 16,
    }, book)


def test_game_id_ignores_identity_headers_and_split_is_stable():
    first = game_id_for(_game("Alice", "Bob"))
    second = game_id_for(_game("Carol", "Dave"))
    assert first == second
    assert split_for_game(first) == split_for_game(second)
    assert split_for_game(first) in VALID_SPLITS


def test_record_is_deterministic_and_detects_tampering(tmp_path):
    game_id = game_id_for(_game())
    kwargs = {
        "game_id": game_id,
        "split": split_for_game(game_id),
        "ply": 23,
        "context": _context(),
        "generation": _generation(),
        "provenance": _provenance(tmp_path),
    }
    first = build_record(**kwargs)
    second = build_record(**kwargs)
    assert first == second
    assert first["schema_id"] == SCHEMA_ID
    assert first["schema_version"] == SCHEMA_VERSION
    assert validate_record(first)

    tampered = json.loads(json.dumps(first, ensure_ascii=False))
    tampered["target"]["why"] = "被修改的解释。"
    assert not validate_record(tampered)


def test_write_jsonl_emits_hash_checked_manifest(tmp_path):
    provenance = _provenance(tmp_path)
    game_id = game_id_for(_game())
    record = build_record(
        game_id=game_id,
        split=split_for_game(game_id),
        ply=23,
        context=_context(),
        generation=_generation(),
        provenance=provenance,
    )
    output = tmp_path / "corpus.jsonl"
    manifest_path = write_jsonl(
        str(output), [record], provenance=provenance,
        stats={"games": 1, "accepted": 1},
    )

    stored = json.loads(output.read_text(encoding="utf-8"))
    manifest = json.loads(open(manifest_path, encoding="utf-8").read())
    assert stored == record
    assert manifest["dataset"]["records"] == 1
    assert manifest["dataset"]["sha256"]
    assert manifest["stats"]["accepted"] == 1
    assert manifest["provenance"]["engine"]["binary"]["filename"] == "stockfish"
    assert "path" not in manifest["provenance"]["engine"]


def test_dataset_command_exports_verified_record(tmp_path, monkeypatch):
    from chess_review import cli

    class FakeEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def metadata(self):
            engine_path = tmp_path / "stockfish"
            engine_path.write_bytes(b"fake")
            return {
                "name": "Stockfish Test", "author": "Test",
                "path": str(engine_path), "depth": 18,
                "movetime_ms": None, "threads": 1, "hash_mb": 16,
            }

    move = SimpleNamespace(
        ply=1, color=chess.WHITE,
        fen_before=chess.STARTING_FEN,
        fen_after=chess.Board().fen(),
        uci="e2e4", best_move_uci="d2d4",
        best_line_san=["d4", "d5"], refutation_line_san=["e5"],
    )
    analysis = SimpleNamespace(
        moves=[move],
        player_color=lambda _name: chess.WHITE,
    )
    monkeypatch.setattr(cli, "Engine", FakeEngine)
    monkeypatch.setattr(cli, "analyze_game", lambda *_args, **_kwargs: analysis)
    monkeypatch.setattr(cli, "significance", lambda *_args: (True, "threw_game"))
    monkeypatch.setattr(cli, "build_explanation_context", lambda *_args: _context())
    monkeypatch.setattr(cli.coach_llm, "available", lambda: True)
    monkeypatch.setattr(cli.coach_llm, "generate_training_explanation",
                        lambda _context: _generation())
    monkeypatch.setattr(cli.OpeningBook, "load", lambda: OpeningBook())

    pgn_path = tmp_path / "game.pgn"
    pgn_path.write_text(
        '[White "Alice"]\n[Black "Bob"]\n[Result "*"]\n\n1. e4 e5 *',
        encoding="utf-8",
    )
    output = tmp_path / "dataset.jsonl"
    args = Namespace(
        pgn=[str(pgn_path)], out=str(output), player=None, limit=None,
        max_samples=10, threshold=100, engine=None, depth=18,
        threads=1, hash_mb=16,
    )
    assert cli.cmd_dataset(args) == 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert validate_record(record)
    assert record["provenance"]["engine"]["threads"] == 1
    assert record["source"]["fen_before"] == chess.STARTING_FEN
    assert record["source"]["played_uci"] == "e2e4"
    manifest = json.loads((tmp_path / "dataset.manifest.json").read_text(encoding="utf-8"))
    assert manifest["stats"]["teacher_calls"] == 1


def test_dataset_parser_rejects_non_positive_call_limit():
    from chess_review.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["dataset", "game.pgn"])
    assert args.threads == 1
    assert args.max_samples == 100
    try:
        parser.parse_args(["dataset", "game.pgn", "--max-samples", "0"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("non-positive --max-samples should be rejected")