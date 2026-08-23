"""Core per-game analysis: centipawn loss, classification, phases, openings."""
from __future__ import annotations

from typing import Optional

import chess
import chess.pgn

from .classify import classify_loss, classify_phase
from .engine import Engine
from .models import GameAnalysis, MoveAnalysis
from .opening_book import OpeningBook


def _headers(game: chess.pgn.Game) -> dict:
    return {k: v for k, v in game.headers.items()}


def analyze_game(
    game: chess.pgn.Game,
    engine: Engine,
    book: Optional[OpeningBook] = None,
    progress: bool = False,
) -> GameAnalysis:
    """Analyze one game and return a populated :class:`GameAnalysis`."""
    headers = _headers(game)
    moves = list(game.mainline_moves())

    # Opening theory detection (independent of the engine).
    opening = book.detect(moves) if book is not None else None

    board = game.board()
    result = GameAnalysis(
        white=headers.get("White", "?"),
        black=headers.get("Black", "?"),
        result=headers.get("Result", "*"),
        date=headers.get("Date", headers.get("UTCDate", "")),
        event=headers.get("Event", ""),
        site=headers.get("Site", ""),
        eco=(opening.eco if opening else headers.get("ECO", "")),
        opening_name=(opening.name if opening else headers.get("Opening", "")),
        headers=headers,
    )
    if opening is not None:
        result.deviation_ply = opening.deviation_ply
        result.deviation_side = opening.deviation_side
        result.deviation_move_san = opening.deviation_move_san
        result.deviation_book_san = opening.book_move_san

    deviation_ply = opening.deviation_ply if opening else None
    book_loaded = book is not None and book.loaded

    # First analysis: the starting position.
    prev = engine.analyse(board)

    total = len(moves)
    for idx, move in enumerate(moves):
        ply = idx + 1
        mover = board.turn
        move_number = board.fullmove_number
        fen_before = board.fen()
        san = board.san(move)
        uci = move.uci()

        # A move is "in book" if it is played before the first deviation ply
        # (or the game never left theory). Used for both the phase and the field.
        in_book = deviation_ply is not None and ply < deviation_ply
        if deviation_ply is None and opening is not None:
            in_book = True  # never left theory
        phase = classify_phase(board, in_book=in_book, book_loaded=book_loaded)

        best_move = prev.best_move
        best_move_san = board.san(best_move) if best_move is not None else ""
        best_is_capture = board.is_capture(best_move) if best_move is not None else False
        best_is_check = board.gives_check(best_move) if best_move is not None else False

        # Convert the engine's principal variation into SAN for explanations.
        best_line_san: list[str] = []
        if prev.pv:
            tmp = board.copy(stack=False)
            for mv in prev.pv[:6]:
                try:
                    best_line_san.append(tmp.san(mv))
                    tmp.push(mv)
                except (ValueError, AssertionError):
                    break
        played_is_capture = board.is_capture(move)
        eval_before_mover = prev.cp_mover
        eval_before_white = prev.cp_white
        mate_before = prev.mate_mover

        board.push(move)
        cur = engine.analyse(board)

        # cur is scored from the new side-to-move (the opponent); convert to
        # the mover's perspective by negating the White-POV consistently.
        eval_after_white = cur.cp_white
        eval_after_mover = eval_after_white if mover == chess.WHITE else -eval_after_white
        mate_after = None
        if cur.mate_mover is not None:
            # cur.mate_mover is from opponent POV -> negate for the mover.
            mate_after = -cur.mate_mover

        cp_loss = max(0, eval_before_mover - eval_after_mover)

        result.moves.append(
            MoveAnalysis(
                ply=ply,
                move_number=move_number,
                color=mover,
                san=san,
                uci=uci,
                fen_before=fen_before,
                fen_after=board.fen(),
                eval_before_mover=eval_before_mover,
                eval_after_mover=eval_after_mover,
                eval_before_white=eval_before_white,
                eval_after_white=eval_after_white,
                cp_loss=cp_loss,
                best_move_uci=best_move.uci() if best_move else "",
                best_move_san=best_move_san,
                phase=phase,
                classification=classify_loss(cp_loss),
                best_is_capture=best_is_capture,
                best_is_check=best_is_check,
                played_is_capture=played_is_capture,
                in_book=in_book,
                mate_before=mate_before,
                mate_after=mate_after,
                best_line_san=best_line_san,
            )
        )

        prev = cur
        if progress:
            print(f"\r  analyzing move {ply}/{total}", end="", flush=True)

    if progress and total:
        print()

    result.final_eval_white = prev.cp_white
    return result
