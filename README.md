# chess-review

A coach-facing chess game review agent. It runs every move of a game through
Stockfish, measures **centipawn loss**, flags blunders and mistakes, detects
where the game **left opening theory**, and produces two kinds of reports:

1. **Single-game review** — a fast, printable report of one game (per-side ACPL,
   phase breakdown, the turning point, and each critical moment with a board
   diagram, the engine's best move, and a lichess analysis link). A coach can
   read it without replaying the game.
2. **Player tracking report** — aggregates many games for one player into the
   same kind of diagnosis used in the sample coaching report: where points are
   lost by phase, middlegame failure rate by move number, conversion of winning
   positions, outcome by evaluation entering the endgame, resilience when
   behind, and the biggest blunders.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Setup (engine + opening book)

```bash
chess-review setup
```

This locates Stockfish (installing via Homebrew on macOS if missing) and
downloads the lichess opening database used for theory-deviation detection.
If you already have Stockfish, you can point to it any time with `--engine`.

## Single-game review

```bash
chess-review review examples/sample.pgn --game-index 0 --out reports
# focus critical moments on one player:
chess-review review game.pgn --player "Test Player" --game-index all
# dual perspective (student review): full report for BOTH sides
chess-review review game.pgn --both
```

Outputs `reports/<white>-vs-<black>-<date>.md` and `.html`. The HTML report
embeds board diagrams (green arrow = engine's best move, red = move played).
With `--both`, the report contains a separate 白方视角 / 黑方视角 section, each
with its own narrative summary, turning point, and per-move explanations.

## Web frontend (drag-and-drop)

```bash
chess-review web            # then open http://127.0.0.1:8000
chess-review web --port 8123
```

Drag a `.pgn` onto the page (or paste PGN text) and pick a mode:

1. **学员对局分析（双方视角）** — one game, full report for both sides.
2. **某位棋手对局（单方视角）** — one game, focused on a named player.
3. **棋手历史回测** — import many games and aggregate one player's long-term
   stats (as much as the current player report supports; more coming).

The report renders inline and can be opened in a new tab.

## Player tracking report

```bash
chess-review report examples/sample.pgn --player "Test Player" --out reports
# multiple files, deeper search:
chess-review report 2025/*.pgn --player "Test Player" --depth 16
```

Outputs `reports/<player>-report.md` and `.html`.

## Key options

| Option | Meaning |
|---|---|
| `--engine PATH` | Path to a Stockfish binary (otherwise auto-detected). |
| `--depth N` | Engine search depth (default 13; raise for accuracy, lower for speed). |
| `--threads N` | Engine threads (default 1). |
| `--format md,html` | Which report formats to write. |
| `--threshold CP` | (review) minimum centipawn loss to list as a critical moment. |
| `--limit N` | (report) analyze only the first N games for a quick pass. |

## How it works

- **Centipawn loss**: each position is evaluated once at a fixed depth; a move's
  loss is the best available evaluation minus the evaluation after the move
  played (from the mover's perspective), clamped so forced mates don't distort
  aggregates.
- **Classification**: blunder ≥ 200cp, mistake 100–200cp, inaccuracy 50–100cp.
- **Phase**: opening = first 15 moves; endgame = non-pawn material ≤ 26 (≤ 20
  with queens on); otherwise middlegame.
- **Opening theory**: the game is walked against the lichess opening database;
  the first move that leaves every known line is the deviation point.

## Development

```bash
pip install -e . pytest
pytest            # engine-free unit tests
```

## Layout

```
src/chess_review/
  analysis.py      per-move engine analysis (centipawn loss, phases)
  metrics.py       aggregate a player's games into a tracking report
  engine.py        Stockfish wrapper (normalized scores)
  opening_book.py  theory-deviation detection
  classify.py      loss thresholds + phase detection
  render.py        Markdown + HTML rendering (board diagrams, lichess links)
  templates/       Jinja2 HTML templates
  cli.py           `setup`, `review`, `report`
```
