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

## Coach dashboard

The coach workspace at `http://127.0.0.1:8000/coach` summarizes every
canonical person in the history database. It shows archive and analysis
coverage, long-term progress metrics, and the sync state of each public account.
Use **新建学员** in the workspace to create a canonical profile and register
exact PGN names or platform usernames as aliases; CLI creation remains available
for scripted imports.

Open a student's **数据来源** section to add games:

- **Chess.com / Lichess**: choose **绑定平台**, enter the exact username, and
  the first official API sync starts immediately. Each sync inspects at most the
  latest 200 standard games; later syncs fetch only new or changed archives.
- **ChessBase Players**: save the player URL as a reference, then export games
  to PGN and use **导入 PGN**. Add the exact name used in its PGN files (for
  example `Wu,S`) while binding the link. ChessBase Players does not publish a
  stable PGN API, so the application does not scrape its pages.
- **Other tournament sites or databases**: download PGN and use **导入 PGN**.
  The importer keeps only games whose White or Black header exactly matches one
  of the student's registered aliases, skips unrelated games, and deduplicates
  repeated imports.

Use **引擎分析** on a student's profile to analyze the latest 20, 40, or 100
games. The dashboard submits one game per request and saves each result
immediately. Stopping, refreshing, or restarting the server does not discard
completed work; starting the same window again skips its cached games and
continues. Trend metrics become available after 20 dated games are analyzed.
The student detail view also exposes the full cached report: evidence-backed
strengths and weaknesses, training priorities, phase accuracy, middlegame risk
windows, winning-position conversion, resilience, endgame outcomes, and links
to representative blunders. A diagnosis is withheld below 10 analyzed games;
phase conclusions require at least 30 moves and scenario conclusions require at
least 5 qualifying games.

```bash
export CHESS_REVIEW_HISTORY_DB=data/player-history.sqlite
chess-review web
```

Local loopback access works without credentials. Before exposing the workspace
remotely, set a strong `CHESS_REVIEW_COACH_TOKEN`; `/coach` and its API then use
HTTP Basic authentication (`coach` as the username, token as the password). A remote
request is rejected when the token is absent. On Fly.io, keep the history
database on the mounted volume and configure authentication before deployment:

```bash
fly secrets set CHESS_REVIEW_COACH_TOKEN='<strong-random-token>' \
  -a ricky-chess-review
```

## Player tracking report

```bash
chess-review report examples/sample.pgn --player "Test Player" --out reports
# multiple files, deeper search:
chess-review report 2025/*.pgn --player "Test Player" --depth 16
```

Outputs `reports/<player>-report.md` and `.html`.

For an ongoing player archive, use the versioned history database instead of
re-running every PGN whenever new games arrive:

```bash
# Create one canonical person. Keep the returned person_id.
chess-review history person add --name "Sibo Wu" \
  --alias "Wu, Sibo" \
  --db data/player-history.sqlite

# Bind official public accounts. Use a provider's stable ID when available.
chess-review history account add --person person_<id> \
  --source lichess --username <lichess_username> \
  --db data/player-history.sqlite
chess-review history account add --person person_<id> \
  --source chess.com --external-id <chess.com_player_id> \
  --username <chess.com_username> \
  --db data/player-history.sqlite

# Keep the returned account_id, then fetch only new or changed archives.
export CHESS_REVIEW_USER_AGENT="chess-review/0.1 (coach@example.com)"
chess-review history sync --account <account_id> \
  --max-games 200 \
  --db data/player-history.sqlite

# Import is cheap and idempotent: repeated PGNs are deduplicated.
chess-review history ingest games/2026-*.pgn \
  --db data/player-history.sqlite

# Only games missing from this exact engine/book/database profile are analyzed.
# person-id includes every registered PGN alias for the student.
chess-review history analyze --person-id person_<id> \
  --db data/player-history.sqlite \
  --limit 20 \
  --depth 18 --threads 1 \
  --master-db data/master-openings.sqlite

# Rendering a cached report does not start Stockfish.
chess-review history report --person-id person_<id> \
  --db data/player-history.sqlite \
  --out reports

chess-review history info --player "Test Player" \
  --db data/player-history.sqlite
```

The cache profile records the Stockfish binary and search settings,
`python-chess` version, analysis source hashes, opening-book hashes, and master
database identity. Changing any of them creates a separate profile, so reports
never silently mix results produced at different depths or by different engine
versions. Canonical people can reserve multiple exact, case-insensitive PGN
aliases and bind multiple public accounts. Lichess sync uses a timestamp cursor;
Chess.com sync uses monthly archives and ETags. Set `LICHESS_TOKEN` only when a
Lichess account or endpoint requires authentication; tokens are never accepted
as command-line arguments. Sources without an official public game API, such as
365Chess, can be retained as profile references but are not scraped.

History reports compare equal chronological windows when at least 20 dated games
are available: 10 vs 10 initially, growing to 20 vs 20. The report labels the
overall direction and each metric as improving, stable, declining, mixed, or
insufficient, with sample size and confidence. Undated games remain in lifetime
totals but are excluded from trend windows. Legacy `--player` commands remain
available for archives that have not yet created canonical people. For history
analysis, `--limit` selects the newest games first.

## Versioned training dataset

Export engine-grounded, two-pass teacher examples for later fine-tuning:

```bash
chess-review dataset games/*.pgn \
  --out datasets/chess-review.jsonl \
  --max-samples 100
```

The exporter uses local Stockfish with one thread by default, rejects any LLM
response that violates the fact or output contract, and keeps every game in a
single deterministic train/dev/test split. It writes canonical JSONL plus a
sidecar manifest containing engine settings and hashes for the engine binary,
prompts, detectors, and opening books. A configured LLM teacher is required.

## Local master opening database

Download one or more monthly PGN ZIP files from the
[Lichess Elite Database](https://database.nikonoel.fr/), then build a compact,
transposition-aware SQLite opening index:

```bash
chess-review master-db build downloads/lichess_elite_2025-*.zip \
  --out data/master-openings.sqlite \
  --max-ply 40
```

The builder streams compressed archives, stores no player identities, removes
duplicate games, and records source SHA-256 hashes and build filters. Existing
databases are protected unless `--force` is supplied. If building from an
unfiltered PGN source, add `--min-rating 2300` (both players must qualify).

Inspect the provenance or query a position directly:

```bash
chess-review master-db info --db data/master-openings.sqlite
chess-review master-db query --db data/master-openings.sqlite \
  --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

Enable the database when generating a game or player report:

```bash
chess-review review games/example.pgn --master-db data/master-openings.sqlite
chess-review report games/*.pgn --player "Player Name" \
  --master-db data/master-openings.sqlite
```

For the web app, set `CHESS_REVIEW_MASTER_DB=data/master-openings.sqlite`.
Reports show master-game frequency and result statistics beside Stockfish's
choice. Popularity is empirical evidence only and never changes engine scores,
move classifications, or the reported engine-best move. Positions with fewer
than 50 indexed games are omitted; positions with fewer than 200 are marked as
small samples.

## Key options

| Option | Meaning |
|---|---|
| `--engine PATH` | Path to a Stockfish binary (otherwise auto-detected). |
| `--depth N` | Engine search depth (default 13; raise for accuracy, lower for speed). |
| `--threads N` | Engine threads (default 1). |
| `--master-db PATH` | Optional local SQLite master opening database. |
| `--format md,html` | Which report formats to write. |
| `--threshold CP` | (review) minimum centipawn loss to list as a critical moment. |
| `--limit N` | Limit the selected command; history analysis uses the newest N games. |
| `--max-games N` | (history sync) inspect at most the newest N standard games (default 200). |

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
