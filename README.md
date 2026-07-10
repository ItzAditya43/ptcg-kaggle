# Pokémon TCG AI Battle Challenge — Agent Repo

Rule-based agents and evaluation tooling for the **Pokémon TCG AI Battle Challenge**
(Kaggle × The Pokémon Company × Matsuo Lab (松尾研) × HEROZ).

The task: write `agent(obs_dict) -> list[int]` that plays the standard-format
Pokémon Trading Card Game and wins on an automated Elo ladder. This repo holds the
shipped ladder agent (`agents/search_agent`), a set of earlier/experimental deck
agents, the shared `cg/` engine bindings, and a `tools/` suite for local evaluation.

> Official competition materials (the engine, sample notebooks, card data, rules)
> are **not** included here — they are available on the Kaggle competition page and
> placed locally under `docs/official/models/cg-lib/`. See [Setup](#setup--local-dev).

---

## Project overview

- **Engine:** the official `cabt` environment (`kaggle_environments` 1.30.1) plus the
  bundled `cg/` Python bindings around the compiled `libcg` engine. Each agent dir
  carries its own copy of `cg/` so a submission is self-contained.
- **Architecture:** `agents/<name>/` holds one deck + one `main.py` policy;
  `agents/_base/` holds the shared `BasePolicy` (inherited by newer agents) and
  `GenericPolicy` (turns any decklist into a cabt opponent); `tools/` holds the
  evaluation/meta harness; `docs/` and `research/` hold strategy notes and
  superseded experiments.
- **Shipped agent:** `agents/search_agent` — an Alakazam + Dudunsparce single-prize
  pilot augmented with belief-state opponent modeling (Phase 3) and an endgame
  exhaustive solver (Phase 5). See [Current shipped agent](#current-shipped-agent--search_agent).

---

## Current shipped agent — `search_agent`

`agents/search_agent` is the agent that has been submitted to Kaggle (the repo's
designated ladder agent as of this writing). It is an Alakazam + Dudunsparce
single-prize deck piloted by `AlakazamPolicy` (a self-contained legacy-style policy,
not a `BasePolicy` subclass), with three search layers layered on top:

1. **Belief-state opponent modeling (`BeliefTracker`, Phase 3).** A weighted
   hypothesis over the opponent's archetype (crustle / lucario / abomasnow /
   dragapult / iono), updated each turn from revealed board state (Pokémon played,
   energy types, attacks used, discards). Used to sample the opponent's hand/deck
   from the correct distribution inside the search, instead of a naive guess.
2. **Endgame exhaustive solver (Phase 5).** When the remaining state is small
   (`total_pokemon + total_prizes <= ENDGAME_THRESHOLD = 8`), it switches from
   sampling to exhaustive lookahead (up to 20 plies) and picks the action that
   maximizes win probability.
3. **Search-augmented MAIN tiebreaker.** For `MAIN`-context decisions where the
   baseline's top-2 scores are within 15% (uncertainty), it runs 1-ply search
   (`_search_main`) using the belief tracker and overrides the pick if search finds
   a better action. Gated by `SEARCH_ENABLED`.

**`SEARCH_ENABLED` is `True`** in the committed `agents/search_agent/main.py`
(commit `097978b…`, 2026-07-10). The Phase 3 validation harness
(`tools/phase3_validation.py`) measured **89.9%** win rate for the search agent vs a
**79.7%** frozen Phase-2 baseline, over **1000 games** (200 × 5 archetypes), with
**non-overlapping Wilson 95% CIs** per archetype — the basis for shipping.
Methodology: each seat is an independently `exec`'d module (isolated RNG/deck/card
state), seats alternate to cancel first/second-player advantage, opponents are the
consensus decks from `cabt_eval`.

**Kaggle submission status (as of 2026-07-10):** `search_agent` is the current
submitted agent in this repo. Live ladder Elo / scoring status is not stored here —
verify with `venv/bin/kaggle competitions submissions pokemon-tcg-ai-battle`. The
repo's git history shows the last submission-related commits on 2026-07-07
(Alakazam v3 + garchomp) and the search-agent enable on 2026-07-09/10; confirm the
currently-scored pair via the Kaggle CLI before relying on any number.

---

## Other agents

| Agent | Description | Status |
|---|---|---|
| `baseline` | Crustle deck, self-contained reference agent (frozen 79.7% baseline). | baseline reference |
| `alakazam` | Alakazam + Dudunsparce single-prize pilot (legacy `AlakazamPolicy`). | active (pre-search primary) |
| `alakazam_mist` | `alakazam` + 2 Mist Energy mirror tech vs Powerful-Hand counters. | experimental variant |
| `bellibolt` | Iono's Bellibolt ex — simple Lightning engine (historically best ladder). | active / experimental |
| `typhlosion` | Ethan's Typhlosion + Dudunsparce Stage-2 combo (`QuilavaPolicy`). | active / experimental |
| `dragapult` | Dragapult ex — official sample pilot + robust `agent()` wrapper. | active (early primary) |
| `dragapult_nobonus` | `dragapult` variant without the bonus build. | experimental variant |
| `garchomp` | Garchomp ex — `GarchompPolicy(BasePolicy)`, divergence-mined. | active (Phase 3) |
| `megastarmie` | Mega Starmie ex + Cinderace — `MegaStarmiePolicy(BasePolicy)`. | superseded by v2 |
| `megastarmie_v2` | Current Mega Starmie pilot (`BasePolicy` subclass). | active |
| `trevenant` | Hop's Trevenant single-prize aggro (`TrevenantPolicy(BasePolicy)`). | demoted |
| `mewtwo` | Team Rocket's Mewtwo ex (`MewtwoExPolicy(BasePolicy)`). | generic / opponent |
| `chandelure` | GenericPolicy over its `deck.csv` (no `build_submission.sh`). | opponent / generic |
| `froslass` | GenericPolicy over its `deck.csv` (no `build_submission.sh`). | opponent / generic |
| `grimmsnarl` | GenericPolicy over its `deck.csv` (no `build_submission.sh`). | opponent / generic |
| `kangaskhan` | GenericPolicy over its `deck.csv` (no `build_submission.sh`). | opponent / generic |
| `ogerpon` | GenericPolicy over its `deck.csv` (no `build_submission.sh`). | opponent / generic |
| `lucario_v3` | Shared community Lucario pilot with a small search overlay (no `build_submission.sh`). | opponent / reference |

`chandelure`, `froslass`, `grimmsnarl`, `kangaskhan`, `ogerpon` are GenericPolicy
wrappers used as cabt opponents (top-tier field decks); they have `main.py` +
`deck.csv` but no `build_submission.sh` and are not primary submissions.

---

## Repo structure

```
README.md                 # this file
CLAUDE.md                 # project guide for the coding agent (different audience — kept separate)
requirements.txt
agents/
  _base/                  # shared BasePolicy (ABC) + GenericPolicy + make_agent/make_generic_agent
  search_agent/           # ★ SHIPPED agent: Alakazam + belief-state search + endgame solver
    main.py deck.csv build_submission.sh cg/
  baseline/               # Crustle reference agent (frozen 79.7% baseline)
  alakazam/ alakazam_mist/ bellibolt/ typhlosion/ dragapult/ dragapult_nobonus/
  garchomp/ megastarmie/ megastarmie_v2/ trevenant/ mewtwo/ lucario_v3/
  chandelure/ froslass/ grimmsnarl/ kangaskhan/ ogerpon/   # GenericPolicy opponents (no build script)
tools/                    # evaluation + meta harness (see table below)
docs/
  strategy/               # TC strategy write-ups (牌組策略.md, 訓練家牌應用.md)
  per_deck_policy_plan.md # per-deck policy spec
  optimization_protocol.md imitation_learning_pipeline.md (superseded)
  official/               # GITIGNORED: engine cg-lib, sample notebooks, card data
research/                 # early/superseded experiments (MCTS, RL, imitation trainers, deck DB)
web/                      # human-vs-AI browser sandbox (server.py + index.html)
```

---

## Harness & evaluation tools

| Script | What it does |
|---|---|
| `tools/cabt_ab.py` | A/B two of our agent dirs against each other in the official `cabt` env. Loads each as a **pre-built callable** (not a directory path) so module-level state (deck, card data, RNG) is isolated per seat; seats alternate. |
| `tools/cabt_eval.py` | Evaluate our agent in the official `cabt` env vs sample/consensus opponents (crustle/lucario/abomasnow/dragapult/iono/mirror). |
| `tools/cabt_gauntlet.py` | Run an agent against the real current top-100 field, prevalence-weighted. |
| `tools/round_robin.py` | Round-robin A/B across all our agent dirs in the `cabt` env. |
| `tools/check_agent.py` | Generic invariant checker — asserts no energy over-fill, no crashes/fallbacks, legal selections; reports whether an agent is on `BasePolicy`. Run after any agent change. |
| `tools/phase3_validation.py` | Phase 3 validation: search_agent vs frozen 79.7% baseline, 200 games × 5 archetypes, Wilson 95% CIs. |
| `tools/meta_eval.py` | Phase 2 meta-weighted deck evaluation harness. |
| `tools/nightly_recalibrate.py` | Phase 4 nightly recalibration — updates field-share weights and belief priors from new episode data. |
| `tools/endgame_test_set.py` | Phase 5 — extract endgame test positions from real replay data. |
| `tools/meta_analyze.py` | Daily ladder episode → archetype distribution, WR, matchup matrix. |
| `tools/autopsy.py` | One-shot daily pipeline: download episodes + leaderboard → meta + divergence → `/tmp/autopsy/<date>/`. |
| `tools/replay_divergence.py` | Replay top-pilot games through our agent → SelectContext-bucketed agree% (where we pilot differently). |
| `tools/divergence_decode.py` | Like `replay_divergence` but decodes each disagreement into card/attack/option names and aggregates human-vs-ours picks. |
| `tools/baseline_test.py` | Run our agent vs a fixed set of GenericPolicy opponents. |
| `tools/battle_analyze.py` | Battle analyzer — logs event streams and analyzes games (optimization-loop foundation). |
| `tools/analyze_ms_losses.py` | Analyze Megastarmie losses vs specific opponents. |
| `tools/debug_mewtwo.py` | Play 1 Mewtwo game vs Dragapult, log every decision to stdout. |

### Subprocess-isolation gotcha (why `cabt_ab.py` uses callables, not dir paths)

`kaggle_environments`' `build_agent()` does **not** support directory-path agent
loading for the custom `cabt` env: it `open()`s the directory as a file
(`IsADirectoryError`), falls back to the path string, then `compile()`s it as source
(`SyntaxError`). So `cabt_ab.py` / `phase3_validation.py` read each `main.py`,
`compile()` it with `__file__` explicitly set, and `exec()` it into a fresh module
namespace. This also gives **per-seat state isolation** (each agent gets its own
deck/card-data/RNG/DIAG), which is what makes A/B and validation numbers trustworthy
— see [Known gotchas](#known-gotchas).

---

## Known gotchas

Two real bugs were hit and fixed; both will bite again if undocumented.

**(a) In-process RNG / state contamination (old harness).** The original harness
loaded both seats from a shared module (e.g. `get_last_callable`), so module-level
state — deck, card data, RNG, `DIAG` — was shared between the two players. That
contaminated the opponent's RNG with the agent's and produced unreliable win rates.
Fix: each agent is loaded as an **independent `exec`'d module** with its own state
(`load_agent_callable` in `cabt_ab.py`), and every `main.py` does
`sys.path.insert(0, _HERE)` so it finds its bundled `cg/` regardless of cwd. Always
use the callable-loading tools, never an in-process shared import, for A/B or
validation.

**(b) `__file__` NameError under Kaggle's exec-based loader.** Kaggle's runner
`exec`s the agent file into an empty namespace where `__file__` is **undefined**, so
`os.path.dirname(os.path.abspath(__file__))` raises `NameError` and the submission
crashes on import. Fix (commit `097978b`, applied to `search_agent`, `baseline`,
`alakazam`): guard the reference —
```python
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
```
and the tools that `exec` agent source set `__file__` explicitly in the exec env.
The deck-path resolver also falls back to `deck.csv` / `/kaggle_simulations/agent/deck.csv`.

---

## Submission workflow

```bash
# 1. Build a submission tarball (point CG_LIB_PATH at the engine's cg/ folder)
CG_LIB_PATH="$(pwd)/docs/official/models/cg-lib/cg" \
  bash agents/search_agent/build_submission.sh

# 2. Verify tarball contents: top-level main.py + deck.csv + cg/, no leaks
tar -tzf agents/search_agent/submission.tar.gz | head
#   expect: ./main.py  ./deck.csv  ./cg/...
#   build_submission.sh already excludes __pycache__/*.pyc/.git/*.bak/.opp_cache_*.csv

# 3. Submit to the ladder
venv/bin/kaggle competitions submit pokemon-tcg-ai-battle \
  -f agents/search_agent/submission.tar.gz -m "message"

# 4. Check score (latest 2 submissions are scored; 5/day limit)
venv/bin/kaggle competitions submissions pokemon-tcg-ai-battle
```

Competition page: <https://www.kaggle.com/competitions/pokemon-tcg-ai-battle>
(submission = `main.py` + `deck.csv` + `cg/`; 5/day; latest 2 scored).

---

## Setup / local dev

The official engine and assets are not redistributed. From the Kaggle competition
page, download the starter materials and place the local engine at
`docs/official/models/cg-lib/`.

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/pip install kaggle-environments==1.30.1   # same version as the ladder
```

Reproduce the validation locally:

```bash
# A/B two agents (isolated seats, alternating first/second)
venv/bin/python tools/cabt_ab.py agents/search_agent agents/baseline 200

# Full Phase 3 validation: search_agent vs frozen 79.7% baseline, Wilson CIs
venv/bin/python tools/phase3_validation.py

# Belief-tracker smoke test: confirm the tracker updates from a revealed board
venv/bin/python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('sa', 'agents/search_agent/main.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
bt = m.BeliefTracker()
print('initial belief:', {k: round(v,3) for k,v in bt.get_belief().items()})
"
```

Play against an agent in the browser sandbox:

```bash
venv/bin/python web/server.py     # open http://localhost:8000
```

---

## Strategy write-ups

`docs/strategy/` (Traditional Chinese):
- `牌組策略.md` — every deck's strategy, meta distribution, rock-paper-scissors map,
  expected-WR tables, deck-building checklist.
- `訓練家牌應用.md` — every Trainer card by category with application notes and combos.

`CLAUDE.md` is the coding-agent project guide (current status, meta notes, engine
API reference) and is intentionally **kept separate** from this README — different
audience, different update cadence.

---

*Not affiliated with The Pokémon Company or Kaggle. Pokémon and card names are
trademarks of their respective owners; this is independent competition work.*