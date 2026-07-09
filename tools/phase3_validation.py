#!/usr/bin/env python3
"""Phase 3 Validation: belief-state search_agent vs frozen Phase 2 baseline.

Measures search_agent (SEARCH_ENABLED=True) against each of the 5 archetype
opponents for 200 games each. Reports Wilson 95% CIs per archetype and aggregate,
then compares against the 79.7% frozen baseline.

Usage:
  venv/bin/python tools/phase3_validation.py

Output:
  Per-archetype and aggregate win rates with Wilson CIs, plus Ship/Flat/Worse verdict.
"""
import sys, os, math, json, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT + '/docs/official/models/cg-lib')

# ── Archetypes (matching cabt_eval / meta_eval) ─────────────────────────────
ARCHE_TYPES = ['crustle', 'lucario', 'abomasnow', 'dragapult', 'iono']
FIELD_WEIGHTS = {'crustle': 0.50, 'lucario': 0.18, 'abomasnow': 0.12,
                 'dragapult': 0.12, 'iono': 0.08}

# Frozen baseline (from Phase 2 measurement)
BASELINE_WR = 0.797   # 79.7%
GAMES_PER_ARCH = 200

# ── Wilson 95% confidence interval ──────────────────────────────────────────
def wilson_ci(wins, total, z=1.96):
    """Compute Wilson score 95% CI for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    p_hat = wins / total
    denominator = 1 + z**2 / total
    centre = (p_hat + z**2 / (2 * total)) / denominator
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2))) / denominator
    return (centre - margin, centre + margin)

# ── Load our agent (search_agent with SEARCH_ENABLED=True) ──────────────────
def load_search_agent():
    """Load search_agent with SEARCH_ENABLED forced True."""
    agent_dir = 'agents/search_agent'
    full_dir = ROOT + '/' + agent_dir
    
    # Ensure cg lib exists
    if not os.path.exists(full_dir + '/cg'):
        import shutil
        shutil.copytree(ROOT + '/docs/official/models/cg-lib/cg', full_dir + '/cg')
    
    # Read source and patch SEARCH_ENABLED to True
    main_py = full_dir + '/main.py'
    source = open(main_py).read()
    # Force SEARCH_ENABLED = True (do NOT modify the file, just the compiled code)
    source_replaced = source.replace(
        'SEARCH_ENABLED = False  # Phase 1 gate not passed',
        'SEARCH_ENABLED = True   # Phase 3 validation override'
    )
    
    cur = os.getcwd(); os.chdir(ROOT)
    try:
        code = compile(source_replaced, main_py, 'exec')
        env = {'__file__': main_py, '__name__': '__main__'}
        exec(code, env)
        callables = [v for v in env.values() if callable(v)]
        if not callables:
            raise RuntimeError("No callable found in search_agent/main.py")
        cb = callables[-1]
    finally:
        os.chdir(cur)
    
    # Verify deck loaded
    md = getattr(cb, '__globals__', {}).get('my_deck')
    assert md and len(md) == 60, f'search_agent deck failed to load ({md and len(md)})'
    
    # Verify SEARCH_ENABLED is True
    se = cb.__globals__.get('SEARCH_ENABLED')
    print(f"  [load] SEARCH_ENABLED = {se}", flush=True)
    
    return cb

# ── Prep opponent (reuse cabt_eval's machinery) ────────────────────────────
import importlib.util
_spec = importlib.util.spec_from_file_location('cabt_eval', ROOT + '/tools/cabt_eval.py')
cabt_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cabt_eval)
prep_opponent = cabt_eval.prep_opponent

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Phase 3 Validation: Search Agent vs Frozen Baseline")
    print("=" * 65)
    print(f"\n  Frozen baseline: {BASELINE_WR*100:.1f}%")
    print(f"  Games per archetype: {GAMES_PER_ARCH}")
    print(f"  Total games planned: {GAMES_PER_ARCH * len(ARCHE_TYPES)}")
    print()

    # Load search agent (SEARCH_ENABLED=True)
    print("[1/3] Loading search agent with SEARCH_ENABLED=True...", flush=True)
    our_cb = load_search_agent()
    print("  OK", flush=True)

    from kaggle_environments import make

    # Per-archetype results
    results = {}
    total_wins = 0
    total_games = 0  # decisive only

    for arch_idx, arch in enumerate(ARCHE_TYPES):
        print(f"\n[{arch_idx+1}/{len(ARCHE_TYPES)}] Running vs {arch.upper()}...", flush=True)
        
        # Prep opponent
        oppfile = prep_opponent(arch)
        print(f"  Opponent: {oppfile}", flush=True)

        w = [0, 0, 0]  # wins, losses, draws
        for g in range(GAMES_PER_ARCH):
            env = make('cabt')
            order = [our_cb, oppfile] if g % 2 == 0 else [oppfile, our_cb]
            try:
                res = env.run(order)
                r = [s.get('reward') for s in res[-1]]
                us = 0 if g % 2 == 0 else 1
                ru, ro = r[us], r[1 - us]
                if ru is None:
                    w[1] += 1
                elif ro is None:
                    w[0] += 1
                elif ru > ro:
                    w[0] += 1
                elif ro > ru:
                    w[1] += 1
                else:
                    w[2] += 1
            except Exception as e:
                w[1] += 1  # error counts as loss
                print(f"  WARN: game {g+1} errored: {e}", flush=True)

            if (g + 1) % 50 == 0:
                decisive = w[0] + w[1]
                wr = (w[0] / decisive * 100) if decisive else 0
                print(f"    ... {g+1}/{GAMES_PER_ARCH}: {w[0]}W/{w[1]}L/{w[2]}D = {wr:.1f}%", flush=True)

        decisive = w[0] + w[1]
        wr = (w[0] / decisive * 100) if decisive else 0.0
        ci_low, ci_high = wilson_ci(w[0], decisive)
        
        results[arch] = {
            'wins': w[0], 'losses': w[1], 'draws': w[2],
            'decisive': decisive, 'wr': wr / 100.0,
            'ci_low': ci_low, 'ci_high': ci_high
        }
        total_wins += w[0]
        total_games += decisive

        print(f"  => {arch:12s}: {w[0]}W/{w[1]}L/{w[2]}D  ({decisive} decisive)")
        print(f"     WR: {wr:.1f}%  [95% CI: {ci_low*100:.1f}% – {ci_high*100:.1f}%]")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  AGGREGATE RESULTS")
    print(f"{'=' * 65}")
    
    agg_wr = (total_wins / total_games) if total_games else 0.0
    agg_ci_low, agg_ci_high = wilson_ci(total_wins, total_games)
    baseline_ci_low, baseline_ci_high = wilson_ci(
        int(BASELINE_WR * total_games), total_games
    )
    # But that's wrong — baseline was measured on different N. Instead, compute
    # baseline CI for the same N we achieved.
    # Actually for the baseline reference we need: if the baseline had the same sample
    # size, what would its CI be? We'll compute it from the binomial.
    # Baseline was measured on some N — we don't know it, so use normal approx.
    # Better: use the baseline WR and our total_games to compute its CI at this N.
    baseline_se = math.sqrt(BASELINE_WR * (1 - BASELINE_WR) / total_games)
    baseline_ci_low = BASELINE_WR - 1.96 * baseline_se
    baseline_ci_high = BASELINE_WR + 1.96 * baseline_se

    print(f"\n  Search Agent (SEARCH_ENABLED=True)")
    print(f"    Total: {total_wins}W / {total_games - total_wins}L / {GAMES_PER_ARCH * len(ARCHE_TYPES) - (total_wins + total_games - total_wins)}D")
    print(f"    Decisive games: {total_games}")
    print(f"    Win rate:       {agg_wr*100:.1f}%")
    print(f"    Wilson 95% CI:  [{agg_ci_low*100:.1f}% – {agg_ci_high*100:.1f}%]")
    print()
    print(f"  Frozen Baseline (Phase 2)")
    print(f"    Win rate:       {BASELINE_WR*100:.1f}%")
    print(f"    CI at N={total_games}: [{baseline_ci_low*100:.1f}% – {baseline_ci_high*100:.1f}%]")
    print()

    # ── Decision ──────────────────────────────────────────────────────────────
    print(f"{'=' * 65}")
    print("  PER-ARCHETYPE VERDICT")
    print(f"{'=' * 65}")
    
    ci_overlap_any = False
    ci_strictly_above_any = False
    
    for arch in ARCHE_TYPES:
        r = results[arch]
        wr = r['wr']
        cl, ch = r['ci_low'], r['ci_high']
        
        # Compare: is our ENTIRE CI above the baseline point estimate?
        strictly_above = cl > BASELINE_WR
        ci_overlaps = not (cl > baseline_ci_high or ch < baseline_ci_low)
        point_above = wr > BASELINE_WR
        
        if strictly_above and not ci_overlaps:
            verdict = "SHIPS ✓"
        elif point_above:
            verdict = "Flat (directionally better)"
        elif wr < BASELINE_WR - 0.02:
            verdict = "WORSE ✗"
        else:
            verdict = "Flat (within noise)"
        
        print(f"  {arch:12s}: {wr*100:5.1f}% [{cl*100:4.1f}%–{ch*100:4.1f}%] vs {BASELINE_WR*100:.1f}% baseline — {verdict}")
        
        if strictly_above:
            ci_strictly_above_any = True
        if ci_overlaps:
            ci_overlap_any = True
    
    # Aggregate verdict
    print(f"\n{'─' * 65}")
    agg_strictly_above = agg_ci_low > baseline_ci_high
    agg_ci_overlaps = not (agg_ci_low > baseline_ci_high or agg_ci_high < baseline_ci_low)
    
    if agg_strictly_above:
        agg_verdict = "SHIPS ✓ — entire aggregate CI above baseline CI"
    elif agg_ci_overlaps:
        agg_verdict = "Flat — CIs overlap, insufficient evidence"
    elif agg_wr > BASELINE_WR:
        agg_verdict = "Directionally better but CIs overlap"
    else:
        agg_verdict = "WORSE ✗ — aggregate CI entirely below baseline"
    
    print(f"\n  AGGREGATE VERDICT: {agg_verdict}")
    print(f"  Search Agent aggregate: {agg_wr*100:.1f}% [{agg_ci_low*100:.1f}% – {agg_ci_high*100:.1f}%]")
    print(f"  Baseline aggregate:     {BASELINE_WR*100:.1f}% [{baseline_ci_low*100:.1f}% – {baseline_ci_high*100:.1f}%]")
    print(f"{'=' * 65}")

    # ── Endgame subset (if tools/endgame_test_set.py has data) ────────────────
    print(f"\n\n  NOTE: Endgame subset evaluation requires real episode data")
    print(f"  (tools/endgame_test_set.py needs a real .zip — not available here).")
    print(f"  Run separately with: venv/bin/python tools/endgame_test_set.py <episode.zip>")

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        'baseline_wr': BASELINE_WR,
        'games_per_archetype': GAMES_PER_ARCH,
        'results': {k: {kk: round(vv, 6) if isinstance(vv, float) else vv
                        for kk, vv in v.items()}
                    for k, v in results.items()},
        'aggregate': {
            'wins': total_wins,
            'decisive': total_games,
            'wr': round(agg_wr, 6),
            'ci_low': round(agg_ci_low, 6),
            'ci_high': round(agg_ci_high, 6),
            'baseline_ci_low': round(baseline_ci_low, 6),
            'baseline_ci_high': round(baseline_ci_high, 6),
            'verdict': agg_verdict,
        }
    }
    out_path = '/tmp/phase3_validation_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Full results saved to: {out_path}")

    # Final summary line
    print(f"\n{'=' * 65}")
    print(f"  FINAL: {agg_verdict}")
    print(f"{'=' * 65}\n")


if __name__ == '__main__':
    main()