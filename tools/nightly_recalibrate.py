#!/usr/bin/env python3
"""Phase 4 — Nightly Recalibration Job.

Pulls the latest daily top-episode replay export, re-derives the archetype
field-share distribution from it, and updates:
  - The belief-state tracker's prior weights (Phase 3)
  - meta_eval.py's FIELD_WEIGHTS for future deck/agent evaluation

Outputs a short diff report each run: old field-share % vs new, per archetype,
plus any newly-detected archetype not in the current 5 (the field can shift —
a 6th archetype crossing some minimum threshold, e.g. 5%, should be flagged,
not silently ignored).

Usage:
  # Manual trigger (default: use latest episode zip in /tmp/ep*):
  venv/bin/python tools/nightly_recalibrate.py

  # With explicit episode zip:
  venv/bin/python tools/nightly_recalibrate.py /tmp/ep19/pokemon-tcg-ai-battle-episodes-2026-06-19.zip

  # With leaderboard for top-tier slice:
  venv/bin/python tools/nightly_recalibrate.py --lb /tmp/lb

  # Scheduled (cron): add to crontab
  # 0 2 * * * cd /path/to/repo && venv/bin/python tools/nightly_recalibrate.py --cron >> /tmp/nightly_recal.log 2>&1

Gate: this phase passes on "runs cleanly, produces a correct diff report" —
not a win-rate number. Test it once against the same 6-17 episode data you
already have to confirm it reproduces the known current distribution correctly
before trusting it on new data.
"""
import sys, os, json, csv, glob, zipfile, argparse, warnings
from collections import Counter, defaultdict
from datetime import datetime
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/docs/official/models/cg-lib')
from cg.api import all_card_data

CT = {c.cardId: c for c in all_card_data()}

# ── Archetype label logic (mirrors meta_analyze.py) ─────────────────────────
SUPPORT = ('Fezandipiti', 'Dudunsparce', 'Dunsparce', 'Shaymin', 'Fan Rotom', 'Rotom',
           'Dedenne', 'Genesect', 'Lumineon', 'Radiant', 'Mew ', 'Snorlax', 'Bibarel',
           'Bidoof', 'Lechonk', 'Squawkabilly')

def _is_support(pid):
    nm = CT[pid].name or ''
    return any(s in nm for s in SUPPORT)

def dk(deck):
    """Archetype label for a 60-card list."""
    cc = Counter(deck)
    poke = [(k, v) for k, v in cc.items() if k in CT and k < 1000 and CT[k].hp]
    core = [p for p in poke if not _is_support(p[0])] or poke
    for rank in ('megaEx', 'ex'):
        cand = [p for p in sorted(core, key=lambda x: -x[1]) if getattr(CT[p[0]], rank, 0)]
        if cand:
            return CT[cand[0][0]].name
    for stage in ('stage2', 'stage1'):
        cand = [p for p in sorted(core, key=lambda x: -x[1]) if getattr(CT[p[0]], stage, 0)]
        if cand:
            return CT[cand[0][0]].name
    if not core:
        return '?'
    return CT[max(core, key=lambda x: x[1])[0]].name

# ── Known archetype names (the current 5) ───────────────────────────────────
# These must match the keys in meta_eval.py FIELD_WEIGHTS and the
# belief-state tracker's ARCH_SIGNATURE_POKEMON.
KNOWN_ARCHETYPES = {
    'Crustle':           'crustle',
    'Mega Lucario ex':   'lucario',
    'Mega Abomasnow ex': 'abomasnow',
    'Dragapult ex':      'dragapult',
    "Iono\u2019s Bellibolt ex": 'iono',   # Iono's Bellibolt (unicode apostrophe)
}

# Reverse mapping: short key -> display name
ARCH_DISPLAY = {v: k for k, v in KNOWN_ARCHETYPES.items()}
ARCH_DISPLAY.update({
    'crustle': 'Crustle',
    'lucario': 'Mega Lucario ex',
    'abomasnow': 'Mega Abomasnow ex',
    'dragapult': 'Dragapult ex',
    'iono': "Iono's Bellibolt ex",
})

# Minimum threshold for flagging a new archetype
NEW_ARCH_THRESHOLD = 0.05  # 5%

# ── Paths ───────────────────────────────────────────────────────────────────
META_EVAL_PATH = os.path.join(ROOT, 'tools', 'meta_eval.py')
AGENT_MAIN_PATH = os.path.join(ROOT, 'agents', 'search_agent', 'main.py')
REPORT_DIR = '/tmp/nightly_recalibrate'

# ── Episode iteration ───────────────────────────────────────────────────────
def iter_games(zip_path, max_n=0):
    """Yield (deckA, deckB, winner_idx_or_None) per episode."""
    z = zipfile.ZipFile(zip_path)
    names = [x for x in z.namelist() if x.endswith('.json')]
    n = 0
    for nm in names:
        if max_n and n >= max_n:
            break
        try:
            d = json.loads(z.read(nm))
            rw = d['rewards']
            decks = [d['steps'][1][0]['action'], d['steps'][1][1]['action']]
            if not (isinstance(decks[0], list) and len(decks[0]) == 60):
                continue
            who = None if rw[0] == rw[1] else (0 if rw[0] > rw[1] else 1)
            n += 1
            yield decks[0], decks[1], who
        except Exception:
            continue

# ── Field analysis ──────────────────────────────────────────────────────────
def analyze_field(zip_path, max_n=0):
    """Analyze episode zip and return field distribution + per-archetype win rates."""
    field_app = Counter()
    field_win = Counter()
    games = 0
    for dA, dB, who in iter_games(zip_path, max_n):
        if who is None:
            continue
        games += 1
        for i, deck in enumerate([dA, dB]):
            label = dk(deck)
            field_app[label] += 1
            if who == i:
                field_win[label] += 1
    return field_app, field_win, games

# ── Update meta_eval.py FIELD_WEIGHTS ───────────────────────────────────────
def update_meta_eval(new_weights):
    """Rewrite meta_eval.py's FIELD_WEIGHTS dict with new values."""
    path = META_EVAL_PATH
    with open(path) as f:
        content = f.read()
    
    # Build the new FIELD_WEIGHTS dict string
    weight_lines = []
    for arch in ['crustle', 'lucario', 'abomasnow', 'dragapult', 'iono']:
        w = new_weights.get(arch, 0)
        weight_lines.append(f"    '{arch}':    {w:.2f},")
    weight_str = '\n'.join(weight_lines)
    
    import re
    # Replace the FIELD_WEIGHTS dict
    pattern = r'FIELD_WEIGHTS = \{[^}]+\}'
    replacement = f'FIELD_WEIGHTS = {{\n{weight_str}\n}}'
    new_content = re.sub(pattern, replacement, content, count=1, flags=re.DOTALL)
    
    with open(path, 'w') as f:
        f.write(new_content)
    print(f'  [update] {path} — FIELD_WEIGHTS updated')

# ── Update search_agent belief priors ───────────────────────────────────────
def update_belief_priors(new_weights):
    """Rewrite search_agent/main.py's FIELD_PRIORS dict with new values."""
    path = AGENT_MAIN_PATH
    with open(path) as f:
        content = f.read()
    
    # Build the new FIELD_PRIORS dict string
    weight_lines = []
    for arch in ['crustle', 'lucario', 'abomasnow', 'dragapult', 'iono']:
        w = new_weights.get(arch, 0)
        weight_lines.append(f"    '{arch}':   {w:.2f},")
    weight_str = '\n'.join(weight_lines)
    
    import re
    # Replace the FIELD_PRIORS dict
    pattern = r'FIELD_PRIORS = \{[^}]+\}'
    replacement = f'FIELD_PRIORS = {{\n{weight_str}\n}}'
    new_content = re.sub(pattern, replacement, content, count=1, flags=re.DOTALL)
    
    with open(path, 'w') as f:
        f.write(new_content)
    print(f'  [update] {path} — FIELD_PRIORS updated')

# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Nightly recalibration job — Phase 4')
    ap.add_argument('zip', nargs='?', default=None,
                    help='Episode zip path (default: latest /tmp/ep*/*.zip)')
    ap.add_argument('--lb', default='/tmp/lb', help='Leaderboard directory for Elo data')
    ap.add_argument('--max', type=int, default=0, help='Max episodes to analyze (0=all)')
    ap.add_argument('--cron', action='store_true', help='Running from cron (less verbose)')
    ap.add_argument('--dry-run', action='store_true', help='Analyze and report only, do not update files')
    args = ap.parse_args()
    
    # Find episode zip
    zip_path = args.zip
    if zip_path is None:
        # Find the latest /tmp/ep*/*.zip
        candidates = sorted(glob.glob('/tmp/ep*/*.zip'))
        if not candidates:
            print('[ERROR] No episode zip found. Provide path or download first.')
            print('  venv/bin/kaggle datasets download kaggle/pokemon-tcg-ai-battle-episodes-2026-06-19 -p /tmp/ep19')
            sys.exit(1)
        zip_path = candidates[-1]
    
    print(f'[nightly] Analyzing {zip_path}', flush=True)
    
    # Analyze field
    field_app, field_win, total_games = analyze_field(zip_path, args.max)
    if total_games == 0:
        print('[ERROR] No decisive games found in zip.')
        sys.exit(1)
    
    total_decks = sum(field_app.values())
    print(f'  Total decisive games: {total_games}')
    print(f'  Total deck appearances: {total_decks}')
    
    # Compute current field shares
    current_weights = {}
    for arch_key, short_key in KNOWN_ARCHETYPES.items():
        n = field_app.get(arch_key, 0)
        wr = field_win.get(arch_key, 0) / max(n, 1) * 100
        share = n / max(total_decks, 1)
        current_weights[short_key] = share
        print(f'  {arch_key:30s} field={share*100:5.1f}%  winrate={wr:5.1f}%  n={n}')
    
    # Check for new archetypes
    new_archetypes = []
    for label, n in field_app.most_common(20):
        if label in KNOWN_ARCHETYPES:
            continue
        share = n / max(total_decks, 1)
        if share >= NEW_ARCH_THRESHOLD:
            new_archetypes.append((label, share, n))
    
    if new_archetypes:
        print(f'\n  ⚠ NEW ARCHETYPE(S) DETECTED (>{NEW_ARCH_THRESHOLD*100:.0f}% field):')
        for label, share, n in new_archetypes:
            wr = field_win.get(label, 0) / max(n, 1) * 100
            print(f'    {label:30s} field={share*100:5.1f}%  winrate={wr:5.1f}%  n={n}')
    
    # Load old weights from meta_eval.py
    old_weights = {}
    try:
        with open(META_EVAL_PATH) as f:
            content = f.read()
        import re
        match = re.search(r'FIELD_WEIGHTS\s*=\s*\{([^}]+)\}', content, re.DOTALL)
        if match:
            for line in match.group(1).strip().split('\n'):
                line = line.strip().rstrip(',')
                if ':' in line:
                    k, v = line.split(':', 1)
                    old_weights[k.strip().strip("'\"")] = float(v.strip())
    except Exception as e:
        print(f'  [warn] Could not read old weights from meta_eval.py: {e}')
        old_weights = {k: 0 for k in KNOWN_ARCHETYPES.values()}
    
    # Diff report
    print(f'\n  === DIFF REPORT ===')
    print(f'  {"Archetype":30s} {"Old%":>7} {"New%":>7} {"Δ":>7}')
    for short_key in ['crustle', 'lucario', 'abomasnow', 'dragapult', 'iono']:
        old_pct = old_weights.get(short_key, 0) * 100
        new_pct = current_weights.get(short_key, 0) * 100
        delta = new_pct - old_pct
        display = ARCH_DISPLAY.get(short_key, short_key)
        print(f'  {display:30s} {old_pct:6.1f}% {new_pct:6.1f}% {delta:+6.1f}%')
    
    if new_archetypes:
        print(f'\n  ⚠ NEW ARCHETYPES (>{NEW_ARCH_THRESHOLD*100:.0f}%):')
        for label, share, n in new_archetypes:
            print(f'    {label:30s} {share*100:6.1f}%')
    
    # Save report
    os.makedirs(REPORT_DIR, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {
        'date': date_str,
        'zip': os.path.basename(zip_path),
        'total_games': total_games,
        'old_weights': old_weights,
        'new_weights': current_weights,
        'new_archetypes': [{'name': l, 'share': s, 'n': c} for l, s, c in new_archetypes],
        'per_archetype': {label: {'n': field_app.get(label, 0), 'win': field_win.get(label, 0)}
                          for label in field_app},
    }
    report_path = os.path.join(REPORT_DIR, f'recalibrate_{date_str}.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n  Report saved: {report_path}')
    
    # Update files (unless dry-run)
    if not args.dry_run:
        print(f'\n  Updating files...')
        update_meta_eval(current_weights)
        update_belief_priors(current_weights)
        print(f'  Done.')
    else:
        print(f'\n  [dry-run] No files updated.')
    
    # Cron-friendly summary
    if args.cron:
        print(f'[nightly] {date_str}: {total_games} games, '
              f'weights: {", ".join(f"{k}={v*100:.0f}%" for k, v in current_weights.items())}')
        if new_archetypes:
            print(f'[nightly] NEW: {", ".join(f"{l} ({s*100:.0f}%)" for l, s, _ in new_archetypes)}')


if __name__ == '__main__':
    main()