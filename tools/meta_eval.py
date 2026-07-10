"""Meta-weighted deck evaluation harness (Phase 2).

Evaluates a candidate agent/deck against a WEIGHTED sample of opponent decks
drawn proportionally to real observed field frequency, instead of one fixed
"meta" deck. This isolates DECK quality (not agent quality) by piloting both
sides with the SAME agent.

Since the live Kaggle episode exports are not available in this environment,
the field-frequency prior is taken from the consensus deck lists already
derived in tools/cabt_eval.py (Crustle / Lucario / Abomasnow / Dragapult /
Iono's Bellibolt) — these are the real, most-common winning decks from the
6-17 episode analysis. The frequency weights below are the documented field
shares from docs/strategy (Crustle ~50%, the rest splitting the remainder).
If you have a fresh episode zip, re-run tools/meta_analyze.py and update
FIELD_WEIGHTS with the new numbers.

Usage:
  venv/bin/python tools/meta_eval.py <our_agent_dir> [games_per_archetype]
"""
import sys, os, json, warnings
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT + '/docs/official/models/cg-lib')

from collections import Counter

# Field frequency prior (documented in docs/strategy/牌組策略.md):
# Crustle ballooned to ~50% of the field; the rest split the remainder.
# These are DIRECTIONAL — refresh from meta_analyze.py output when available.
# New archetypes from ladder observation (empirical, not hardcoded meta)
FIELD_WEIGHTS = {
    'crustle':    0.40,   # Reduced to make room for new archetypes
    'lucario':    0.15,
    'abomasnow':  0.10,
    'dragapult':  0.10,
    'iono':       0.07,
    'starmie':    0.03,   # Mega Starmie ex
    'froslass':   0.03,   # Mega Froslass ex
    'garchomp':   0.03,   # Cynthia's Garchomp ex
    'cinderace':  0.03,   # Cinderace ex + Duraludon/Archaludon
    'ogerpon':    0.03,   # Teal Mask Ogerpon ex
    'unknown':    0.06,   # Fallback for unseen archetypes
}

# Reuse cabt_eval's consensus-deck machinery (deterministic, pinned decks).
import importlib.util
_spec = importlib.util.spec_from_file_location('cabt_eval', ROOT + '/tools/cabt_eval.py')
cabt_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cabt_eval)
CONSENSUS = cabt_eval.CONSENSUS
NB = cabt_eval.NB


def _prep_opponent(opp):
    return cabt_eval.prep_opponent(opp)


def main():
    our_dir = sys.argv[1] if len(sys.argv) > 1 else 'agents/search_agent'
    gpa = int(sys.argv[2]) if len(sys.argv) > 2 else 40   # games per archetype

    # Build the weighted opponent schedule.
    sched = []
    for arch, w in FIELD_WEIGHTS.items():
        n = max(1, round(gpa * w * 10))   # scale up so weights are meaningful
        sched += [arch] * n
    total = len(sched)
    print(f'[meta_eval] {our_dir} vs weighted field (weights={FIELD_WEIGHTS}) '
          f'{total} games', flush=True)

    # Load our agent (pre-built callable, reads its own deck.csv once).
    our = ROOT + '/' + our_dir + '/main.py'
    if not os.path.exists(ROOT + '/' + our_dir + '/cg'):
        import shutil
        shutil.copytree(ROOT + '/docs/official/models/cg-lib/cg', ROOT + '/' + our_dir + '/cg')
    from kaggle_environments.agent import get_last_callable
    cur = os.getcwd(); os.chdir(ROOT)
    try:
        our_cb = get_last_callable(open(our).read(), path=our)
    finally:
        os.chdir(cur)
    md = getattr(our_cb, '__globals__', {}).get('my_deck')
    assert md and len(md) == 60, f'our deck failed to load ({md and len(md)})'

    from kaggle_environments import make
    per_arch = Counter()
    per_arch_w = Counter()
    w = [0, 0, 0]
    for i, arch in enumerate(sched):
        oppfile = _prep_opponent(arch)
        env = make('cabt')
        order = [our_cb, oppfile] if i % 2 == 0 else [oppfile, our_cb]
        res = env.run(order)
        r = [s.get('reward') for s in res[-1]]
        us = 0 if i % 2 == 0 else 1
        ru, ro = r[us], r[1 - us]
        if ru is None: w[1] += 1
        elif ro is None: w[0] += 1
        elif ru > ro: w[0] += 1
        elif ro > ru: w[1] += 1
        else: w[2] += 1
        per_arch[arch] += 1
        if ru is not None and ru > ro: per_arch_w[arch] += 1
        print(f'  [{i+1}/{total}] {arch}: us={ru} opp={ro}', flush=True)

    t = w[0] + w[1]
    print(f'\n[meta_eval] {our_dir} vs weighted field: {w[0]}W/{w[1]}L/{w[2]}D = {w[0]/t*100:.1f}%' if t else 'no decisive')
    print('  per-archetype win rate:')
    for arch in FIELD_WEIGHTS:
        ga = per_arch[arch]
        if ga:
            print(f'    {arch:12s} {per_arch_w[arch]}/{ga} = {per_arch_w[arch]/ga*100:.0f}%  (field {FIELD_WEIGHTS[arch]*100:.0f}%)')


if __name__ == '__main__':
    main()