"""Round-robin A/B for all our agent dirs in the official cabt env.

Runs every pairing (A vs B) for `games` matches, alternating seats so first/second
player advantage cancels. Prints a win-rate matrix and an overall ranking.

Usage: venv/bin/python tools/round_robin.py [games] [agent_dir ...]
  games: matches per pairing (default 200)
  agent_dirs: restrict to a subset (default: all agents/* with a main.py + deck.csv)
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT + '/docs/official/models/cg-lib')

import glob

DEFAULT_AGENTS = []
for d in sorted(glob.glob(ROOT + '/agents/*/')):
    if os.path.exists(d + 'main.py') and os.path.exists(d + 'deck.csv'):
        DEFAULT_AGENTS.append('agents/' + os.path.basename(d.rstrip('/')))


def load_cb(agent_dir):
    our = ROOT + '/' + agent_dir + '/main.py'
    if not os.path.exists(ROOT + '/' + agent_dir + '/cg'):
        import shutil
        shutil.copytree(ROOT + '/docs/official/models/cg-lib/cg', ROOT + '/' + agent_dir + '/cg')
    from kaggle_environments.agent import get_last_callable
    cur = os.getcwd()
    os.chdir(ROOT + '/' + agent_dir)
    try:
        cb = get_last_callable(open(our).read(), path=our)
    finally:
        os.chdir(cur)
    md = getattr(cb, '__globals__', {}).get('my_deck')
    assert md and len(md) == 60, f'{agent_dir} deck failed to load ({md and len(md)})'
    return cb


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    agents = sys.argv[2:] or DEFAULT_AGENTS
    print(f'[round-robin] agents={agents} games/pairing={games}', flush=True)

    cbs = {a: load_cb(a) for a in agents}
    from kaggle_environments import make

    # wins[a][b] = number of times a beat b
    wins = {a: {b: 0 for b in agents} for a in agents}
    losses = {a: {b: 0 for b in agents} for a in agents}
    draws = {a: {b: 0 for b in agents} for a in agents}

    pairings = [(a, b) for i, a in enumerate(agents) for b in agents[i + 1:]]
    total = len(pairings) * games
    done = 0
    for a, b in pairings:
        ca, cb = cbs[a], cbs[b]
        for g in range(games):
            env = make('cabt')
            order = [ca, cb] if g % 2 == 0 else [cb, ca]
            res = env.run(order)
            r = [s.get('reward') for s in res[-1]]
            us = 0 if g % 2 == 0 else 1
            ru, ro = r[us], r[1 - us]
            if ru is None:
                losses[a][b] += 1
            elif ro is None:
                wins[a][b] += 1
            elif ru > ro:
                wins[a][b] += 1
            elif ro > ru:
                losses[a][b] += 1
            else:
                draws[a][b] += 1
            done += 1
            if done % 20 == 0 or done == total:
                print(f'  progress {done}/{total}', flush=True)

    def _win_rate(ax, bx):
        """a's decisive win % vs b using the forward-or-reverse data."""
        fwd_dec = wins[ax][bx] + losses[ax][bx]
        rev_dec = wins[bx][ax] + losses[bx][ax]
        a_wins = wins[ax][bx] + losses[bx][ax]  # a beats b = forward wins + reverse losses
        a_losses = losses[ax][bx] + wins[bx][ax]
        dec = a_wins + a_losses
        return (a_wins / dec * 100) if dec else 0.0

    print('\n=== Round-robin win-rate matrix (row beats column, % of decisive games) ===')
    hdr = 'agent'.ljust(22) + ''.join(b.split('/')[-1][:8].rjust(10) for b in agents)
    print(hdr)
    for a in agents:
        row = a.split('/')[-1][:20].ljust(22)
        for b in agents:
            if a == b:
                row += '    -   '.rjust(10)
            else:
                row += f'{_win_rate(a, b):6.1f}%'.rjust(10)
        print(row)

    # Overall score: total decisive wins / total decisive games (symmetric)
    print('\n=== Overall ranking (decisive win rate across all pairings) ===')
    overall = []
    for a in agents:
        a_wins = 0
        a_losses = 0
        a_draws = 0
        for b in agents:
            if a == b:
                continue
            a_wins += wins[a][b] + losses[b][a]
            a_losses += losses[a][b] + wins[b][a]
            a_draws += draws[a][b] + draws[b][a]
        dec = a_wins + a_losses
        pct = (a_wins / dec * 100) if dec else 0.0
        overall.append((a, a_wins, a_losses, a_draws, pct))
    overall.sort(key=lambda x: -x[4])
    for rank, (a, w, l, d, pct) in enumerate(overall, 1):
        print(f'  {rank}. {a.split("/")[-1]:20s} {pct:6.1f}%  ({w}W/{l}L/{d}D)')
    print(f'\n[round-robin] WINNER (freeze as baseline): {overall[0][0]}', flush=True)


if __name__ == '__main__':
    main()