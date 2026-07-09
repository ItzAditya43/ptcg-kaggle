#!/usr/bin/env python3
"""Phase 5 — Build endgame test set from real replay data.

Extracts genuinely close late-game/prize-race situations from real ladder
episode replays. These are situations where:
  - Both players have <= 3 remaining prizes
  - Total remaining Pokémon on both sides <= 5
  - The game is still undecided (not about to end next turn)

Output: a JSON file with the extracted observations and associated metadata,
ready to be used for evaluating the endgame solver's performance.

Usage:
  venv/bin/python tools/endgame_test_set.py <episode_zip> [--max-games 500] [--output /tmp/endgame_set.json]
"""
import sys, os, json, zipfile, argparse, warnings
from collections import Counter
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + '/docs/official/models/cg-lib')
from cg.api import to_observation_class, all_card_data

CT = {c.cardId: c for c in all_card_data()}

# Endgame criteria
MAX_PRIZES_PER_SIDE = 3  # both sides must have <= 3 prizes
MAX_TOTAL_POKEMON = 5     # total Pokémon on both sides

# Support Pokémon (draw/search engines that aren't the win condition)
SUPPORT = ('Fezandipiti', 'Dudunsparce', 'Dunsparce', 'Shaymin', 'Fan Rotom', 'Rotom',
           'Dedenne', 'Genesect', 'Lumineon', 'Radiant', 'Mew ', 'Snorlax', 'Bibarel',
           'Bidoof', 'Lechonk', 'Squawkabilly')

def _is_support(pid):
    nm = CT[pid].name or ''
    return any(s in nm for s in SUPPORT)

def dk(deck):
    """Archetype label."""
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


def is_endgame_observation(obs_dict):
    """Check if an observation represents an endgame state."""
    try:
        obs = to_observation_class(obs_dict)
        st = obs.current
        if st is None or st.result != -1:
            return False
        
        me = st.players[st.yourIndex]
        op = st.players[1 - st.yourIndex]
        
        # Both sides must have <= 3 prizes
        if len(me.prize) > MAX_PRIZES_PER_SIDE or len(op.prize) > MAX_PRIZES_PER_SIDE:
            return False
        
        # Count total Pokémon
        total_pokemon = 0
        for p in (me.active + me.bench):
            if p is not None:
                total_pokemon += 1
        for p in (op.active + op.bench):
            if p is not None:
                total_pokemon += 1
        
        if total_pokemon > MAX_TOTAL_POKEMON:
            return False
        
        # Must be a MAIN decision (not setup/draw/etc.)
        if obs.select is None or obs.select.context != 0:  # MAIN = 0
            return False
        
        return True
    except Exception:
        return False


def extract_endgame_observations(zip_path, max_games=500):
    """Extract endgame observations from episode replays.
    
    For each game, we step through observations and capture those that
    meet the endgame criteria. We only take the first qualifying observation
    per game (not every turn of the endgame) to keep the test set diverse.
    """
    z = zipfile.ZipFile(zip_path)
    names = [x for x in z.namelist() if x.endswith('.json')]
    
    endgames = []
    games_checked = 0
    games_with_endgame = 0
    
    for nm in names:
        if max_games and games_checked >= max_games:
            break
        try:
            d = json.loads(z.read(nm))
            rw = d['rewards']
            if rw[0] == rw[1]:
                continue  # skip draws
            
            winner = 0 if rw[0] > rw[1] else 1
            decks = [d['steps'][1][0]['action'], d['steps'][1][1]['action']]
            if not (isinstance(decks[0], list) and len(decks[0]) == 60):
                continue
            
            games_checked += 1
            
            # Step through the game observations
            steps = d['steps']
            found_endgame = False
            for t in range(1, len(steps) - 1):
                for pi in range(2):
                    obs = steps[t][pi].get('observation')
                    if obs is None:
                        continue
                    try:
                        if is_endgame_observation(obs):
                            # Collect metadata
                            p0_arch = dk(decks[0])
                            p1_arch = dk(decks[1])
                            
                            endgames.append({
                                'game_index': games_checked - 1,
                                'episode': nm,
                                'turn': t,
                                'player_index': pi,
                                'observation': obs,
                                'winner': winner,
                                'player0_archetype': p0_arch,
                                'player1_archetype': p1_arch,
                                'player0_deck': decks[0],
                                'player1_deck': decks[1],
                            })
                            found_endgame = True
                            break  # one observation per game
                    except Exception:
                        continue
                if found_endgame:
                    break
            
            if found_endgame:
                games_with_endgame += 1
                
            if games_checked % 100 == 0:
                print(f'  Checked {games_checked} games, found {games_with_endgame} with endgame states', flush=True)
                
        except Exception:
            continue
    
    print(f'  Total: {games_checked} games, {games_with_endgame} endgame states extracted', flush=True)
    return endgames


def main():
    ap = argparse.ArgumentParser(description='Extract endgame test set from real replays')
    ap.add_argument('zip', help='Episode zip path')
    ap.add_argument('--max-games', type=int, default=500, help='Max games to check')
    ap.add_argument('--output', default='/tmp/endgame_test_set.json', help='Output JSON path')
    args = ap.parse_args()
    
    print(f'[endgame] Extracting from {args.zip}', flush=True)
    endgames = extract_endgame_observations(args.zip, args.max_games)
    
    # Save the metadata (not the full observations to keep file size manageable)
    meta_only = []
    for eg in endgames:
        meta_only.append({
            k: v for k, v in eg.items() if k != 'observation'
        })
    
    with open(args.output, 'w') as f:
        json.dump({
            'source_zip': os.path.basename(args.zip),
            'total_endgames': len(endgames),
            'endgames': meta_only,
        }, f, indent=2)
    
    # Also save the full observations separately (may be large)
    full_path = args.output.replace('.json', '_full.json')
    with open(full_path, 'w') as f:
        json.dump({
            'source_zip': os.path.basename(args.zip),
            'total_endgames': len(endgames),
            'endgames': endgames,
        }, f, indent=2)
    
    print(f'\n[endgame] Saved {len(endgames)} endgame states:')
    print(f'  Metadata: {args.output}')
    print(f'  Full:     {full_path}')
    
    # Summary stats
    archetype_counts = Counter()
    for eg in endgames:
        pi = eg['player_index']
        arch_key = f'player{pi}_archetype'
        archetype_counts[eg[arch_key]] += 1
    
    print(f'\n  Archetype distribution (focal player):')
    for arch, n in archetype_counts.most_common():
        print(f'    {arch:30s} {n}')


if __name__ == '__main__':
    main()