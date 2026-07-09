"""A/B two of OUR agent dirs against each other in the official cabt env.

Loads each agent as a pre-built callable (NOT a directory path), because
kaggle_environments' build_agent() does NOT support directory-path agent
loading for custom envs like cabt — it tries to open() the directory as a
file, gets IsADirectoryError, falls back to the path string, then tries to
compile() it as Python source code, producing a SyntaxError.

Each agent is loaded independently so module-level state (deck, card data)
is isolated per seat. Seats alternate every game so first/second player
advantage cancels out.

Usage: venv/bin/python tools/cabt_ab.py <dirA> <dirB> [games]
  e.g.  venv/bin/python tools/cabt_ab.py agents/alakazam_mist agents/alakazam 40
  Reports A's win-rate vs B.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT + '/docs/official/models/cg-lib')


def verify_agent_dir(agent_dir):
    """Verify agent directory has required structure."""
    full_dir = ROOT + '/' + agent_dir
    main_py = full_dir + '/main.py'
    deck_csv = full_dir + '/deck.csv'
    if not os.path.isdir(full_dir):
        raise FileNotFoundError(f"Agent directory not found: {full_dir}")
    if not os.path.exists(main_py):
        raise FileNotFoundError(f"Agent missing main.py: {main_py}")
    if not os.path.exists(deck_csv):
        raise FileNotFoundError(f"Agent missing deck.csv: {deck_csv}")
    # Verify cg lib exists (copy if needed, like cabt_eval does)
    if not os.path.exists(full_dir + '/cg'):
        import shutil
        shutil.copytree(ROOT + '/docs/official/models/cg-lib/cg', full_dir + '/cg')
    return full_dir


def load_agent_callable(agent_dir):
    """Load an agent as a pre-built callable (NOT a directory path).

    IMPORTANT: Does NOT use get_last_callable() from kaggle_environments because
    that function creates an empty exec env dict, so __file__ is undefined in the
    agent code. Instead, we exec() the source ourselves with __file__ explicitly
    set — this mirrors how CPython sets module.__file__ when loading from a path.

    Each call to this function creates an independent module with its own state
    (deck, card data, DIAG), so the two seats have isolated RNG and state.
    """
    main_py = ROOT + '/' + agent_dir + '/main.py'
    cur = os.getcwd()
    os.chdir(ROOT)
    try:
        source = open(main_py).read()
        # Build a __file__-aware exec env to prevent
        #   NameError: name '__file__' is not defined
        # when agent code calls os.path.dirname(os.path.abspath(__file__)).
        code = compile(source, main_py, 'exec')
        env = {
            '__file__': main_py,
            '__name__': '__main__',
        }
        exec(code, env)
        callables = [v for v in env.values() if callable(v)]
        if not callables:
            raise RuntimeError(f"No callable found in {agent_dir}/main.py")
        cb = callables[-1]
    finally:
        os.chdir(cur)
    # Verify the agent loaded its deck correctly
    md = getattr(cb, '__globals__', {}).get('my_deck')
    assert md and len(md) == 60, f'{agent_dir} deck failed to load ({md and len(md)})'
    return cb


def main():
    dirA = sys.argv[1]
    dirB = sys.argv[2]
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 40

    # Verify both agent directories exist
    verify_agent_dir(dirA)
    verify_agent_dir(dirB)

    # Load each agent as an independent callable (not a directory path)
    agentA = load_agent_callable(dirA)
    agentB = load_agent_callable(dirB)

    print(f"[cabt_ab] Callable-agent A/B:")
    print(f"[cabt_ab]   A: {dirA}")
    print(f"[cabt_ab]   B: {dirB}")
    print(f"[cabt_ab]   Games: {games}")

    from kaggle_environments import make
    w = [0, 0, 0]  # A wins, B wins, draws
    for g in range(games):
        env = make('cabt')
        # Alternate seats to cancel out first/second player advantage
        order = [agentA, agentB] if g % 2 == 0 else [agentB, agentA]
        res = env.run(order)
        r = [s.get('reward') for s in res[-1]]
        us = 0 if g % 2 == 0 else 1
        ru, ro = r[us], r[1 - us]
        if ru is None: w[1] += 1
        elif ro is None: w[0] += 1
        elif ru > ro: w[0] += 1
        elif ro > ru: w[1] += 1
        else: w[2] += 1
        print(f'  game {g+1}/{games}: A={ru} B={ro}  [{w[0]}W/{w[1]}L/{w[2]}D]', flush=True)
    t = w[0] + w[1]
    if t:
        print(f'[cabt_ab] {dirA} (A) vs {dirB} (B): {w[0]}W/{w[1]}L/{w[2]}D = A {w[0]/t*100:.1f}%')
    else:
        print('[cabt_ab] no decisive games')


if __name__ == '__main__':
    main()
