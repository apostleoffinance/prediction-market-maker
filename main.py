
import os, json, time
from market_state import MarketState
from execution_engine import ExecutionEngine
from logger import write_report

def build_markets():
    markets = {
        'inflation_gt_20': MarketState('inflation_gt_20', initial_mid=0.30),
        'election_candidate_a': MarketState('election_candidate_a', initial_mid=0.55),
        'team_x_wins': MarketState('team_x_wins', initial_mid=0.50)
    }
    for m in markets.values():
        m.inventory_limit = 200.0
        m.exposure_limit = 10000.0
        m.spread = 0.05
    return markets

def run_demo():
    markets = build_markets()
    engine = ExecutionEngine(markets, rng_seed=123)
    trace = engine.run(steps=200)
    out_dir = os.path.dirname(__file__)
    csv_path = os.path.join(out_dir, 'simulation_report.csv')
    write_report(markets, csv_path)
    print('Simulation complete. Report written to:', csv_path)
    with open(os.path.join(out_dir, 'trace.json'), 'w') as f:
        json.dump(trace, f, indent=2)
    for name, s in markets.items():
        print(name, s.snapshot())
    return csv_path

if __name__ == '__main__':
    run_demo()
