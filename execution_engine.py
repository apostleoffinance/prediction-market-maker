
import random, math
from typing import List, Dict, Any
from market_maker import MarketMaker
from market_state import MarketState

class ExecutionEngine:
    def __init__(self, markets: Dict[str, MarketState], rng_seed: int = 42):
        self.markets = markets
        self.mms = {name: MarketMaker(state) for name, state in markets.items()}
        self.time = 0
        self.rng = random.Random(rng_seed)

    def simulate_order_flow(self, market_name: str) -> List[Dict[str, Any]]:
        state = self.markets[market_name]
        orders = []
        n = self.rng.randint(1,3)
        for _ in range(n):
            # bias toward mid: higher mid -> more buys, lower mid -> more sells
            prob = state.mid + self.rng.uniform(-0.15, 0.15)
            side = 'buy' if prob > 0.5 else 'sell'
            size = max(1.0, min(30.0, abs(self.rng.gauss(6,2))))
            price = 1.0 if side == 'buy' else 0.0
            orders.append({'side': side, 'size': size, 'price': price})
        return orders

    def step(self):
        results = {}
        for name, mm in self.mms.items():
            orders = self.simulate_order_flow(name)
            fills = mm.on_tick(orders)
            for side, size, price in fills:
                signed = size if side == 'buy' else -size
                prev_mid = mm.state.mid
                mm.state.pnl += -signed * (price - prev_mid)
                mm.state.peak_pnl = max(mm.state.peak_pnl, mm.state.pnl)
                dd = mm.state.peak_pnl - mm.state.pnl
                mm.state.max_drawdown = max(mm.state.max_drawdown, dd)
            # small mean reversion
            mm.state.mid = mm.state.mid * 0.995 + 0.5 * 0.005
            results[name] = {
                'fills': fills,
                'mid': mm.state.mid,
                'inventory': mm.state.inventory,
                'pnl': mm.state.pnl,
                'spread': mm.state.spread
            }
        self.time += 1
        return results

    def run(self, steps: int = 100):
        trace = []
        for _ in range(steps):
            trace.append(self.step())
        return trace
