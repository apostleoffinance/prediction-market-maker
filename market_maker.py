
import math
from typing import Tuple, List
from market_state import MarketState

class MarketMaker:
    def __init__(self, state: MarketState, config: dict = None):
        self.state = state
        self.cfg = config or {}
        self.imbalance_window: List[float] = []
        self.window_size = int(self.cfg.get('window_size', 20))
        self.base_spread = float(self.cfg.get('base_spread', self.state.spread))
        self.min_spread = float(self.cfg.get('min_spread', 0.01))
        self.max_spread = float(self.cfg.get('max_spread', 0.5))
        self.inventory_skew = float(self.cfg.get('inventory_skew', 0.001))

    def quote(self) -> Tuple[float, float, float]:
        mid = self.state.mid
        imbalance = sum(self.imbalance_window[-self.window_size:]) if self.imbalance_window else 0.0
        abs_imb = abs(imbalance)
        spread = max(self.min_spread, min(self.base_spread * (1 + abs_imb/10.0 + abs(self.state.inventory)*self.inventory_skew), self.max_spread))
        skew = self.state.inventory * self.inventory_skew
        mid_shaded = min(0.99, max(0.01, mid - skew))
        bid = max(0.0, mid_shaded - spread/2.0)
        ask = min(1.0, mid_shaded + spread/2.0)
        size = max(1.0, min(20.0, 10.0 - abs(self.state.inventory)/10.0))
        self.state.spread = spread
        return bid, ask, size

    def on_fill(self, side: str, size: float):
        delta = size if side == 'buy' else -size
        self.imbalance_window.append(delta)
        if len(self.imbalance_window) > max(100, self.window_size*4):
            self.imbalance_window = self.imbalance_window[-self.window_size*4:]
        alpha = 0.05
        flow = delta
        self.state.mid = min(0.99, max(0.01, self.state.mid + alpha * (flow / (10.0 + abs(flow)))))
        inv = self.state.inventory
        if abs(inv) > self.state.inventory_limit * 0.8:
            correction = -math.copysign(0.05, inv)
            self.state.mid = min(0.99, max(0.01, self.state.mid + correction))

    def on_tick(self, market_order_flow: List[dict]):
        fills = []
        bid, ask, size = self.quote()
        for order in market_order_flow:
            side = order['side']
            qty = order['size']
            # Taker aggressiveness: if buyer, they hit ask; if seller, they hit bid
            if side == 'buy' and order.get('price', 1.0) >= ask:
                fills.append(('sell', qty, ask))
            elif side == 'sell' and order.get('price', 0.0) <= bid:
                fills.append(('buy', qty, bid))
        for side, qty, price in fills:
            self.state.record_fill(side=side, size=qty, price=price)
            # call on_fill with perspective of trades we took (we 'bought' when side=='buy')
            self.on_fill('buy' if side == 'buy' else 'sell', qty)
        return fills
