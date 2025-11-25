
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Fill:
    side: str
    size: float
    price: float
    timestamp: float = field(default_factory=time.time)

class MarketState:
    def __init__(self, name: str, initial_mid: float = 0.5):
        self.name = name
        self.mid = float(initial_mid)  # mid probability (0..1)
        self.spread = 0.05             # absolute spread (probability points)
        self.inventory = 0.0
        self.exposure = 0.0
        self.pnl = 0.0
        self.fills: List[Fill] = []
        self.fill_count = 0
        self.notional = 0.0
        self.max_drawdown = 0.0
        self.peak_pnl = 0.0
        # risk parameters
        self.inventory_limit = 100.0
        self.exposure_limit = 10000.0
        self.fee = 0.0  # fees per trade (notional)
        self.extra = {}

    def record_fill(self, side: str, size: float, price: float):
        fill = Fill(side=side, size=size, price=price)
        self.fills.append(fill)
        self.fill_count += 1
        self.notional += abs(size) * price
        if side == 'buy':
            self.inventory += size
        else:
            self.inventory -= size
        self.exposure = abs(self.inventory) * self.mid

    def snapshot(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'mid': self.mid,
            'spread': self.spread,
            'inventory': self.inventory,
            'exposure': self.exposure,
            'pnl': self.pnl,
            'fill_count': self.fill_count,
            'notional': self.notional,
        }
