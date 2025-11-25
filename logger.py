
import csv, os
from typing import Dict, Any

def write_report(states: Dict[str, Any], out_path: str):
    rows = []
    for name, s in states.items():
        rows.append({
            'market': name,
            'mid': s.mid,
            'spread': s.spread,
            'inventory': s.inventory,
            'pnl': s.pnl,
            'fill_count': s.fill_count,
            'notional': s.notional,
            'max_drawdown': s.max_drawdown
        })
    keys = rows[0].keys() if rows else []
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(keys))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
