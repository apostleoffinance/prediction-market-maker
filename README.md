# Quant Execution Bot - Market Making System

A production-grade market-making execution bot for binary event contracts (prediction markets) with dynamic pricing, inventory management, and real-time risk controls.

## 📋 Table of Contents
- [Architecture Overview](#architecture-overview)
- [How to Run Simulations](#how-to-run-simulations)
- [Interpreting Results](#interpreting-results)
- [Key Features](#key-features)
- [Further Optimizations](#further-optimizations)

---

## 🏗️ Architecture Overview

### Project Structure
```
quant_bot_submission/
├── main.py                    # Entry point and simulation orchestrator
├── market_state.py            # Market state container and trade recording
├── market_maker.py            # Core quoting logic and adaptation algorithms
├── execution_engine.py        # Simulation driver and order flow generator
├── logger.py                  # CSV report writer
├── dashboard.py               # Streamlit visualization dashboard
├── requirements.txt           # Python dependencies
├── simulation_report.csv      # Final metrics (generated)
├── trace.json                 # Time-series data (generated)
└── DESIGN_NOTE.txt           # Technical design documentation
```

### Core Components

**1. MarketState** (`market_state.py`)
- Maintains state for each market: mid price, spread, inventory, exposure, PnL
- Records all fills with timestamps
- Tracks performance metrics: notional, drawdown, fill count

**2. MarketMaker** (`market_maker.py`)
- **Quoting Logic**: Generates bid/ask prices based on mid, spread, and inventory
- **Adaptive Behavior**: Responds to order flow imbalance and position risk
- **Risk Management**: Widens spreads and skews prices to manage inventory

**3. ExecutionEngine** (`execution_engine.py`)
- Simulates realistic market order flow (biased by current mid price)
- Matches taker orders against market maker quotes
- Updates PnL and tracks drawdown per tick
- Applies mean reversion to mid prices

---

##  How to Run Simulations

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Setup Instructions

**1. Create and activate virtual environment:**
```bash
cd quant_bot_submission
python3 -m venv prediction_market
source prediction_market/bin/activate  # On macOS/Linux
# prediction_market\Scripts\activate  # On Windows
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the simulation:**
```bash
python3 main.py
```

**Expected Output:**
```
Simulation complete. Report written to: /path/to/simulation_report.csv
inflation_gt_20 {'name': 'inflation_gt_20', 'mid': 0.5076, 'spread': 0.1122, ...}
election_candidate_a {'name': 'election_candidate_a', 'mid': 0.5161, ...}
team_x_wins {'name': 'team_x_wins', 'mid': 0.4852, ...}
```

**Output Files:**
- `simulation_report.csv` - Final metrics for all markets
- `trace.json` - Step-by-step time-series data (200 ticks)

### Customizing Simulations

**Modify simulation parameters in `main.py`:**

```python
def build_markets():
    markets = {
        'inflation_gt_20': MarketState('inflation_gt_20', initial_mid=0.30),
        'election_candidate_a': MarketState('election_candidate_a', initial_mid=0.55),
        'team_x_wins': MarketState('team_x_wins', initial_mid=0.50)
    }
    for m in markets.values():
        m.inventory_limit = 200.0      # Adjust position limits
        m.exposure_limit = 10000.0     # Adjust exposure limits
        m.spread = 0.05                # Adjust base spread
    return markets

# Change number of simulation steps
trace = engine.run(steps=200)  # Default: 200 ticks
```

**Configure MarketMaker behavior:**

```python
# In execution_engine.py or create config
config = {
    'window_size': 20,           # Imbalance window size
    'base_spread': 0.05,         # Base spread width
    'min_spread': 0.01,          # Minimum allowed spread
    'max_spread': 0.5,           # Maximum allowed spread
    'inventory_skew': 0.001      # Inventory skew factor
}
mm = MarketMaker(state, config=config)
```

---

## Interpreting Results

### Dashboard Visualization

**Launch the interactive dashboard:**
```bash
streamlit run dashboard.py
```

Access at: `http://localhost:8501`

**Dashboard Features:**
- 📈 **Summary Metrics**: Total PnL, fills, notional across all markets
- 📊 **Market Comparison**: Visual comparison of PnL and inventory
- ⏱️ **Time Series**: Evolution of mid price, inventory, PnL, spread
- 🎯 **Performance Metrics**: Per-market detailed statistics

### CSV Report Analysis

**Sample `simulation_report.csv`:**
```csv
market,mid,spread,inventory,pnl,fill_count,notional,max_drawdown
inflation_gt_20,0.5076,0.1122,83.86,189.76,413,1052.91,0.71
election_candidate_a,0.5161,0.0716,-24.30,193.38,404,1282.85,0.12
team_x_wins,0.4852,0.1812,15.58,210.58,403,1209.04,0.06
```

**Key Metrics Explained:**

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **mid** | Final mid price (probability) | Market's terminal probability estimate (0-1) |
| **spread** | Final spread width | Reflects final market uncertainty and inventory risk |
| **inventory** | Net position | Positive = long, Negative = short. Target: near zero |
| **pnl** | Profit & Loss | Revenue from market making minus position losses |
| **fill_count** | Number of trades | Higher = more active market participation |
| **notional** | Total volume traded | Sum of (price × size) for all fills |
| **max_drawdown** | Peak-to-trough PnL decline | Risk metric - lower is better |

### Performance Indicators

**✅ Good Performance:**
- Positive PnL across all markets
- Low max drawdown (< $1-2)
- Inventory near zero or manageable (<50% of limit)
- Spread adapting to market conditions (0.05-0.20 range)
- High fill count relative to simulation steps

**⚠️ Warning Signs:**
- Negative PnL (adverse selection or poor pricing)
- High max drawdown (>$5) indicates risk management issues
- Inventory near limit (>150) suggests position accumulation
- Very wide spreads (>0.3) indicate defensive mode
- Low fill count (<100 in 200 steps) suggests uncompetitive quotes

### Trace Analysis

**`trace.json` provides tick-by-tick data:**
```json
{
  "inflation_gt_20": {
    "fills": [["buy", 6.19, 0.275]],
    "mid": 0.3200,
    "inventory": 6.19,
    "pnl": 0.27,
    "spread": 0.05
  }
}
```

**Use for:**
- Debugging specific trade sequences
- Analyzing adaptation speed to market events
- Identifying risk management trigger points
- Backtesting strategy variations

---

## 🎯 Key Features

### Dynamic Pricing Algorithm

**Quote Generation:**
```python
1. Calculate imbalance from recent trades (buy/sell pressure)
2. Widen spread: base_spread × (1 + |imbalance|/10 + |inventory|×skew)
3. Skew mid price against inventory position
4. Generate: bid = mid_shaded - spread/2, ask = mid_shaded + spread/2
5. Size inversely proportional to inventory
```

### Adaptive Behavior

**On Fill Response:**
- Updates rolling imbalance window
- Moves mid price toward order flow (5% learning rate)
- Defensive correction when inventory >80% of limit
- Reduces quote sizes as position grows

### Risk Management

1. **Spread Widening**: Increases with imbalance & inventory
2. **Quote Skewing**: Prices discourage further inventory accumulation
3. **Size Reduction**: Smaller quotes when inventory is large
4. **Mid Correction**: Defensive repricing at risk thresholds
5. **Mean Reversion**: 0.5% pull toward neutral (0.5) per tick

---

## 🚀 Further Optimizations

### 1. **Matching Engine & Order Book** (Priority: HIGH)

**Current State:**
- Simplified immediate fill model
- No partial fills or queue simulation
- Price-taking orders only

**Optimization:**
```python
class LimitOrderBook:
    def __init__(self):
        self.bids = SortedDict()  # price -> [(size, order_id, timestamp)]
        self.asks = SortedDict()
        
    def add_order(self, side, price, size, order_id):
        # Queue orders by price-time priority
        
    def match(self, side, price, size):
        # Implement proper matching logic
        # Return partial fills if insufficient liquidity
```

**Benefits:**
- Realistic market microstructure
- Better slippage modeling
- Queue position dynamics
- More accurate PnL estimation

---

### 2. **Advanced Statistical Models** (Priority: HIGH)

**Current:** Simple alpha × flow update to mid price

**Optimization:**

**A. Bayesian Probability Updating**
```python
class BayesianPriceModel:
    def update_belief(self, prior, flow_signal, confidence):
        # Use conjugate priors for efficient updates
        posterior = (prior * prior_weight + signal * signal_weight) / total_weight
        return posterior
```

**B. Flow Toxicity Detection**
```python
def calculate_vpin(self):
    # Volume-Synchronized Probability of Informed Trading
    buy_vol = sum(sizes for side, sizes in window if side == 'buy')
    sell_vol = sum(sizes for side, sizes in window if side == 'sell')
    return abs(buy_vol - sell_vol) / (buy_vol + sell_vol)
```

**C. Mean-Reverting Process (Ornstein-Uhlenbeck)**
```python
def update_mid_ou_process(self, current, theta=0.1, mu=0.5, dt=1):
    # dX = theta * (mu - X) * dt + sigma * dW
    reversion = theta * (mu - current) * dt
    flow_signal = self.calculate_flow_signal()
    return current + reversion + flow_signal
```

**Benefits:**
- More sophisticated probability estimation
- Better differentiation between informed vs noise trades
- Reduced adverse selection

---

### 3. **Machine Learning Integration** (Priority: MEDIUM)

**A. Reinforcement Learning for Dynamic Quoting**
```python
import torch
import torch.nn as nn

class QuotingAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  # [spread_adjustment, skew]
        )
    
    def forward(self, state):
        # state: [mid, inventory, imbalance, recent_pnl, volatility]
        return self.network(state)

# Train using PPO or DQN
# Reward = PnL - penalty * inventory^2 - penalty * spread_width
```

**B. LSTM for Flow Prediction**
```python
class FlowPredictor(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(input_size=10, hidden_size=64, num_layers=2)
        self.fc = nn.Linear(64, 1)  # Predict next flow direction
    
    def forward(self, sequence):
        # sequence: historical [price, size, side, time_delta, ...]
        lstm_out, _ = self.lstm(sequence)
        return self.fc(lstm_out[-1])
```

**Benefits:**
- Learn optimal policies from historical data
- Adaptive to changing market regimes
- Non-linear relationship modeling

---

### 4. **Performance & Scalability** (Priority: HIGH)

**A. Rust/C++ Critical Path**
```rust
// execution_core.rs
pub struct FastMatcher {
    bids: BTreeMap<OrderedFloat<f64>, Vec<Order>>,
    asks: BTreeMap<OrderedFloat<f64>, Vec<Order>>,
}

impl FastMatcher {
    pub fn match_order(&mut self, order: &Order) -> Vec<Fill> {
        // High-performance matching with zero-copy
    }
}

// Python binding with PyO3
#[pyfunction]
fn match_orders(orders: Vec<Order>) -> PyResult<Vec<Fill>> {
    // Expose to Python
}
```

**B. Asynchronous I/O**
```python
import asyncio
import websockets

class AsyncMarketMaker:
    async def handle_order_stream(self, websocket):
        async for message in websocket:
            order = parse_order(message)
            fills = await self.process_order(order)
            await websocket.send(json.dumps(fills))
    
    async def quote_updater(self):
        while True:
            quotes = self.generate_quotes()
            await self.broadcast_quotes(quotes)
            await asyncio.sleep(0.1)  # 100ms update frequency
```

**Benefits:**
- Sub-millisecond latency for high-frequency trading
- Handle 10k+ orders per second
- Real-time quote updates

---

### 5. **Risk Management Enhancements** (Priority: HIGH)

**A. Portfolio-Level Risk**
```python
class PortfolioRiskManager:
    def __init__(self):
        self.correlation_matrix = np.eye(n_markets)
        
    def calculate_var(self, positions, confidence=0.95):
        # Value at Risk calculation
        portfolio_volatility = np.sqrt(
            positions.T @ self.correlation_matrix @ positions
        )
        return norm.ppf(1 - confidence) * portfolio_volatility
    
    def rebalance_if_needed(self, markets):
        var = self.calculate_var([m.inventory for m in markets])
        if var > self.var_limit:
            # Reduce exposure in correlated markets
            self.aggressive_inventory_reduction()
```

**B. Dynamic Position Limits**
```python
def adaptive_inventory_limit(self, volatility, liquidity):
    # Scale limits based on market conditions
    base_limit = 200
    vol_factor = 1.0 / (1.0 + volatility)
    liq_factor = liquidity / 1000.0
    return base_limit * vol_factor * liq_factor
```

**C. Circuit Breakers**
```python
class CircuitBreaker:
    def __init__(self):
        self.pnl_threshold = -100  # Stop if loss > $100
        self.drawdown_threshold = 0.15  # 15% from peak
        self.quote_error_limit = 5  # Consecutive errors
        
    def should_halt_trading(self, state):
        if state.pnl < self.pnl_threshold:
            logger.critical("PnL circuit breaker triggered")
            return True
        # Additional checks...
```

**Benefits:**
- Multi-dimensional risk control
- Correlated position management
- Automatic emergency stops

---

### 6. **Testing & Validation** (Priority: HIGH)

**A. Unit Tests**
```python
# test_market_maker.py
import pytest

def test_quote_generation():
    state = MarketState('test', initial_mid=0.5)
    mm = MarketMaker(state)
    bid, ask, size = mm.quote()
    
    assert 0 <= bid < ask <= 1
    assert ask - bid >= mm.min_spread
    assert size > 0

def test_inventory_skew():
    state = MarketState('test', initial_mid=0.5)
    state.inventory = 100  # Long position
    mm = MarketMaker(state)
    bid1, ask1, _ = mm.quote()
    
    # Quotes should be skewed down to discourage more buys
    assert (bid1 + ask1) / 2 < 0.5
```

**B. Integration Tests**
```python
def test_full_simulation_reproducibility():
    markets1 = build_markets()
    engine1 = ExecutionEngine(markets1, rng_seed=123)
    trace1 = engine1.run(steps=100)
    
    markets2 = build_markets()
    engine2 = ExecutionEngine(markets2, rng_seed=123)
    trace2 = engine2.run(steps=100)
    
    # Same seed should produce identical results
    assert trace1 == trace2
```

**C. Backtesting Framework**
```python
class Backtester:
    def __init__(self, historical_data):
        self.data = historical_data
        
    def run_strategy(self, strategy, start_date, end_date):
        results = []
        for tick in self.data.iter_range(start_date, end_date):
            quotes = strategy.generate_quotes(tick.state)
            fills = self.match_against_historical(quotes, tick.orders)
            results.append(self.calculate_metrics(fills))
        return BacktestReport(results)
```

**Benefits:**
- Catch regressions early
- Validate strategy changes
- Build confidence in production deployment

---

### 7. **Monitoring & Observability** (Priority: MEDIUM)

**A. Real-Time Metrics**
```python
from prometheus_client import Counter, Histogram, Gauge

fills_counter = Counter('mm_fills_total', 'Total fills', ['market', 'side'])
pnl_gauge = Gauge('mm_pnl', 'Current PnL', ['market'])
latency_histogram = Histogram('mm_quote_latency_seconds', 'Quote generation time')

@latency_histogram.time()
def generate_quotes(self):
    # Measure performance
    quotes = self.quote()
    return quotes
```

**B. Structured Logging**
```python
import structlog

logger = structlog.get_logger()

def on_fill(self, side, size, price):
    logger.info(
        "fill_executed",
        market=self.state.name,
        side=side,
        size=size,
        price=price,
        inventory_after=self.state.inventory,
        pnl=self.state.pnl
    )
```

**C. Alerting**
```python
class AlertManager:
    def check_anomalies(self, state):
        if state.pnl < -50:
            self.send_alert("HIGH", "PnL drop detected")
        if abs(state.inventory) > state.inventory_limit * 0.9:
            self.send_alert("MEDIUM", "Inventory near limit")
```

**Benefits:**
- Real-time performance visibility
- Quick incident response
- Historical analysis capabilities

---

### 8. **Market Data Integration** (Priority: MEDIUM)

**A. WebSocket Feeds**
```python
async def connect_to_exchange():
    async with websockets.connect(EXCHANGE_WS_URL) as ws:
        await ws.send(json.dumps({"action": "subscribe", "markets": ["BTC"]}))
        async for msg in ws:
            data = json.parse(msg)
            await process_market_data(data)
```

**B. Historical Data Pipeline**
```python
class MarketDataStore:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        
    def store_tick(self, market, timestamp, bid, ask, last_price, volume):
        self.conn.execute("""
            INSERT INTO ticks VALUES (?, ?, ?, ?, ?, ?)
        """, (market, timestamp, bid, ask, last_price, volume))
```

---

## 📚 Additional Resources

**Design Documentation:**
- See `DESIGN_NOTE.txt` for detailed architecture explanation
- See `dashboard.py` for visualization code

**Code Style:**
- Type hints throughout for clarity
- Modular design for easy extension
- Configuration-driven behavior

**Future Work:**
- Implement portfolio optimization
- Add multi-asset correlation analysis
- Build live trading connector for production deployment

---

## 📧 Questions?

For questions about implementation details or deployment strategies, please refer to the code comments or reach out for clarification.

**Current Implementation Status:**
✅ Core market making logic  
✅ Risk management framework  
✅ Simulation engine  
✅ Visualization dashboard  
✅ CSV reporting  

**Ready for Production with:**
- Order book integration
- Real-time data feeds
- Enhanced risk controls
- Comprehensive testing suite