# strategies/xs_meanrev.py
# PD-Ratio = 1.428
# True_PD = 0.565
# Activity = 30%

import math
import backtrader as bt
"""
python main.py --strategy xs_meanrev --data-dir DATA/PART1 --output-dir output/xsmeanrev --param bottom_pct=0.10 --param meanrev_window=5 --param day_top_n=1 --param entry_units_cap=500 --param atr_k=2.3 --param cooldown=2 --param two_bar_exit=True --debug
"""

"""
XsMeanReversion
---------------
- Use overspend_guard before sending market orders.
- Use place_market and place_limit helpers (they're injected by the wrapper).
"""

class XsMeanReversion(bt.Strategy):
    """
    Cross-sectional mean reversion trading strategy with tight ranking and reversal confirmation.
    
    Entry Logic:
    - Identify laggards: bottom 10% of symbols by N-bar return
    - Require reversal: close > previous close AND close > open
    - Volume gate: today's true range > 5-bar median true range
    - Max 1 symbol per day
    
    Exit Logic:
    - Trailing stop: previous bar's close - 2.3*ATR
    - Exit triggered after 2 consecutive bars below stop
    - Severe breach near end-window (0.75*ATR buffer)
    
    Risk Constraints:
    - 50% of cash deployed per trade (buffer 100k)
    - Max 80% of portfolio in live positions
    - Daily budget: 40% of cash
    - Per-symbol daily limit: 40% of daily budget
    - Position size capped at 2500 units
    """
    
    params = dict(
        # Portfolio Risk Management
        cash_frac=0.50,                    # Fraction of available cash to deploy per trade
        cash_buffer=100000.0,              # Minimum cash to keep in reserve
        max_exposure_frac=0.80,            # Max portfolio invested (80% of portfolio value)

        # Daily Trading Constraints
        day_budget_frac=0.40,              # Max fraction of cash to spend per day across all trades
        day_top_n=1,                       # Max number of symbols to trade per day
        day_symbol_cap_notional=0.40,      # Per-symbol share of daily budget (40% of day_cap per symbol)

        # Individual Trade Sizing
        max_units_per_symbol=2500,         # Hard cap on units per symbol
        entry_units_cap=500,               # Max initial units on entry (before ATR sizing)
        risk_notional_per_leg=25000.0,     # Risk target, used to derive units from ATR

        # Mean Reversion Selection
        meanrev_window=5,                  # N-bar lookback for selecting laggards
        bottom_pct=0.10,                   # Bottom decile: 10% worst performers
        min_tr_range=5,                    # Only trade if true range > 5-bar median

        # Exit & Trailing Stop
        atr_len=14,                        # ATR period for volatility-based stops
        atr_k=2.3,                         # Trailing stop = recent_close - 2.3*ATR
        two_bar_exit=True,                 # Exit after 2 bars below trailing stop (vs 1 bar)
        cooldown=2,                        # Bars to wait before re-entering same symbol after exit

        # End-window guards and optional breaker
        risk_free_bars=7,                  # No new entries; severe exits only - reducing tail risk
        end_taper_bars=14,                 # Taper sizing/budget here
        severe_breach_mult=0.75,           # 0.75*ATR buffer near end

        dd_window=0,                       # Set >0 (e.g., 30) to enable breaker
        dd_thresh_frac=0.04,               # 4% of start equity
        dd_block_bars=10,                  # Block new entries for K bars

        printlog=False,                    # Logging of fills and rejections
    )

    def __init__(self):
        
        # Technical Indicators
        self.atr = {d: bt.indicators.ATR(d, period=int(self.p.atr_len)) for d in self.datas}
        
        # State tracking: trail, cooldown period, breach count, entry price, entry size
        self.state = {d: dict(trail=None, cool_until=-math.inf, breach_count=0, entry_px=None, entry_units=0) for d in self.datas}
        self.prev_pos = {d: 0.0 for d in self.datas}
        
        # Daily budget tracking
        self.day_spent_notional = 0.0
        self.last_calendar_date = None
        self.day_symbol_spent = {}

        # Circuit breaker state
        self.start_value = None
        self.roll_peak = None
        self.roll_dd = 0.0
        self.block_until_bar = -math.inf

    def ready(self):
        """Check if there are historical bars for all indicators."""
        need = max(6, int(self.p.meanrev_window), int(self.p.atr_len))
        return len(self) >= need

    def reset_day_if_needed(self):
        """Reset daily budget counter when calendar date changes."""
        cur = self.datas[0].datetime.date(0)
        if self.last_calendar_date != cur:
            self.day_spent_notional = 0.0
            self.last_calendar_date = cur
            self.day_symbol_spent = {}

    def units_from_risk(self, d):
        """
        Calculate position sizing using ATR based risk model.
        units = risk_notional_per_leg / (ATR*price)
        """
        atr_val = float(self.atr[d][0]) if self.atr[d] is not None else 0.0
        price = float(d.close[0])
        if atr_val <= 0 or price <= 0:
            return 0
        return max(0, math.floor(self.p.risk_notional_per_leg / atr_val / price))

    def next(self):
        if not self.ready():
            return
        
        # Initialize breaker baseline
        if self.start_value is None:
            try:
                self.start_value = float(self.broker.getvalue())
            except Exception:
                self.start_value = 1.0
            self.roll_peak = self.start_value

        self.reset_day_if_needed()
        tbar = len(self)

        # Bars left until end
        try:
            bars_left = self.datas[0].buflen() - len(self)
        except Exception:
            bars_left = 999999
        
        block_new_entries = bars_left <= int(self.p.risk_free_bars)
        final_week = bars_left <= int(self.p.end_taper_bars)
        
        # Taper sizing/budget near end (linear; floor at 0.3)
        if final_week and not block_new_entries:
            span = max(1, int(self.p.end_taper_bars) - int(self.p.risk_free_bars))
            over = max(0, bars_left - int(self.p.risk_free_bars))
            taper_scale = max(0.3, over / span)
        else:
            taper_scale = 1.0
        
        day_top_n_eff = 0 if block_new_entries else 1
        
        # Tapered daily budget
        cash = float(self.broker.getcash())
        day_cap = cash * float(self.p.day_budget_frac) * float(taper_scale)

        # Entry Logic: Cross-sectional mean reversion (worst decile, strong reversal, big true range)
        perf = {d: float(d.close[0]) - float(d.close[-int(self.p.meanrev_window)]) for d in self.datas}
        ranked = sorted(perf, key=perf.get)  # Ascending; worst first
        cutoff = max(1, int(self.p.bottom_pct * len(ranked)))
        laggards = set(ranked[:cutoff])

        # Candidate Search
        candidates = []
        for d in laggards:
            if block_new_entries:
                continue
            st = self.state[d]
            pos = self.getposition(d).size
            if pos > 0 or tbar < st['cool_until']:  # Skip if pos already in use or in cooldown period
                continue
            
            # Require reversal: close > previous close AND close > open
            if float(d.close[0]) > float(d.close[-1]) and float(d.close[0]) > float(d.open[0]):
                # Must be meaningful reversal (today's true range > 5-bar median)
                t_range = abs(float(d.high[0]) - float(d.low[0]))
                tr_median = sorted([abs(float(d.high[-k]) - float(d.low[-k])) for k in range(5)])[2]
                if t_range > tr_median:
                    candidates.append(d)
        
        candidates = candidates[:day_top_n_eff] if day_top_n_eff else candidates

        # Executing entries with conservative sizing (+ taper) and per-symbol cap
        for d in candidates:
            price = float(d.close[0])
            entry_cap = int(self.p.entry_units_cap)
            atr_units = self.units_from_risk(d)
            units = min(entry_cap, atr_units)
            
            # Apply taper near end
            units = int(max(1, math.floor(units * float(taper_scale))))

            cash_now = float(self.broker.getcash())
            port_val = float(self.broker.getvalue())
            invested_est = port_val - cash_now
            headroom_val = max(0.0, float(self.p.max_exposure_frac) * port_val - invested_est)
            units_cash = math.floor(max(0.0, cash_now * float(self.p.cash_frac) - float(self.p.cash_buffer)) / price) if price > 0 else 0
            units_headroom = math.floor(headroom_val / price) if price > 0 else 0

            units = max(0, min(units, units_cash, units_headroom, int(self.p.max_units_per_symbol)))
            pos = self.getposition(d).size
            delta = float(units) - float(pos)
            if delta <= 0:
                continue

            # Check daily budget and per-symbol limit
            notional = price * delta
            sym_spent = self.day_symbol_spent.get(d._name, 0.0)
            sym_cap = float(self.p.day_symbol_cap_notional) * day_cap
            if sym_spent + notional > sym_cap:
                continue
            if self.day_spent_notional + notional > day_cap:
                continue

            # Execute trade
            self.buy(data=d, size=delta)
            self.day_spent_notional += notional
            self.day_symbol_spent[d._name] = sym_spent + notional

            # Initialize trailing stop at previous bar's close minus ATR buffer
            st = self.state[d]
            trail = float(d.close[-1])
            if self.atr[d] is not None:
                trail = max(trail, float(d.close[-1]) - float(self.p.atr_k) * float(self.atr[d][0]))
            st['trail'] = trail
            st['breach_count'] = 0

        # Exiting active positions (severe-breach near end)
        for d in self.datas:
            pos = self.getposition(d).size
            if pos <= 0:
                continue
            
            st = self.state[d]
            
            # Check trailing stop doesn't go down
            new_trail = float(d.close[-1])
            if self.atr[d] is not None:
                new_trail = max(new_trail, float(d.close[-1]) - float(self.p.atr_k) * float(self.atr[d][0]))
            st['trail'] = max(st['trail'] if st['trail'] is not None else -math.inf, new_trail)
            
            # Check for exit signal with optional severe breach near end
            if self.atr[d] is not None:
                atr_now = float(self.atr[d][0])
                buffer_mult = float(self.p.severe_breach_mult) if block_new_entries else 0.25
                breach = float(d.close[0]) < float(st['trail']) - buffer_mult * atr_now
            else:
                breach = float(d.close[0]) < float(st['trail'])
            
            st['breach_count'] = st['breach_count'] + 1 if breach else 0
            
            # Exit if breached trailing stop for 2 bars
            need_bars = 2 if self.p.two_bar_exit else 1
            if st['breach_count'] >= need_bars and not block_new_entries:
                self.sell(data=d, size=pos)
                st['cool_until'] = tbar + int(self.p.cooldown)
                st['breach_count'] = 0
                continue

    def notify_order(self, order):
        """Handle order execution callbacks."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        d = order.data
        if order.status in [order.Completed]:
            price = order.executed.price
            size = order.executed.size
            prev_pos = self.prev_pos.get(d, 0.0)
            new_pos = self.getposition(d).size
            if order.isbuy() and prev_pos <= 0 and new_pos > 0:
                st = self.state[d]
                st['entry_px'] = float(price)
                st['entry_units'] = abs(float(size))
            self.prev_pos[d] = new_pos
