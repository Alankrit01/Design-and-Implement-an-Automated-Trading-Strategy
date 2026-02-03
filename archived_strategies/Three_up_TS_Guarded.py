# strategies/s_threeup_trail_sized.py
# PD-Ratio = 0.484
"""
TemplateStrategy
----------------
- Use overspend_guard before sending market orders.
- Use place_market and place_limit helpers (they're injected by the wrapper).
"""

import backtrader as bt
import math

class TeamStrategy(bt.Strategy):
    """
    Multi-symbol momentum trading strategy with advanced risk management features.
    
    Entry Logic:
    - Symbol must show 2+ up days in last 3 bars
    - Current close >= 5-bar high (breakout)
    - Volume z-score >= 0 (above average volume)
    - Price above rising 20-SMA (uptrend confirmation)
    
    Exit Logic:
    - Trailing stop: previous bar's close - 2.2*ATR
    - Exit triggered after 2 consecutive bars below stop
    - Severe breach near end-window (0.75*ATR buffer)
    
    Pyramiding:
    - Add 50% of entry size when profit reaches 1.5*ATR
    - Max 1 addition per position
    - Disabled in final 14 bars
    
    Risk Constraints:
    - 50% of cash deployed per trade (buffer 100k)
    - Max 80% of portfolio in live positions
    - Daily budget: 40% of cash
    - Per-symbol daily limit: 50% of daily budget
    - Position size capped at 5000 units
    """
    
    params = dict(
        # Portfolio Risk Management 
        cash_frac=0.50,                    # Fraction of available cash to deploy per trade
        cash_buffer=100000.0,              # Minimum cash to keep in reserve
        max_exposure_frac=0.80,            # Max portfolio invested (80% of portfolio value)

        # Daily Trading Constraints 
        day_budget_frac=0.40,              # Max fraction of cash to spend per day across all trades
        day_top_n=2,                       # Max number of symbols to trade per day
        day_symbol_cap_notional=0.50,      # Per-symbol share of daily budget (50% of day_cap per symbol)

        # Individual Trade Sizing 
        max_units_per_symbol=5000,         # Hard cap on units per symbol
        entry_units_cap=1500,              # Max initial units on entry (before ATR sizing)
        risk_per_leg=50000.0,              # Risk target, used to derive units from ATR

        # Adding on Strength
        add_pct_of_entry=0.50,             # Add position at 50% of original entry size
        add_trigger_atr=1.5,               # Add when profit reaches 1.5 * ATR above entry
        max_additions=1,                   # Maximum number of times to add to a position

        # Entry Conditions
        up_window=3,                       # Look back 3 bars for uptrend check
        up_days_min=2,                     # Need at least 2 up days in window to qualify
        breakout_lookback=5,               # Compare current close to 5-bar high
        ma_len=20,                         # SMA length for trend confirmation (close above SMA and SMA rising)

        # Volume Gate 
        vol_window=20,                     # Window for volume z-score calculation
        vol_z_min=0.0,                     # Require volume z-score >= 0 (average or above)

        # Exit & Trailing Stop 
        atr_len=14,                        # ATR period for volatility-based stops
        atr_k=2.2,                         # Trailing stop = recent_close - 2.2*ATR
        two_bar_exit=True,                 # Exit after 2 bars below trailing stop (vs 1 bar)
        cooldown=3,                        # Bars to wait before re-entering same symbol after exit

        # End-window guards and optional breaker
        risk_free_bars=7,                  # no new entries; severe exits only - Reducing tail risk
        end_taper_bars=14,                 # taper sizing/budget here
        severe_breach_mult=0.75,           # 0.75*ATR buffer near end

        dd_window=0,                       # set >0 (e.g., 30) to enable breaker
        dd_thresh_frac=0.04,               # 4% of start equity
        dd_block_bars=10,                  # block new entries for K bars

        printlog=False,                    # logging of fills and rejections
    )

    def __init__(self):
        
        # Technical Indicators
        self.sma = {}  
        self.atr = {}                     # Average True Range for volatility/sizing
        self.vol_ma = {}
        self.vol_std = {}
        self.highest_close = {}    
        
        # trail, cooldown period, breach count, entry price, entry size, pyramiding
        self.state = {}
        self.prev_pos = {}
        
        for d in self.datas:
            self.sma[d] = bt.indicators.SimpleMovingAverage(d.close, period=int(self.p.ma_len)) if self.p.ma_len else None
            self.atr[d] = bt.indicators.ATR(d, period=int(self.p.atr_len)) if self.p.atr_len else None
            self.vol_ma[d] = bt.indicators.SimpleMovingAverage(d.volume, period=int(self.p.vol_window))
            self.vol_std[d] = bt.indicators.StandardDeviation(d.volume, period=int(self.p.vol_window))
            self.highest_close[d] = bt.indicators.Highest(d.close, period=int(self.p.breakout_lookback)) if self.p.breakout_lookback else None

            self.state[d] = dict(trail=None, cool_until=-math.inf, breach_count=0, entry_px=None, entry_units=0, adds_done=0)
            self.prev_pos[d] = 0.0
        
        # Track day spent 
        self.day_spent = 0.0
        self.last_calendar_date = None
        self.day_symbol_spent = {}

        # Circuit breaker state
        self.start_value = None
        self.roll_peak = None
        self.roll_dd = 0.0
        self.block_until_bar = -math.inf
        
    def ready(self):
        """Check if we have enough historical bars for all indicators."""
        need = max(         # Lookback 3 bars
            3,
            int(self.p.up_window),
            int(self.p.vol_window),
            int(self.p.ma_len) if self.p.ma_len else 0,
            int(self.p.breakout_lookback) if self.p.breakout_lookback else 0,
            int(self.p.atr_len) if self.p.atr_len else 0,
        )
        return len(self) >= need
            
    def reset_day(self):
        """Reset daily budget counter when calendar date changes."""
        cur = self.datas[0].datetime.date(0)
        if self.last_calendar_date != cur:
            self.day_spent = 0.0
            self.last_calendar_date = cur
            self.day_symbol_spent = {}
            
    def entry_score(self, d, ups, vol_z):
        """
        Calculate entry signal strength by combining multiple indicators.
        
        Parameters:
            ups : number of up days in recent window
            breakout_distance : how far above 5 bar high (% gain)
            vol_z : volume normalised to recent average
        
        Returns: float score (higher is better)
        """
        score = float(ups)
        if self.highest_close[d] is not None:
            prev_high = float(self.highest_close[d][-1])
            if prev_high > 0:
                dist = max(0.0, float(d.close[0]) - prev_high)
                score += 0.5 * (dist / max(prev_high, 1e-9))
        score += 0.25 * max(min(vol_z, 3.0), -3.0)
        return score
    
    def units_from_risk(self, d):
        """
        Calculate position sizing using ATR based risk model. 
        units = risk_per_leg / (ATR*price)
        """
        atr_val = float(self.atr[d][0]) if self.atr[d] is not None else 0.0
        price_intent = float(d.close[0])
        if atr_val <= 0 or price_intent <= 0:
            return 0
        return max(0, math.floor(self.p.risk_per_leg / atr_val / price_intent))
    
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
        
        self.reset_day()
        tbar = len(self)    # Current bar number

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

        max_additions_effective = 0 if final_week else int(self.p.max_additions)
        day_top_n_eff = 0 if block_new_entries else (1 if final_week else (int(self.p.day_top_n) if self.p.day_top_n else 0))

        # Tapered daily budget and optional circuit breaker
        cash = float(self.broker.getcash())
        day_cap = cash * float(self.p.day_budget_frac) * float(taper_scale)

        if int(self.p.dd_window) > 0 and self.start_value:
            cur_val = float(self.broker.getvalue())
            self.roll_peak = max(self.roll_peak, cur_val)
            dd = (self.roll_peak - cur_val) / max(1e-9, self.start_value)
            self.roll_dd = dd
            if dd >= float(self.p.dd_thresh_frac):
                self.block_until_bar = len(self) + int(self.p.dd_block_bars)
        block_for_dd = len(self) < self.block_until_bar
        
        # Candidate Search
        candidates = []
        for d in self.datas:
            if block_new_entries or block_for_dd:
                continue
            st = self.state[d]  
            pos = self.getposition(d).size
            if pos > 0 or tbar < st['cool_until']:      # skip if pos already in use or in cooldown period
                continue
            
            # Entry Signal 1 : Uptrend in recent bars
            ups = sum(1 for k in range(1, int(self.p.up_window) + 1) if float(d.close[0]) > float(d.close[-k]))
            up_ok = ups >= int(self.p.up_days_min)
            
            # Entry Signal 2 : High Volume
            v = float(d.volume[0])
            mu = float(self.vol_ma[d][0])
            sd_raw = float(self.vol_std[d][0])
            sd = sd_raw if sd_raw != 0 else 1.0  
            vol_z = (v - mu) / sd
            vol_ok = vol_z >= float(self.p.vol_z_min)
            
            # Entry Signal 3 : Breakout
            brk_ok = True
            if self.highest_close[d] is not None:
                brk_ok = float(d.close[0]) >= float(self.highest_close[d][-1])
                
            # Entry Signal 4 : Uptrend in MA (price>SMA and SMA going up)
            trend_ok = True
            if self.sma[d] is not None:
                trend_ok = float(d.close[0]) > float(self.sma[d][0]) and float(self.sma[d][0]) > float(self.sma[d][-1])
                
            if up_ok and trend_ok and vol_ok and brk_ok:
                score = self.entry_score(d, ups, vol_z)
                candidates.append((score, d, pos)) 
                
        # Rank scores and only enter top 2 candidates per day
        candidates.sort(key=lambda x: x[0], reverse=True)  
        N = int(day_top_n_eff) if day_top_n_eff and day_top_n_eff > 0 else len(candidates)
        chosen = candidates[:N]
        
        # Normalise scores for position sizing
        if len(chosen) > 1:
            max_s = chosen[0][0]
            min_s = chosen[-1][0]
            span = max(max_s - min_s, 1e-9)
            norm = {d: (s - min_s) / span for s, d, _ in chosen}
        elif len(chosen) == 1:
            s, d1, _ = chosen[0]
            norm = {d1: 1.0}
        else:
            norm = {}
        
        # Executing entries with conservative sizing (+ taper) and per-symbol cap
        for s, d, pos in chosen:
            score_norm = norm.get(d, 0.0)
            price_intent = float(d.close[0])  
            
            # starting with normalised score * entry cap
            entry_cap = int(self.p.entry_units_cap)
            base_units_raw = int(max(1, math.floor(score_norm * entry_cap)))
            atr_units = self.units_from_risk(d)     # Limiting to ATR based risk sizing
            units = base_units_raw if atr_units == 0 else min(base_units_raw, atr_units)

            # apply taper near end
            units = int(max(1, math.floor(units * float(taper_scale))))
            
            cash_now = float(self.broker.getcash())
            port_val = float(self.broker.getvalue())
            invested_est = port_val - cash_now
            headroom_val = max(0.0, float(self.p.max_exposure_frac) * port_val - invested_est)
            units_cash = math.floor(max(0.0, cash_now * float(self.p.cash_frac) - float(self.p.cash_buffer)) / price_intent) if price_intent > 0 else 0
            units_headroom = math.floor(headroom_val / price_intent) if price_intent > 0 else 0
            
            units = max(0, min(units, units_cash, units_headroom, int(self.p.max_units_per_symbol)))  
            delta = float(units) - float(pos)
            if delta <= 0:
                continue
            
            # Check daily budget and per-symbol limit
            notional = price_intent * delta
            sym_spent = self.day_symbol_spent.get(d._name, 0.0)
            sym_cap = float(self.p.day_symbol_cap_notional) * day_cap
            if sym_spent + notional > sym_cap:
                continue
            if self.day_spent + notional > day_cap:
                continue
            
            # Execute trade
            self.buy(data=d, size=delta)
            self.day_spent += notional
            self.day_symbol_spent[d._name] = sym_spent + notional
            
            # Initialize trailing stop at previous bar's close minus ATR buffer
            st = self.state[d]
            trail = float(d.close[-1])
            if self.atr[d] is not None:
                trail = max(trail, float(d.close[-1]) - float(self.p.atr_k) * float(self.atr[d][0]))
            st['trail'] = trail
            st['breach_count'] = 0
            
        # Exiting and Pyramiding active positions (severe-breach near end; adds disabled in final week)
        for d in self.datas:
            pos = self.getposition(d).size
            if pos <= 0:
                continue
            
            # check trailing stop doesnt go down 
            st = self.state[d]
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
            
            # Pyramiding
            if int(max_additions_effective) > 0 and st['entry_px'] is not None and self.atr[d] is not None and st['adds_done'] < int(max_additions_effective):  
                atr_val = float(self.atr[d][0])
                if atr_val > 0:
                    # Add when profit >= 1.5*ATR above entry price
                    gain = float(d.close[0]) - float(st['entry_px']) 
                    if gain >= float(self.p.add_trigger_atr) * atr_val:
                        # Add 50% of original entry size
                        add_units = int(max(1, math.floor(float(self.p.add_pct_of_entry) * float(st['entry_units']))))
                        cur_units = int(pos)
                        add_units = min(add_units, int(self.p.max_units_per_symbol) - cur_units)
                        price_intent = float(d.close[0])  
                        add_notional = price_intent * add_units
                        cash_now = float(self.broker.getcash())
                        day_cap_now = cash_now * float(self.p.day_budget_frac) * float(taper_scale)
                        if add_units > 0 and self.day_spent + add_notional <= day_cap_now:
                            self.buy(data=d, size=add_units)
                            self.day_spent += add_notional
                            st['adds_done'] += 1
                            
    def notify_order(self, order):
        """Handle order execution callbacks."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        d = order.data
        if order.status in [order.Completed]:
            side = "BUY" if order.isbuy() else "SELL"
            price = order.executed.price
            size = order.executed.size
            prev_pos = self.prev_pos.get(d, 0.0)  
            new_pos = self.getposition(d).size
            if order.isbuy() and prev_pos <= 0 and new_pos > 0:
                st = self.state[d]  
                st['entry_px'] = float(price)
                st['entry_units'] = abs(float(size))
                st['adds_done'] = 0
            self.prev_pos[d] = new_pos  
            if self.p.printlog:
                dt = self.datas[0].datetime.date(0)
                print(f"{dt.isoformat()} {d._name} FILL {side} px={price:.6g} size={size:.4f}")
        elif order.status in [order.Canceled, order.Rejected]:
            if self.p.printlog:
                dt = self.datas[0].datetime.date(0)
                print(f"{dt.isoformat()} {d._name} ORDER {order.Status[order.status]}")
