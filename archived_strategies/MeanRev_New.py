import math
import backtrader as bt

'''
python main.py --strategy MeanRev_New --data-dir DATA/PART1 --output-dir output/MR_NEW --debug
'''

'''
{
  "final_value": 1010004.2500000001,
  "bankrupt": false,
  "bankrupt_date": null,
  "open_pnl_pd_ratio": 1.2657120622568092,
  "true_pd_ratio": 1.2285705513938494,
  "activity_pct": 83.3,
  "end_policy": "liquidate",
  "s_mult": 2.0
}
=== Trade Analyzer Stats ===
Total closed trades: 6
Wins: 4
Losses: 2
'''

class MeanRev_New(bt.Strategy):
    params = dict(
        entry_units_cap=1000,
        atr_len=14,
        atr_k=2.3,
        cooldown=2,
        two_bar_exit=True,
        meanrev_window=5,
        top_pct=0.10,  # fraction for cross-sectional selection
        # Entry params
        volume_z=1.0,
        short_return=-0.03,
        long_return=0.0,
        bounce_pct=0.012,
        short_return_short=0.03,
        long_return_short=0.0,
        drop_pct_short=0.012,
        # Portfolio / PD
        cash_frac=0.60,
        cash_buffer=100000.0,
        max_exposure_frac=0.90,
        day_budget_frac=0.6,
        day_symbol_cap_notional=0.60,
        max_units_per_symbol=7000,
        risk_notional_per_leg=70000.0,
        risk_free_bars=7,
        end_taper_bars=14,
        severe_breach_mult=0.75,
        dd_window=0,
        dd_thresh_frac=0.04,
        dd_block_bars=10,
        printlog=False,
    )

    def __init__(self):
        self.atr = {d: bt.indicators.ATR(d, period=int(self.p.atr_len)) for d in self.datas}
        self.vol_avg = {d: bt.indicators.SimpleMovingAverage(d.volume, period=20) for d in self.datas}
        self.vol_std = {d: bt.indicators.StandardDeviation(d.volume, period=20) for d in self.datas}
        self.state = {d: dict(trail=None, cool_until=-math.inf, breach_count=0, entry_px=None, entry_units=0) for d in self.datas}
        self.prev_pos = {d: 0.0 for d in self.datas}
        self.day_spent_notional = 0.0
        self.last_calendar_date = None
        self.day_symbol_spent = {}
        self.start_value = None

    def ready(self):
        return len(self) >= 31

    def reset_day(self):
        cur = self.datas[0].datetime.date(0)
        if self.last_calendar_date != cur:
            self.day_spent_notional = 0.0
            self.last_calendar_date = cur
            self.day_symbol_spent = {}

    def units_from_risk(self, d):
        atr_val = float(self.atr[d][0]) if self.atr[d] is not None else 0.0
        price = float(d.close[0])
        if atr_val <= 0 or price <= 0:
            return 0
        return max(0, math.floor(self.p.risk_notional_per_leg / atr_val / price))

    def next(self):
        if not self.ready():
            return
        self.reset_day()
        tbar = len(self)
        try:
            bars_left = self.datas[0].buflen() - len(self)
        except Exception:
            bars_left = 999999

        # --- Initialize start value ---
        if self.start_value is None:
            try:
                self.start_value = float(self.broker.getvalue())
            except Exception:
                self.start_value = 1.0

        # --- Exit all positions a month before the end *if in profit* ---
        month_bars = 21  # or however many trading bars ≈ 1 month for your data
        cur_value = float(self.broker.getvalue())
        in_profit = cur_value > self.start_value
        if bars_left <= month_bars and in_profit:
            for d in self.datas:
                pos = self.getposition(d).size
                if pos > 0:
                    self.sell(data=d, size=pos)
                if pos < 0:
                    self.buy(data=d, size=abs(pos))
            return

        # --- Hard forced exit for last 'risk_free_bars' regardless of profit ---
        if bars_left <= int(self.p.risk_free_bars):
            for d in self.datas:
                pos = self.getposition(d).size
                if pos > 0:
                    self.sell(data=d, size=pos)
                if pos < 0:
                    self.buy(data=d, size=abs(pos))
            return

        # --- Main Entry Logic ---
        perf = {d: (float(d.close[0]) - float(d.close[-self.p.meanrev_window])) / float(d.close[-self.p.meanrev_window]) for d in self.datas}
        ranked = sorted(perf, key=perf.get)
        n = len(ranked)
        bottom_cut = max(1, int(self.p.top_pct * n))
        top_cut = max(1, int(self.p.top_pct * n))
        laggards = set(ranked[:bottom_cut])
        leaders = set(ranked[-top_cut:])

        for d in self.datas:
            state = self.state[d]
            pos = self.getposition(d).size
            if pos != 0 or tbar < state['cool_until']:
                continue

            close = float(d.close[0])
            prev_close = float(d.close[-1])
            openp = float(d.open[0])
            volume_z = (float(d.volume[0]) - float(self.vol_avg[d][0])) / (float(self.vol_std[d][0]) or 1)
            short_return = (close - float(d.close[-self.p.meanrev_window])) / float(d.close[-self.p.meanrev_window])
            long_return = (close - float(d.close[-30])) / float(d.close[-30])
            low_20 = min([float(d.low[-i]) for i in range(20)])
            bounce_from_low = (close - low_20) / low_20
            high_20 = max([float(d.high[-i]) for i in range(20)])
            drop_from_high = (high_20 - close) / high_20

            # LONGS: bottom X%
            if d in laggards:
                go_long = (
                    volume_z >= self.p.volume_z and
                    short_return <= self.p.short_return and
                    long_return >= self.p.long_return and
                    close > openp and
                    bounce_from_low >= self.p.bounce_pct
                )
                if go_long:
                    self._enter_trade(d, size_sign=1)
                    continue

            # SHORTS: top X%
            if d in leaders:
                go_short = (
                    volume_z >= self.p.volume_z and
                    short_return >= self.p.short_return_short and
                    long_return >= self.p.long_return_short and
                    close < openp and
                    drop_from_high >= self.p.drop_pct_short
                )
                if go_short:
                    self._enter_trade(d, size_sign=-1)
                    continue

        # --- Exits ---
        for d in self.datas:
            pos = self.getposition(d).size
            if pos == 0: continue
            state = self.state[d]
            new_trail = float(d.close[-1])
            if self.atr[d] is not None:
                new_trail = max(new_trail, float(d.close[-1]) - float(self.p.atr_k) * float(self.atr[d][0]))
            state['trail'] = max(state['trail'] if state['trail'] is not None else -math.inf, new_trail)
            if self.atr[d] is not None:
                atr_now = float(self.atr[d][0])
                buffer_mult = float(self.p.severe_breach_mult)
                breach = (float(d.close[0]) < float(state['trail']) - buffer_mult * atr_now
                          if pos > 0 else float(d.close[0]) > float(state['trail']) + buffer_mult * atr_now)
            else:
                breach = (float(d.close[0]) < float(state['trail']) if pos > 0 else float(d.close[0]) > float(state['trail']))
            state['breach_count'] = state['breach_count'] + 1 if breach else 0
            need_bars = 2 if self.p.two_bar_exit else 1
            if state['breach_count'] >= need_bars:
                self.close(data=d)
                state['cool_until'] = tbar + int(self.p.cooldown)
                state['breach_count'] = 0

    def _enter_trade(self, d, size_sign):
        price = float(d.close[0])
        entry_cap = int(self.p.entry_units_cap)
        atr_units = self.units_from_risk(d)
        units = min(entry_cap, atr_units)
        units = int(max(1, math.floor(units)))
        units *= size_sign

        cash_now = float(self.broker.getcash())
        port_val = float(self.broker.getvalue())
        invested_est = port_val - cash_now
        headroom_val = max(0.0, float(self.p.max_exposure_frac) * port_val - invested_est)
        units_cash = math.floor(max(0.0, cash_now * float(self.p.cash_frac) - float(self.p.cash_buffer)) / abs(price)) if price != 0 else 0
        units_headroom = math.floor(headroom_val / abs(price)) if price != 0 else 0

        units = max(-int(self.p.max_units_per_symbol), min(units, units_cash, units_headroom, int(self.p.max_units_per_symbol)))
        notional = abs(price * units)
        sym_spent = self.day_symbol_spent.get(d._name, 0.0)
        sym_cap = float(self.p.day_symbol_cap_notional) * port_val
        if sym_spent + notional > sym_cap:
            return
        if self.day_spent_notional + notional > port_val * self.p.day_budget_frac:
            return

        # Actually place the order:
        if units > 0:
            self.buy(data=d, size=units)
        else:
            self.sell(data=d, size=abs(units))
        self.day_spent_notional += notional
        self.day_symbol_spent[d._name] = sym_spent + notional
        state = self.state[d]
        trail = float(d.close[-1])
        if self.atr[d] is not None:
            trail = max(trail, float(d.close[-1]) - float(self.p.atr_k) * float(self.atr[d][0]))
        state['trail'] = trail
        state['breach_count'] = 0

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        d = order.data
        if order.status in [order.Completed]:
            price = order.executed.price
            size = order.executed.size
            prev_pos = self.prev_pos.get(d, 0.0)
            new_pos = self.getposition(d).size
            if order.isbuy() and prev_pos <= 0 and new_pos > 0:
                state = self.state[d]
                state['entry_px'] = float(price)
                state['entry_units'] = abs(float(size))
            self.prev_pos[d] = new_pos
