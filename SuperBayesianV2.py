# No tapered exits 
# Unlabelled trades

'''
Runs 4 BayesianTradingStrategy configurations in parallel, each with a different
bayesian_lookback (40, 60, 90, 120) and tuned parameters.

Meta-layer (BayesianModelSelector):
- Tracks rolling Sharpe for each config
- Uses a Dirichlet-Bayesian belief to allocate capital weights
- Starts equal (25% each), gradually concentrates into the best performer
- At end, data is consumed by main.py to print a ranked table

Run:
python main.py --strategy SuperBayesianV2 --data-dir DATA/Realstocks --output-dir output/SuperBayes2
'''

'''
{
  "final_value": 1180833.9771859995,
  "bankrupt": false,
  "bankrupt_date": null,
  "open_pnl_pd_ratio": 2.744452504789552,
  "true_pd_ratio": 2.720415116107024,
  "activity_pct": 91.8,
  "end_policy": "liquidate",
  "s_mult": 2.0
}
=== Trade Analyzer Stats ===
Total closed trades: 14
Wins: 7
Losses: 7

=== Trade Breakdown by Strategy Type ===

Mean Reversion Trades Long:
  Total trades: 0
  Wins: 0
  Losses: 0
  Total PnL: $0.00

Trend Following Trades Long:
  Total trades: 0
  Wins: 0
  Losses: 0
  Total PnL: $0.00

Mean Reversion Trades Short:
  Total trades: 0
  Wins: 0
  Losses: 0
  Total PnL: $0.00

Trend Following Trades Shorts:
  Total trades: 0
  Wins: 0
  Losses: 0
  Total PnL: $0.00

Other/Unlabeled Trades:
  Total trades: 14
  Wins: 7
  Losses: 7
  Total PnL: $186453.75
  Win Rate: 50.0%
'''

import backtrader as bt
import math
import numpy as np
from scipy import stats
from collections import defaultdict, deque

# ==============================================================================
# BAYESIAN REGIME DETECTOR (shared, lookback-aware)
# ==============================================================================
class BayesianRegimeDetector(object):
    def __init__(self, lookback=60, alpha=2.0, beta=2.0):
        self.lookback = lookback
        self.alpha = alpha
        self.beta = beta
        self.trend_observations = 0
        self.mr_observations = 0
        self.price_history = deque(maxlen=lookback)

    def update(self, close):
        self.price_history.append(close)
        mw = self.lookback // 3
        if len(self.price_history) >= mw:
            recent_change = (self.price_history[-1] - self.price_history[-mw]) / self.price_history[-mw]
            if abs(recent_change) > 0.02:
                self.trend_observations += 1
            else:
                self.mr_observations += 1

    def get_regime(self):
        mw = self.lookback // 3
        if len(self.price_history) < mw:
            return {'uptrend': 0.33, 'downtrend': 0.33, 'sideways': 0.34}

        recent_slice = list(self.price_history)[-mw:]
        early_slice = (
            list(self.price_history)[-(mw * 2):-mw]
            if len(self.price_history) >= mw * 2
            else recent_slice
        )

        recent_avg = np.mean(recent_slice)
        early_avg = np.mean(early_slice)
        total = self.trend_observations + self.mr_observations
        if total < 10:
            return {'uptrend': 0.33, 'downtrend': 0.33, 'sideways': 0.34}

        trendy_alpha = self.alpha + self.trend_observations
        mr_alpha = self.beta + self.mr_observations
        trendy_prob = trendy_alpha / (trendy_alpha + mr_alpha)

        if recent_avg > early_avg:
            uptrend_prob = trendy_prob * 0.7
            downtrend_prob = trendy_prob * 0.3
        else:
            uptrend_prob = trendy_prob * 0.3
            downtrend_prob = trendy_prob * 0.7

        return {
            'uptrend': uptrend_prob,
            'downtrend': downtrend_prob,
            'sideways': 1 - trendy_prob,
        }

# ==============================================================================
# BAYESIAN SIGNAL STRENGTH
# ==============================================================================
class BayesianSignalStrength(object):
    def __init__(self, prior_accuracy=0.50, window=50):
        self.prior_accuracy = prior_accuracy
        self.window = window
        self.success = 0
        self.total_trades = 0
        self.history = deque(maxlen=window)

    def update_signal(self, signal_fired, trade_won):
        if signal_fired:
            self.success += 1 if trade_won else 0
            self.total_trades += 1
            self.history.append(1.0 if trade_won else 0.0)

    def get_accuracy(self, alpha=1.0, beta=1.0):
        if self.total_trades == 0:
            return self.prior_accuracy
        alphapost = alpha + self.success
        betapost = beta + (self.total_trades - self.success)
        return alphapost / (alphapost + betapost)

# ==============================================================================
# META-LAYER: BAYESIAN MODEL SELECTOR
# ==============================================================================
class BayesianModelSelector(object):
    """
    Maintains a Dirichlet belief distribution over N model configs.
    Each config starts with equal weight (alpha=1).
    """

    def __init__(self, config_names, update_interval=20, sharpe_window=30):
        self.config_names = config_names
        self.n = len(config_names)
        self.update_interval = update_interval
        self.sharpe_window = sharpe_window

        self.dirichlet_alphas = {name: 1.0 for name in config_names}
        self.pnl_history = {name: deque(maxlen=sharpe_window) for name in config_names}
        self.weight_history = []   # (bar, {name: weight})
        self.bar_count = 0
        self.config_stats = {
            name: {
                'trades': 0,
                'wins': 0,
                'total_pnl': 0.0,
                'pnl_series': [],
                'capital_allocated': 0.0,
            }
            for name in config_names
        }

    def record_trade(self, config_name, pnl_pct, won):
        if config_name not in self.pnl_history:
            return
        self.pnl_history[config_name].append(pnl_pct)
        s = self.config_stats[config_name]
        s['trades'] += 1
        s['total_pnl'] += pnl_pct
        s['pnl_series'].append(pnl_pct)
        if won:
            s['wins'] += 1

    def record_capital(self, config_name, notional):
        self.config_stats[config_name]['capital_allocated'] += notional

    def _rolling_sharpe(self, config_name):
        hist = list(self.pnl_history[config_name])
        if len(hist) < 3:
            return 0.0
        arr = np.array(hist)
        std = np.std(arr)
        if std < 1e-9:
            return 0.1 if np.mean(arr) > 0 else 0.0
        return float(np.mean(arr) / std)

    def update_beliefs(self):
        sharpes = {name: max(0.0, self._rolling_sharpe(name)) for name in self.config_names}
        total_sharpe = sum(sharpes.values())
        if total_sharpe < 1e-9:
            return
        for name in self.config_names:
            boost = sharpes[name] / total_sharpe
            self.dirichlet_alphas[name] = max(
                0.1, self.dirichlet_alphas[name] + boost
            )

    def get_weights(self):
        total = sum(self.dirichlet_alphas.values())
        return {name: self.dirichlet_alphas[name] / total for name in self.config_names}

    def step(self, bar):
        self.bar_count += 1
        if self.bar_count % self.update_interval == 0:
            self.update_beliefs()
        weights = self.get_weights()
        self.weight_history.append((bar, dict(weights)))

    def final_sharpe(self, config_name):
        series = self.config_stats[config_name]['pnl_series']
        if len(series) < 2:
            return float('nan')
        arr = np.array(series)
        std = np.std(arr)
        if std < 1e-9:
            return float('nan')
        return float(np.mean(arr) / std)

    def get_report_data(self):
        """
        Build structured report used by main.py
        """
        weights = self.get_weights()
        rows = []
        for name in self.config_names:
            s = self.config_stats[name]
            trades = s['trades']
            wins = s['wins']
            losses = trades - wins
            win_rate = (wins / trades * 100.0) if trades > 0 else 0.0
            total_pnl = s['total_pnl']
            sharpe = self.final_sharpe(name)
            avg_pnl = (total_pnl / trades) if trades > 0 else 0.0
            final_weight = weights[name] * 100.0
            alpha = self.dirichlet_alphas[name]

            rows.append({
                'name': name,
                'trades': trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl_pct': total_pnl * 100.0,
                'avg_pnl_pct': avg_pnl * 100.0,
                'sharpe': sharpe,
                'final_weight': final_weight,
                'dirichlet_alpha': alpha,
                'capital_allocated': s['capital_allocated'],
            })

        import math as _math
        rows.sort(
            key=lambda r: r['sharpe'] if not _math.isnan(r['sharpe']) else -999,
            reverse=True,
        )

        return {
            'rows': rows,
            'weight_history': list(self.weight_history),
            'config_names': list(self.config_names),
        }

    # You can keep print_final_report for debugging if you like, but main.py uses get_report_data.


# ==============================================================================
# CONFIG DEFINITIONS (unchanged)
# ==============================================================================
CONFIGS = {
    'LB40': dict(
        atr_len=10,
        atr_k=2.1,
        meanrev_window=5,
        top_pct=0.20,
        volume_z=0.8,
        short_return=-0.015,
        long_return=0.0,
        bounce_pct=0.015,
        up_window=5,
        up_days_min=3,
        breakout_lookback=6,
        ma_len=15,
        vol_z_min=0.6,
        cash_frac=0.55,
        cash_buffer=100000.0,
        max_exposure_frac=0.80,
        day_budget_frac=0.30,
        day_symbol_cap_notional=0.40,
        max_units_per_symbol=4500,
        entry_units_cap=700,
        risk_notional_per_leg=60000.0,
        cooldown=3,
        two_bar_exit=True,
        risk_free_bars=7,
        end_taper_bars=14,
        severe_breach_mult=0.60,
        bayesian_lookback=40,
        regime_learning_enabled=True,
        signal_learning_enabled=True,
        min_confidence_mr=0.44,
        min_confidence_trend=0.44,
        sideways_threshold=0.35,
        trending_threshold=0.55,
        printlog=False,
    ),
    'LB60': dict(
        atr_len=14,
        atr_k=2.3,
        meanrev_window=5,
        top_pct=0.10,
        volume_z=1.0,
        short_return=-0.03,
        long_return=0.0,
        bounce_pct=0.015,
        up_window=5,
        up_days_min=3,
        breakout_lookback=7,
        ma_len=20,
        vol_z_min=0.8,
        cash_frac=0.55,
        cash_buffer=100000.0,
        max_exposure_frac=0.80,
        day_budget_frac=0.30,
        day_symbol_cap_notional=0.40,
        max_units_per_symbol=4500,
        entry_units_cap=700,
        risk_notional_per_leg=60000.0,
        cooldown=4,
        two_bar_exit=True,
        risk_free_bars=7,
        end_taper_bars=14,
        severe_breach_mult=0.60,
        bayesian_lookback=60,
        regime_learning_enabled=True,
        signal_learning_enabled=True,
        min_confidence_mr=0.42,
        min_confidence_trend=0.42,
        sideways_threshold=0.30,
        trending_threshold=0.50,
        printlog=False,
    ),
    'LB90': dict(
        atr_len=21,
        atr_k=2.7,
        meanrev_window=7,
        top_pct=0.25,
        volume_z=0.7,
        short_return=-0.018,
        long_return=0.0,
        bounce_pct=0.018,
        up_window=7,
        up_days_min=4,
        breakout_lookback=14,
        ma_len=30,
        vol_z_min=1.0,
        cash_frac=0.55,
        cash_buffer=100000.0,
        max_exposure_frac=0.80,
        day_budget_frac=0.30,
        day_symbol_cap_notional=0.40,
        max_units_per_symbol=4500,
        entry_units_cap=700,
        risk_notional_per_leg=60000.0,
        cooldown=5,
        two_bar_exit=True,
        risk_free_bars=10,
        end_taper_bars=21,
        severe_breach_mult=0.60,
        bayesian_lookback=90,
        regime_learning_enabled=True,
        signal_learning_enabled=True,
        min_confidence_mr=0.40,
        min_confidence_trend=0.40,
        sideways_threshold=0.27,
        trending_threshold=0.45,
        printlog=False,
    ),
    'LB120': dict(
        atr_len=30,
        atr_k=3.0,
        meanrev_window=30,
        top_pct=0.10,
        volume_z=0.9,
        short_return=-0.030,
        long_return=0.0,
        bounce_pct=0.025,
        up_window=20,
        up_days_min=13,
        breakout_lookback=30,
        ma_len=75,
        vol_z_min=0.6,
        cash_frac=0.55,
        cash_buffer=100000.0,
        max_exposure_frac=0.80,
        day_budget_frac=0.30,
        day_symbol_cap_notional=0.40,
        max_units_per_symbol=4500,
        entry_units_cap=700,
        risk_notional_per_leg=60000.0,
        cooldown=10,
        two_bar_exit=True,
        risk_free_bars=14,
        end_taper_bars=30,
        severe_breach_mult=0.60,
        bayesian_lookback=120,
        regime_learning_enabled=True,
        signal_learning_enabled=True,
        min_confidence_mr=0.38,
        min_confidence_trend=0.38,
        sideways_threshold=0.25,
        trending_threshold=0.40,
        printlog=False,
    ),
}

# ==============================================================================
# CHILD STRATEGY
# ==============================================================================
class ChildStrategy(object):
    def __init__(self, name, cfg, datas, broker, indicators):
        self.name = name
        self.cfg = cfg
        self.datas = datas
        self.broker = broker
        self.lb = cfg['bayesian_lookback']
        self.mw = self.lb // 3

        self.atr = indicators['atr']
        self.vol_avg = indicators['vol_avg']
        self.vol_std = indicators['vol_std']
        self.sma = indicators['sma']
        self.lowest = indicators['lowest']

        self.trend_filters = {d: BayesianRegimeDetector(lookback=self.lb) for d in datas}
        self.mr_signal_strength = {d: BayesianSignalStrength(prior_accuracy=0.48) for d in datas}
        self.tf_signal_strength = {d: BayesianSignalStrength(prior_accuracy=0.48) for d in datas}

        self.state = {
            d: dict(
                trail=None,
                cool_until=-math.inf,
                breach_count=0,
                entry_px=None,
                entry_units=0,
                entry_type='',
                entry_bar=0,
                regime_history=deque(maxlen=self.mw),
            )
            for d in datas
        }

        self.positions = {}
        self.day_spent_notional = 0.0
        self.day_symbol_spent = {}
        self.last_calendar_date = None
        self.closed_trades = []

    def ready(self, bar_count):
        return bar_count >= int(self.lb * 0.67)

    def reset_day(self, cur_date):
        if self.last_calendar_date != cur_date:
            self.day_spent_notional = 0.0
            self.day_symbol_spent = {}
            self.last_calendar_date = cur_date

    def units_from_risk(self, d):
        atr_val = float(self.atr[d][0]) if self.atr[d] is not None else 0.0
        price = float(d.close[0])
        if atr_val <= 0 or price <= 0:
            return 0
        return max(0, math.floor(self.cfg['risk_notional_per_leg'] / atr_val / price))

    # --- signal checkers (MR/TF long/short) remain identical to your original code ---
    def check_mr_long(self, d):
        close = float(d.close[0])
        openp = float(d.open[0])
        vol_z = (float(d.volume[0]) - float(self.vol_avg[d][0])) / (float(self.vol_std[d][0]) or 1.0)
        short_ret = (
            close - float(d.close[-self.cfg['meanrev_window']])
        ) / float(d.close[-self.cfg['meanrev_window']])
        low_n = min(float(d.low[-i]) for i in range(1, self.mw + 1))
        dist_from_low = (close - low_n) / low_n if low_n > 0 else 0
        signals = {
            "volume_spike": vol_z >= self.cfg['volume_z'],
            "short_return": short_ret <= self.cfg['short_return'],
            "green_candle": close > openp,
            "near_low": dist_from_low <= self.cfg['bounce_pct'],
        }
        return sum(signals.values()) >= 3, signals

    def check_mr_short(self, d):
        close = float(d.close[0])
        openp = float(d.open[0])
        vol_z = (float(d.volume[0]) - float(self.vol_avg[d][0])) / (float(self.vol_std[d][0]) or 1.0)
        ret_n = (close - float(d.close[-self.cfg['meanrev_window']])) / float(d.close[-self.cfg['meanrev_window']])
        high_n = max(float(d.high[-i]) for i in range(1, self.mw + 1))
        dist_from_high = (high_n - close) / high_n if high_n > 0 else 0
        signals = {
            "volume_climactic": vol_z >= 1.0,
            "overbought": ret_n >= 0.04,
            "red_candle": close < openp,
            "near_high": dist_from_high <= 0.02,
        }
        return sum(signals.values()) >= 3, signals

    def check_tf_long(self, d):
        close = float(d.close[0])
        ups = sum(
            1
            for k in range(1, self.cfg['up_window'] + 1)
            if float(d.close[0]) > float(d.close[-k])
        )
        up_ok = ups >= self.cfg['up_days_min']
        v, mu = float(d.volume[0]), float(self.vol_avg[d][0])
        sd = float(self.vol_std[d][0]) if float(self.vol_std[d][0]) != 0 else 1.0
        vol_ok = (v - mu) / sd >= self.cfg['vol_z_min']
        high_n = max(float(d.high[-k]) for k in range(1, self.cfg['breakout_lookback'] + 1))
        brk_ok = close >= high_n
        sma = self.sma[d]
        trend_ok = True
        if sma is not None:
            s0, s1, s2 = float(sma[0]), float(sma[-1]), float(sma[-2])
            trend_ok = close > s0 and (s0 > s1 > s2)
        signals = {"ups": up_ok, "volume": vol_ok, "breakout_up": brk_ok, "sma_trend_up": trend_ok}
        return sum(signals.values()) >= 3, signals

    def check_tf_short(self, d):
        close = float(d.close[0])
        downs = sum(
            1
            for k in range(1, self.cfg['up_window'] + 1)
            if float(d.close[0]) < float(d.close[-k])
        )
        down_ok = downs >= self.cfg['up_days_min']
        v, mu = float(d.volume[0]), float(self.vol_avg[d][0])
        sd = float(self.vol_std[d][0]) if float(self.vol_std[d][0]) != 0 else 1.0
        vol_ok = (v - mu) / sd >= self.cfg['vol_z_min']
        low_n = min(float(d.low[-k]) for k in range(1, self.cfg['breakout_lookback'] + 1))
        brk_ok = close <= low_n
        sma = self.sma[d]
        trend_ok = True
        if sma is not None:
            s0, s1, s2 = float(sma[0]), float(sma[-1]), float(sma[-2])
            trend_ok = close < s0 and (s0 < s1 < s2)
        signals = {"downs": down_ok, "volume": vol_ok, "breakdown": brk_ok, "sma_trend_down": trend_ok}
        return sum(signals.values()) >= 3, signals

    # ------------------------------------------------------------------
    # Virtual trade execution
    # ------------------------------------------------------------------
    def get_trade_intent(self, d, capital_weight, bars_left, tbar, perf, laggards):
        state = self.state[d]
        pos = self.positions.get(d._name, {}).get('size', 0)
        if pos != 0 or tbar < state['cool_until']:
            return None

        self.trend_filters[d].update(float(d.close[0]))
        regime = self.trend_filters[d].get_regime()
        state['regime_history'].append(regime)

        price = float(d.close[0])
        size_sign = 0
        entry_type = ''

        # Mean reversion regime
        if regime['sideways'] >= self.cfg['sideways_threshold']:
            if d in laggards:
                ok, _ = self.check_mr_long(d)
                if ok:
                    size_sign = 1
                    entry_type = 'mean_reversion_long'
            if size_sign == 0 and regime['downtrend'] > regime['uptrend']:
                if perf.get(d, 0) >= 0.05:
                    ok, _ = self.check_mr_short(d)
                    if ok:
                        size_sign = -1
                        entry_type = 'mean_reversion_short'

        # Trend regime
        if size_sign == 0 and (regime['uptrend'] + regime['downtrend']) >= self.cfg['trending_threshold']:
            if regime['downtrend'] > regime['uptrend']:
                ok, _ = self.check_tf_short(d)
                if ok:
                    size_sign = -1
                    entry_type = 'trend_following_short'
            else:
                ok, _ = self.check_tf_long(d)
                if ok:
                    size_sign = 1
                    entry_type = 'trend_following_long'

        if size_sign == 0:
            return None

        atr_units = self.units_from_risk(d)
        units = min(self.cfg['entry_units_cap'], atr_units)
        units = int(max(1, math.floor(units * capital_weight)))
        units = max(1, min(units, self.cfg['max_units_per_symbol']))
        units *= size_sign

        notional = abs(price * units)
        return {
            'action': 'buy' if size_sign > 0 else 'sell',
            'size': abs(units),
            'size_sign': size_sign,
            'entry_type': entry_type,
            'config': self.name,
            'data': d,
            'notional': notional,
            'price': price,
        }

    def record_entry(self, d, size_sign, entry_type, price):
        atr_val = float(self.atr[d][0]) if self.atr[d] is not None else 0.0
        if size_sign > 0:
            trail = max(price, price - self.cfg['atr_k'] * atr_val)
        else:
            trail = min(price, price + self.cfg['atr_k'] * atr_val)

        self.state[d]['trail'] = trail
        self.state[d]['breach_count'] = 0
        self.state[d]['entry_type'] = entry_type
        self.state[d]['entry_px'] = price

        self.positions[d._name] = {
            'size': size_sign,
            'entry_px': price,
            'entry_type': entry_type,
        }

    def check_exit(self, d, tbar):
        pos_info = self.positions.get(d._name)
        if pos_info is None:
            return False

        state = self.state[d]
        close = float(d.close[0])
        trail = state.get('trail')
        if trail is None:
            return False

        side = pos_info['size']
        atr_val = float(self.atr[d][0]) if self.atr[d] is not None else 0.0
        severe_k = self.cfg['severe_breach_mult'] * atr_val

        breached = (side > 0 and close < trail) or (side < 0 and close > trail)
        severe = (side > 0 and close < trail - severe_k) or (side < 0 and close > trail + severe_k)

        if severe:
            return True

        if breached:
            state['breach_count'] = state.get('breach_count', 0) + 1
            if self.cfg['two_bar_exit'] and state['breach_count'] >= 2:
                return True
            elif not self.cfg['two_bar_exit']:
                return True
        else:
            state['breach_count'] = 0

        if side > 0:
            state['trail'] = max(trail, close - self.cfg['atr_k'] * atr_val)
        else:
            state['trail'] = min(trail, close + self.cfg['atr_k'] * atr_val)

        return False

    def record_exit(self, d, exit_price, selector):
        pos_info = self.positions.pop(d._name, None)
        if pos_info is None:
            return

        entry_px = pos_info['entry_px']
        side = pos_info['size']

        if entry_px and entry_px > 0:
            pnl_pct = side * (exit_price - entry_px) / entry_px
            won = pnl_pct > 0
            selector.record_trade(self.name, pnl_pct, won)

        self.state[d]['cool_until'] = float('inf')
        self.state[d]['trail'] = None


# ==============================================================================
# SUPER STRATEGY
# ==============================================================================
class SuperBayesian(bt.Strategy):
    """
    Runs 4 ChildStrategies (LB40, LB60, LB90, LB120) in parallel.
    """

    params = dict(
        selector_update_interval=20,
        sharpe_window=30,
        global_max_exposure=0.80,
        global_cash_buffer=100000.0,
        risk_free_bars=7,
        printlog=False,
    )

    def __init__(self):
        config_names = list(CONFIGS.keys())
        self.selector = BayesianModelSelector(
            config_names=config_names,
            update_interval=self.p.selector_update_interval,
            sharpe_window=self.p.sharpe_window,
        )

        # Indicators per config
        self._indicator_sets = {}
        for name, cfg in CONFIGS.items():
            lb = cfg['bayesian_lookback']
            mw = lb // 3
            self._indicator_sets[name] = {
                'atr': {d: bt.indicators.ATR(d, period=cfg['atr_len']) for d in self.datas},
                'vol_avg': {d: bt.indicators.SimpleMovingAverage(d.volume, period=mw) for d in self.datas},
                'vol_std': {d: bt.indicators.StandardDeviation(d.volume, period=mw) for d in self.datas},
                'sma': {d: bt.indicators.SimpleMovingAverage(d.close, period=cfg['ma_len']) for d in self.datas},
                'lowest': {d: bt.indicators.Lowest(d.close, period=mw) for d in self.datas},
            }

        # Children (logic / exits / internal state)
        self.children = {}
        for name, cfg in CONFIGS.items():
            self.children[name] = ChildStrategy(
                name=name,
                cfg=cfg,
                datas=self.datas,
                broker=self.broker,
                indicators=self._indicator_sets[name],
            )

        # Bookkeeping for orders and open positions
        self.pending_orders = {}      # ref -> (config_name, data, size_sign, entry_type)
        self.open_positions = {}      # (config_name, data._name) -> dict

        self.start_value = None
        self.last_calendar_date = None
        self.day_spent_notional = 0.0
        self.day_symbol_spent = {}

        # *** REPORTING-LEVEL STATE (for TradeTypeAnalyzer) ***
        # One entry per data feed, with an entry_type field which will be
        # read by main.py's TradeTypeAnalyzer when a trade closes.
        self.state = {
            d: dict(entry_type='')
            for d in self.datas
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def reset_day(self):
        cur = self.datas[0].datetime.date(0)
        if self.last_calendar_date != cur:
            self.day_spent_notional = 0.0
            self.day_symbol_spent = {}
            self.last_calendar_date = cur

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def next(self):
        bar = len(self)
        self.reset_day()

        if self.start_value is None:
            try:
                self.start_value = float(self.broker.getvalue())
            except Exception:
                self.start_value = 1.0

        # Bars remaining
        try:
            bars_left = self.datas[0].buflen() - bar
        except Exception:
            bars_left = 999_999

        # Final liquidation window
        if bars_left <= self.p.risk_free_bars:
            for d in self.datas:
                pos = self.getposition(d).size
                if pos > 0:
                    self.sell(data=d, size=pos)
                elif pos < 0:
                    self.buy(data=d, size=abs(pos))
            return

        # Update selector and get per-config capital weights
        self.selector.step(bar)
        weights = self.selector.get_weights()

        # Market-wide performance for laggard filter (use LB60 meanrev_window)
        ref_window = CONFIGS['LB60']['meanrev_window']
        try:
            perf = {
                d: (float(d.close[0]) - float(d.close[-ref_window])) / float(d.close[-ref_window])
                for d in self.datas
            }
        except Exception:
            perf = {d: 0.0 for d in self.datas}

        ranked = sorted(perf, key=perf.get)
        n = len(ranked)
        bottom_cut = max(1, int(CONFIGS['LB60']['top_pct'] * n))
        laggards = set(ranked[:bottom_cut])

        port_val = float(self.broker.getvalue())
        cash_now = float(self.broker.getcash())
        invested = port_val - cash_now
        global_headroom = max(0.0, self.p.global_max_exposure * port_val - invested)

        # Loop over child configs and data feeds
        for name, child in self.children.items():
            if not child.ready(bar):
                continue

            weight = weights[name]
            cfg = child.cfg

            for d in self.datas:
                # ----- EXIT LOGIC -----
                if d._name in child.positions:
                    wants_exit = child.check_exit(d, bar)
                    if wants_exit:
                        pos_key = (name, d._name)
                        if pos_key in self.open_positions:
                            pos = self.getposition(d).size
                            if pos > 0:
                                o = self.sell(data=d, size=pos)
                            elif pos < 0:
                                o = self.buy(data=d, size=abs(pos))
                            else:
                                o = None
                        else:
                            o = None

                        if o:
                            # Exit order: tag with 'exit'
                            self.pending_orders[o.ref] = (name, d, 0, 'exit')
                            exit_price = float(d.close[0])
                            child.record_exit(d, exit_price, self.selector)
                            child.state[d]['cool_until'] = bar + cfg['cooldown']
                            self.open_positions.pop(pos_key, None)
                        continue

                # ----- ENTRY LOGIC -----
                intent = child.get_trade_intent(d, weight, bars_left, bar, perf, laggards)
                if intent is None:
                    continue

                notional = intent['notional']
                sym_spent = self.day_symbol_spent.get(d._name, 0.0)
                sym_cap = cfg['day_symbol_cap_notional'] * port_val

                if (
                    sym_spent + notional > sym_cap
                    or self.day_spent_notional + notional > port_val * cfg['day_budget_frac']
                    or notional > global_headroom
                    or cash_now < notional + self.p.global_cash_buffer
                ):
                    continue

                size = intent['size']
                if intent['action'] == 'buy':
                    o = self.buy(data=d, size=size)
                else:
                    o = self.sell(data=d, size=size)

                if o:
                    # Store all info needed to reconstruct entry at fill time
                    self.pending_orders[o.ref] = (
                        name,
                        d,
                        intent['size_sign'],
                        intent['entry_type'],
                    )
                    self.selector.record_capital(name, notional)
                    self.day_spent_notional += notional
                    self.day_symbol_spent[d._name] = sym_spent + notional
                    global_headroom -= notional
                    cash_now -= notional

    # ------------------------------------------------------------------
    # Order callback – STRATEGY-LEVEL REPORTING MOVED HERE
    # ------------------------------------------------------------------
    def notify_order(self, order):
        # Ignore initial states
        if order.status in [order.Submitted, order.Accepted]:
            return

        ref = order.ref
        if ref not in self.pending_orders:
            return

        name, d, size_sign, entry_type = self.pending_orders.pop(ref)

        # ----- EXIT ORDERS -----
        if entry_type == 'exit':
            # Only clear label once the exit actually completes
            if order.status == order.Completed and d in self.state:
                self.state[d]['entry_type'] = ''
            return

        # ----- ENTRY ORDERS -----
        if order.status != order.Completed or size_sign == 0:
            # Rejected / canceled / margin failed – nothing opened
            return

        price = float(order.executed.price)
        child = self.children[name]

        # Let the child maintain its own internal state for logic / exits
        child.record_entry(d, size_sign, entry_type, price)

        # Track open position per (config, symbol)
        self.open_positions[(name, d._name)] = {
            'size': size_sign,
            'entry_px': price,
            'entry_type': entry_type,
        }

        # *** KEY LINE FOR REPORTING ***
        # Tag the main strategy's state for this data feed. TradeTypeAnalyzer
        # in main.py will read this at trade close time.
        if d in self.state:
            self.state[d]['entry_type'] = entry_type

    # ------------------------------------------------------------------
    # Trade callback (optional; you can keep it minimal)
    # ------------------------------------------------------------------
    def notify_trade(self, trade):
        # You don't need to tag trade._entry_type anymore for your analyzer,
        # because it reads self.strategy.state[data]['entry_type'].
        # You can leave this empty or use it for debugging/logging.
        pass

    def stop(self):
        """
        End-of-backtest hook.
        Reporting is handled externally in main.py via analyzers.
        """
        pass
