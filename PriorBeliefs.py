# Fixes applied:
#   1. Trailing stop initialisation corrected for shorts (was clamping trail to entry price)
#   2. Cooldown enforcement added before re-entry (was missing, blocking re-entries silently)
#   3. Budget/label writes moved outside `if o:` block (fixes buffered-order brokers)
#   4. Cooldown bar recorded on exit so re-entry timer is accurate

'''
FIXED SHORTING LOGIC - More balanced than One_Super and doesnt return massive losses in Bear Market like One_Super.
However worse performance on PART1, PART2 and Combined PART1 and PART2 compared to One_Super.

Holding Period Still Extends to multiple Years.

Added max_hold_bars to exit conditions

Dirichlet Weighting still over four lookbacks and continously shifts allocation toward the lookback which has the best 
rolling Sharpe Ratio with a min %age for all strategies
'''

'''
python main.py --strategy PriorBeliefs --data-dir DATA/Set3 --output-dir output/PRIOR
'''

'''
{
  "final_value": 1125835.784,
  "bankrupt": false,
  "bankrupt_date": null,
  "open_pnl_pd_ratio": 1.6457531046478708,
  "true_pd_ratio": 1.5979822892828346,
  "activity_pct": 91.8,
  "end_policy": "liquidate",
  "s_mult": 2.0
}
=== Trade Analyzer Stats ===
Total closed trades: 8
Wins: 3
Losses: 5

=== Trade Breakdown by Strategy Type ===

Mean Reversion Trades Long:
  Total trades: 2
  Wins: 2
  Losses: 0
  Total PnL: $27722.59
  Win Rate: 100.0%

Trend Following Trades Long:
  Total trades: 5
  Wins: 1
  Losses: 4
  Total PnL: $102922.00
  Win Rate: 20.0%

Mean Reversion Trades Short:
  Total trades: 0
  Wins: 0
  Losses: 0
  Total PnL: $0.00

Trend Following Trades Shorts:
  Total trades: 1
  Wins: 0
  Losses: 1
  Total PnL: $-453.67
  Win Rate: 0.0%

--- Portfolio Total ---
  Total positions : 8
  Total wins      : 3
  Overall PnL     : $130,190.92
  Overall win rate: 37.5%

Symbol     Type                     EntryDate    ExitDate              PnL  W/L
--------------------------------------------------------------------------------------------------------------
series_6   trend_following_long     2070-02-21   2072-08-27        -49.00  LOSS
series_3   trend_following_short    2070-02-22   2072-08-27       -453.67  LOSS
series_4   mean_reversion_long      2070-02-22   2072-08-27     20,307.00  WIN
series_1   trend_following_long     2070-02-23   2072-08-27    -20,230.00  LOSS
series_9   mean_reversion_long      2070-02-23   2072-08-27      7,415.59  WIN
series_2   trend_following_long     2070-02-24   2072-08-27    -15,257.25  LOSS
series_5   trend_following_long     2070-02-24   2072-08-27    146,475.00  WIN
series_10  trend_following_long     2070-03-07   2072-08-27     -8,016.75  LOSS
'''

import backtrader as bt
import math
import numpy as np
from collections import defaultdict, deque

class BayesianRegimeDetector:
    def __init__(self, lookback=60, alpha=2.0, beta=2.0):
        self.lookback   = lookback
        self.alpha      = alpha
        self.beta       = beta
        self.prices     = deque(maxlen=lookback)
        # Rolling window of obs — only the last `lookback` bars count.
        # A running total would accumulate 1000+ obs and swamp the prior.
        self._obs       = deque(maxlen=lookback)   # each entry: 1=trend, 0=mr

    @property
    def trend_obs(self):
        return sum(self._obs)

    @property
    def mr_obs(self):
        return len(self._obs) - sum(self._obs)

    def update(self, close):
        self.prices.append(close)
        mw = self.lookback // 3
        if len(self.prices) >= mw:
            change = (self.prices[-1] - self.prices[-mw]) / self.prices[-mw]
            self._obs.append(1 if abs(change) > 0.02 else 0)

    def extract_state(self):
        # Transfer the last `lookback` observations as a compact list.
        # On load, trend_obs/mr_obs are derived from this so they reflect
        # only the recent window — not 1000 accumulated bars.
        return {'obs': list(self._obs)}

    def load_state(self, state):
        self._obs = deque(state['obs'], maxlen=self.lookback)
        # prices left empty — refills naturally from PART 2 bars

    def get_regime(self):
        mw = self.lookback // 3
        if len(self.prices) < mw:
            return {'uptrend': 0.33, 'downtrend': 0.33, 'sideways': 0.34}
        recent = list(self.prices)[-mw:]
        early  = list(self.prices)[-(mw * 2):-mw] if len(self.prices) >= mw * 2 else recent
        r_avg  = np.mean(recent)
        e_avg  = np.mean(early)
        total  = self.trend_obs + self.mr_obs
        if total < 10:
            return {'uptrend': 0.33, 'downtrend': 0.33, 'sideways': 0.34}
        t_alpha    = self.alpha + self.trend_obs
        m_alpha    = self.beta  + self.mr_obs
        trend_prob = t_alpha / (t_alpha + m_alpha)
        if r_avg > e_avg:
            up_p, dn_p = trend_prob * 0.7, trend_prob * 0.3
        else:
            up_p, dn_p = trend_prob * 0.3, trend_prob * 0.7
        return {'uptrend': up_p, 'downtrend': dn_p, 'sideways': 1 - trend_prob}

class BayesianSignalStrength:
    def __init__(self, prior=0.50, window=50):
        self.prior   = prior
        self.window  = window
        self.success = 0
        self.total   = 0
        self.history = deque(maxlen=window)

    def update(self, fired, won):
        if fired:
            self.success += 1 if won else 0
            self.total   += 1
            self.history.append(1.0 if won else 0.0)

    def extract_state(self):
        return {
            'success': self.success,
            'total':   self.total,
            'history': list(self.history),
        }

    def load_state(self, state):
        self.success = state['success']
        self.total   = state['total']
        self.history = deque(state['history'], maxlen=self.window)

    def accuracy(self, alpha=1.0, beta=1.0):
        if self.total == 0:
            return self.prior
        a = alpha + self.success
        b = beta  + (self.total - self.success)
        return a / (a + b)

class BayesianModelSelector:
    def __init__(self, config_names, update_interval=20, sharpe_window=30):
        self.config_names     = config_names
        self.update_interval  = update_interval
        self.sharpe_window    = sharpe_window
        self.alphas           = {n: 1.0 for n in config_names}
        self.pnl_history      = {n: deque(maxlen=sharpe_window) for n in config_names}
        self.weight_history   = []
        self.bar_count        = 0
        self.stats            = {
            n: {'trades': 0, 'wins': 0, 'total_pnl': 0.0, 'pnl_series': [], 'capital': 0.0}
            for n in config_names
        }

    def extract_state(self):
        return {
            'alphas':      dict(self.alphas),
            'pnl_history': {n: list(v) for n, v in self.pnl_history.items()},
            'bar_count':   self.bar_count,
            'stats': {
                n: {
                    **s,
                    'pnl_series': list(s['pnl_series']),
                }
                for n, s in self.stats.items()
            },
        }

    def load_state(self, state):
        self.alphas    = state['alphas']
        self.bar_count = state['bar_count']
        for n, hist in state['pnl_history'].items():
            if n in self.pnl_history:
                self.pnl_history[n] = deque(hist, maxlen=self.sharpe_window)
        for n, s in state['stats'].items():
            if n in self.stats:
                self.stats[n] = {**s, 'pnl_series': list(s['pnl_series'])}

    def record_trade(self, name, pnl_pct, won):
        if name not in self.pnl_history:
            return
        self.pnl_history[name].append(pnl_pct)
        s = self.stats[name]
        s['trades']    += 1
        s['total_pnl'] += pnl_pct
        s['pnl_series'].append(pnl_pct)
        if won:
            s['wins'] += 1

    def record_capital(self, name, notional):
        self.stats[name]['capital'] += notional

    def _sharpe(self, name):
        hist = list(self.pnl_history[name])
        if len(hist) < 3:
            return 0.0
        arr = np.array(hist)
        std = np.std(arr)
        if std < 1e-9:
            return 0.1 if np.mean(arr) > 0 else 0.0
        return float(np.mean(arr) / std)

    def update_beliefs(self):
        sharpes = {n: max(0.0, self._sharpe(n)) for n in self.config_names}
        total   = sum(sharpes.values())
        if total < 1e-9:
            return
        for n in self.config_names:
            self.alphas[n] = max(0.1, self.alphas[n] + sharpes[n] / total)

    def get_weights(self):
        total = sum(self.alphas.values())
        return {n: self.alphas[n] / total for n in self.config_names}

    def step(self, bar):
        self.bar_count += 1
        if self.bar_count % self.update_interval == 0:
            self.update_beliefs()
        self.weight_history.append((bar, dict(self.get_weights())))

CONFIGS = {
    'LB40': dict(
        atr_len=10, atr_k=2.1, meanrev_window=5, top_pct=0.20,
        volume_z=0.8, short_return=-0.015, long_return=0.0,
        bounce_pct=0.015, up_window=5, up_days_min=3,
        breakout_lookback=6, ma_len=15, vol_z_min=0.6,
        cash_frac=0.55, cash_buffer=100000.0, max_exposure_frac=0.80,
        day_budget_frac=0.30, day_symbol_cap_notional=0.40,
        max_units_per_symbol=4500, entry_units_cap=700,
        risk_notional_per_leg=60000.0, cooldown=3,
        risk_free_bars=7, severe_breach_mult=0.60,
        bayesian_lookback=40, regime_learning_enabled=True,
        signal_learning_enabled=True, min_confidence_mr=0.44,
        min_confidence_trend=0.44, sideways_threshold=0.35,
        trending_threshold=0.55, max_hold_bars=60,
    ),
    'LB60': dict(
        atr_len=14, atr_k=2.3, meanrev_window=5, top_pct=0.10,
        volume_z=1.0, short_return=-0.03, long_return=0.0,
        bounce_pct=0.015, up_window=5, up_days_min=3,
        breakout_lookback=7, ma_len=20, vol_z_min=0.8,
        cash_frac=0.55, cash_buffer=100000.0, max_exposure_frac=0.80,
        day_budget_frac=0.30, day_symbol_cap_notional=0.40,
        max_units_per_symbol=4500, entry_units_cap=700,
        risk_notional_per_leg=60000.0, cooldown=4,
        risk_free_bars=7, severe_breach_mult=0.60,
        bayesian_lookback=60, regime_learning_enabled=True,
        signal_learning_enabled=True, min_confidence_mr=0.42,
        min_confidence_trend=0.42, sideways_threshold=0.30,
        trending_threshold=0.50, max_hold_bars=90,
    ),
    'LB90': dict(
        atr_len=21, atr_k=2.7, meanrev_window=7, top_pct=0.25,
        volume_z=0.7, short_return=-0.018, long_return=0.0,
        bounce_pct=0.018, up_window=7, up_days_min=4,
        breakout_lookback=14, ma_len=30, vol_z_min=1.0,
        cash_frac=0.55, cash_buffer=100000.0, max_exposure_frac=0.80,
        day_budget_frac=0.30, day_symbol_cap_notional=0.40,
        max_units_per_symbol=4500, entry_units_cap=700,
        risk_notional_per_leg=60000.0, cooldown=5,
        risk_free_bars=10, severe_breach_mult=0.60,
        bayesian_lookback=90, regime_learning_enabled=True,
        signal_learning_enabled=True, min_confidence_mr=0.40,
        min_confidence_trend=0.40, sideways_threshold=0.27,
        trending_threshold=0.45, max_hold_bars=120,
    ),
    'LB120': dict(
        atr_len=30, atr_k=3.0, meanrev_window=30, top_pct=0.10,
        volume_z=0.9, short_return=-0.030, long_return=0.0,
        bounce_pct=0.025, up_window=20, up_days_min=13,
        breakout_lookback=30, ma_len=75, vol_z_min=0.6,
        cash_frac=0.55, cash_buffer=100000.0, max_exposure_frac=0.80,
        day_budget_frac=0.30, day_symbol_cap_notional=0.40,
        max_units_per_symbol=4500, entry_units_cap=700,
        risk_notional_per_leg=60000.0, cooldown=10,
        risk_free_bars=14, severe_breach_mult=0.60,
        bayesian_lookback=120, regime_learning_enabled=True,
        signal_learning_enabled=True, min_confidence_mr=0.38,
        min_confidence_trend=0.38, sideways_threshold=0.25,
        trending_threshold=0.40, max_hold_bars=180,
    ),
}

# ---------------------------------------------------------------------------
# PART 1 → PART 2 state transfer
#
# HOW TO USE:
#   1. Run PART 1 normally with PART1_STATE = {}  (empty dict = cold start)
#   2. At the end of the run, the strategy prints a ready-to-paste block to
#      stdout that looks like:
#
#        # ---- PASTE BELOW into PART1_STATE ----
#        PART1_STATE = { ... }
#        # ---- END PASTE ----
#
#   3. Copy that entire dict and replace the empty PART1_STATE = {} below.
#   4. Switch --data-dir to PART2.  The strategy auto-loads the prior
#      knowledge on bar 0.  No positions are carried over.
# ---------------------------------------------------------------------------
PART1_STATE = {}
class SignalEvaluator:
    """
    One instance per config.  Holds per-instrument regime detectors and signal
    strength trackers.  Does NOT own any position state.
    """
    def __init__(self, name, cfg, datas, indicators):
        self.name = name
        self.cfg  = cfg
        lb        = cfg['bayesian_lookback']
        mw        = lb // 3
        self.mw   = mw

        self.atr     = indicators['atr']
        self.vol_avg = indicators['vol_avg']
        self.vol_std = indicators['vol_std']
        self.sma     = indicators['sma']

        self.regime   = {d: BayesianRegimeDetector(lookback=lb) for d in datas}
        self.mr_sig   = {d: BayesianSignalStrength(prior=0.48)  for d in datas}
        self.tf_sig   = {d: BayesianSignalStrength(prior=0.48)  for d in datas}

    def extract_state(self):
        # Only regime observation counts per instrument.
        # mr_sig/tf_sig skipped — with only ~8 trades they are still at prior.
        return {
            d._name: self.regime[d].extract_state() for d in self.regime
        }

    def load_state(self, state):
        for d in self.regime:
            if d._name in state:
                self.regime[d].load_state(state[d._name])
        self._warmed = True

    def ready(self, bar):
        lb = self.cfg['bayesian_lookback']
        mw = lb // 3
        if getattr(self, '_warmed', False):
            # Prior beliefs loaded — only wait for indicators to have enough bars.
            # vol_avg/vol_std need mw bars, ATR needs atr_len, SMA needs ma_len.
            indicator_min = max(mw, self.cfg['atr_len'], self.cfg['ma_len'])
            return bar >= indicator_min
        # Cold start — wait for both indicators and enough price history for regime.
        return bar >= int(lb * 0.67)

    # ---------------------------------------------------------------- signals
    def _vol_z(self, d):
        v  = float(d.volume[0])
        mu = float(self.vol_avg[d][0])
        sd = float(self.vol_std[d][0]) or 1.0
        return (v - mu) / sd

    def check_mr_long(self, d):
        close   = float(d.close[0])
        openp   = float(d.open[0])
        vol_z   = self._vol_z(d)
        mw      = self.mw
        short_ret = (close - float(d.close[-self.cfg['meanrev_window']])) / float(d.close[-self.cfg['meanrev_window']])
        low_n   = min(float(d.low[-i]) for i in range(1, mw + 1))
        dist    = (close - low_n) / low_n if low_n > 0 else 0
        hits = (
            (vol_z   >= self.cfg['volume_z'])
            + (short_ret <= self.cfg['short_return'])
            + (close  > openp)
            + (dist   <= self.cfg['bounce_pct'])
        )
        return hits >= 3

    def check_mr_short(self, d):
        close   = float(d.close[0])
        openp   = float(d.open[0])
        vol_z   = self._vol_z(d)
        mw      = self.mw
        ret_n   = (close - float(d.close[-self.cfg['meanrev_window']])) / float(d.close[-self.cfg['meanrev_window']])
        high_n  = max(float(d.high[-i]) for i in range(1, mw + 1))
        dist    = (high_n - close) / high_n if high_n > 0 else 0
        hits = (
            (vol_z  >= 1.0)
            + (ret_n >= 0.04)
            + (close < openp)
            + (dist  <= 0.02)
        )
        return hits >= 3

    def check_tf_long(self, d):
        close  = float(d.close[0])
        ups    = sum(1 for k in range(1, self.cfg['up_window'] + 1) if close > float(d.close[-k]))
        vol_ok = self._vol_z(d) >= self.cfg['vol_z_min']
        high_n = max(float(d.high[-k]) for k in range(1, self.cfg['breakout_lookback'] + 1))
        brk_ok = close >= high_n
        sma    = self.sma[d]
        trend_ok = True
        if sma is not None:
            s0, s1, s2 = float(sma[0]), float(sma[-1]), float(sma[-2])
            trend_ok   = close > s0 and s0 > s1 > s2
        hits = (ups >= self.cfg['up_days_min']) + vol_ok + brk_ok + trend_ok
        return hits >= 3

    def check_tf_short(self, d):
        close   = float(d.close[0])
        downs   = sum(1 for k in range(1, self.cfg['up_window'] + 1) if close < float(d.close[-k]))
        vol_ok  = self._vol_z(d) >= self.cfg['vol_z_min']
        low_n   = min(float(d.low[-k]) for k in range(1, self.cfg['breakout_lookback'] + 1))
        brk_ok  = close <= low_n
        sma     = self.sma[d]
        trend_ok = True
        if sma is not None:
            s0, s1, s2 = float(sma[0]), float(sma[-1]), float(sma[-2])
            trend_ok   = close < s0 and s0 < s1 < s2
        hits = (downs >= self.cfg['up_days_min']) + vol_ok + brk_ok + trend_ok
        return hits >= 3

    def get_signal(self, d, weight, perf, laggards):
        """
        Returns (size_sign, entry_type) or (0, '') if no signal.
        Updates regime detector but does NOT modify any position state.
        """
        cfg    = self.cfg
        self.regime[d].update(float(d.close[0]))
        r      = self.regime[d].get_regime()
        sign   = 0
        etype  = ''

        # Mean reversion
        if r['sideways'] >= cfg['sideways_threshold']:
            if d in laggards and self.check_mr_long(d):
                sign, etype = 1, 'mean_reversion_long'
            elif sign == 0 and r['downtrend'] > r['uptrend'] and perf.get(d, 0) >= 0.05:
                if self.check_mr_short(d):
                    sign, etype = -1, 'mean_reversion_short'

        # Trend following
        if sign == 0 and (r['uptrend'] + r['downtrend']) >= cfg['trending_threshold']:
            if r['downtrend'] > r['uptrend']:
                if self.check_tf_short(d):
                    sign, etype = -1, 'trend_following_short'
            else:
                if self.check_tf_long(d):
                    sign, etype = 1, 'trend_following_long'

        return sign, etype

    def notify_result(self, d, entry_type, won):
        """Called after a position closes so signal strength can be updated."""
        is_mr = 'mean_reversion' in entry_type
        is_tf = 'trend_following' in entry_type
        self.mr_sig[d].update(is_mr, won)
        self.tf_sig[d].update(is_tf, won)

class SuperBayesian(bt.Strategy):
    """
    SuperBayesianV4 — single-manager architecture.

    Key design:
      - ONE open position per instrument at a time (enforced at broker level).
      - Four SignalEvaluators (LB40/60/90/120) vote on each bar.
      - The highest-weight config whose signal fires wins.
      - All position state (trail, entry price, entry_type, holding bar count)
        is stored in self._pos dict keyed by data._name — no child ambiguity.
      - Exit logic is centralised in _check_exit().
      - notify_order/notify_trade interface is identical to V3 for analyzer
        compatibility.
    """

    params = dict(
        selector_update_interval=20,
        sharpe_window=30,
        global_max_exposure=0.80,
        global_cash_buffer=100000.0,
        risk_free_bars=7,
        printlog=False,
        warm_state=None,   # pass extract_state() output here for PART 2
    )

    # ------------------------------------------------------------------
    def __init__(self):
        config_names  = list(CONFIGS.keys())
        self.selector = BayesianModelSelector(
            config_names    = config_names,
            update_interval = self.p.selector_update_interval,
            sharpe_window   = self.p.sharpe_window,
        )

        # Build indicators and signal evaluators
        self._indicators = {}
        self.evaluators  = {}
        for name, cfg in CONFIGS.items():
            lb = cfg['bayesian_lookback']
            mw = lb // 3
            ind = {
                'atr':     {d: bt.indicators.ATR(d, period=cfg['atr_len'])                      for d in self.datas},
                'vol_avg': {d: bt.indicators.SimpleMovingAverage(d.volume, period=mw)            for d in self.datas},
                'vol_std': {d: bt.indicators.StandardDeviation(d.volume, period=mw)              for d in self.datas},
                'sma':     {d: bt.indicators.SimpleMovingAverage(d.close, period=cfg['ma_len'])  for d in self.datas},
            }
            self._indicators[name] = ind
            self.evaluators[name]  = SignalEvaluator(name, cfg, self.datas, ind)

        # ----------------------------------------------------------------
        # SINGLE position registry — keyed by data._name
        # Each entry: {
        #   'size_sign': +1/-1,
        #   'entry_px': float,
        #   'entry_type': str,
        #   'entry_bar': int,
        #   'trail': float,
        #   'breach_count': int,
        #   'config': str,       # which config opened this position
        # }
        # ----------------------------------------------------------------
        self._pos = {}

        # Orders in flight: order.ref -> ('entry'|'exit', data, size_sign, entry_type, config)
        self._pending = {}

        # Deferred label clear (same mechanism as V3 for analyzer compatibility)
        self._clear_label_next_bar = set()
        self._exit_sizes           = {}   # sym -> share count at exit signal time

        # Top-level state read by TradeTypeAnalyzer / PositionTypeAnalyzer
        self.state = {d: {'entry_type': ''} for d in self.datas}

        # Daily budget tracking
        self._day_notional = 0.0
        self._day_symbol   = {}
        self._last_date    = None
        self._start_value  = None

        # Cooldown tracking — sym -> bar number of last exit
        self._last_exit_bar: dict = {}

        # If warm state was injected (PART 2 run), restore it immediately
        if self.p.warm_state is not None:
            self.load_state(self.p.warm_state)
        elif PART1_STATE:
            self.load_state(PART1_STATE)

    # ------------------------------------------------------------------
    def _reset_day(self):
        cur = self.datas[0].datetime.date(0)
        if self._last_date != cur:
            self._day_notional = 0.0
            self._day_symbol   = {}
            self._last_date    = cur

    # ------------------------------------------------------------------
    def _units_from_risk(self, d, cfg):
        atr_val = float(self._indicators[cfg['bayesian_lookback'] and 'LB40' or 'LB40']['atr'][d][0])
        # Use the evaluator's own ATR
        return 0  # placeholder — see usage below

    # ------------------------------------------------------------------
    def _atr(self, config_name, d):
        try:
            return float(self._indicators[config_name]['atr'][d][0])
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def _check_exit(self, sym, d, bar):
        """
        Returns True if the open position on `d` should be closed this bar.
        Also updates the trailing stop in place.
        """
        pos = self._pos.get(sym)
        if pos is None:
            return False

        close     = float(d.close[0])
        side      = pos['size_sign']
        trail     = pos['trail']
        cfg       = CONFIGS[pos['config']]
        atr_val   = self._atr(pos['config'], d)
        severe_k  = cfg['severe_breach_mult'] * atr_val

        # Max holding period
        if bar - pos['entry_bar'] >= cfg['max_hold_bars']:
            return True

        breached = (side > 0 and close < trail) or (side < 0 and close > trail)
        severe   = (side > 0 and close < trail - severe_k) or (side < 0 and close > trail + severe_k)

        if severe:
            return True

        if breached:
            pos['breach_count'] += 1
            if cfg.get('two_bar_exit', True) and pos['breach_count'] >= 2:
                return True
            elif not cfg.get('two_bar_exit', True):
                return True
        else:
            pos['breach_count'] = 0
            # Advance trail
            if side > 0:
                pos['trail'] = max(trail, close - cfg['atr_k'] * atr_val)
            else:
                pos['trail'] = min(trail, close + cfg['atr_k'] * atr_val)

        return False

    # ------------------------------------------------------------------
    def next(self):
        bar = len(self)
        self._reset_day()

        # Deferred label clear — runs one bar AFTER notify_trade so the analyzer
        # always sees the correct label when notify_trade fires.
        for d in list(self._clear_label_next_bar):
            self.state[d]['entry_type'] = ''
        self._clear_label_next_bar.clear()

        # Re-assert labels for all currently open positions every bar.
        for sym, pos_info in self._pos.items():
            d_match = next((d for d in self.datas if d._name == sym), None)
            if d_match is not None:
                self.state[d_match]['entry_type'] = pos_info['entry_type']

        if self._start_value is None:
            try:
                self._start_value = float(self.broker.getvalue())
            except Exception:
                self._start_value = 1.0

        try:
            bars_left = self.datas[0].buflen() - bar
        except Exception:
            bars_left = 999_999

        # ---- End-of-period forced liquidation ----
        if bars_left <= self.p.risk_free_bars:
            for d in self.datas:
                pos = self.getposition(d).size
                if pos > 0:
                    self.sell(data=d, size=pos)
                elif pos < 0:
                    self.buy(data=d, size=abs(pos))
            return

        self.selector.step(bar)
        weights = self.selector.get_weights()

        # Compute cross-sectional performance for MR laggard detection
        ref_window = CONFIGS['LB60']['meanrev_window']
        try:
            perf = {
                d: (float(d.close[0]) - float(d.close[-ref_window])) / float(d.close[-ref_window])
                for d in self.datas
            }
        except Exception:
            perf = {d: 0.0 for d in self.datas}

        ranked     = sorted(perf, key=perf.get)
        n          = len(ranked)
        bottom_cut = max(1, int(CONFIGS['LB60']['top_pct'] * n))
        laggards   = set(ranked[:bottom_cut])

        port_val        = float(self.broker.getvalue())
        cash_now        = float(self.broker.getcash())
        invested        = port_val - cash_now
        global_headroom = max(0.0, self.p.global_max_exposure * port_val - invested)

        for d in self.datas:
            sym      = d._name
            has_pos  = sym in self._pos
            broker_sz = self.getposition(d).size

            # ---- Stale position guard ----
            # If strategy thinks it has a position but broker is flat (e.g. a
            # pending order never filled and was cancelled), clean up cleanly.
            if has_pos and broker_sz == 0:
                pos_info   = self._pos.pop(sym)
                entry_type = pos_info['entry_type']
                config     = pos_info['config']
                # Notify selector with break-even result (we don't know PnL)
                self.selector.record_trade(config, 0.0, False)
                self.evaluators[config].notify_result(d, entry_type, False)
                self._clear_label_next_bar.add(d)
                has_pos = False

            # ---- EXIT ----
            if has_pos:
                if self._check_exit(sym, d, bar):
                    broker_pos = self.getposition(d).size
                    if broker_pos > 0:
                        o = self.sell(data=d, size=broker_pos)
                    elif broker_pos < 0:
                        o = self.buy(data=d, size=abs(broker_pos))
                    else:
                        o = None

                    # Write exit tracking unconditionally — sell/buy return None
                    if broker_pos != 0:
                        self._exit_sizes[sym] = abs(broker_pos)
                        self._clear_label_next_bar.add(d)

                    if o:
                        pos_info = self._pos[sym]
                        self._pending[o.ref] = (
                            'exit', d, pos_info['size_sign'],
                            pos_info['entry_type'], pos_info['config'],
                        )
                continue   # never enter on same bar as exit signal

            # ---- ENTRY — pick best-weight config whose signal fires ----
            # Respect cooldown — don't re-enter too soon after an exit
            last_exit = self._last_exit_bar.get(sym, -9999)
            min_cooldown = min(cfg_val['cooldown'] for cfg_val in CONFIGS.values())
            if bar - last_exit < min_cooldown:
                continue

            best_sign   = 0
            best_etype  = ''
            best_config = ''
            best_weight = -1.0

            for name, ev in self.evaluators.items():
                if not ev.ready(bar):
                    continue
                sign, etype = ev.get_signal(d, weights[name], perf, laggards)
                if sign == 0:
                    continue
                if weights[name] > best_weight:
                    best_sign, best_etype, best_config, best_weight = sign, etype, name, weights[name]

            if best_sign == 0:
                continue

            cfg      = CONFIGS[best_config]
            atr_val  = self._atr(best_config, d)
            price    = float(d.close[0])
            if atr_val <= 0 or price <= 0:
                continue

            # Size purely from ATR-based risk — weight influences which config
            # wins the entry, not how many shares are bought.
            raw_units = max(0, math.floor(cfg['risk_notional_per_leg'] / atr_val / price))
            units     = min(cfg['entry_units_cap'], raw_units)
            units     = max(1, min(units, cfg['max_units_per_symbol']))
            notional  = price * units

            sym_spent = self._day_symbol.get(sym, 0.0)
            sym_cap   = cfg['day_symbol_cap_notional'] * port_val
            if (
                sym_spent + notional > sym_cap
                or self._day_notional + notional > port_val * cfg['day_budget_frac']
                or notional > global_headroom
                or cash_now < notional + self.p.global_cash_buffer
            ):
                continue

            if best_sign > 0:
                o = self.buy(data=d, size=units)
            else:
                o = self.sell(data=d, size=units)

            # Write label and update budgets unconditionally — buy/sell may return
            # None in some broker implementations (buffered market orders), so we
            # must not gate these writes on `if o:`.
            self.state[d]['entry_type'] = best_etype
            self.selector.record_capital(best_config, notional)
            self._day_notional          += notional
            self._day_symbol[sym]        = sym_spent + notional
            global_headroom             -= notional
            cash_now                    -= notional

            if o:
                self._pending[o.ref] = ('entry', d, best_sign, best_etype, best_config)

    # ------------------------------------------------------------------
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        ref = order.ref
        if ref not in self._pending:
            return

        kind, d, size_sign, entry_type, config = self._pending.pop(ref)
        sym = d._name
        cfg = CONFIGS[config]

        if kind == 'exit':
            if order.status == order.Completed:
                pos_info = self._pos.pop(sym, None)
                if pos_info:
                    exit_price = float(order.executed.price)
                    side       = pos_info['size_sign']
                    entry_px   = pos_info['entry_px']
                    pnl_pct    = side * (exit_price - entry_px) / entry_px if entry_px > 0 else 0.0
                    won        = pnl_pct > 0
                    self.selector.record_trade(config, pnl_pct, won)
                    self.evaluators[config].notify_result(d, entry_type, won)
                    # Record cooldown so we don't immediately re-enter
                    self._last_exit_bar[sym] = len(self)
            return

        # Entry order
        if order.status != order.Completed:
            if d in self.state:
                self.state[d]['entry_type'] = ''
            return

        price   = float(order.executed.price)
        atr_val = self._atr(config, d)
        if size_sign > 0:
            # Long: trail starts below entry price
            trail = price - cfg['atr_k'] * atr_val
        else:
            # Short: trail starts above entry price
            trail = price + cfg['atr_k'] * atr_val

        self._pos[sym] = {
            'size_sign':   size_sign,
            'entry_px':    price,
            'entry_type':  entry_type,
            'entry_bar':   len(self),
            'trail':       trail,
            'breach_count': 0,
            'config':      config,
        }
        if d in self.state:
            self.state[d]['entry_type'] = entry_type

    # ------------------------------------------------------------------
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        data = trade.data
        sym  = data._name
        label_in_state = self.state[data]['entry_type'] if data in self.state else 'NO_STATE_KEY'
        if data in self.state:
            self._clear_label_next_bar.add(data)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # State transfer helpers — call extract_state() at the end of PART 1
    # and load_state() at the start of PART 2 (before any next() runs).
    # ------------------------------------------------------------------
    def extract_state(self):
        # Compact state: just trend_obs/mr_obs per instrument per config.
        # This is all that's needed to give PART 2 prior regime beliefs.
        # The selector alphas are barely moved from 1.0 after ~8 trades, not worth transferring.
        return {
            name: ev.extract_state() for name, ev in self.evaluators.items()
        }

    def load_state(self, state):
        # state is {config_name: {instrument_name: {trend_obs, mr_obs}}}
        for name, ev_state in state.items():
            if name in self.evaluators:
                self.evaluators[name].load_state(ev_state)

    # ------------------------------------------------------------------
    def stop(self):
        # Only save when running PART 1 (PART1_STATE empty = cold start).
        if PART1_STATE:
            return

        import os

        state     = self.extract_state()
        # Use repr() so the output is a valid Python literal with no formatting ambiguity.
        state_str = "PART1_STATE = " + repr(state)

        script_path = os.path.abspath(__file__)
        txt_path    = os.path.join(os.path.dirname(script_path), "part1_state.txt")
        with open(txt_path, "w") as f:
            f.write(state_str + "\n")
        print("[SuperShortFixed] Prior beliefs saved to: " + txt_path)
        print("[SuperShortFixed] Open part1_state.txt, copy the PART1_STATE = {...} line,")
        print("[SuperShortFixed] paste it over PART1_STATE = {} in this file, then run PART 2.")