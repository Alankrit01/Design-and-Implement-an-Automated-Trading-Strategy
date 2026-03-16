# Fixes applied:
#  Trailing stop initialisation corrected for shorts (was clamping trail to entry price)
#  Cooldown enforcement added before re-entry (was missing, blocking re-entries silently)
#  Budget/label writes moved outside `if o:` block (fixes buffered-order brokers)
#  Cooldown bar recorded on exit so re-entry timer is accurate

'''
SuperShortFixed with Global Market Regime detection for risk on/off 
Added max_hold_bars to exit conditions

Part 2 - 8 of 13 trades won but only 20k PnL

Dirichlet Weighting still over four lookbacks and continously shifts allocation toward the lookback which has the best 
rolling Sharpe Ratio with a min %age for all strategies
'''

'''
python main.py --strategy GlobalRegime --data-dir DATA/PART1 --output-dir output/Global
'''

'''
{
  "final_value": 1224671.4186648799,
  "bankrupt": false,
  "bankrupt_date": null,
  "open_pnl_pd_ratio": 4.265257312194384,
  "true_pd_ratio": 3.361967063842396,
  "activity_pct": 91.8,
  "end_policy": "liquidate",
  "s_mult": 2.0
}
=== Trade Analyzer Stats ===
Total closed trades: 42
Wins: 15
Losses: 27

=== Trade Breakdown by Strategy Type ===

Mean Reversion Trades Long:
  Total trades: 1
  Wins: 1
  Losses: 0
  Total PnL: $122623.88
  Win Rate: 100.0%

Trend Following Trades Long:
  Total trades: 12
  Wins: 4
  Losses: 8
  Total PnL: $92826.71
  Win Rate: 33.3%

Mean Reversion Trades Short:
  Total trades: 1
  Wins: 1
  Losses: 0
  Total PnL: $10145.50
  Win Rate: 100.0%

Trend Following Trades Shorts:
  Total trades: 28
  Wins: 9
  Losses: 19
  Total PnL: $9069.65
  Win Rate: 32.1%

--- Portfolio Total ---
  Total positions : 42
  Total wins      : 15
  Overall PnL     : $234,665.74
  Overall win rate: 35.7%

Symbol     Type                     EntryDate    ExitDate              PnL  W/L
--------------------------------------------------------------------------------------------------------------
series_6   trend_following_short    2070-02-21   2070-07-16     -1,449.30  LOSS
series_4   trend_following_short    2070-02-22   2070-04-22     -5,831.57  LOSS
series_3   trend_following_short    2070-02-22   2072-08-27       -237.09  LOSS
series_9   trend_following_short    2070-02-23   2070-05-10     -2,106.98  LOSS
series_1   trend_following_short    2070-02-23   2070-06-06       -244.62  LOSS
series_2   trend_following_short    2070-02-24   2070-05-30        683.75  WIN
series_5   mean_reversion_long      2070-02-24   2072-08-27    122,623.88  WIN
series_10  trend_following_short    2070-03-07   2070-05-20       -171.50  LOSS
series_7   trend_following_long     2070-03-10   2070-05-25       -942.53  LOSS
series_8   trend_following_short    2070-03-13   2070-05-09     -2,495.89  LOSS
series_4   trend_following_long     2070-04-22   2070-05-23     -1,857.25  LOSS
series_8   trend_following_long     2070-05-09   2070-07-11     -4,801.87  LOSS
series_9   trend_following_short    2070-05-10   2070-05-25        514.23  WIN
series_10  trend_following_short    2070-05-20   2070-05-30        276.00  WIN
series_4   trend_following_short    2070-05-23   2070-06-05       -809.42  LOSS
series_9   trend_following_short    2070-05-25   2070-06-19     -1,106.80  LOSS
series_10  trend_following_short    2070-05-30   2070-06-30     -8,451.00  LOSS
series_2   mean_reversion_short     2070-05-30   2070-06-30     10,145.50  WIN
series_7   trend_following_long     2070-06-05   2070-06-19        150.00  WIN
series_4   trend_following_long     2070-06-05   2070-07-12       -861.56  LOSS
series_1   trend_following_short    2070-06-06   2072-08-27     39,057.28  WIN
series_7   trend_following_long     2070-06-25   2070-07-18        -28.03  LOSS
series_9   trend_following_short    2070-06-25   2070-07-18     -1,014.78  LOSS
series_2   trend_following_long     2070-06-30   2070-07-28       -327.00  LOSS
series_8   trend_following_long     2070-07-11   2072-08-27      5,722.75  WIN
series_4   trend_following_long     2070-07-12   2072-08-27     95,995.30  WIN
series_6   trend_following_long     2070-07-16   2071-07-19       -626.52  LOSS
series_10  trend_following_short    2070-07-19   2071-07-05     -1,859.75  LOSS
series_2   trend_following_short    2070-07-28   2071-06-18       -678.00  LOSS
series_2   trend_following_short    2071-06-18   2071-08-06      1,287.00  WIN
series_9   trend_following_short    2071-06-30   2071-07-03       -182.40  LOSS
series_9   trend_following_short    2071-07-03   2071-07-10        521.25  WIN
series_7   trend_following_long     2071-07-03   2071-07-10        -45.59  LOSS
series_10  trend_following_short    2071-07-05   2071-07-22        115.50  WIN
series_9   trend_following_short    2071-07-18   2071-07-21        329.96  WIN
series_6   trend_following_short    2071-07-19   2071-08-15       -150.66  LOSS
series_9   trend_following_short    2071-07-21   2071-08-16     -1,225.08  LOSS
series_7   trend_following_long     2071-07-21   2071-08-16        449.02  WIN
series_10  trend_following_short    2071-07-22   2071-08-06       -902.25  LOSS
series_6   trend_following_short    2071-08-15   2072-08-27        138.79  WIN
series_2   trend_following_short    2071-08-17   2072-08-27     -3,122.00  LOSS
series_10  trend_following_short    2071-08-23   2072-08-27     -1,815.00  LOSS
'''

import backtrader as bt
import math
import numpy as np
from collections import defaultdict, deque
from itertools import combinations

# GLobal Market Regime
class GlobalMarketMonitor:
    """
    Aggregates price/volume data from ALL N data series every bar and exposes:
        market_breadth: fraction of series trading above their own N-bar SMA
        avg_momentum: cross-sectional mean return over momentum_window bars
        breadth_momentum: EMA of breadth change (acceleration)
        realised_vol_index: cross-sectional mean of rolling 20-bar stdev of returns
        avg_corr: mean pairwise rolling correlation (cohesion / contagion proxy)
        dispersion: cross-sectional stdev of returns (high = pairs opportunity)
    """

    def __init__(self, n_series, breadth_window=20, momentum_window=10, vol_window=20, corr_window=30, ema_alpha=0.1):
        self.n = n_series
        self.bw = breadth_window
        self.mw = momentum_window
        self.vw = vol_window
        self.cw = corr_window
        self.ema_alpha = ema_alpha

        # rolling price history per series —> store index-keyed (populated in update)
        self._prices: dict[int, deque] = {i: deque(maxlen=max(breadth_window, corr_window) + 5)
                                          for i in range(n_series)}

        # computed metrics (updated each bar)
        self.market_breadth = 0.5
        self.avg_momentum = 0.0
        self.breadth_momentum = 0.0   # EMA of breadth delta
        self.realised_vol_index = 0.0
        self.avg_corr = 0.0
        self.dispersion = 0.0   # cross-sectional return stdev

        self.prev_breadth = 0.5
        self.bar_count = 0

    def update(self, closes: list[float]):
        """
        Call once per bar with the list of close prices for all series, in the same order as self.datas.
        """
        self.bar_count += 1
        for i, c in enumerate(closes):
            self._prices[i].append(c)

        n = self.n
        breadth_hits = 0
        momentums = []
        vol_vals = []

        for i in range(n):
            px = list(self._prices[i])
            if len(px) < 2:
                continue

            # Breadth: price above N-bar SMA
            if len(px) >= self.bw:
                sma = np.mean(px[-self.bw:])
                if px[-1] > sma:
                    breadth_hits += 1

            # Momentum
            if len(px) >= self.mw + 1:
                ret = (px[-1] - px[-(self.mw + 1)]) / px[-(self.mw + 1)]
                momentums.append(ret)

            # Realised vol
            if len(px) >= self.vw + 1:
                rets = np.diff(px[-self.vw - 1:]) / np.array(px[-self.vw - 1:-1])
                vol_vals.append(float(np.std(rets)))

        counted = sum(1 for i in range(n) if len(self._prices[i]) >= self.bw)
        self.market_breadth = breadth_hits / counted if counted > 0 else 0.5
        self.avg_momentum = float(np.mean(momentums)) if momentums else 0.0
        self.dispersion = float(np.std(momentums)) if len(momentums) > 1 else 0.0
        self.realised_vol_index = float(np.mean(vol_vals)) if vol_vals else 0.0

        # EMA of breadth change
        delta = self.market_breadth - self.prev_breadth
        self.breadth_momentum = (self.ema_alpha * delta + (1 - self.ema_alpha) * self.breadth_momentum)
        self.prev_breadth = self.market_breadth

        # Mean pairwise correlation
        if self.bar_count >= self.cw + 1:
            ret_series = []
            for i in range(n):
                px = list(self._prices[i])
                if len(px) >= self.cw + 1:
                    r = np.diff(px[-self.cw - 1:]) / np.array(px[-self.cw - 1:-1])
                    ret_series.append(r)
            if len(ret_series) >= 2:
                corr_vals = []
                for a, b in combinations(range(len(ret_series)), 2):
                    c = float(np.corrcoef(ret_series[a], ret_series[b])[0, 1])
                    if not math.isnan(c):
                        corr_vals.append(c)
                self.avg_corr = float(np.mean(corr_vals)) if corr_vals else 0.0

    
    def summary(self) -> dict:
        return {
            'breadth': self.market_breadth,
            'momentum': self.avg_momentum,
            'breadth_mom': self.breadth_momentum,
            'vol_index': self.realised_vol_index,
            'avg_corr': self.avg_corr,
            'dispersion': self.dispersion,
        }

# Risk ON/OFF
class RiskRegimeClassifier:
    """
    Synthesises GlobalMarketMonitor metrics into a single market-wide Risk-On / Risk-Off / Neutral state, 
    plus a scalar risk_score ∈ [0, 1] (1 = fully risk-on).

    Thresholds are intentionally asymmetric to avoid whipsawing:
      RISK_ON  requires breadth > ON_THRESH  AND momentum > 0 AND vol low
      RISK_OFF requires breadth < OFF_THRESH OR  vol spike  OR sharp neg momentum
      NEUTRAL  otherwise

    A hysteresis counter (confirm_bars) prevents flip-flopping:
    the regime must be indicated for N consecutive bars before switching.
    """

    RISK_ON  = 'RISK_ON'
    RISK_OFF = 'RISK_OFF'
    NEUTRAL  = 'NEUTRAL'

    def __init__(self,
                 on_breadth_thresh = 0.60,
                 off_breadth_thresh = 0.38,
                 on_momentum_min = 0.005,
                 off_momentum_max = -0.010,
                 vol_spike_mult = 2.0,
                 confirm_bars = 3):

        self.on_bt = on_breadth_thresh
        self.off_bt = off_breadth_thresh
        self.on_mom = on_momentum_min
        self.off_mom = off_momentum_max
        self.vol_mult = vol_spike_mult
        self.confirm = confirm_bars

        self.regime = self.NEUTRAL
        self.risk_score = 0.65           # generous default during warmup
        self.pending = self.NEUTRAL   # candidate next regime
        self.pending_bars = 0
        self.vol_history = deque(maxlen=60)
        # Warmup blackout: breadth is unreliable until SMA windows fill.
        # Hold NEUTRAL with a permissive risk_score for N bars so the regime classifier cannot block early TF-long entries.
        self.warup_bars = 0
        self.warmup_reqd = 25   # > breadth_window (default 20)
    
    def update(self, gm: GlobalMarketMonitor):
        """
        Call after GlobalMarketMonitor.update() each bar.
        """
        self.warup_bars += 1

        # During SMA warmup breadth/momentum are unreliable: lock to NEUTRAL with a permissive risk_score so early TF-long entries are not blocked.
        if self.warup_bars < self.warmup_reqd:
            self.regime = self.NEUTRAL
            self.risk_score = 0.65
            return

        breadth = gm.market_breadth
        momentum = gm.avg_momentum
        vol = gm.realised_vol_index
        breadth_mom = gm.breadth_momentum

        self.vol_history.append(vol)
        vol_baseline = float(np.mean(self.vol_history)) if self.vol_history else vol
        vol_spike = vol > self.vol_mult * vol_baseline if vol_baseline > 1e-9 else False

        # classify candidate regime 
        if (breadth >= self.on_bt
                and momentum >= self.on_mom
                and not vol_spike
                and breadth_mom >= -0.02):
            candidate = self.RISK_ON

        elif (breadth <= self.off_bt
              or momentum <= self.off_mom
              or vol_spike):
            candidate = self.RISK_OFF

        else:
            candidate = self.NEUTRAL

        # hysteresis 
        if candidate == self.pending:
            self.pending_bars += 1
        else:
            self.pending = candidate
            self.pending_bars = 1

        if self.pending_bars >= self.confirm:
            self.regime = self.pending

        # continuous risk score (for position sizing)
        # Breadth component [0,1]
        b_score = max(0.0, min(1.0, (breadth - 0.30) / 0.50))
        # Momentum component [0,1]
        m_score = max(0.0, min(1.0, (momentum + 0.02) / 0.04))
        # Vol penalty [0,1] — higher vol = lower score
        if vol_baseline > 1e-9:
            v_pen = max(0.0, min(1.0, 1.0 - (vol / (vol_baseline * self.vol_mult))))
        else:
            v_pen = 1.0
        self.risk_score = 0.45 * b_score + 0.35 * m_score + 0.20 * v_pen

    
    def is_risk_on(self) -> bool: return self.regime == self.RISK_ON
    def is_risk_off(self) -> bool: return self.regime == self.RISK_OFF
    def is_neutral(self) -> bool: return self.regime == self.NEUTRAL

    def entry_scale(self, base_frac: float = 1.0) -> float:
        """
        Multiplicative scaling factor for position size based on risk regime.
        RISK_ON  = 0.50–1.0×  (base_frac * risk_score, floor 0.50)
        NEUTRAL  = 0.30–0.80×
        RISK_OFF = 0.25–0.55×  floor raised from 0.0 so that TF-long positions that fire despite the regime 
        still get meaningful size. The original strategy's edge comes from multi-year trend holds; cutting size to ~0 kills 
        the PnL even when direction is right.
        """
        if self.is_risk_on():
            return base_frac * max(0.5, self.risk_score)
        elif self.is_neutral():
            return base_frac * max(0.3, self.risk_score * 0.8)
        else:  # RISK_OFF — reduce longs but don't zero them
            return base_frac * max(0.25, self.risk_score * 0.55)

    def allows_long(self)  -> bool:
        """Directional longs only when NOT firmly risk-off."""
        return not self.is_risk_off()

    def allows_short(self) -> bool:
        """Shorts always allowed; risk-off encourages them."""
        return True

    def pair_trade_bias(self) -> str:
        """
        Hint for the pair engine:
          RISK_OFF  = prefer long-low / short-high (spread compression)
          RISK_ON   = prefer long-high / short-low (momentum spread)
          NEUTRAL   = symmetric
        """
        if self.is_risk_off():
            return 'compress'
        elif self.is_risk_on():
            return 'momentum'
        return 'symmetric'


# Pair tracking to find correlation between assets

class PairTracker:
    """
    Maintains a rolling pairwise correlation + spread Z-score for every combination of the N data series.  
    The strategy can query this to decide whether to trade a pair (long one leg / short the other).

    Pair-trade entry signal fires when:
      1. |rolling_corr| >= corr_thresh (series are co-moving or contra-moving)
      2. |spread_z| >= z_entry (spread has diverged)
      3. spread_z direction is consistent with the risk regime bias

    Exit fires when |spread_z| <= z_exit  OR  corr breaks down (< corr_break).
    """

    def __init__(self, n_series, names,
                 corr_window=40, spread_window=30,
                 corr_thresh=0.55, corr_break=0.25,
                 z_entry=1.8, z_exit=0.4):

        self.n = n_series
        self.names = names   # list of data._name strings
        self.cw = corr_window
        self.sw = spread_window
        self.corr_thresh = corr_thresh
        self.corr_break = corr_break
        self.z_entry = z_entry
        self.z_exit = z_exit

        # price history per series
        self._prices: dict[int, deque] = {i: deque(maxlen=corr_window + spread_window + 5) for i in range(n_series)}

        # cached metrics per pair (i,j) with i<j
        self._corr: dict[tuple, float] = {}
        self.spread_z: dict[tuple, float] = {}
        self.spread_history: dict[tuple, deque] = { (i, j): deque(maxlen=spread_window) for i, j in combinations(range(n_series), 2) }
        self.bar_count = 0

    def update(self, closes: list[float]):
        """Call each bar with all closes (same order as datas)."""
        self.bar_count += 1
        for i, c in enumerate(closes):
            self._prices[i].append(c)

        if self.bar_count < self.cw + 2:
            return

        for i, j in combinations(range(self.n), 2):
            px_i = list(self._prices[i])
            px_j = list(self._prices[j])
            if len(px_i) < self.cw + 1 or len(px_j) < self.cw + 1:
                continue

            # Rolling correlation of returns
            r_i = np.diff(px_i[-self.cw - 1:]) / np.array(px_i[-self.cw - 1:-1])
            r_j = np.diff(px_j[-self.cw - 1:]) / np.array(px_j[-self.cw - 1:-1])
            # Guard: flat series has std=0 = corrcoef produces NaN + RuntimeWarning
            if np.std(r_i) < 1e-9 or np.std(r_j) < 1e-9:
                self._corr[(i, j)] = 0.0
            else:
                c = float(np.corrcoef(r_i, r_j)[0, 1])
                self._corr[(i, j)] = c if not math.isnan(c) else 0.0

            # Log-price spread
            if px_i[-1] > 0 and px_j[-1] > 0:
                spread = math.log(px_i[-1]) - math.log(px_j[-1])
                self.spread_history[(i, j)].append(spread)

            hist = list(self.spread_history[(i, j)])
            if len(hist) >= self.sw:
                mu = np.mean(hist)
                sd = np.std(hist)
                self.spread_z[(i, j)] = (hist[-1] - mu) / sd if sd > 1e-9 else 0.0
            else:
                self.spread_z[(i, j)] = 0.0
    
    def get_pair_signal(self, i: int, j: int, risk_bias: str = 'symmetric') -> dict:
        """
        Returns a dict:
          {
            'tradeable': bool,
            'long_leg':  int (index),   # the leg to BUY
            'short_leg': int (index),   # the leg to SELL
            'corr':      float,
            'spread_z':  float,
            'reason':    str,
          }
        """
        key = (min(i, j), max(i, j))
        corr = self._corr.get(key, 0.0)
        sz = self.spread_z.get(key, 0.0)

        if abs(corr) < self.corr_thresh:
            return {'tradeable': False, 'corr': corr, 'spread_z': sz, 'reason': 'corr_too_low'}

        if abs(sz) < self.z_entry:
            return {'tradeable': False, 'corr': corr, 'spread_z': sz, 'reason': 'spread_within_band'}

        # Direction: if corr > 0 (co-integrated), spread > z_entry means i expensive vs j = short i / long j (mean-reversion)
        if corr >= self.corr_thresh:
            if sz > self.z_entry:
                long_leg, short_leg = j, i   # i expensive = short i
            elif sz < -self.z_entry:
                long_leg, short_leg = i, j
            else:
                return {'tradeable': False, 'corr': corr, 'spread_z': sz, 'reason': 'spread_within_band'}

        else:
            # Negative correlation — divergence trade
            if risk_bias == 'compress':
                # Fade the divergence
                long_leg, short_leg = (i, j) if sz < 0 else (j, i)
            elif risk_bias == 'momentum':
                # Ride the divergence
                long_leg, short_leg = (j, i) if sz > 0 else (i, j)
            else:
                long_leg, short_leg = (j, i) if sz > 0 else (i, j)

        return {
            'tradeable':  True,
            'long_leg':   long_leg,
            'short_leg':  short_leg,
            'corr':       corr,
            'spread_z':   sz,
            'reason':     'pair_signal',
        }

    def should_exit_pair(self, i: int, j: int) -> bool:
        """Return True when the pair trade should be closed."""
        key = (min(i, j), max(i, j))
        corr = self._corr.get(key, 0.0)
        sz = self.spread_z.get(key, 0.0)
        corr_collapsed = abs(corr) < self.corr_break
        spread_mean_rev = abs(sz) <= self.z_exit
        return corr_collapsed or spread_mean_rev

    def corr(self, i: int, j: int) -> float:
        key = (min(i, j), max(i, j))
        return self._corr.get(key, 0.0)

    def spread_z(self, i: int, j: int) -> float:
        key = (min(i, j), max(i, j))
        return self.spread_z.get(key, 0.0)

class BayesianRegimeDetector:
    def __init__(self, lookback=60, alpha=2.0, beta=2.0):
        self.lookback = lookback
        self.alpha = alpha
        self.beta = beta
        self.trend_obs = 0
        self.mr_obs = 0
        self.prices = deque(maxlen=lookback)

    def update(self, close):
        self.prices.append(close)
        mw = self.lookback // 3
        if len(self.prices) >= mw:
            change = (self.prices[-1] - self.prices[-mw]) / self.prices[-mw]
            if abs(change) > 0.02:
                self.trend_obs += 1
            else:
                self.mr_obs += 1

    def get_regime(self):
        mw = self.lookback // 3
        if len(self.prices) < mw:
            return {'uptrend': 0.33, 'downtrend': 0.33, 'sideways': 0.34}
        recent = list(self.prices)[-mw:]
        early = list(self.prices)[-(mw * 2):-mw] if len(self.prices) >= mw * 2 else recent
        r_avg = np.mean(recent)
        e_avg = np.mean(early)
        total = self.trend_obs + self.mr_obs
        if total < 10:
            return {'uptrend': 0.33, 'downtrend': 0.33, 'sideways': 0.34}
        t_alpha = self.alpha + self.trend_obs
        m_alpha = self.beta  + self.mr_obs
        trend_prob = t_alpha / (t_alpha + m_alpha)
        if r_avg > e_avg:
            up_p, dn_p = trend_prob * 0.7, trend_prob * 0.3
        else:
            up_p, dn_p = trend_prob * 0.3, trend_prob * 0.7
        return {'uptrend': up_p, 'downtrend': dn_p, 'sideways': 1 - trend_prob}

class BayesianSignalStrength:
    def __init__(self, prior=0.50, window=50):
        self.prior = prior
        self.window  = window
        self.success = 0
        self.total = 0
        self.history = deque(maxlen=window)

    def update(self, fired, won):
        if fired:
            self.success += 1 if won else 0
            self.total += 1
            self.history.append(1.0 if won else 0.0)

    def accuracy(self, alpha=1.0, beta=1.0):
        if self.total == 0:
            return self.prior
        a = alpha + self.success
        b = beta  + (self.total - self.success)
        return a / (a + b)

class BayesianModelSelector:
    def __init__(self, config_names, update_interval=20, sharpe_window=30):
        self.config_names = config_names
        self.update_interval = update_interval
        self.sharpe_window = sharpe_window
        self.alphas = {n: 1.0 for n in config_names}
        self.pnl_history = {n: deque(maxlen=sharpe_window) for n in config_names}
        self.weight_history = []
        self.bar_count = 0
        self.stats = {
            n: {'trades': 0, 'wins': 0, 'total_pnl': 0.0, 'pnl_series': [], 'capital': 0.0} for n in config_names}

    def record_trade(self, name, pnl_pct, won):
        if name not in self.pnl_history:
            return
        self.pnl_history[name].append(pnl_pct)
        s = self.stats[name]
        s['trades'] += 1
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
        total = sum(sharpes.values())
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
        day_budget_frac=0.30, day_symbol_cap_notional=0.12,   # tightened: was 0.40
        max_units_per_symbol=4500, entry_units_cap=700,
        risk_notional_frac=0.055,          # NEW: fraction of portfolio per leg
        risk_notional_per_leg=60000.0,     # kept as fallback floor
        cooldown=3,
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
        day_budget_frac=0.30, day_symbol_cap_notional=0.12,   # tightened: was 0.40
        max_units_per_symbol=4500, entry_units_cap=700,
        risk_notional_frac=0.055,
        risk_notional_per_leg=60000.0,
        cooldown=4,
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
        day_budget_frac=0.30, day_symbol_cap_notional=0.12,   # tightened: was 0.40
        max_units_per_symbol=4500, entry_units_cap=700,
        risk_notional_frac=0.055,
        risk_notional_per_leg=60000.0,
        cooldown=5,
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
        day_budget_frac=0.30, day_symbol_cap_notional=0.12,   # tightened: was 0.40
        max_units_per_symbol=4500, entry_units_cap=700,
        risk_notional_frac=0.055,
        risk_notional_per_leg=60000.0,
        cooldown=10,
        risk_free_bars=14, severe_breach_mult=0.60,
        bayesian_lookback=120, regime_learning_enabled=True,
        signal_learning_enabled=True, min_confidence_mr=0.38,
        min_confidence_trend=0.38, sideways_threshold=0.25,
        trending_threshold=0.40, max_hold_bars=180,
    ),
}

class SignalEvaluator:
    """
    One instance per config.  Holds per-instrument regime detectors and signal
    strength trackers.  Does NOT own any position state.
    """
    def __init__(self, name, cfg, datas, indicators):
        self.name = name
        self.cfg  = cfg
        lb = cfg['bayesian_lookback']
        mw = lb // 3
        self.mw = mw

        self.atr = indicators['atr']
        self.vol_avg = indicators['vol_avg']
        self.vol_std = indicators['vol_std']
        self.sma = indicators['sma']

        self.regime = {d: BayesianRegimeDetector(lookback=lb) for d in datas}
        self.mr_sig = {d: BayesianSignalStrength(prior=0.48) for d in datas}
        self.tf_sig = {d: BayesianSignalStrength(prior=0.48) for d in datas}

    def ready(self, bar):
        return bar >= int(self.cfg['bayesian_lookback'] * 0.67)

    def _vol_z(self, d):
        v = float(d.volume[0])
        mu = float(self.vol_avg[d][0])
        sd = float(self.vol_std[d][0]) or 1.0
        return (v - mu) / sd

    def check_mr_long(self, d):
        close = float(d.close[0])
        openp = float(d.open[0])
        vol_z = self._vol_z(d)
        mw = self.mw
        short_ret = (close - float(d.close[-self.cfg['meanrev_window']])) / float(d.close[-self.cfg['meanrev_window']])
        low_n = min(float(d.low[-i]) for i in range(1, mw + 1))
        dist = (close - low_n) / low_n if low_n > 0 else 0
        hits = (
            (vol_z   >= self.cfg['volume_z'])
            + (short_ret <= self.cfg['short_return'])
            + (close  > openp)
            + (dist   <= self.cfg['bounce_pct'])
        )
        return hits >= 3

    def check_mr_short(self, d):
        close = float(d.close[0])
        openp = float(d.open[0])
        vol_z = self._vol_z(d)
        mw = self.mw
        ret_n = (close - float(d.close[-self.cfg['meanrev_window']])) / float(d.close[-self.cfg['meanrev_window']])
        high_n = max(float(d.high[-i]) for i in range(1, mw + 1))
        dist = (high_n - close) / high_n if high_n > 0 else 0
        hits = (
            (vol_z  >= 1.0)
            + (ret_n >= 0.04)
            + (close < openp)
            + (dist  <= 0.02)
        )
        return hits >= 3

    def check_tf_long(self, d):
        close = float(d.close[0])
        ups = sum(1 for k in range(1, self.cfg['up_window'] + 1) if close > float(d.close[-k]))
        vol_ok = self._vol_z(d) >= self.cfg['vol_z_min']
        high_n = max(float(d.high[-k]) for k in range(1, self.cfg['breakout_lookback'] + 1))
        brk_ok = close >= high_n
        sma = self.sma[d]
        trend_ok = True
        if sma is not None:
            s0, s1, s2 = float(sma[0]), float(sma[-1]), float(sma[-2])
            trend_ok = close > s0 and s0 > s1 > s2
        hits = (ups >= self.cfg['up_days_min']) + vol_ok + brk_ok + trend_ok
        return hits >= 3

    def check_tf_short(self, d):
        close = float(d.close[0])
        downs = sum(1 for k in range(1, self.cfg['up_window'] + 1) if close < float(d.close[-k]))
        vol_ok = self._vol_z(d) >= self.cfg['vol_z_min']
        low_n = min(float(d.low[-k]) for k in range(1, self.cfg['breakout_lookback'] + 1))
        brk_ok = close <= low_n
        sma = self.sma[d]
        trend_ok = True
        if sma is not None:
            s0, s1, s2 = float(sma[0]), float(sma[-1]), float(sma[-2])
            trend_ok = close < s0 and s0 < s1 < s2
        hits = (downs >= self.cfg['up_days_min']) + vol_ok + brk_ok + trend_ok
        return hits >= 3

    def get_signal(self, d, weight, perf, laggards,
                   risk_classifier: 'RiskRegimeClassifier' = None,
                   global_monitor:  'GlobalMarketMonitor'  = None):
        """
        Returns (size_sign, entry_type) or (0, '') if no signal.
        Updates regime detector but does NOT modify any position state.

        risk_classifier and global_monitor, when provided, gate and adjust
        the signal based on the cross-series market-wide regime:
          - RISK_OFF  blocks new directional longs; permits/promotes shorts
          - RISK_ON   grants full access to both longs and shorts
          - NEUTRAL   applies partial gating
          - High dispersion (pairs opportunity) loosens MR short threshold
        """
        cfg = self.cfg
        self.regime[d].update(float(d.close[0]))
        r = self.regime[d].get_regime()
        sign = 0
        etype = ''

        # Global regime gates 
        risk_on = True   # default: no gating if classifier absent
        risk_off = False
        dispersion = 0.0
        breadth = 0.5
        if risk_classifier is not None:
            risk_on = risk_classifier.allows_long()
            risk_off = risk_classifier.is_risk_off()
        if global_monitor is not None:
            dispersion = global_monitor.dispersion
            breadth = global_monitor.market_breadth

        # Loosen MR short threshold when cross-sectional dispersion is high
        # (high dispersion = more mean-reversion opportunities cross-series)
        mr_short_ret_thresh = 0.04
        if dispersion > 0.03:
            mr_short_ret_thresh = 0.03   # easier to trigger MR short

        # In RISK_OFF: MR longs disabled (we don't want to catch falling knives);
        #              TF longs disabled; shorts of both types remain active.
        # In high-breadth (RISK_ON): bias toward longs, suppress shorts slightly.
        allow_mr_long = risk_on and not risk_off
        allow_mr_short = True                           # always allowed
        allow_tf_long = risk_on and not risk_off
        allow_tf_short = True                           # always allowed
        # When market is broadly strong, raise the bar for shorts
        if breadth > 0.70:
            allow_mr_short = False   # very broad rally = suppress MR shorts
            allow_tf_short = breadth < 0.85   # extreme breadth blocks tf shorts too

        # Mean reversion
        if r['sideways'] >= cfg['sideways_threshold']:
            if allow_mr_long and d in laggards and self.check_mr_long(d):
                sign, etype = 1, 'mean_reversion_long'
            elif (sign == 0 and allow_mr_short
                  and r['downtrend'] > r['uptrend']
                  and perf.get(d, 0) >= (mr_short_ret_thresh if not risk_off else 0.025)):
                if self.check_mr_short(d):
                    sign, etype = -1, 'mean_reversion_short'

        # Trend following
        if sign == 0 and (r['uptrend'] + r['downtrend']) >= cfg['trending_threshold']:
            if r['downtrend'] > r['uptrend']:
                if allow_tf_short and self.check_tf_short(d):
                    sign, etype = -1, 'trend_following_short'
            else:
                if allow_tf_long and self.check_tf_long(d):
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
    Key design:
      ONE open position per instrument at a time (enforced at broker level).
      Four SignalEvaluators (LB40/60/90/120) vote on each bar.
      The highest-weight config whose signal fires wins.
      All position state (trail, entry price, entry_type, holding bar count) is stored in self._pos dict keyed by data._name — no child ambiguity.
      Exit logic is centralised in check_exit().
    """

    params = dict(
        selector_update_interval=20,
        sharpe_window=30,
        global_max_exposure=0.80,
        global_cash_buffer=100000.0,
        risk_free_bars=7,
        printlog=False,
        # Global market awareness
        gm_breadth_window=20,
        gm_momentum_window=10,
        gm_corr_window=30,
        # Risk regime
        rr_on_breadth=0.60,
        rr_off_breadth=0.38,
        rr_confirm_bars=3,
        # Pair trading
        pt_corr_window=40,
        pt_spread_window=30,
        pt_corr_thresh=0.55,
        pt_z_entry=1.8,
        pt_z_exit=0.4,
        pt_maxopen_pairs=3, # max simultaneous pair trades
        pt_notional_frac=0.12, # fraction of portfolio per pair leg
    )

    def __init__(self):
        config_names  = list(CONFIGS.keys())
        self.selector = BayesianModelSelector(
            config_names = config_names,
            update_interval = self.p.selector_update_interval,
            sharpe_window = self.p.sharpe_window,
        )

        # Build indicators and signal evaluators
        self.indicators = {}
        self.evaluators = {}
        for name, cfg in CONFIGS.items():
            lb = cfg['bayesian_lookback']
            mw = lb // 3
            ind = {
                'atr': {d: bt.indicators.ATR(d, period=cfg['atr_len']) for d in self.datas},
                'vol_avg': {d: bt.indicators.SimpleMovingAverage(d.volume, period=mw) for d in self.datas},
                'vol_std': {d: bt.indicators.StandardDeviation(d.volume, period=mw) for d in self.datas},
                'sma': {d: bt.indicators.SimpleMovingAverage(d.close, period=cfg['ma_len']) for d in self.datas},
            }
            self.indicators[name] = ind
            self.evaluators[name] = SignalEvaluator(name, cfg, self.datas, ind)

        # Global market awareness components
        n_series = len(self.datas)
        self.global_monitor = GlobalMarketMonitor(
            n_series = n_series,
            breadth_window = self.p.gm_breadth_window,
            momentum_window = self.p.gm_momentum_window,
            corr_window = self.p.gm_corr_window,
        )
        self.risk_classifier = RiskRegimeClassifier(
            on_breadth_thresh = self.p.rr_on_breadth,
            off_breadth_thresh = self.p.rr_off_breadth,
            confirm_bars = self.p.rr_confirm_bars,
        )
        self.pair_tracker = PairTracker(
            n_series = n_series,
            names = [d._name for d in self.datas],
            corr_window = self.p.pt_corr_window,
            spread_window = self.p.pt_spread_window,
            corr_thresh = self.p.pt_corr_thresh,
            z_entry = self.p.pt_z_entry,
            z_exit = self.p.pt_z_exit,
        )

        # Index map: data._name = position in self.datas
        self.data_index: dict[str, int] = {d._name: i for i, d in enumerate(self.datas)}

        # Open pair trades: key=(sym_long, sym_short), value=dict with entry info
        self.open_pairs: dict[tuple, dict] = {}

        # Single-leg position registry
        self._pos = {}

        # Orders in flight: order.ref -> ('entry'|'exit', data, size_sign, entry_type, config)
        self.pending = {}

        # Deferred label clear (same mechanism as V3 for analyzer compatibility)
        self.clear_label_next_bar = set()
        self.exit_sizes = {}   # sym -> share count at exit signal time

        # Top-level state read by TradeTypeAnalyzer / PositionTypeAnalyzer
        self.state = {d: {'entry_type': ''} for d in self.datas}

        # Daily budget tracking
        self.day_notional = 0.0
        self.day_symbol = {}
        self.last_date = None
        self.start_value = None

        # Cooldown tracking — sym -> bar number of last exit
        self.last_exit_bar: dict = {}

    
    def reset_day(self):
        cur = self.datas[0].datetime.date(0)
        if self.last_date != cur:
            self.day_notional = 0.0
            self.day_symbol = {}
            self.last_date = cur

    
    def units_from_risk(self, d, cfg):
        atr_val = float(self.indicators[cfg['bayesian_lookback'] and 'LB40' or 'LB40']['atr'][d][0])
        # Use the evaluator's own ATR
        return 0  # placeholder — see usage below

    
    def atr(self, config_name, d):
        try:
            return float(self.indicators[config_name]['atr'][d][0])
        except Exception:
            return 0.0

    
    def check_exit(self, sym, d, bar):
        """
        Returns True if the open position on `d` should be closed this bar.
        Also updates the trailing stop in place.

        Exit conditions (any one triggers):
          1. max_hold_bars exceeded (hard cap — always enforced)
          2. Severe breach (close beyond trail by severe_k)
          3. Two-bar trailing stop breach
          4. Profit-lock: position has > 2× initial risk profit AND held
             > half of max_hold_bars = tighten trail multiplier to 0.5×
             so gains are protected and capital is freed for new signals.
        """
        pos = self._pos.get(sym)
        if pos is None:
            return False

        close = float(d.close[0])
        side = pos['size_sign']
        trail = pos['trail']
        cfg = CONFIGS[pos['config']]
        atr_val = self.atr(pos['config'], d)
        severe_k = cfg['severe_breach_mult'] * atr_val
        bars_held = bar - pos['entry_bar']

        # 1. Hard max holding period — always exit regardless of profit
        if bars_held >= cfg['max_hold_bars']:
            return True

        # 4. Profit-lock: tighten trail when well into profit and half-time reached
        half_max = cfg['max_hold_bars'] // 2
        entry_px = pos['entry_px']
        unrealised_pct = side * (close - entry_px) / entry_px if entry_px > 0 else 0.0
        atr_k = cfg['atr_k']
        if bars_held >= half_max and unrealised_pct > 0.10:
            # Tighten: use 50% of normal ATR multiplier
            atr_k = cfg['atr_k'] * 0.50

        breached = (side > 0 and close < trail) or (side < 0 and close > trail)
        severe = (side > 0 and close < trail - severe_k) or (side < 0 and close > trail + severe_k)

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
            # Advance trail using (possibly tightened) atr_k
            if side > 0:
                pos['trail'] = max(trail, close - atr_k * atr_val)
            else:
                pos['trail'] = min(trail, close + atr_k * atr_val)

        return False

    def next(self):
        bar = len(self)
        self.reset_day()

        # Deferred label clear — runs one bar AFTER notify_trade so the analyzer
        # always sees the correct label when notify_trade fires.
        for d in list(self.clear_label_next_bar):
            self.state[d]['entry_type'] = ''
        self.clear_label_next_bar.clear()

        # Re-assert labels for all currently open positions every bar.
        for sym, pos_info in self._pos.items():
            d_match = next((d for d in self.datas if d._name == sym), None)
            if d_match is not None:
                self.state[d_match]['entry_type'] = pos_info['entry_type']

        if self.start_value is None:
            try:
                self.start_value = float(self.broker.getvalue())
            except Exception:
                self.start_value = 1.0

        try:
            bars_left = self.datas[0].buflen() - bar
        except Exception:
            bars_left = 999_999

        # End-of-period forced liquidation 
        if bars_left <= self.p.risk_free_bars:
            for d in self.datas:
                pos = self.getposition(d).size
                if pos > 0:
                    self.sell(data=d, size=pos)
                elif pos < 0:
                    self.buy(data=d, size=abs(pos))
            return

        #  STEP 1: Update global market awareness
        closes = [float(d.close[0]) for d in self.datas]
        self.global_monitor.update(closes)
        self.risk_classifier.update(self.global_monitor)
        self.pair_tracker.update(closes)

        gm = self.global_monitor
        rc = self.risk_classifier
        pt = self.pair_tracker
        regime  = rc.regime
        r_score = rc.risk_score
        pt_bias = rc.pair_trade_bias()

        if self.p.printlog:
            print(f"[Bar {bar}] Regime={regime} Score={r_score:.3f} "
                  f"Breadth={gm.market_breadth:.2f} Mom={gm.avg_momentum:.4f} "
                  f"Disp={gm.dispersion:.4f} AvgCorr={gm.avg_corr:.3f}")

        self.selector.step(bar)
        weights = self.selector.get_weights()

        #  STEP 2: Pair trade management (exits first, then new entries)
        self.manage_pair_exots(bar)
        self.scan_paor_entries(bar, pt_bias)

        #  STEP 3: Cross-sectional performance for MR laggard detection
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

        #  STEP 4: Single-leg directional trading (risk-regime aware)
        for d in self.datas:
            sym = d._name
            has_pos = sym in self._pos
            broker_sz = self.getposition(d).size

            # Stale position guard
            if has_pos and broker_sz == 0:
                pos_info = self._pos.pop(sym)
                entry_type = pos_info['entry_type']
                config = pos_info['config']
                self.selector.record_trade(config, 0.0, False)
                self.evaluators[config].notify_result(d, entry_type, False)
                self.clear_label_next_bar.add(d)
                has_pos = False

            # EXIT
            if has_pos:
                if self.check_exit(sym, d, bar):
                    broker_pos = self.getposition(d).size
                    if broker_pos > 0:
                        o = self.sell(data=d, size=broker_pos)
                    elif broker_pos < 0:
                        o = self.buy(data=d, size=abs(broker_pos))
                    else:
                        o = None

                    if broker_pos != 0:
                        self.exit_sizes[sym] = abs(broker_pos)
                        self.clear_label_next_bar.add(d)

                    if o:
                        pos_info = self._pos[sym]
                        self.pending[o.ref] = (
                            'exit', d, pos_info['size_sign'],
                            pos_info['entry_type'], pos_info['config'],
                        )
                continue   # never enter on same bar as exit signal

            # Skip if already one leg of an open pair trade
            if self.sym_in_open_pair(sym):
                continue

            # ENTRY — Cooldown
            last_exit = self.last_exit_bar.get(sym, -9999)
            min_cooldown = min(cfg_val['cooldown'] for cfg_val in CONFIGS.values())
            if bar - last_exit < min_cooldown:
                continue

            best_sign = 0
            best_etype = ''
            best_config = ''
            best_weight = -1.0

            for name, ev in self.evaluators.items():
                if not ev.ready(bar):
                    continue
                # Pass global awareness into each evaluator
                sign, etype = ev.get_signal(
                    d, weights[name], perf, laggards,
                    risk_classifier=rc,
                    global_monitor=gm,
                )
                if sign == 0:
                    continue
                if weights[name] > best_weight:
                    best_sign, best_etype, best_config, best_weight = sign, etype, name, weights[name]

            if best_sign == 0:
                continue

            cfg = CONFIGS[best_config]
            atr_val = self.atr(best_config, d)
            price = float(d.close[0])
            if atr_val <= 0 or price <= 0:
                continue

            # Portfolio-adaptive risk notional
            # Use a fixed fraction of current portfolio value so sizing scales with account growth and doesn't 
            # become stale as prices rise across dataset periods.  Floor at the original fixed value.
            risk_notional = max(
                cfg['risk_notional_per_leg'],
                port_val * cfg.get('risk_notional_frac', 0.055),
            )
            raw_units = max(0, math.floor(risk_notional / atr_val / price))
            units = min(cfg['entry_units_cap'], raw_units)
            units = max(1, min(units, cfg['max_units_per_symbol']))

            # Apply risk-regime position scaling
            scale = rc.entry_scale(base_frac=1.0)
            # For longs, additionally scale by risk_score to reduce size in deteriorating markets; shorts are less curtailed.
            if best_sign > 0:
                units = max(1, math.floor(units * scale))
            else:
                # Shorts: allow slightly larger size in risk-off (trend continuation)
                short_scale = 1.0 + (0.20 if rc.is_risk_off() else 0.0)
                units = max(1, math.floor(units * min(1.0, scale * short_scale)))

            notional = price * units

            sym_spent = self.day_symbol.get(sym, 0.0)
            sym_cap = cfg['day_symbol_cap_notional'] * port_val
            if (
                sym_spent + notional > sym_cap
                or self.day_notional + notional > port_val * cfg['day_budget_frac']
                or notional > global_headroom
                or cash_now < notional + self.p.global_cash_buffer
            ):
                continue

            if best_sign > 0:
                o = self.buy(data=d, size=units)
            else:
                o = self.sell(data=d, size=units)

            self.state[d]['entry_type'] = best_etype
            self.selector.record_capital(best_config, notional)
            self.day_notional += notional
            self.day_symbol[sym] = sym_spent + notional
            global_headroom -= notional
            cash_now -= notional

            if o:
                self.pending[o.ref] = ('entry', d, best_sign, best_etype, best_config)

    
    #  PAIR TRADE HELPERS
    def sym_in_open_pair(self, sym: str) -> bool:
        """True if sym is already one leg of an open pair trade."""
        for (sl, ss) in self.open_pairs:
            if sym in (sl, ss):
                return True
        return False

    def manage_pair_exots(self, bar: int):
        """Close any pair trades whose spread has mean-reverted or correlation broke."""
        to_close = []
        for key, info in list(self.open_pairs.items()):
            sym_long, sym_short = key
            i = self.data_index[sym_long]
            j = self.data_index[sym_short]

            should_exit = self.pair_tracker.should_exit_pair(i, j)
            max_bars = info.get('max_bars', 120)
            aged_out = (bar - info['entry_bar']) >= max_bars

            if should_exit or aged_out:
                to_close.append(key)

        for key in to_close:
            info = self.open_pairs.pop(key)
            sym_long, sym_short = key
            d_long = next((d for d in self.datas if d._name == sym_long),  None)
            d_short = next((d for d in self.datas if d._name == sym_short), None)

            if d_long and self.getposition(d_long).size > 0:
                self.sell(data=d_long,  size=self.getposition(d_long).size)
            if d_short and self.getposition(d_short).size < 0:
                self.buy(data=d_short,  size=abs(self.getposition(d_short).size))

            if self.p.printlog:
                reason = 'spread_reverted_or_corr_break'
                print(f"  [Pair EXIT] {sym_long}↑/{sym_short}↓  reason={reason}")

    def scan_paor_entries(self, bar: int, pt_bias: str):
        """
        Scan all N*(N-1)/2 pairs for trade opportunities.
        Respects:
          - max open pair trades (self.p.pt_maxopen_pairs)
          - neither leg already in a single-leg position or another pair
          - sufficient cash and headroom
        """
        if len(self.open_pairs) >= self.p.pt_maxopen_pairs:
            return

        port_val = float(self.broker.getvalue())
        cash_now = float(self.broker.getcash())
        leg_notional = port_val * self.p.pt_notional_frac

        # Build index = data lookup
        idx2data = {i: d for i, d in enumerate(self.datas)}

        scored_pairs = []
        for i, j in combinations(range(len(self.datas)), 2):
            sig = self.pair_tracker.get_pair_signal(i, j, risk_bias=pt_bias)
            if not sig['tradeable']:
                continue
            # Rank by absolute spread Z (larger = more attractive)
            scored_pairs.append((abs(sig['spread_z']), i, j, sig))

        scored_pairs.sort(reverse=True)   # best spread first

        for _, i, j, sig in scored_pairs:
            if len(self.open_pairs) >= self.p.pt_maxopen_pairs:
                break

            sym_long = self.pair_tracker.names[sig['long_leg']]
            sym_short = self.pair_tracker.names[sig['short_leg']]

            # Skip if either leg is busy
            if sym_long in self._pos or sym_short in self._pos:
                continue
            if self.sym_in_open_pair(sym_long) or self.sym_in_open_pair(sym_short):
                continue

            d_long = idx2data[sig['long_leg']]
            d_short = idx2data[sig['short_leg']]

            price_long = float(d_long.close[0])
            price_short = float(d_short.close[0])
            if price_long <= 0 or price_short <= 0:
                continue

            units_long  = max(1, math.floor(leg_notional / price_long))
            units_short = max(1, math.floor(leg_notional / price_short))
            total_cash  = units_long * price_long + units_short * price_short

            if cash_now < total_cash + self.p.global_cash_buffer:
                continue

            o_long  = self.buy(data=d_long,  size=units_long)
            o_short = self.sell(data=d_short, size=units_short)

            key = (sym_long, sym_short)
            self.open_pairs[key] = {
                'entry_bar': bar,
                'entry_z': sig['spread_z'],
                'corr': sig['corr'],
                'max_bars': 90,
                'units_long': units_long,
                'units_short': units_short,
            }
            cash_now -= total_cash

            if self.p.printlog:
                print(f"  [Pair ENTRY] {sym_long}↑/{sym_short}↓  "
                      f"corr={sig['corr']:.3f} z={sig['spread_z']:.2f} "
                      f"bias={pt_bias}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        ref = order.ref
        if ref not in self.pending:
            return

        kind, d, size_sign, entry_type, config = self.pending.pop(ref)
        sym = d._name
        cfg = CONFIGS[config]

        if kind == 'exit':
            if order.status == order.Completed:
                pos_info = self._pos.pop(sym, None)
                if pos_info:
                    exit_price = float(order.executed.price)
                    side = pos_info['size_sign']
                    entry_px = pos_info['entry_px']
                    pnl_pct = side * (exit_price - entry_px) / entry_px if entry_px > 0 else 0.0
                    won = pnl_pct > 0
                    self.selector.record_trade(config, pnl_pct, won)
                    self.evaluators[config].notify_result(d, entry_type, won)
                    # Record cooldown so we don't immediately re-enter
                    self.last_exit_bar[sym] = len(self)
            return

        # Entry order
        if order.status != order.Completed:
            if d in self.state:
                self.state[d]['entry_type'] = ''
            return

        price = float(order.executed.price)
        atr_val = self.atr(config, d)
        if size_sign > 0:
            # Long: trail starts below entry price
            trail = price - cfg['atr_k'] * atr_val
        else:
            # Short: trail starts above entry price
            trail = price + cfg['atr_k'] * atr_val

        self._pos[sym] = {
            'size_sign': size_sign,
            'entry_px': price,
            'entry_type': entry_type,
            'entry_bar': len(self),
            'trail':trail,
            'breach_count': 0,
            'config': config,
        }
        if d in self.state:
            self.state[d]['entry_type'] = entry_type
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        data = trade.data
        sym  = data._name
        label_in_state = self.state[data]['entry_type'] if data in self.state else 'NO_STATE_KEY'
        if data in self.state:
            self.clear_label_next_bar.add(data)

    def stop(self):
        pass