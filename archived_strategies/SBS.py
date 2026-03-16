'''
python main.py --strategy SBS --data-dir DATA/PART1 --output-dir output/SBS

SuperBayesian: Dirichlet-weighted ensemble of 4 Bayesian regime detectors
Lookbacks: 40, 60, 90, 120 days
Each lookback group has its own BayesianRegimeDetector and BayesianSignalStrength trackers.
Dirichlet weights are updated based on each group's recent predictive accuracy.
Final regime probabilities = Dirichlet-weighted average across all 4 groups.
Trade classification: mean_reversion_long, mean_reversion_short,
                      trend_following_long, trend_following_short
'''

'''
{
  "final_value": 1021028.100772,
  "bankrupt": false,
  "bankrupt_date": null,
  "open_pnl_pd_ratio": 0.9336542583075648,
  "true_pd_ratio": 0.8940790422155553,
  "activity_pct": 91.0,
  "end_policy": "liquidate",
  "s_mult": 2.0
}
=== Trade Analyzer Stats ===
Total closed trades: 12
Wins: 7
Losses: 5

=== Trade Breakdown by Strategy Type ===

Mean Reversion Trades Long:
  Total trades: 1
  Wins: 1
  Losses: 0
  Total PnL: $340.38
  Win Rate: 100.0%

Trend Following Trades Long:
  Total trades: 10
  Wins: 5
  Losses: 5
  Total PnL: $19237.75
  Win Rate: 50.0%

Mean Reversion Trades Short:
  Total trades: 0
  Wins: 0
  Losses: 0
  Total PnL: $0.00

Trend Following Trades Shorts:
  Total trades: 1
  Wins: 1
  Losses: 0
  Total PnL: $2002.00
  Win Rate: 100.0%

--- Portfolio Total ---
  Total positions : 12
  Total wins      : 7
  Overall PnL     : $21,580.13
  Overall win rate: 58.3%

Symbol     Type                     EntryDate    ExitDate              PnL  W/L
--------------------------------------------------------------------------------------------------------------
series_4   trend_following_long     2070-02-22   2072-08-13      3,444.00  WIN
series_3   trend_following_long     2070-02-22   2072-08-20        -51.38  LOSS
series_1   trend_following_long     2070-02-23   2072-08-20     -5,810.00  LOSS
series_5   trend_following_long     2070-02-24   2072-08-13     16,317.00  WIN
series_2   trend_following_long     2070-02-24   2072-08-20     -3,062.50  LOSS
series_6   trend_following_long     2070-03-04   2072-08-13         84.00  WIN
series_9   trend_following_long     2070-03-10   2072-08-13      6,146.84  WIN
series_7   trend_following_long     2070-03-10   2072-08-20        -92.96  LOSS
series_8   trend_following_long     2070-03-13   2072-08-13      2,318.75  WIN
series_10  trend_following_short    2070-03-15   2072-08-20      2,002.00  WIN
series_6   trend_following_long     2072-08-14   2072-08-20        -56.00  LOSS
series_5   mean_reversion_long      2072-08-15   2072-08-18        340.38  WIN
'''

import backtrader as bt
import math
import numpy as np
from scipy import stats
from collections import defaultdict, deque

class BayesianRegimeDetector(object):
    """Classify market into UPTREND / DOWNTREND / SIDEWAYS using Beta-Binomial model.

    One instance per lookback group. The momentum window is lookback // 3.
    """

    def __init__(self, lookback=90, alpha=2.0, beta=2.0):
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
            recent_change = (
                (self.price_history[-1] - self.price_history[-mw])
                / self.price_history[-mw]
            )
            if abs(recent_change) > 0.02:
                self.trend_observations += 1
            else:
                self.mr_observations += 1

    def get_regime(self):
        mw = self.lookback // 3
        if len(self.price_history) < mw:
            return {'uptrend': 0.33, 'downtrend': 0.33, 'sideways': 0.34}

        recent_20 = list(self.price_history)[-mw:]
        early_20 = (
            list(self.price_history)[-(mw * 2):-mw]
            if len(self.price_history) >= mw * 2
            else recent_20
        )
        recent_avg = np.mean(recent_20)
        early_avg = np.mean(early_20)

        total = self.trend_observations + self.mr_observations
        if total < 10:
            return {'uptrend': 0.33, 'downtrend': 0.33, 'sideways': 0.34}

        trendy_alpha = self.alpha + self.trend_observations
        mr_alpha = self.beta + self.mr_observations
        trendy_prob = trendy_alpha / (trendy_alpha + mr_alpha)
        sideways_prob = 1 - trendy_prob

        if recent_avg > early_avg:
            uptrend_prob = trendy_prob * 0.7
            downtrend_prob = trendy_prob * 0.3
        else:
            uptrend_prob = trendy_prob * 0.3
            downtrend_prob = trendy_prob * 0.7

        return {
            'uptrend': uptrend_prob,
            'downtrend': downtrend_prob,
            'sideways': sideways_prob,
        }

class BayesianSignalStrength(object):
    """Beta-Binomial tracker for historical signal accuracy.

    Prior belief = 50%; updated with observed win/loss outcomes.
    """

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

    def get_credible_interval(self, alpha=1.0, beta=1.0, confidence=0.95):
        if self.total_trades == 0:
            return (0.0, 1.0)
        alphapost = alpha + self.success
        betapost = beta + (self.total_trades - self.success)
        lower = stats.beta.ppf((1 - confidence) / 2, alphapost, betapost)
        upper = stats.beta.ppf((1 + confidence) / 2, alphapost, betapost)
        return (lower, upper)

class DirichletWeightManager(object):
    """Maintains and updates Dirichlet concentration parameters for K groups.

    The Dirichlet distribution is a conjugate prior over a K-simplex (a set of
    K non-negative weights that sum to 1).  Each time a group makes a
    successful regime call we increment its concentration parameter; otherwise
    we add a small decay to keep weights adaptive.

    Expected weights are simply  alpha_k / sum(alpha).
    """

    def __init__(self, n_groups=4, prior_concentration=2.0):
        self.n = n_groups
        # Symmetric Dirichlet prior — equal weight to all groups initially
        self.alphas = np.full(n_groups, prior_concentration, dtype=float)

    def update(self, group_idx, success: bool, decay: float = 0.02):
        """Update concentration for group_idx after an observation.

        Args:
            group_idx: Index of the lookback group (0-3).
            success:   True if group's regime call was profitable.
            decay:     Small amount subtracted from all other groups to keep
                       weights adaptive over time (prevents staleness).
        """
        if success:
            self.alphas[group_idx] += 1.0
        else:
            # Slight decay on the unsuccessful group only
            self.alphas[group_idx] = max(0.5, self.alphas[group_idx] - 0.1)

        # Apply global decay to all groups (keeps weights from growing unbounded)
        self.alphas = np.maximum(0.5, self.alphas * (1 - decay) + decay * 0.5)

    def get_weights(self):
        """Return normalised expected weights E[theta] = alpha / sum(alpha)."""
        return self.alphas / self.alphas.sum()

    def __repr__(self):
        w = self.get_weights()
        return (
            f"DirichletWeights(alphas={np.round(self.alphas, 3)}, "
            f"weights={np.round(w, 3)})"
        )

# Each group defines the complete parameter configuration for its lookback.
# Groups are indexed 0..3 corresponding to lookbacks [40, 60, 90, 120].
PARAM_GROUPS = [
    # ── Group 0: lookback 40 (Short) ──────────────────────────────────────────
    dict(
        bayesian_lookback=40,
        atr_len=10,
        atr_k=2.1,
        meanrev_window=5,
        top_pct=0.20,
        volume_z=0.8,
        short_return=-0.015,
        bounce_pct=0.015,
        up_window=5,
        up_days_min=3,
        breakout_lookback=6,
        ma_len=15,
        vol_z_min=0.6,
        cooldown=3,
        risk_free_bars=7,
        end_taper_bars=14,
        min_confidence_mr=0.44,
        min_confidence_trend=0.44,
        sideways_threshold=0.35,
        trending_threshold=0.55,
    ),
    # ── Group 1: lookback 60 (Medium-Short) ───────────────────────────────────
    dict(
        bayesian_lookback=60,
        atr_len=14,
        atr_k=2.4,
        meanrev_window=7,
        top_pct=0.20,
        volume_z=0.75,
        short_return=-0.020,
        bounce_pct=0.020,
        up_window=7,
        up_days_min=4,
        breakout_lookback=10,
        ma_len=21,
        vol_z_min=0.7,
        cooldown=4,
        risk_free_bars=10,
        end_taper_bars=18,
        min_confidence_mr=0.42,
        min_confidence_trend=0.42,
        sideways_threshold=0.30,
        trending_threshold=0.50,
    ),
    # ── Group 2: lookback 90 (Medium-Long) ────────────────────────────────────
    dict(
        bayesian_lookback=90,
        atr_len=21,
        atr_k=2.7,
        meanrev_window=7,
        top_pct=0.25,
        volume_z=0.7,
        short_return=-0.018,
        bounce_pct=0.018,
        up_window=7,
        up_days_min=4,
        breakout_lookback=14,
        ma_len=30,
        vol_z_min=1.0,
        cooldown=5,
        risk_free_bars=10,
        end_taper_bars=21,
        min_confidence_mr=0.40,
        min_confidence_trend=0.40,
        sideways_threshold=0.27,
        trending_threshold=0.45,
    ),
    # ── Group 3: lookback 120 (Long) ──────────────────────────────────────────
    dict(
        bayesian_lookback=120,
        atr_len=30,
        atr_k=3.0,
        meanrev_window=30,
        top_pct=0.10,
        volume_z=0.9,
        short_return=-0.030,
        bounce_pct=0.025,
        up_window=20,
        up_days_min=13,
        breakout_lookback=30,
        ma_len=75,
        vol_z_min=0.6,
        cooldown=10,
        risk_free_bars=14,
        end_taper_bars=30,
        min_confidence_mr=0.38,
        min_confidence_trend=0.38,
        sideways_threshold=0.25,
        trending_threshold=0.40,
    ),
]

LOOKBACKS = [pg['bayesian_lookback'] for pg in PARAM_GROUPS]   # [40, 60, 90, 120]
N_GROUPS = len(PARAM_GROUPS)

class SuperBayesianStrategy(bt.Strategy):
    """
    Super Bayesian strategy with Dirichlet-weighted ensemble of 4 lookback groups.

    Architecture
    ────────────
    • 4 BayesianRegimeDetectors  (one per lookback group)
    • 4 × 2 BayesianSignalStrength trackers (MR + TF per group, per symbol)
    • 1 DirichletWeightManager per symbol  (tracks which lookback is most reliable)

    Regime Aggregation
    ──────────────────
    Each group produces a regime distribution {uptrend, downtrend, sideways}.
    Final distribution = Dirichlet-weighted linear combination of all 4 groups.

    Signal Filtering
    ────────────────
    • Mean reversion signals must pass accuracy threshold from the *leading group*
      (highest Dirichlet weight) for MR.
    • Trend following signals must pass accuracy threshold from the *leading group*
      for TF.

    Trade Classification
    ────────────────────
    Every trade is tagged with one of:
        mean_reversion_long  | mean_reversion_short
        trend_following_long | trend_following_short
    The tag is stored on the trade object so the TradeTypeAnalyzer in main.py
    can classify it correctly.
    """

    params = dict(
        # Portfolio management (shared across all groups)
        cash_frac=0.55,
        cash_buffer=100000.0,
        max_exposure_frac=0.80,
        day_budget_frac=0.30,
        day_symbol_cap_notional=0.40,
        max_units_per_symbol=4500,
        entry_units_cap=700,
        risk_notional_per_leg=60000.0,
        severe_breach_mult=0.60,
        two_bar_exit=True,
        long_return=0.0,

        # Dirichlet prior concentration (symmetric)
        dirichlet_prior=2.0,
        # Decay rate for Dirichlet weight adaptation
        dirichlet_decay=0.02,

        # Regime learning / signal learning toggles
        regime_learning_enabled=True,
        signal_learning_enabled=True,

        printlog=False,

        # COMP396 injected config (do not edit)
        _comp396=None,
    )

    def __init__(self):
        # Shared indicators (use the largest lookback as baseline)
        max_lb = max(LOOKBACKS)       # 120
        max_mw = max_lb // 3         # 40

        self.atr = {}
        self.vol_avg = {}
        self.vol_std = {}
        self.sma = {}       # keyed (d, group_idx)
        self.highest = {}   # keyed (d, group_idx)
        self.lowest = {}    # keyed (d, group_idx)

        for d in self.datas:
            # ATR and volume stats — use largest window to cover all groups
            self.atr[d] = bt.indicators.ATR(d, period=max(pg['atr_len'] for pg in PARAM_GROUPS))
            self.vol_avg[d] = bt.indicators.SimpleMovingAverage(d.volume, period=max_mw)
            self.vol_std[d] = bt.indicators.StandardDeviation(d.volume, period=max_mw)

            # Per-group indicators that depend on group-specific lengths
            for gi, pg in enumerate(PARAM_GROUPS):
                lb = pg['bayesian_lookback']
                mw = lb // 3
                self.sma[(d, gi)] = bt.indicators.SimpleMovingAverage(
                    d.close, period=int(pg['ma_len'])
                )
                self.highest[(d, gi)] = bt.indicators.Highest(
                    d.close, period=int(pg['breakout_lookback'])
                )
                self.lowest[(d, gi)] = bt.indicators.Lowest(
                    d.close, period=mw
                )

        # Per-symbol Bayesian state (4 detectors + signal trackers)
        self.regime_detectors = {}       # (d, gi) → BayesianRegimeDetector
        self.mr_signal_strength = {}     # (d, gi) → BayesianSignalStrength
        self.tf_signal_strength = {}     # (d, gi) → BayesianSignalStrength
        self.dirichlet = {}              # d → DirichletWeightManager

        for d in self.datas:
            self.dirichlet[d] = DirichletWeightManager(
                n_groups=N_GROUPS,
                prior_concentration=float(self.p.dirichlet_prior),
            )
            for gi, pg in enumerate(PARAM_GROUPS):
                self.regime_detectors[(d, gi)] = BayesianRegimeDetector(
                    lookback=pg['bayesian_lookback']
                )
                self.mr_signal_strength[(d, gi)] = BayesianSignalStrength(
                    prior_accuracy=0.48
                )
                self.tf_signal_strength[(d, gi)] = BayesianSignalStrength(
                    prior_accuracy=0.48
                )

        # Per-symbol trade state
        self.state = {
            d: dict(
                trail=None,
                cool_until=-math.inf,
                breach_count=0,
                entry_px=None,
                entry_units=0,
                entry_type='',
                entry_bar=0,
                entry_group=-1,          # which lookback group triggered the trade
                regime_history=deque(maxlen=40),
            )
            for d in self.datas
        }

        self.prev_pos = {d: 0.0 for d in self.datas}
        self.day_spent_notional = 0.0
        self.last_calendar_date = None
        self.day_symbol_spent = {}
        self.start_value = None
        # Persistent map: trade.ref → entry_type
        # trade.ref is the same integer on both the open and close notify_trade
        # calls, unlike the trade object itself which is recreated at close.
        self._trade_entry_type = {}

    def ready(self):
        """Require enough bars for the smallest lookback group to warm up."""
        min_lb = min(LOOKBACKS)
        return len(self) >= int(min_lb * 0.67)

    def reset_day(self):
        cur = self.datas[0].datetime.date(0)
        if self.last_calendar_date != cur:
            self.day_spent_notional = 0.0
            self.last_calendar_date = cur
            self.day_symbol_spent = {}

    def units_from_risk(self, d, atr_len):
        """ATR-based position sizing using per-group ATR length."""
        atr_val = float(self.atr[d][0]) if self.atr[d] is not None else 0.0
        price = float(d.close[0])
        if atr_val <= 0 or price <= 0:
            return 0
        return max(0, math.floor(self.p.risk_notional_per_leg / atr_val / price))

    def get_aggregate_regime(self, d):
        """Compute Dirichlet-weighted blend of 4 group regime distributions.

        Returns:
            dict with keys 'uptrend', 'downtrend', 'sideways' (sum ≈ 1)
            and 'weights' (the 4 Dirichlet weights used).
        """
        weights = self.dirichlet[d].get_weights()   # shape (4,)
        agg = {'uptrend': 0.0, 'downtrend': 0.0, 'sideways': 0.0}

        for gi in range(N_GROUPS):
            regime = self.regime_detectors[(d, gi)].get_regime()
            w = float(weights[gi])
            agg['uptrend'] += w * regime['uptrend']
            agg['downtrend'] += w * regime['downtrend']
            agg['sideways'] += w * regime['sideways']

        agg['weights'] = weights
        return agg

    def check_mr_long(self, d, gi):
        pg = PARAM_GROUPS[gi]
        close = float(d.close[0])
        openp = float(d.open[0])
        mw = pg['bayesian_lookback'] // 3

        vol_avg = float(self.vol_avg[d][0])
        vol_std = float(self.vol_std[d][0]) or 1.0
        volume_z = (float(d.volume[0]) - vol_avg) / vol_std

        window = int(pg['meanrev_window'])
        short_return = (
            (close - float(d.close[-window])) / float(d.close[-window])
            if len(d) > window else 0.0
        )

        low_n = min(float(d.low[-i]) for i in range(1, mw + 1)) if len(d) > mw else close
        dist_from_low = (close - low_n) / low_n if low_n != 0 else 0.0

        signals = {
            'volume_spike': volume_z >= pg['volume_z'],
            'short_return': short_return <= pg['short_return'],
            'green_candle': close > openp,
            'near_low': dist_from_low <= pg['bounce_pct'],
        }
        ok = sum(signals.values()) >= 3
        return ok, signals

    def check_mr_short(self, d, gi):
        pg = PARAM_GROUPS[gi]
        close = float(d.close[0])
        openp = float(d.open[0])
        mw = pg['bayesian_lookback'] // 3

        vol_avg = float(self.vol_avg[d][0])
        vol_std = float(self.vol_std[d][0]) or 1.0
        volume_z = (float(d.volume[0]) - vol_avg) / vol_std

        window = int(pg['meanrev_window'])
        ret_n = (
            (close - float(d.close[-window])) / float(d.close[-window])
            if len(d) > window else 0.0
        )

        high_n = max(float(d.high[-i]) for i in range(1, mw + 1)) if len(d) > mw else close
        dist_from_high = (high_n - close) / high_n if high_n != 0 else 0.0

        signals = {
            'volume_climactic': volume_z >= 1.0,
            'overbought': ret_n >= 0.04,
            'red_candle': close < openp,
            'near_high': dist_from_high <= 0.02,
        }
        ok = sum(signals.values()) >= 3
        return ok, signals

    def check_tf_long(self, d, gi):
        pg = PARAM_GROUPS[gi]
        close = float(d.close[0])

        ups = sum(
            1 for k in range(1, int(pg['up_window']) + 1)
            if len(d) > k and float(d.close[0]) > float(d.close[-k])
        )
        up_ok = ups >= int(pg['up_days_min'])

        vol_avg = float(self.vol_avg[d][0])
        vol_std = float(self.vol_std[d][0]) or 1.0
        vol_z = (float(d.volume[0]) - vol_avg) / vol_std
        vol_ok = vol_z >= pg['vol_z_min']

        brk_lb = int(pg['breakout_lookback'])
        high_n = max(float(d.high[-k]) for k in range(1, brk_lb + 1)) if len(d) > brk_lb else close
        brk_ok = close >= high_n

        sma_ind = self.sma[(d, gi)]
        trend_ok = True
        if sma_ind is not None and len(sma_ind) >= 3:
            s0, s1, s2 = float(sma_ind[0]), float(sma_ind[-1]), float(sma_ind[-2])
            trend_ok = close > s0 and (s0 > s1 > s2)

        signals = {
            'ups': up_ok,
            'volume': vol_ok,
            'breakout_up': brk_ok,
            'sma_trend_up': trend_ok,
        }
        ok = sum(signals.values()) >= 3
        return ok, signals

    def check_tf_short(self, d, gi):
        pg = PARAM_GROUPS[gi]
        close = float(d.close[0])

        downs = sum(
            1 for k in range(1, int(pg['up_window']) + 1)
            if len(d) > k and float(d.close[0]) < float(d.close[-k])
        )
        down_ok = downs >= int(pg['up_days_min'])

        vol_avg = float(self.vol_avg[d][0])
        vol_std = float(self.vol_std[d][0]) or 1.0
        vol_z = (float(d.volume[0]) - vol_avg) / vol_std
        vol_ok = vol_z >= pg['vol_z_min']

        brk_lb = int(pg['breakout_lookback'])
        low_n = min(float(d.low[-k]) for k in range(1, brk_lb + 1)) if len(d) > brk_lb else close
        brk_ok = close <= low_n

        sma_ind = self.sma[(d, gi)]
        trend_ok = True
        if sma_ind is not None and len(sma_ind) >= 3:
            s0, s1, s2 = float(sma_ind[0]), float(sma_ind[-1]), float(sma_ind[-2])
            trend_ok = close < s0 and (s0 < s1 < s2)

        signals = {
            'downs': down_ok,
            'volume': vol_ok,
            'breakdown': brk_ok,
            'sma_trend_down': trend_ok,
        }
        ok = sum(signals.values()) >= 3
        return ok, signals

    def next(self):
        if not self.ready():
            return

        self.reset_day()
        tbar = len(self)

        # Initialise start value once
        if self.start_value is None:
            try:
                self.start_value = float(self.broker.getvalue())
            except Exception:
                self.start_value = 1.0

        # Bars remaining
        try:
            bars_left = self.datas[0].buflen() - len(self)
        except Exception:
            bars_left = 999999

        # ── Force-exit in final risk-free window (use largest group's setting) ──
        max_rfb = max(pg['risk_free_bars'] for pg in PARAM_GROUPS)
        if bars_left <= max_rfb:
            for d in self.datas:
                pos = self.getposition(d).size
                if pos > 0:
                    self.sell(data=d, size=pos)
                elif pos < 0:
                    self.buy(data=d, size=abs(pos))
            return

        # ── Update regime detectors for every symbol ───────────────────────────
        for d in self.datas:
            close = float(d.close[0])
            for gi in range(N_GROUPS):
                self.regime_detectors[(d, gi)].update(close)

        # ── Rank symbols for mean reversion (use medium lookback = group 2) ────
        ref_pg = PARAM_GROUPS[2]   # 90-day group as reference for ranking
        ref_window = int(ref_pg['meanrev_window'])
        perf = {}
        for d in self.datas:
            if len(d) > ref_window and float(d.close[-ref_window]) != 0:
                perf[d] = (
                    (float(d.close[0]) - float(d.close[-ref_window]))
                    / float(d.close[-ref_window])
                )
            else:
                perf[d] = 0.0

        ranked = sorted(perf, key=perf.get)
        n = len(ranked)

        # ── Entry logic ────────────────────────────────────────────────────────
        for d in self.datas:
            state = self.state[d]
            pos = self.getposition(d).size

            if pos != 0 or tbar < state['cool_until']:
                continue

            # Aggregate Dirichlet-weighted regime
            regime = self.get_aggregate_regime(d)
            weights = regime['weights']
            state['regime_history'].append(regime)

            # Determine leading group (highest weight) for each signal type
            leading_gi = int(np.argmax(weights))
            pg_lead = PARAM_GROUPS[leading_gi]

            # ── MEAN REVERSION ─────────────────────────────────────────────────
            if regime['sideways'] >= pg_lead['sideways_threshold']:

                # MR LONG: laggard stock
                bottom_cut = max(1, int(pg_lead['top_pct'] * n))
                laggards = set(ranked[:bottom_cut])

                if d in laggards:
                    # Check signal across all groups; require leading group to pass
                    ok_lead, _ = self.check_mr_long(d, leading_gi)
                    if ok_lead:
                        # Also verify signal accuracy threshold
                        mr_acc = self.mr_signal_strength[(d, leading_gi)].get_accuracy()
                        if mr_acc >= pg_lead['min_confidence_mr']:
                            self.enter_trade(
                                d, size_sign=1,
                                entry_type='mean_reversion_long',
                                group_idx=leading_gi,
                            )
                            continue

                # MR SHORT: recent strong winner, bearish tilt
                if regime['downtrend'] > regime['uptrend']:
                    perf_ref = perf.get(d, 0.0)
                    if perf_ref >= 0.05:
                        ok_lead, _ = self.check_mr_short(d, leading_gi)
                        if ok_lead:
                            mr_acc = self.mr_signal_strength[(d, leading_gi)].get_accuracy()
                            if mr_acc >= pg_lead['min_confidence_mr']:
                                self.enter_trade(
                                    d, size_sign=-1,
                                    entry_type='mean_reversion_short',
                                    group_idx=leading_gi,
                                )
                                continue

            # ── TREND FOLLOWING ────────────────────────────────────────────────
            if (regime['uptrend'] + regime['downtrend']) >= pg_lead['trending_threshold']:

                # TF SHORT: downtrend dominant
                if regime['downtrend'] > regime['uptrend']:
                    ok_lead, _ = self.check_tf_short(d, leading_gi)
                    if ok_lead:
                        tf_acc = self.tf_signal_strength[(d, leading_gi)].get_accuracy()
                        if tf_acc >= pg_lead['min_confidence_trend']:
                            self.enter_trade(
                                d, size_sign=-1,
                                entry_type='trend_following_short',
                                group_idx=leading_gi,
                            )
                            continue

                # TF LONG: uptrend dominant
                else:
                    ok_lead, _ = self.check_tf_long(d, leading_gi)
                    if ok_lead:
                        tf_acc = self.tf_signal_strength[(d, leading_gi)].get_accuracy()
                        if tf_acc >= pg_lead['min_confidence_trend']:
                            self.enter_trade(
                                d, size_sign=1,
                                entry_type='trend_following_long',
                                group_idx=leading_gi,
                            )
                            continue

        # ── Final month: exit profitable longs early ───────────────────────────
        if bars_left <= 21:
            for d in self.datas:
                pos = self.getposition(d).size
                if pos <= 0:
                    continue
                state = self.state[d]
                entry_px = state['entry_px']
                if entry_px and entry_px > 0:
                    pnl_pct = (float(d.close[0]) - entry_px) / entry_px
                    if pnl_pct > 0.005:
                        self.close(data=d)
                        continue

        # ── Final week: liquidate everything ──────────────────────────────────
        if bars_left <= 5:
            for d in self.datas:
                pos = self.getposition(d).size
                if pos > 0:
                    self.close(data=d)
            return

    def enter_trade(self, d, size_sign, entry_type='', group_idx=0):
        """Execute a trade using the parameter set of the given lookback group."""
        pg = PARAM_GROUPS[group_idx]
        price = float(d.close[0])

        # Position sizing
        entry_cap = int(pg['entry_units_cap']) if 'entry_units_cap' in pg else int(self.p.entry_units_cap)
        # Use shared param for entry_units_cap (not in PARAM_GROUPS to keep them clean)
        entry_cap = int(self.p.entry_units_cap)
        atr_units = self.units_from_risk(d, pg['atr_len'])
        units = min(entry_cap, atr_units)
        units = int(max(1, math.floor(units)))
        units *= size_sign

        # Cash constraints
        cash_now = float(self.broker.getcash())
        port_val = float(self.broker.getvalue())
        invested_est = port_val - cash_now
        headroom_val = max(
            0.0,
            float(self.p.max_exposure_frac) * port_val - invested_est,
        )
        units_cash = (
            math.floor(
                max(0.0, cash_now * float(self.p.cash_frac) - float(self.p.cash_buffer))
                / abs(price)
            )
            if price != 0
            else 0
        )
        units_headroom = math.floor(headroom_val / abs(price)) if price != 0 else 0

        units = max(
            -int(self.p.max_units_per_symbol),
            min(units, units_cash, units_headroom, int(self.p.max_units_per_symbol)),
        )

        notional = abs(price * units)
        sym_spent = self.day_symbol_spent.get(d._name, 0.0)
        sym_cap = float(self.p.day_symbol_cap_notional) * port_val

        if (
            sym_spent + notional > sym_cap
            or self.day_spent_notional + notional > port_val * self.p.day_budget_frac
        ):
            return

        # ── Set state BEFORE placing the order ────────────────────────────────
        # notify_order (and notify_trade for the open event) may fire
        # synchronously in some framework wrappers, so entry_type and
        # entry_group must be in state before the order is submitted.
        state = self.state[d]
        atr_k   = pg['atr_k']
        trail   = float(d.close[-1]) if len(d) > 1 else price
        if self.atr[d] is not None:
            atr_val = float(self.atr[d][0])
            if size_sign > 0:
                trail = max(trail, float(d.close[-1]) - atr_k * atr_val)
            else:
                trail = min(trail, float(d.close[-1]) + atr_k * atr_val)

        state['trail']       = trail
        state['breach_count'] = 0
        state['entry_type']  = entry_type
        state['entry_bar']   = len(self)
        state['entry_group'] = group_idx
        state['cool_until']  = len(self) + pg['cooldown']

        # Execute order
        if units > 0:
            self.buy(data=d, size=units)
        else:
            self.sell(data=d, size=abs(units))

        # Update spend tracking
        self.day_spent_notional += notional
        self.day_symbol_spent[d._name] = sym_spent + notional

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status != order.Completed:
            return

        d = order.data
        prev_pos = self.prev_pos.get(d, 0.0)
        new_pos  = self.getposition(d).size
        self.prev_pos[d] = new_pos

        opening_long  = order.isbuy()  and prev_pos <= 0 and new_pos > 0
        opening_short = order.issell() and prev_pos >= 0 and new_pos < 0

        if not (opening_long or opening_short):
            return

        st = self.state[d]
        st['entry_px']    = float(order.executed.price)
        st['entry_units'] = abs(float(order.executed.size))

        entry_type = st['entry_type']
        group_idx  = st['entry_group']

        # Dirichlet update at position-open time, based on regime match
        if group_idx >= 0 and self.p.regime_learning_enabled:
            is_mr = 'mean_reversion' in entry_type
            regime = self.get_aggregate_regime(d)
            regime_match = (
                regime['sideways'] >= 0.25 if is_mr
                else (regime['uptrend'] + regime['downtrend']) >= 0.40
            )
            self.dirichlet[d].update(
                group_idx=group_idx,
                success=regime_match,
                decay=float(self.p.dirichlet_decay),
            )

        # Signal strength: optimistic open-time mark (corrected at close)
        if group_idx >= 0 and self.p.signal_learning_enabled:
            if 'mean_reversion' in entry_type:
                self.mr_signal_strength[(d, group_idx)].update_signal(True, True)
            else:
                self.tf_signal_strength[(d, group_idx)].update_signal(True, True)

    def notify_trade(self, trade):
        d     = trade.data
        state = self.state[d]

        if not trade.isclosed:
            # Position just opened. Store entry_type keyed by trade.ref —
            # this ref is identical on the closing call so we can look it up.
            entry_type = state['entry_type']
            if entry_type:
                self._trade_entry_type[trade.ref] = entry_type
                trade._entry_type = entry_type   # also stamp for analyzer
            return

        # Position closed. Retrieve entry_type via trade.ref (reliable across
        # open/close events), then stamp it onto this close-event trade object
        # so the TradeTypeAnalyzer can read it.
        entry_type = self._trade_entry_type.get(trade.ref, '')
        if entry_type:
            trade._entry_type = entry_type

        group_idx = state['entry_group']
        trade_won = trade.pnl > 0

        # Correct signal strength with real outcome
        if group_idx >= 0 and self.p.signal_learning_enabled and entry_type:
            is_mr   = 'mean_reversion' in entry_type
            tracker = (
                self.mr_signal_strength[(d, group_idx)] if is_mr
                else self.tf_signal_strength[(d, group_idx)]
            )
            tracker.success      = max(0, tracker.success - 1)
            tracker.total_trades = max(0, tracker.total_trades - 1)
            tracker.update_signal(signal_fired=True, trade_won=trade_won)

        # Clean up the ref entry now that the position is fully closed
        self._trade_entry_type.pop(trade.ref, None)

        # Reset per-symbol state
        state['entry_type']   = ''
        state['entry_group']  = -1
        state['entry_px']     = None
        state['entry_units']  = 0
        state['trail']        = None
        state['breach_count'] = 0
