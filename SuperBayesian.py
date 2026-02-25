'''
python main.py --strategy SuperBayesian --data-dir DATA/PART1 --output-dir output/SuperBayes
'''

# Check exit conditions towards end period
# Check parameters for MR and TF Short

'''
{
  "final_value": 1371372.9365009987,
  "bankrupt": false,
  "bankrupt_date": null,
  "open_pnl_pd_ratio": 2.6648096129653633,
  "true_pd_ratio": 2.6248374368252807,
  "activity_pct": 91.9,
  "end_policy": "liquidate",
  "s_mult": 2.0
}
=== Trade Analyzer Stats ===
Total closed trades: 16
Wins: 8
Losses: 8

=== Trade Breakdown by Strategy Type ===

Mean Reversion Trades Long:
  Total trades: 0
  Wins: 0
  Losses: 0
  Total PnL: $0.00

Trend Following Trades Long:
  Total trades: 8
  Wins: 8
  Losses: 0
  Total PnL: $403739.35
  Win Rate: 100.0%

Mean Reversion Trades Short:
  Total trades: 0
  Wins: 0
  Losses: 0
  Total PnL: $0.00

Trend Following Trades Shorts:
  Total trades: 8
  Wins: 0
  Losses: 8
  Total PnL: $-20984.41
  Win Rate: 0.0%
'''

import backtrader as bt
import math
import numpy as np
from scipy import stats
from collections import defaultdict, deque

class BayesianRegimeDetector(object):
    """Classify market into 3 regime types:
    UPTREND: Strong directional movement up -> favor trend-following
    DOWNTREND: Strong directional movement down -> avoid most trades
    SIDEWAYS: Low momentum, mean-reverting environment -> favor mean reversion
    
    Uses Beta-Binomial Bayesian model:
    Updates priors (alpha, beta) with observations of trending vs sideways behavior
    Computes posterior probability distribution over regimes
    Uses momentum (20-bar % change) to classify bars as trending or sideways
    """
    
    def __init__(self, lookback=60, alpha=2.0, beta=2.0):
        """Initialise Bayesian Trend Filter

        Args:
            lookback: Number of bars in price history Defaults to 60.
            alpha: prior count for trend regime Defaults to 2.0.
            beta: prior count for sideways regime Defaults to 2.0.
        """
        
        self.lookback = lookback
        self.alpha = alpha                          # Bayesian Prior for strength of TF belief
        self.beta = beta                            # Bayesian Prior for strength of MR belief
        self.trend_observations = 0                 # Count of bars with high momentum
        self.mr_observations = 0                    # Count of bars with low momentum
        self.price_history = deque(maxlen=lookback) # Keep last N prices
        
    def update(self, close):
        """Add price observation and classify as trending or sideways

        1. Store the closing price
        2. If 20+ bars, calculate 20-bar momentum
        3. If momentum > 2%, classify as TRENDING bar
        4. Otherwise classify as SIDEWAYS bar
        
        Args:
            close: Current bar's closing price
        """
        
        self.price_history.append(close)
        mw = self.lookback//3
        if len(self.price_history)>=mw:
            # Calculate 20 bar momentum as %change      if price[0] = 100 and price[20] = 98, momentum = (100-98)/98
            recent_change = (self.price_history[-1] - self.price_history[-mw]) / self.price_history[-mw]
            if abs(recent_change)>0.02:     # If momentum > 2%, trending else sideways
                self.trend_observations+=1
            else:
                self.mr_observations+=1
                
    def get_regime(self):
        """Bayesian Posterior Probabilities for market regimes
        
        1. Use Beta-Binomial conjugate prior/posterior
        2. Posterior alpha = prior alpha + trending observations
        3. Posterior beta = prior beta + sideways observations
        4. Trend probability = alpha_post / (alpha_post + beta_post)
        5. Split trend probability into uptrend/downtrend based on price direction
        
        Returns:
            Dictionary with probability 
        """
        
        mw = self.lookback//3
        if len(self.price_history)<mw:
            return{'uptrend':0.33, 'downtrend':0.33, 'sideways':0.34}   # Uninformed prior
        
        recent_20 = list(self.price_history)[-mw:]   # Last 20 brs
        early_20 = list(self.price_history)[-(mw*2):-mw] if len(self.price_history)>=mw*2 else recent_20  # Check bars 20-40 if available
        recent_avg = np.mean(recent_20)
        early_avg = np.mean(early_20)
        
        total = self.trend_observations+self.mr_observations
        if total<10:
            return{'uptrend':0.33, 'downtrend':0.33, 'sideways':0.34}   # Uninformed prior
        
        # Update posterior using observations
        trendy_alpha = self.alpha + self.trend_observations         # Prior + trending bars observed
        mr_alpha = self.beta + self.mr_observations                 # Prior + sideways bars observed
        trendy_prob = trendy_alpha / (trendy_alpha + mr_alpha)
        sideways_prob = 1 - trendy_prob
        
        # Split trending prob b/w uptrend and downtrend based on whether prices rose or fell
        if recent_avg>early_avg:    # More likely to be uptrend
            uptrend_prob = trendy_prob*0.7
            downtrend_prob = trendy_prob*0.3
        else:
            uptrend_prob = trendy_prob*0.3
            downtrend_prob = trendy_prob*0.7
            
        return {'uptrend':uptrend_prob, 'downtrend':downtrend_prob, 'sideways':sideways_prob}
    

class BayesianSignalStrength(object):
    """Track historical accuracy of individual trading signals 
    
    Beta-Binomial Bayesian model:
    Prior belief = 50%, use posterior accuracy to filter low-quality signals
    Each time signal fires: record if it was profitable or not and update posterior accuracy estimate
    
    Example: If mean-reversion signal fired 10 times and was profitable 7 times,posterior accuracy ≈ 70%, 
    so we might require 42% posterior accuracy threshold
    """
    
    def __init__(self, prior_accuracy=0.50, window=50):
        # Assume each signal is 50% accurate before checking
        self.prior_accuracy = prior_accuracy
        self.window = window    # Lookback on last N signal outcomes
        self.success = 0        # How many times signal led to winning trade
        self.total_trades = 0   # Total no of trades when signal is fired
        
        self.history = deque(maxlen=window)    # Keep history of last 50 trades (1.0 = win, 0.0 = loss)
        
    def update_signal(self, signal_fried, trade_won):
        """Call after trade close to update signal reliabuility 

        Args:
            signal_fried (Boolean)
            trade_won (Boolean)
        """
        if signal_fried:
            self.success+=1 if trade_won else 0
            self.total_trades+=1
            self.history.append(1.0 if trade_won else 0.0)
            
    def get_accuracy(self, alpha=1.0, beta=1.0):
        """Estimate of accuracy of signal

        Args:
            alpha: Prior strength for success outcomes Defaults to 1.0
            beta:  Prior strength for failure outcomes  Defaults to 1.0
            
        Returns: Probability b/w 0 and 1:
            0 if unreliable 
            1 if reliable
        """
        
        if self.total_trades == 0:      # No trades occured 
            return self.prior_accuracy
        
        # Posterior = Prior + Likelihood
        # alphapost = prior success + observed success
        # betapost = prior failure + observed failure
        alphapost = alpha + self.success
        betapost = beta + (self.total_trades - self.success)
        return alphapost/(alphapost+betapost)   # Expected Value of Beta distribution
    
class BayesianModelSelector(object):
    """
    Manages capital allocation across the 4 Lookbacks(config) (LB40/60/90/120).

    Dirichlet based belief updating:
      Each config has an alpha parameter in a Dirichlet distribution. Weights are computed by normalizing all alphas.
      When a config performs well (positive rolling Sharpe), its alpha is boosted proportionally. 
      Over time the distribution concentrates on the best config.

    Starting state: all alphas = 1.0 -> equal 25% weight each.
    As trading progresses: winning configs accumulate larger alphas → larger weight.

    Parameters:
      config_names     : list of config name strings
      update_interval  : re-score beliefs every N bars (default 20)
      sharpe_window    : rolling window (in trades) for Sharpe calculation (default 30)
    """
    
    def __init__(self, config_names, update_interval=20, sharpe_window=30):
        self.config_names = config_names
        self.n = len(config_names)
        self.update_interval = update_interval
        self.sharpe_window = sharpe_window
        
        self.dirichlet_alphas = {name: 1.0 for name in config_names}
        self.pnl_history = {name: deque(maxlen=sharpe_window) for name in config_names} # Rolling PnL %age history
        self.weight_history = []
        self.bar_count = 0
        self.config_stats = {
            name: {
                "trades":0,
                "wins":0,
                "total_pnl":0,
                "pnl_series":[],
                "Captail_allocated":0
            } for name in config_names
        }
        
    def record_trade(self,config_name, pnl_pct, won):
        if config_name not in self.pnl_history:
            return
        self.pnl_history[config_name].append(pnl_pct)
        s = self.config_stats[config_name]
        s["trades"]+=1
        s["total_pnl"]+=pnl_pct
        s["pnl_series"].append(pnl_pct)
        if won:
            s["wins"]+=1
            
    def record_captial(self,config_name,notional):
        self.config_stats[config_name]["capital_allocated"]+=notional   #Track capital deployed by each confg
        
    def rolling_sharpe(self,config_name):
        """Calculate rolling Sharpe ratio for config (mean/std)

        Returns:
            0.0: if less than 3 trades
            0.1: if std approx 0 but mean > 0 (no volatility)
        """
        history=list(self.pnl_history[config_name])
        if len(history)<3:
            return 0.0
        arr = np.array(history)
        std = np.std(arr)
        if std<1e-9:
            return 0.1 if np.mean(arr)>0 else 0.0
        return float(np.mean(arr)/std)
    
    def update_beliefs(self):
        """
        Normalise Rolling sharpe for each config to sum to 1
        Add boost for better peforming configs
        Floor alpha for a config at 0.1 to ensure the config still trades at lower weight
        """
        sharpes = {name: max(0.0, self.rolling_sharpe(name)) for name in self.config_names}
        total_sharpe = sum(sharpes.values())
        if total_sharpe < 1e-9:     # All configs have -ve sharpe
            return                  
        for name in self.config_names:
            boost = sharpes[name] / total_sharpe
            self.dirichlet_alphas[name] = max(0.1, self.dirichlet_alphas[name] + boost) # Update confid alpha

    def get_weights(self):
        total = sum(self.dirichlet_alphas.values())
        return {name: self.dirichlet_alphas[name] / total for name in self.config_names}    # Convert alpha into weight

    def step(self, bar):
        self.bar_count += 1
        if self.bar_count % self.update_interval == 0:
            self.update_beliefs()       # Update beliefs every "update_interval" bars
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


class ChildStrategy(object):
    """
    Regime Detection PER SYMBOL with Position sizing via ATR-based risk (units_from_risk)
    """
    def __init__(self,name,cfg,datas,broker,indicators):
        self.name = name
        self.cfg = cfg
        self.datas = datas
        self.broker = broker
        self.lb = cfg['bayesian_lookback']
        self.mw = self.lb//3
        
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
        """
        Returns True only after at least 67% of bayesian_lookback bars have elapsed.
        e.g. LB40 starts after bar 26; LB120 starts after bar 80.
        """
        return bar_count >= int(self.lb * 0.67)

    def reset_day(self, cur_date):
        if self.last_calendar_date != cur_date:
            self.day_spent_notional = 0.0
            self.day_symbol_spent   = {}
            self.last_calendar_date = cur_date
            
    def units_from_risk(self, d):
        """
        Calculate position sizing using ATR based risk model.
        units = risk_notional_per_leg / (ATR*price)
        """
        atr_val = float(self.atr[d][0]) if self.atr[d] is not None else 0.0
        price = float(d.close[0])
        if atr_val <= 0 or price <= 0:
            return 0
        return max(0, math.floor(self.cfg['risk_notional_per_leg'] / atr_val / price))
    

    # MEAN REVERSION
    
    def check_mr_long(self, d):
        close = float(d.close[0])
        openp = float(d.open[0])

        volume_z = (float(d.volume[0]) - float(self.vol_avg[d][0])) / (float(self.vol_std[d][0]) or 1.0)
        short_return = (close - float(d.close[-self.cfg['meanrev_window']])) / float(d.close[-self.cfg['meanrev_window']]) # Return over last 5 bars

        # n-bar low and distance from it
        low_n = min(float(d.low[-i]) for i in range(1, self.mw+1))
        dist_from_low = (close - low_n) / low_n if low_n > 0 else 0 # 0 near low, >0 above

        signals = {
            "volume_spike": volume_z >= self.cfg['volume_z'],
            "short_return": short_return <= self.cfg['short_return'],
            "green_candle": close > openp,
            "near_low": dist_from_low <= self.cfg['bounce_pct'],
        }

        ok = sum(signals.values()) >= 3  # require 3 of 4
        return ok, signals

    def check_mr_short(self, d):
        close = float(d.close[0])
        openp = float(d.open[0])

        volume_z = (float(d.volume[0]) - float(self.vol_avg[d][0])) / (float(self.vol_std[d][0]) or 1.0)
        return_n = (close - float(d.close[-self.cfg['meanrev_window']])) / float(d.close[-self.cfg['meanrev_window']])

        # n-bar low and distance from it
        high_n = max(float(d.low[-i]) for i in range(1, self.mw+1))
        dist_from_high = (high_n - close) / high_n if high_n > 0 else 0

        signals = {
            "volume_climactic": volume_z >= 1.0,
            "overbought": return_n <= 0.04,
            "red_candle": close < openp,
            "near_high": dist_from_high <= 0.02,
        }

        ok = sum(signals.values()) >= 3  # require 3 of 4
        return ok, signals


    # TREND FOLLOWING
    
    def check_tf_long(self, d):
        close = float(d.close[0])

        # Up closes in last N bars
        ups = sum(
            1 for k in range(1, self.cfg['up_window'] + 1)
            if float(d.close[0]) > float(d.close[-k])
        )
        up_ok = ups >= self.cfg['up_days_min']

        # Volume
        v = float(d.volume[0])
        mu = float(self.vol_avg[d][0])
        sd = float(self.vol_std[d][0]) if float(self.vol_std[d][0]) != 0.0 else 1.0
        vol_z = (v - mu) / sd
        vol_ok = vol_z >= self.cfg['vol_z_min'] 

        # Breakout: new N-bar high 
        high_n = max(float(d.high[-k]) for k in range(1, self.cfg['breakout_lookback'] + 1))
        brk_ok = close >= high_n

        # SMA trend
        sma = self.sma[d]
        trend_ok = True
        if sma is not None:
            s0 = float(sma[0])
            s1 = float(sma[-1])
            s2 = float(sma[-2])
            trend_ok = close > s0 and (s0 > s1 > s2)

        signals = {
            "ups": up_ok,
            "volume": vol_ok,
            "breakout_up": brk_ok,
            "sma_trend_up": trend_ok,
        }

        ok = sum(signals.values()) >= 3
        return ok, signals

    def check_tf_short(self, d):
        close = float(d.close[0])

        downs = sum(
            1 for k in range(1, self.cfg['up_window'] + 1)
            if float(d.close[0]) < float(d.close[-k])
        )
        down_ok = downs >= self.cfg['up_days_min']

        # Volume
        v = float(d.volume[0])
        mu = float(self.vol_avg[d][0])
        sd = float(self.vol_std[d][0]) if float(self.vol_std[d][0]) != 0.0 else 1.0
        vol_z = (v - mu) / sd
        vol_ok = vol_z >= self.cfg['vol_z_min'] 

        # Breakout: new N-bar low 
        low_n = min(float(d.high[-k]) for k in range(1, self.cfg['breakout_lookback'] + 1))
        brk_ok = close <= low_n

        # SMA trend
        sma = self.sma[d]
        trend_ok = True
        if sma is not None:
            s0 = float(sma[0])
            s1 = float(sma[-1])
            s2 = float(sma[-2])
            trend_ok = close < s0 and (s0 < s1 < s2)

        signals = {
            "downs": down_ok,
            "volume": vol_ok,
            "breakout_down": brk_ok,
            "sma_trend_down": trend_ok,
        }

        ok = sum(signals.values()) >= 3
        return ok, signals
    
    def get_trade_intent(self,d,capital_weight,bars_left,tbar,perf,laggards):
        """
        Entry Logic - Decides whether child wants to trade on symbol d on current bar
        
        Desicision Trree:
        Update regime detector
            If sideways regime:
                MR_Long if laggard and 3 of 4 checks passed
                MR_SHORT if downtrend and 5% up recently and 3 of 4 checks passed
            If Trending regime:
                TR_LONG if 3 of 4 checks passsed
                TF_SHORT if downtrend and 3 of 4 checks passed
        
        Args:
            capital_weight (float): current Dirichlet weight(0-1)
            bars_left (int): bars remaining
            tbar (int): current bar index
            perf (dict): cross sectional ranking
            laggards (set): bottom top_pct by recent performance
        """
    
        state = self.state[d]
        # If already in a position or in cooldown, skip this symbol
        pos = self.positions.get(d._name, {}).get('size', 0)
        if pos != 0 or tbar < state['cool_until']:
            return None

        # Update regime detector
        self.trend_filters[d].update(float(d.close[0]))
        regime = self.trend_filters[d].get_regime()
        state['regime_history'].append(regime)

        price = float(d.close[0])
        size_sign = 0   # 1 = Long, -1 = short, 0 = no trade
        entry_type = ''

        if regime['sideways'] >= self.cfg['sideways_threshold']:
            if d in laggards:       # Symbol has underperformed
                ok, sigs = self.check_mr_long(d)
                if ok:
                    size_sign = 1
                    entry_type = 'mean_reversion_long'
            if size_sign == 0 and regime['downtrend'] > regime['uptrend']:
                if perf.get(d, 0) >= 0.05:      # MR Short conditions
                    ok, sigs = self.check_mr_short(d)
                    if ok:
                        size_sign = -1
                        entry_type = 'mean_reversion_short'

        if size_sign == 0 and (regime['uptrend'] + regime['downtrend']) >= self.cfg['trending_threshold']:
            if regime['downtrend'] > regime['uptrend']:
                ok, sigs = self.check_tf_short(d)
                if ok:
                    size_sign = -1
                    entry_type = 'trend_following_short'
            else:
                ok, sigs = self.check_tf_long(d)
                if ok:
                    size_sign = 1
                    entry_type = 'trend_following_long'

        if size_sign == 0:
            return None

        atr_units = self.units_from_risk(d)                         # ATR based risk sizing
        units = min(self.cfg['entry_units_cap'], atr_units)
        units = int(max(1, math.floor(units * capital_weight)))     # Scaling by Dirichlet weight
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
    
    def record_entry(self,d,size_sign,entry_type,price):
        """
        Called by SuperBayesian.notify_order() after entry order fills
        Initialises trailing stop and saves position information
        """
        
        atr_val = float(self.atr[d][0]) if self.atr[d] is not None else 0.0
        if size_sign > 0:
            # Long: stop below price
            trail = max(price, price - self.cfg['atr_k'] * atr_val)
        else:
            # Short: stop above price
            trail = min(price, price + self.cfg['atr_k'] * atr_val)
    
        self.state[d]['trail']       = trail
        self.state[d]['breach_count'] = 0
        self.state[d]['entry_type']  = entry_type
        self.state[d]['entry_px']    = price

        # Register position in this child's position tracker
        self.positions[d._name] = {
            'size':       size_sign,
            'entry_px':   price,
            'entry_type': entry_type,
        }
        
    def check_exit(self,d,tbar):
        """
        Evaluate if current position needs to be exited
        
        Conditons:
            Severe breach
            Normal breach + two_bar_exit = True
            Normal breach + two_bar_exit = False
            No Breach
        """
        
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
            state['trail'] = max(trail, close - self.cfg['atr_k'] * atr_val)    # Traling Stoploss only moves up
        else:
            state['trail'] = min(trail, close + self.cfg['atr_k'] * atr_val)    # Trailing Stoploss only moves down

        return False
    
    def record_exit(self,d,exit_price,selector):
        """
        Calculate PnL% on position exit and report to BayesianModelSelector and reset per-symbol state
        """
        pos_info = self.positions.pop(d._name, None)
        if pos_info is None:
            return

        entry_px = pos_info['entry_px']
        side = pos_info['size']

        if entry_px and entry_px > 0:
            pnl_pct = side * (exit_price - entry_px) / entry_px
            won = pnl_pct > 0
            selector.record_trade(self.name, pnl_pct, won)

        # Cooldown and clear Trailing Stoploss
        self.state[d]['cool_until'] = float('inf')
        self.state[d]['trail'] = None
    
class SuperBayesian(bt.Strategy):
    """
    Super strategy that runs all 4 Child Strategies in parallel
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
        # Initialise meta-layer with one Dirichlet alpha per config
        config_names = list(CONFIGS.keys())
        self.selector = BayesianModelSelector(
            config_names=config_names,
            update_interval=self.p.selector_update_interval,
            sharpe_window=self.p.sharpe_window,
        )

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
            
        self.children = {}
        for name, cfg in CONFIGS.items():
            self.children[name] = ChildStrategy(
                name=name,
                cfg=cfg,
                datas=self.datas,
                broker=self.broker,
                indicators=self._indicator_sets[name],
            )

        self.pending_orders = {}
        self.open_positions = {}
        self.start_value = None
        self.last_calendar_date = None
        self.day_spent_notional = 0.0
        self.day_symbol_spent = {}
    
    def reset_day(self):
        """
        Update calendar day
        """
        cur = self.datas[0].datetime.date(0)
        if self.last_calendar_date != cur:
            self.day_spent_notional = 0.0
            self.day_symbol_spent = {}
            self.last_calendar_date = cur
    
    def next(self):
        self.reset_day()
        bar = len(self)    # Current bar
        
        # Bars left until end
        try:
            bars_left = self.datas[0].buflen() - bar
        except Exception:
            bars_left = 999999

        # Initialise Start Value
        if self.start_value is None:
            try:
                self.start_value = float(self.broker.getvalue())
            except Exception:
                self.start_value = 1.0

        # Force exits in the final week
        if bars_left <= int(self.p.risk_free_bars):
            for d in self.datas:
                pos = self.getposition(d).size
                # If long sell to close
                if pos > 0:
                    self.sell(data=d, size=pos)
                # If short buy to cover
                if pos < 0:
                    self.buy(data=d, size=abs(pos))
            return
        
        # Meta Layer 
        self.selector.step(bar)
        weights = self.selector.get_weights()

        # Using Lookback 60 as reference window for ranking, cross sectional perf ranking
        ref_window = CONFIGS['LB60']['meanrev_window']
        try:
            perf = {
                d: (float(d.close[0]) - float(d.close[-ref_window])) / float(d.close[-ref_window])
                for d in self.datas
            }
        except Exception:
            perf = {d: 0.0 for d in self.datas}

        # sort symbols by performance
        ranked = sorted(perf, key=perf.get)
        n = len(ranked)
        bottom_cut = max(1, int(CONFIGS['LB60']['top_pct'] * n))
        laggards = set(ranked[:bottom_cut])

        port_val = float(self.broker.getvalue())
        cash_now = float(self.broker.getcash())
        invested = port_val - cash_now
        global_headroom = max(0.0, self.p.global_max_exposure * port_val - invested)

        for name, child in self.children.items():
            if not child.ready(bar):
                continue

            weight = weights[name]      # Dirichlet capital weight
            cfg = child.cfg

            # Exit Logic
            for d in self.datas:
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
                            o = None        # PARENT DOESNT KNOW POSITION (CHECK ERROR)

                        if o:
                            self.pending_orders[o.ref] = (name, d, 0, 'exit')
                        
                        # Record exit in Child and calculate PnL
                        exit_price = float(d.close[0])
                        child.record_exit(d, exit_price, self.selector)
                        
                        # Cooldown 
                        child.state[d]['cool_until'] = bar + cfg['cooldown']
                        self.open_positions.pop(pos_key, None)
                        continue
                
                # Entry check for new positon
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

                # Submit order
                size = intent['size']
                if intent['action'] == 'buy':
                    o = self.buy(data=d, size=size)
                else:
                    o = self.sell(data=d, size=size)

                if o:
                    self.pending_orders[o.ref] = (name, d, intent['size_sign'], intent['entry_type'])
                    self.selector.record_capital(name, notional)
                    self.day_spent_notional += notional
                    self.day_symbol_spent[d._name] = sym_spent + notional
                    global_headroom -= notional
                    cash_now -= notional

    def notify_order(self, order):
        """Handle order execution callbacks."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        ref = order.ref
        if ref not in self.pending_orders:
            return   

        name, d, size_sign, entry_type = self.pending_orders.pop(ref)

        if order.status == order.Completed and entry_type != 'exit' and size_sign != 0:
            # Entry order filled 
            price = order.executed.price
            child = self.children[name]
            child.record_entry(d, size_sign, entry_type, price)
            # Register in the parent's position registry
            self.open_positions[(name, d._name)] = {
                'size':       size_sign,
                'entry_px':   price,
                'entry_type': entry_type,
            }
