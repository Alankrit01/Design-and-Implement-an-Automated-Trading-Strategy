'''python main.py --strategy BayesianCombined --data-dir DATA/PART1 --output-dir output/bayesian_hybrid --param risk_free_bars=7 --param end_taper_bars=14 --param severe_breach_mult=0.75 --param day_budget_frac=0.40 --param day_symbol_cap_notional=0.50 --param atr_k=2.2 --param two_bar_exit=True --param cooldown=3 --param regime_learning_enabled=True --param signal_learning_enabled=True --param param_learning_enabled=True --param update_frequency=20 --debug'''
# Too Many Trades. Tighten entry cases, analyse and edit logic.
'''
=== Trade Analyzer Stats ===
Total closed trades: 491
Wins: 184
Losses: 307

=== Trade Breakdown by Strategy Type ===

Mean Reversion Trades:
  Total trades: 125
  Wins: 40
  Losses: 84
  Total PnL: $-5628.77
  Win Rate: 32.0%

Trend Following Trades:
  Total trades: 366
  Wins: 139
  Losses: 223
  Total PnL: $15014.92
  Win Rate: 38.0%
'''

import backtrader as bt
import math
import numpy as np
from scipy import stats
from collections import defaultdict, deque

class BayesianCombined(object):
    """
    Bayesian Model that learns market behaviour and differentiates between trending and reverting markets
    """
    
    def __init__(self, alpha=1.0, beta=1.0, window=20):
        self.alpha = alpha      # Likelihood of Trends
        self.beta = beta        # Likelihood of mean reversion
        self.window = window    # Lookback
        
        self.trend_success = 0  # Count of times trending worked
        self.mr_success = 0     # Count of times mean reversion worked
        self.total_obs = 0      # Total observations seen
        
        self.history = deque(maxlen=window) # Keep only last N observations
        
    def update(self, close, prev_close, next_close):
        """
        Evaluate if current market behaviour is mean reverting or trending
        """
        
        momentum = close>prev_close # True if upward momentum
        reversal = next_close < close if momentum else next_close > close # Check if price reversed
        trend_success = 1 if (next_close>close) == momentum else 0 # If momentum continued in same direction --> 1 else 0
        mr_success = 1 if reversal else 0 # If price reversed --> 1 else 0
        
        self.trend_success += trend_success
        self.mr_success += mr_success
        self.total_obs+=1
        self.history.append(trend_success, mr_success)
        
    def get_regime(self):
        """
        Returns P(Market trending | observed data)
        Beta(alpha+trend_successes, beta+mr_successes)
        
        Returns: Probability b/w 0 and 1
        0.0 --> Mean reversion
        0.5 --> uncertain
        1.0 --> Trending
        """
        if self.total_obs == 0:     # No data
            return 0.5
        
        alphapost = self.alpha + self.trend_success
        betapost = self.beta + self.mr_success
        
        # Mean of Beta distribution = alpha/(alpha+beta)
        # Higher trend_success relative to mr_success means higher trend probabulity
        trend_prob = alphapost/(alphapost+betapost)
        return trend_prob
    
    def get_uncertainity(self):
        """
        Confidernce level (high variance = low confidence)
        Returns: Variance value
            0.0 = very confident
            0.25 = max uncertainity 
        """
        if self.total_obs == 0:
            return 0.25         # prob = 0.5
        
        alphapost = self.alpha+self.trend_success
        betapost = self.beta+self.mr_success
        n = alphapost+betapost
        
        variance = (alphapost*betapost)/(n**2 * (n+1)) # Variance = (alpha * beta) / ((alpha+beta)^2 * (alpha+beta+1))
        return variance

class BayesianSignalStrenght(object):
    """
    Track historical accuracy of individual trading signals 
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
            alpha Defaults to 1.0
            beta Defaults to 1.0
            
        Returns: Probability b/w 0 and 1:
            0 if unreliable 
            1 if reliable
        """
        
        if self.total_trades == 0:      # No trades occured 
            return self.prior_accuracy
        
        # Posterior = Prior + Likelihood
        # alphapost = prior success + observed success
        # betapost = prior failure + observed failure
        alphapost = alpha + self.successes
        betapost = beta + (self.total_trades - self.successes)
        return alphapost/(alphapost+betapost)   # Expected Value of Beta distribution
    
    def get_credivle_interval(self, alpha=1.0, beta=1.0, confidence=0.95):
        """Bayesian Credible Interval: range where true accuracy (95% confident)
        Returns: Tuple (lower,upper) 
        Example: (0.50,0.65) means 95% confident true accuracy is between 50-65%
        """
        
        if self.total_trades == 0:
            return(0.0,1.0)     # no trades indicates maximum uncertainity
        
        alphapost = alpha + self.success
        betapost = beta + (self.total_trades - self.success)
        
        lower = stats.beta.ppf((1-confidence)/2, alphapost, betapost)   # 1-confidence/2 = 2/5% of 95% confidence
        upper = stats.beta.ppf((1+confidence)/2, alphapost, betapost)   # 1+confidence/2 = 97.5% of 95% confidence
        
        return (lower, upper)

class BayesianParameterLearner(object):
    """
    Tracks which parameter combination works best
    """    
    def __init__(self, param_grid, prior_success_rate=0.45):
        self.param_grid = param_grid                    # Dictionary of parameter name
        self.prior_success_rate = prior_success_rate    # Prior belief: 45% success rate before evidence
        self.param_outcomed = defaultdict(lambda:{'wins':0, 'total':0}) # dictionary sttoring results for each param combination
        
    def record_outcome(self, params_used, trade_won):
        """Record result of trade that used specific parameters

        Args:
            params_used (dict): Example: {'atr_len': 14, 'threshold': 1.5}
            trade_won (Boolean): did the trade make money?
        """
        key = tuple(sorted(params_used.items()))    # convert dict to tutpple and sort 
        self.param_outcomed[key]['total'] += 1      # total trades +1
        if trade_won:
            self.param_outcomed[key]['wins']+=1     # trades won +1
            
    def get_param_wights(self, alpha=1.0, beta=1.0):
        """Returns success probability for each parameter combination. Higher weight means more likely to be successful

        Returns: Dictionary with normalised weights (sum to 1.0)
        """
        weights = {}
        for key, outcome in self.param_outcomed.items():
            alphapost = alpha + outcome['wins']
            betapost = beta + (outcome['total'] - outcome['wns'])
            success_prob = alphapost/(alphapost+betapost)   # mean of beta distribution gives success probabvility estimate
            weights['key'] = success_prob
        
        total_weight = sum(weights.values()) if weights else 1.0
        if total_weight>0:
            weights = {k:v / total_weight for k, v in weights.items()}
            
        return weights
    
class BayesianTradingStrategy(bt.Strategy):
    params = dict(
        # Mean Reversion Params
        atr_len=14,              # ATR lookback period for volatility
        atr_k=2.3,               # Stop loss distance: 2.3 * ATR below entry
        meanrev_window=5,        # Look back 5 bars for recent performance
        top_pct=0.10,            # Target bottom 10% of stocks 
        volume_z=1.0,            # Volume must be 1.0 std dev above average
        short_return=-0.03,      # Stock must be down 3%+ in last 5 bars
        long_return=0.0,         # Stock must be up 0%+ in last 30 bars 
        bounce_pct=0.012,        # Stock must have bounced 1.2% from 20-bar low
        
        # Trend Following Params
        up_window=3,             # Look back 3 bars
        up_days_min=2,           # Need at least 2 of last 3 bars higher
        breakout_lookback=7,     # Compare current close to highest of last 7 bars
        ma_len=20,               # 20-bar moving average for trend confirmation
        vol_z_min=1.0,           # Volume must be 1.0 std dev above average
        
        # Portfolio Management Params
        cash_frac=0.55,                 # Only use 55% of available cash for new trades
        cash_buffer=100000.0,           # Always keep $100k cash on hand
        max_exposure_frac=0.85,         # Never invest more than 85% of portfolio
        day_budget_frac=0.4,            # Limit daily spending to 40% of portfolio value
        day_symbol_cap_notional=0.50,   # Can't spend more than 50% of portfolio on 1 stock/day
        max_units_per_symbol=5000,      # Max 5000 shares per position
        entry_units_cap=900,            # Cap entry size at 900 shares
        risk_notional_per_leg=60000.0,  # Risk $60k per trade for position sizing
        
        # Exit Params
        cooldown=3,                 # Wait 3 bars before trading same stock again
        two_bar_exit=True,          # Require 2 bars of breach before exiting 
        risk_free_bars=7,           # Close all positions 7 bars before backtest end
        end_taper_bars=14,          # Taper exposure in last 14 bars 
        severe_breach_mult=0.75,    # Exit if price drops 0.75*ATR below trailing stop
        
        # Bayesian Learning Params
        regime_learning_enabled=True,    # Learn if market is trending or mean-reverting
        signal_learning_enabled=True,    # Learn which entry signals are reliable
        param_learning_enabled=True,     # Track which parameter combos work best
        update_frequency=20,             # Update beliefs every 20 bars 
        
        printlog=False,  
    )
        
    def __init__(self):
        self.atr = {d:bt.indicators.ATR(d,period=int(self.p.atr_len)) for d in self.datas} # Volatility Measure
        self.vol_avg = {d:bt.indicators.SimpleMovingAverage(d.volume, period=20) for d in self.datas} # 20 bar moving Average
        self.vol_std = {d:bt.indicators.StandardDeviation(d.volume, period=20) for d in self.datas} # 20 bar volume standard deviation
        self.sma = {d:bt.indicators.SimpleMovingAverage(d.close, period=int(self.p.ma_len)) for d in self.datas} # SMA of close 
        self.highest = {d:bt.indicators.Highest(d.close, period=int(self.p.breakout_lookback)) for d in self.datas} # highest close in lookback
        
        self.regime_models = {d: BayesianCombined() for d in self.datas} # regime model - MR or TF
        self.mr_signal_strength = {d: BayesianSignalStrenght(prior_accuracy=0.50) for d in self.datas} # MR signal reliability
        self.tf_signal_strength = {d: BayesianSignalStrenght(prior_accuracy=0.50) for d in self.datas} # TF signal reliability
        self.param_learner = BayesianParameterLearner({})   # Analysing which parameter combination works best
        
        self.state = {d:dict(
            trail=None,           # Current trailing stop price
            cool_until=-math.inf, # Don't trade until after this bar
            breach_count=0,       # How many bars has price breached stop?
            entry_px=None,        # Price we entered at
            entry_units=0,        # How many units did we buy?
            entry_type='',        # Was this 'trend_following' or 'mean_reversion'?
            entry_bar=0,          # Which bar did we enter?
            active_signals=[]     # Which signals triggered this entry?
        ) for d in self.datas}
        
        self.prev_pos = {d: 0.0 for d in self.datas}  # Track previous position size
        self.day_spent_notional = 0.0  # How much spent today
        self.last_calendar_date = None  # Track calendar date for daily reset
        self.day_symbol_spent = {}     # How much spent on each stock today
        self.start_value = None        # Starting portfolio value
        self.bar_count = 0             # Total bars processed
        
        self.closed_trades = deque(maxlen=100)   # Keep last 100 closed trades 
        
    def ready(self):
        """
        cehck if there is enough data
        Returns: True if there are atleast 31 bars of data (30 bar lookback + 1)
        """
        return len(self) >= 31
    
    def reset_day(self):
        """
        reset daily spending counter for each calendar day
        """
        cur = self.datas[0].datetime.date(0)
        if self.last_calendar_date!=cur:    # reset budget for day change
            self.day_spent_notional=0.0
            self.last_calendar_date=cur
            self.day_symbol_spent={}
            
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
        
    def get_regime_weight(self,d):
        """
        Bayesian Regime weighting

        Returns: Float between 0 and 1
            0.0 = strongly favouring Mean Reversion
            0.5 = neutral/uncertain
            1.0 = strongly favouring Trend Following
        """
        if not self.p.regime_learning_enabled:
            return 0.5
        
        regime_model = self.regime_models[d]    # Get regime model
        trend_prob = regime_model.get_regime()  # get posterior probability stock is trending (0-1)
        uncertainty = regime_model.get_uncertainity()   # get uncertainty (0-0.25)
        
        # If uncertain, push probability toward 0.5 (neutral)
        # Formula: adjusted_prob = 0.5 + (original - 0.5) * (1 - uncertainty*10)
        # If uncertainty = 0.025 (high certainty): multiply by (1 - 0.25) = 0.75 (keep most signal)
        # If uncertainty = 0.20 (low certainty): multiply by (1 - 2.0) = -1.0 (flip to neutral)
        adjusted_prob = 0.5 + (trend_prob - 0.5) * (1.0 - uncertainty * 10)
        return adjusted_prob
    
    def eval_mr_signals(self,d):
        """
        Evaluate MR signal and return True if trade looks good for Mean Reversion

        Signals checked:
        Volume elevated? (volume_z >= 1.0)
        Down 3%+ in last 5 bars?
        Up 0%+ in last 30 bars? (not crashed)
        Close above open (recovery)
        Bounced 1.2%+ from 20-bar low?
        """
        close = float(d.close[0])
        prev_close = float(d.close[-1])
        openp = float(d.open[0])
        
        volume_z = (float(d.volume[0]) - float(self.vol_avg[d][0])) / (float(self.vol_std[d][0]) or 1) # Volume Z-score
        short_return = (close - float(d.close[-self.p.meanrev_window])) / float(d.close[-self.p.meanrev_window]) # Return over last 5 bars
        long_return = (close - float(d.close[-30])) / float(d.close[-30]) # Calculate return over last 30 bars
        low_20 = min([float(d.low[-i]) for i in range(20)]) # Find lowest close in last 20 bars
        bounce_from_low = (close - low_20) / low_20 # Calculate bounce from 20 bar low
        
        mr_signals = {
            'volume': volume_z >= self.p.volume_z,           # Volume spike?
            'short_return': short_return <= self.p.short_return,  # Down enough?
            'long_return': long_return >= self.p.long_return,     # Not crashed?
            'close_above_open': close > openp,               # Closing higher than open?
            'bounce': bounce_from_low >= self.p.bounce_pct,  # Bounced from low?
        }
        
        if self.p.signal_learning_enabled:
            weights = {}
            for signal_name, fired in mr_signals.items():
                strength_model = self.mr_signal_strength[d]
                weights[signal_name] = strength_model.get_accuracy()
            weighted_score = sum(weights[k]*v for k,v in mr_signals.items()) # Example: 0.6*1 + 0.5*1 + 0.7*0 + 0.6*1 + 0.8*1 = 2.9
            threshold = sum(weights.values())*0.6    # Threshold needs to be 60% of max possible score
            signal_valid = weighted_score>=threshold
        else:
            signal_valid=all(mr_signals.values())   # Learning disabled, require all signals to be true
            
        return signal_valid, mr_signals
    
    def eval_tf_signals(self,d):
        """
        Evaluate TF signal and return True if trade looks good for Trend following

        Signals Checked:
        2+ of 3 bars higher? (uptrend)
        Volume Elevated (vol_z>=1.0)
        Price at/above highest close of last 7 bars?
        price>SMA and SMA rising?
        """
        close = float(d.close[0])
        openp = float(d.open[0])
        
        ups = sum(1 for k in range(1, int(self.p.up_window) + 1) # Range of k: 1, 2, 3 (yesterday, 2 days ago, 3 days ago)
                  if float(d.close[0]) > float(d.close[-k]))
        up_ok = ups >= int(self.p.up_days_min)  # Need at least 2 of the 3 bars to be up
        v = float(d.volume[0])
        mu = float(self.vol_avg[d][0])
        sd = float(self.vol_std[d][0]) if float(self.vol_std[d][0]) != 0 else 1.0
        vol_z = (v - mu) / sd   # Z-score = (value-mean)/std dev
        vol_ok = vol_z >= self.p.vol_z_min  # Volume must be 1/0 std dev above average
        brk_ok = close >= float(self.highest[d][-1]) if self.highest[d] is not None else True
        trend_ok = True  # Default: assume true if can't calculate
        sma = self.sma[d]
        if sma is not None:
            s0 = float(sma[0])    # Current SMA
            s1 = float(sma[-1])   # 1 bar ago
            s2 = float(sma[-2])   # 2 bars ago
            trend_ok = close > s0 and (s0 > s1 > s2)   # SMA is rising (s0 > s1 > s2) AND price is above SMA
            
        # Check each signal condition
        trend_signals = {
            'ups': up_ok,          # Uptrend confirmed?
            'volume': vol_ok,      # Volume above average?
            'breakout': brk_ok,    # Breaking out of range?
            'sma_trend': trend_ok, # SMA rising and price above it?
        }
        
        if self.p.signal_learning_enabled:
            weights = {}
            for signal_name, fired in trend_signals.items():  # Get historical accuracy of each trend signal
                strength_model = self.tf_signal_strength[d]
                weights[signal_name] = strength_model.get_accuracy()
            
            weighted_score = sum(weights[k] * v for k, v in trend_signals.items())
            threshold = sum(weights.values()) * 0.6
            signal_valid = weighted_score >= threshold
        else:
            signal_valid = all(trend_signals.values())
        
        return signal_valid, trend_signals
    
    def next(self):
        if not self.ready():
            return
        
        self.reset_day()
        self.bar_count += 1
        tbar = len(self)
        # Bars left until end
        try:
            bars_left = self.datas[0].buflen() - len(self)
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
                # If long, sell to close
                if pos > 0:
                    self.sell(data=d, size=pos)
                # If short, buy to cover
                if pos < 0:
                    self.buy(data=d, size=abs(pos))
            return
        
        # Rank Stocks
        # Calculate 5-bar return for each stock
        perf = {d: (float(d.close[0]) - float(d.close[-self.p.meanrev_window])) / 
                float(d.close[-self.p.meanrev_window]) for d in self.datas}
        
        # Sort stocks from worst to best performer
        ranked = sorted(perf, key=perf.get)
        n = len(ranked)
        
        # Bottom 10%: potential mean reversion candidates
        bottom_cut = max(1, int(self.p.top_pct * n))
        # Top 10%: potential trend following candidates
        top_cut = max(1, int(self.p.top_pct * n))
        
        # Get sets of laggard and leader stocks
        laggards = set(ranked[:bottom_cut])  # Worst 10%
        leaders = set(ranked[-top_cut:])     # Best 10%
        
        
        # Entry
        for d in self.datas:
            state = self.state[d]
            pos = self.getposition(d).size
            
            # Skip if already have position OR still in cooldown
            if pos != 0 or tbar < state['cool_until']:
                continue

            # TF or MR
            regime_weight = self.get_regime_weight(d)
            
            # Use when regime_weight < 0.55 (confident in mean reversion) and stock is in bottom 10% (oversold)
            if regime_weight < 0.55 and d in laggards:
                # Check if mean reversion signals align
                mr_valid, mr_sigs = self.eval_mr_signals(d)
                if mr_valid:
                    # Enter long position (buy)
                    self.enter_trade(d, size_sign=1, entry_type='mean_reversion')
                    # Record which signals triggered this entry
                    state['active_signals'] = list(mr_sigs.keys())
                    continue  # Move to next stock

            # Use when regime_weight > 0.45 (confident in trend following)
            if regime_weight > 0.45:
                # Check if trend following signals align
                trend_valid, trend_sigs = self.eval_tf_signals(d)
                if trend_valid:
                    # Enter long position (buy)
                    self.enter_trade(d, size_sign=1, entry_type='trend_following')
                    # Record which signals triggered this entry
                    state['active_signals'] = list(trend_sigs.keys())
                    continue
            
        # Exit
        for d in self.datas:
            pos = self.getposition(d).size
            # Skip if no position
            if pos == 0:
                continue
            
            state = self.state[d]
            
            # Update Trailing stop: max of previous trail or (previous close - 2.3*ATR)
            new_trail = float(d.close[-1])  # Previous bar close
            if self.atr[d] is not None:
                # Alternative: previous close - 2.3 * current ATR
                new_trail = max(new_trail, float(d.close[-1]) - float(self.p.atr_k) * float(self.atr[d][0]))
            
            # Keep highest trailing stop seen so far
            # Initialise with previous trail or negative infinity if first time
            state['trail'] = max(state['trail'] if state['trail'] is not None else -math.inf, new_trail)
            
            # Check for StopLoss Breach 
            if self.atr[d] is not None:
                atr_now = float(self.atr[d][0])
                # Add small buffer (75% of ATR) before triggering exit
                buffer_mult = float(self.p.severe_breach_mult)
                # Breach if price drops below trail minus buffer
                breach = float(d.close[0]) < float(state['trail']) - buffer_mult * atr_now
            else:
                # Simple breach: just check against trail
                breach = float(d.close[0]) < float(state['trail'])
            
            # Count consecutive bars with breaches
            # Reset to 0 if no breach this bar, otherwise increment
            state['breach_count'] = state['breach_count'] + 1 if breach else 0
            
            # Exit after N consecutive breach bars (default 2)
            need_bars = 2 if self.p.two_bar_exit else 1
            
            # If we have enough breach bars, close the position
            if state['breach_count'] >= need_bars:
                self.close(data=d)  # Sell position
                state['cool_until'] = tbar + int(self.p.cooldown)  # Start cooldown
                state['breach_count'] = 0  # Reset breach counter
                
    def enter_trade(self,d,size_sign,entry_type=''):
        """
        Execute trade

        Args:
            d: data feed to trade
            size_sign: 1 for long, -1 for short
            entry_type: MR or TF
        """
        price = float(d.close[0])   # get current price
        
        # Calculate Position Sizing
        entry_cap = int(self.p.entry_units_cap) # Cap from entry units param
        atr_units = self.units_from_risk(d)     # Calculate units based on risk
        units = min(entry_cap, atr_units)       # min of ATR and entry cap
        units = int(max(1, math.floor(units)))  # Min 1 unit
        units *= size_sign
        
        # Check available cash
        cash_now = float(self.broker.getcash())
        port_val = float(self.broker.getvalue())
        invested_est = port_val - cash_now  # Estimate capital invested
        # Calculate available headroom under max exposure limit
        headroom_val = max(0.0, float(self.p.max_exposure_frac) * port_val - invested_est)
        # Units we can afford from available cash (using only 55% of it)
        units_cash = math.floor(
            max(0.0, cash_now * float(self.p.cash_frac) - float(self.p.cash_buffer)) / abs(price)
        ) if price != 0 else 0
        # Units we can afford from available headroom
        units_headroom = math.floor(headroom_val / abs(price)) if price != 0 else 0

        # Take minimum of all constraints
        # Also cap at max_units_per_symbol
        units = max(
            -int(self.p.max_units_per_symbol),
            min(units, units_cash, units_headroom, int(self.p.max_units_per_symbol))
        )
        
        # Calculate notional value
        notional = abs(price * units)
        # Amount spent on symbol today
        sym_spent = self.day_symbol_spent.get(d._name, 0.0)
        # Cap for single symbol per day: 50% of portfolio
        sym_cap = float(self.p.day_symbol_cap_notional) * port_val
        
        # Skip if this trade would exceed daily limits
        if sym_spent + notional > sym_cap or \
           self.day_spent_notional + notional > port_val * self.p.day_budget_frac:
            return

        # Execute Trade
        if units > 0:
            self.buy(data=d, size=units)  # Go long
        else:
            self.sell(data=d, size=abs(units))  # Go short

        # Update spent tracking
        self.day_spent_notional += notional
        self.day_symbol_spent[d._name] = sym_spent + notional
        
        state = self.state[d]
        # Initialize trailing stop at previous close
        trail = float(d.close[-1])
        # Alternative: previous close - 2.3 * ATR
        if self.atr[d] is not None:
            trail = max(trail, float(d.close[-1]) - float(self.p.atr_k) * float(self.atr[d][0]))
        state['trail'] = trail
        state['breach_count'] = 0
        state['entry_type'] = entry_type
        state['entry_bar'] = len(self)
    
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
