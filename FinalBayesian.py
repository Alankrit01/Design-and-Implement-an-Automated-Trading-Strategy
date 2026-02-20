'''
python main.py --strategy FinalBayesian --data-dir DATA/PART1 --output-dir output/FinalBayes
'''

'''
{
  "final_value": 1020903.5846000001,
  "bankrupt": false,
  "bankrupt_date": null,
  "open_pnl_pd_ratio": 0.9001671470354202,
  "true_pd_ratio": 0.8743873962476292,
  "activity_pct": 95.2,
  "end_policy": "liquidate",
  "s_mult": 2.0
}
=== Trade Analyzer Stats ===
Total closed trades: 12
Wins: 7
Losses: 5

=== Trade Breakdown by Strategy Type ===

Mean Reversion Trades Long:
  Total trades: 2
  Wins: 1
  Losses: 1
  Total PnL: $-2193.00
  Win Rate: 50.0%

Trend Following Trades Long:
  Total trades: 9
  Wins: 5
  Losses: 4
  Total PnL: $20595.75
  Win Rate: 55.6%

Mean Reversion Trades Short:
  Total trades: 0
  Wins: 0
  Losses: 0
  Total PnL: $0.00

Trend Following Trades Shorts:
  Total trades: 1
  Wins: 1
  Losses: 0
  Total PnL: $3074.00
  Win Rate: 100.0%
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
        if len(self.price_history)>=20:
            # Calculate 20 bar momentum as %change      if price[0] = 100 and price[20] = 98, momentum = (100-98)/98
            recent_change = (self.price_history[-1] - self.price_history[-20]) / self.price_history[-20]
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
        
        if len(self.price_history)<20:
            return{'uptrend':0.33, 'downtrend':0.33, 'sideways':0.34}   # Uninformed prior
        
        recent_20 = list(self.price_history)[-20:]   # Last 20 brs
        early_20 = list(self.price_history)[-40:-20] if len(self.price_history)>=40 else recent_20  # Check bars 20-40 if available
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
    
class BayesianTradingStrategy(bt.Strategy):
    params = dict(
        # Mean Reversion Params
        atr_len=14,              # ATR lookback period for volatility
        atr_k=2.3,               # Stop loss distance: 2.3 * ATR below entry
        meanrev_window=5,        # Look back 5 bars for recent performance
        top_pct=0.15,            # Target bottom 15% of stocks 
        volume_z=0.8,            # Volume must be 0.8 std dev above average
        short_return=-0.02,      # Stock must be down 2%+ in last 5 bars
        long_return=0.0,         # Stock must be up 0%+ in last 30 bars 
        bounce_pct=0.015,        # Stock must have bounced 1.5% from 20-bar low
        
        # Trend Following Params
        up_window=5,             # Look back 5 bars
        up_days_min=3,           # Need at least 3 of last 5 bars higher
        breakout_lookback=7,     # Compare current close to highest of last 7 bars
        ma_len=20,               # 20-bar moving average for trend confirmation
        vol_z_min=0.8,           # Volume must be 0.8 std dev above average
        
        # Portfolio Management Params
        cash_frac=0.55,                 # Only use 55% of available cash for new trades
        cash_buffer=100000.0,           # Always keep $100k cash on hand
        max_exposure_frac=0.80,         # Never invest more than 80% of portfolio
        day_budget_frac=0.30,           # Limit daily spending to 30% of portfolio value
        day_symbol_cap_notional=0.40,   # Can't spend more than 40% of portfolio on 1 stock/day
        max_units_per_symbol=4500,      # Max 4500 shares per position
        entry_units_cap=700,            # Cap entry size at 700 shares
        risk_notional_per_leg=60000.0,  # Risk $60k per trade for position sizing
        
        # Exit Params
        cooldown=4,                 # Wait 4 bars before trading same stock again
        two_bar_exit=True,          # Require 2 bars of breach before exiting 
        risk_free_bars=7,           # Close all positions 7 bars before backtest end
        end_taper_bars=14,          # Taper exposure in last 14 bars 
        severe_breach_mult=0.60,    # Exit if price drops 0.60*ATR below trailing stop
        
        # Bayesian Learning Params
        regime_learning_enabled=True,    # Learn if market is trending or mean-reverting
        signal_learning_enabled=True,    # Learn which entry signals are reliable
        min_confidence_mr=0.42,          # Mean reversion signal must have >42% posterior accuracy
        min_confidence_trend=0.42,       # Trend signal must have >42% posterior accuracy
        sideways_threshold=0.30,         # Enter mean reversion if sideways probability >40%
        trending_threshold=0.50,         # Enter trend following if (uptrend + downtrend) >50%
        
        printlog=False,  
    )
    
    def __init__(self):
        self.atr = {d:bt.indicators.ATR(d,period=int(self.p.atr_len)) for d in self.datas} # Volatility Measure
        self.vol_avg = {d:bt.indicators.SimpleMovingAverage(d.volume, period=20) for d in self.datas} # 20 bar moving Average
        self.vol_std = {d:bt.indicators.StandardDeviation(d.volume, period=20) for d in self.datas} # 20 bar volume standard deviation
        self.sma = {d:bt.indicators.SimpleMovingAverage(d.close, period=int(self.p.ma_len)) for d in self.datas} # SMA of close 
        self.highest = {d:bt.indicators.Highest(d.close, period=int(self.p.breakout_lookback)) for d in self.datas} # highest close in lookback
        self.lowest = {d:bt.indicators.Lowest(d.close, period=20) for d in self.datas} # Lowest close in 20 bars
        
        self.trend_filters = {d: BayesianRegimeDetector() for d in self.datas} # regime model - MR or TF
        self.mr_signal_strength = {d: BayesianSignalStrength(prior_accuracy=0.48) for d in self.datas} # MR signal reliability
        self.tf_signal_strength = {d: BayesianSignalStrength(prior_accuracy=0.48) for d in self.datas} # TF signal reliability
        
        self.state = {d:dict(
            trail=None,           # Current trailing stop price
            cool_until=-math.inf, # Don't trade until after this bar
            breach_count=0,       # How many bars has price breached stop?
            entry_px=None,        # Price we entered at
            entry_units=0,        # How many units did we buy?
            entry_type='',        # Was this 'trend_following' or 'mean_reversion'?
            entry_bar=0,          # Which bar did we enter?
            regime_history=deque(maxlen=20)   # Last 20 regime classifications
        ) for d in self.datas}
        
        self.prev_pos = {d: 0.0 for d in self.datas}  # Track previous position size
        self.day_spent_notional = 0.0  # How much spent today
        self.last_calendar_date = None  # Track calendar date for daily reset
        self.day_symbol_spent = {}     # How much spent on each stock today
        self.start_value = None        # Starting portfolio value
        
    def ready(self):
        """
        cehck if there is enough data
        Returns: True if there are atleast 40 bars of data (50 bar lookback + 1)
        """
        return len(self) >= 40
    
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
    
        # === MEAN REVERSION LONG ===
    def check_mr_long(self, d):
        close = float(d.close[0])
        openp = float(d.open[0])

        volume_z = (float(d.volume[0]) - float(self.vol_avg[d][0])) / (float(self.vol_std[d][0]) or 1.0)
        short_return = (close - float(d.close[-self.p.meanrev_window])) / float(d.close[-self.p.meanrev_window]) # Return over last 5 bars

        # 20-bar low and distance from it
        low_20 = min(float(d.low[-i]) for i in range(1, 21))
        dist_from_low = (close - low_20) / low_20  # 0 near low, >0 above

        signals = {
            "volume_spike": volume_z >= self.p.volume_z,          
            "short_return": short_return <= self.p.short_return,
            "green_candle": close > openp,                         # recovery intraday
            "near_20_low": dist_from_low <= self.p.bounce_pct,   
        }

        ok = sum(signals.values()) >= 3  # require 3 of 5
        return ok, signals


    # === MEAN REVERSION SHORT ===
    def check_mr_short(self, d):
        close = float(d.close[0])
        openp = float(d.open[0])

        volume_z = (float(d.volume[0]) - float(self.vol_avg[d][0])) / (float(self.vol_std[d][0]) or 1.0)

        # 5-bar return (positive = overbought)
        ret_5 = (close - float(d.close[-self.p.meanrev_window])) / float(d.close[-self.p.meanrev_window])

        # 20-bar high and distance from it
        high_20 = max(float(d.high[-i]) for i in range(1, 21))
        dist_from_high = (high_20 - close) / high_20  # 0 near high, >0 below

        signals = {
            "volume_climactic": volume_z >= 1.0,                   # a bit stricter for shorts
            "overbought": ret_5 >= 0.04,                           # +4% in 5 bars
            "red_candle": close < openp,
            "near_20_high": dist_from_high <= 0.02,                # within 2% of 20-bar high
        }

        ok = sum(signals.values()) >= 3
        return ok, signals


    # === TREND FOLLOWING LONG ===
    def check_tf_long(self, d):
        close = float(d.close[0])

        # Up closes in last N bars
        ups = sum(
            1 for k in range(1, int(self.p.up_window) + 1)
            if float(d.close[0]) > float(d.close[-k])
        )
        up_ok = ups >= int(self.p.up_days_min)  # 2 of 3

        # Volume
        v = float(d.volume[0])
        mu = float(self.vol_avg[d][0])
        sd = float(self.vol_std[d][0]) if float(self.vol_std[d][0]) != 0.0 else 1.0
        vol_z = (v - mu) / sd
        vol_ok = vol_z >= self.p.vol_z_min    # e.g. 0.8

        # Breakout: new N-bar high (use your breakout_lookback)
        high_n = max(float(d.high[-k]) for k in range(1, int(self.p.breakout_lookback) + 1))
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


    # === TREND FOLLOWING SHORT ===
    def check_tf_short(self, d):
        close = float(d.close[0])

        # Down closes in last N bars
        downs = sum(
            1 for k in range(1, int(self.p.up_window) + 1)
            if float(d.close[0]) < float(d.close[-k])
        )
        down_ok = downs >= int(self.p.up_days_min)

        # Volume
        v = float(d.volume[0])
        mu = float(self.vol_avg[d][0])
        sd = float(self.vol_std[d][0]) if float(self.vol_std[d][0]) != 0.0 else 1.0
        vol_z = (v - mu) / sd
        vol_ok = vol_z >= self.p.vol_z_min

        # Breakdown: new N-bar low
        low_n = min(float(d.low[-k]) for k in range(1, int(self.p.breakout_lookback) + 1))
        brk_down_ok = close <= low_n

        # SMA downtrend
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
            "breakdown": brk_down_ok,
            "sma_trend_down": trend_ok,
        }

        ok = sum(signals.values()) >= 3
        return ok, signals
    
    def next(self):
        if not self.ready():    # Not enough bars
            return
        
        self.reset_day()
        tbar = len(self)    # Current bar
        
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
                # If long sell to close
                if pos > 0:
                    self.sell(data=d, size=pos)
                # If short buy to cover
                if pos < 0:
                    self.buy(data=d, size=abs(pos))
            return

        # Update trend filters for each symbol
        for d in self.datas:
            self.trend_filters[d].update(float(d.close[0]))

        # Rank Stocks
        # Calculate 5-bar return for each stock
        perf = {d: (float(d.close[0]) - float(d.close[-self.p.meanrev_window])) / float(d.close[-self.p.meanrev_window]) 
                for d in self.datas}
        
        # Sort stocks from worst to best performer
        ranked = sorted(perf, key=perf.get)
        n = len(ranked)
        
        bottom_cut = max(1, int(self.p.top_pct * n))    # Bottom 15%: potential mean reversion candidates
        laggards = set(ranked[:bottom_cut])             # Set of symbols that are underperforming

        # Entry
        for d in self.datas:
            state = self.state[d]
            pos = self.getposition(d).size
            if pos != 0 or tbar < state['cool_until']:  # No pyramiding
                continue

            # Get regime for this symbol
            regime = self.trend_filters[d].get_regime()
            state['regime_history'].append(regime)
            
            # MEAN REVERSION
            if regime['sideways'] >= self.p.sideways_threshold:

                # MR LONG: sideways + laggard
                if d in laggards:
                    ok, sigs = self.check_mr_long(d)
                    if ok:
                        self.enter_trade(d, size_sign=1, entry_type='mean_reversion_long')
                        continue

                # MR SHORT: sideways + strong recent winner + bearish tilt
                if regime['downtrend'] > regime['uptrend']:
                    perf_5 = perf[d]
                    if perf_5 >= 0.05:  # +5% in 5 bars
                        ok, sigs = self.check_mr_short(d)
                        if ok:
                            self.enter_trade(d, size_sign=-1, entry_type='mean_reversion_short')
                            continue

            # TREND FOLLOWING
            if (regime['uptrend'] + regime['downtrend']) >= self.p.trending_threshold:

                # TF SHORT in downtrend
                if regime['downtrend'] > regime['uptrend']:
                    ok, sigs = self.check_tf_short(d)
                    if ok:
                        self.enter_trade(d, size_sign=-1, entry_type='trend_following_short')
                        continue

                # TF LONG in uptrend
                else:
                    ok, sigs = self.check_tf_long(d)
                    if ok:
                        self.enter_trade(d, size_sign=1, entry_type='trend_following_long')
                        continue


        # In final month, exit profitable positions and hold losers for recovery
        if bars_left <= 21: 
            for d in self.datas:
                pos = self.getposition(d).size
                if pos <= 0:
                    continue
                
                state = self.state[d]
                current_price = float(d.close[0])
                entry_px = state['entry_px']
                
                if entry_px and entry_px > 0:
                    pnl_pct = (current_price - entry_px) / entry_px
                    if pnl_pct > 0.005:     # Exit winners (>0.5% profit)
                        self.close(data=d)
                        continue
        
        # In final week, liquidate everything at market
        if bars_left <= 5:
            for d in self.datas:
                pos = self.getposition(d).size
                if pos > 0:
                    self.close(data=d)
            return
        
        # Exit logic
        
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
            if size_sign > 0:
                # Long: stop below price
                trail = max(trail, float(d.close[-1]) - float(self.p.atr_k) * float(self.atr[d][0]))
            else:
                # Short: stop above price
                trail = min(trail, float(d.close[-1]) + float(self.p.atr_k) * float(self.atr[d][0]))
        state['trail'] = trail
        state['breach_count'] = 0
        state['entry_type'] = entry_type

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