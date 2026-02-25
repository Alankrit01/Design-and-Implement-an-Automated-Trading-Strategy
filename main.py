# !!!! WORKING FOR SuperBayesian.py 25/02 !!!!
# =============================================================================
# main.py — BT396 Backtrader harness entrypoint
# =============================================================================
# This is the primary executable script for the BT396 framework.
# It acts as the main CLI interface for running trading backtests.
#
# Responsibilities:
#   - Parse command-line arguments and configuration files.
#   - Load 10 aligned CSV data feeds via the framework’s data loader.
#   - Dynamically load and wrap a user strategy with COMP396Base (rules engine).
#   - Configure Backtrader’s Cerebro engine, broker, and analyzers.
#   - Run the backtest, collect results, and output analytics + plots.
#   - Generate summary JSON and diagnostic figures in an output directory.
# =============================================================================

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import backtrader as bt

# -----------------------------------------------------------------------------
# Framework plotting utilities
# These functions are responsible for producing the output plots and summaries
# after a backtest completes. All images and tables are written to output_dir.
# -----------------------------------------------------------------------------
from framework.plotting import (
    save_equity_plot,
    save_per_series_pd,
    save_per_series_plots,
    save_equity_dashboard,
    save_all_series_equity,
    save_portfolio_underwater,
    save_realized_equity_plot,
    save_combined_equity_dashboard,
    save_true_equity_plot
)

# -----------------------------------------------------------------------------
# Default configuration used when no config.yaml/json is found.
# This ensures consistent baseline behavior even without external config.
# -----------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "starting_cash": 1_000_000,            # initial portfolio value
    "commission": 0.0,                     # commission rate per trade (fractional)
    "s_mult": 1.0,                         # multiplier for 20% gap slippage model
    "plot": True,                          # enable plotting by default
    "data_dir": "./DATA/PART1",            # default path for 10 CSVs
    "output_dir": "./output",              # directory to save reports/plots
    "end_policy": "liquidate",             # final-day liquidation behavior
    "strategy": "example_buy_and_hold",    # default strategy module
    "strategy_class": None,                # explicitly specify class if needed
    "strategy_params": {},                 # deprecated; CLI params override instead
    "debug": True,                         # enables verbose logging in the framework
}

# -----------------------------------------------------------------------------
# Config loader (YAML preferred, JSON fallback)
# This function attempts to read a config file and merge it into DEFAULT_CONFIG.
# YAML is used first for readability, falling back to JSON for compatibility.
# -----------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if not path.exists():
        return cfg
    try:
        import yaml  # optional dependency
        with path.open("r") as f:
            cfg.update(yaml.safe_load(f) or {})
        return cfg
    except Exception:
        # fallback: JSON format
        try:
            with path.open("r") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
        return cfg

# -----------------------------------------------------------------------------
# CLI parameter coercion helpers
# These handle dynamic typing for strategy parameter overrides.
# -----------------------------------------------------------------------------
TRUE_SET = {"true", "yes", "on", "1"}
FALSE_SET = {"false", "no", "off", "0"}

def _coerce_value(s: str):
    # Attempts to infer type for CLI-provided strings.
    # Handles JSON, booleans, ints, floats, and comma-separated lists.
    try:
        if s.startswith("[") or s.startswith("{"):
            return json.loads(s)
    except Exception:
        pass

    low = s.strip().lower()
    if low in TRUE_SET:
        return True
    if low in FALSE_SET:
        return False

    # Try integer conversion, guarding against leading-zero strings.
    try:
        if low.startswith("0") and low != "0" and not low.startswith("0."):
            raise ValueError
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            pass

    # Allow comma-separated sequences for multi-value args.
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        return [_coerce_value(p) for p in parts]

    return s  # fallback: raw string

def parse_param_args(param_items: list[str]) -> dict:
    # Converts repeated CLI args like:
    #   --param stake=2 --param risk=0.1
    # into {"stake": 2, "risk": 0.1}
    params = {}
    for item in param_items or []:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k:
            continue
        params[k] = _coerce_value(v)
    return params

# -----------------------------------------------------------------------------
# Directory utility
# -----------------------------------------------------------------------------
def ensure_dir(p: Path):
    # Creates directory recursively if not present.
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Argument parser definition
# This CLI mirrors Backtrader’s general structure but adds COMP396 extensions.
# -----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="BT396 Backtrader harness")

    # Basic configuration options
    ap.add_argument("--config", default="config.yaml",
                    help="Path to YAML/JSON config file (optional).")
    ap.add_argument("--strategy", help="Strategy module name (in /strategies).")
    ap.add_argument("--strategy-class", help="Explicit class name (optional).")

    # Simulation-level overrides
    ap.add_argument("--cash", type=float, help="Starting cash (default 1,000,000).")
    ap.add_argument("--commission", type=float, help="Broker commission (e.g. 0.001).")
    ap.add_argument("--s-mult", type=float, help="Slippage multiplier for gap model.")
    ap.add_argument("--no-plot", action="store_true", help="Disable plotting outputs.")
    ap.add_argument("--data-dir", help="Path to folder containing 10 CSV files.")
    ap.add_argument("--output-dir", help="Folder for logs, plots, summaries.")
    ap.add_argument("--end-policy", choices=["liquidate", "hold"],
                    help="Final-day behavior for open positions.")

    # Strategy parameters (CLI only; replaces YAML per-strategy configs)
    ap.add_argument("--param", action="append", default=[],
                    help=("Strategy parameter override (key=value). Repeatable. "
                          "Values auto-parsed as int/float/bool/JSON/list."))

    # Logging / debugging / date filtering
    ap.add_argument("--debug", action="store_true",
                    help="Enable verbose framework logs (orders, fills, etc).")
    ap.add_argument("--fromdate", type=str, help="Start date YYYY-MM-DD.")
    ap.add_argument("--todate", type=str, help="End date YYYY-MM-DD.")

    return ap.parse_args()


# Added Jan26
# Added Jan26

class TradeTypeAnalyzer(bt.Analyzer):
    """
    Tracks trades separated by direction (long vs short) using realized PnL.
    Does not depend on strategy internals, so it works with SuperBayesian.
    """

    def __init__(self):
        self.trades_by_type = {
            'mean_reversion_long':   {'total': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0},
            'trend_following_long':  {'total': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0},
            'mean_reversion_short':  {'total': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0},
            'trend_following_short': {'total': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0},
        }

    def notify_trade(self, trade):
        """
        Called when a trade closes.

        We cannot see the original 'entry_type' of SuperBayesian's child
        strategies from here, so we classify only by direction:
          - long trades are counted under 'trend_following_long'
          - short trades are counted under 'trend_following_short'

        This still gives you directional breakdown while keeping the
        strategy code untouched.
        """
        if not trade.isclosed:
            return

        # Long if net size was positive during the trade, else short
        size = trade.size  # final net size of the closed trade
        if size > 0:
            key = 'trend_following_long'
        elif size < 0:
            key = 'trend_following_short'
        else:
            # Fallback: classify by PnL sign if size is not available
            key = 'trend_following_long' if trade.pnl >= 0 else 'trend_following_short'

        bucket = self.trades_by_type[key]
        bucket['total'] += 1
        bucket['pnl'] += trade.pnl

        if trade.pnl > 0:
            bucket['wins'] += 1
        elif trade.pnl < 0:
            bucket['losses'] += 1

    def get_analysis(self):
        return self.trades_by_type


# -----------------------------------------------------------------------------
# Main execution entrypoint
# -----------------------------------------------------------------------------
def main():
    # Parse CLI args and config
    args = parse_args()
    cfg = load_config(Path(args.config))

    # Merge CLI overrides (priority over file-based config)
    if args.strategy:                cfg["strategy"] = args.strategy
    if args.strategy_class:          cfg["strategy_class"] = args.strategy_class
    if args.cash is not None:        cfg["starting_cash"] = args.cash
    if args.commission is not None:  cfg["commission"] = args.commission
    if args.s_mult is not None:      cfg["s_mult"] = args.s_mult
    if args.no_plot:                 cfg["plot"] = False
    if args.data_dir:                cfg["data_dir"] = args.data_dir
    if args.output_dir:              cfg["output_dir"] = args.output_dir
    if args.end_policy:              cfg["end_policy"] = args.end_policy

    # Determine debug mode:
    # Enabled explicitly via --debug or automatically when under pytest.
    debug_flag = bool(args.debug or os.environ.get("PYTEST_CURRENT_TEST"))

    # Add project root to sys.path so that framework/ and strategies/ are importable.
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))

    # Lazy imports (prevents circular deps if framework modules import back)
    from framework.data_loader import add_10_csv_feeds
    from framework.strategies_loader import load_strategy_class
    from framework.analyzers import OpenOpenPnL, RealizedPnL, PDRatio, Activity, TruePortfolioPD
    from framework.strategy_base import COMP396BrokerConfig

    # Resolve I/O directories
    data_dir = (root / cfg["data_dir"]).resolve()
    output_dir = (root / cfg["output_dir"]).resolve()
    ensure_dir(output_dir)

    # -------------------------------------------------------------------------
    # Configure Backtrader’s Cerebro engine
    # -------------------------------------------------------------------------
    # Cerebro orchestrates the broker, strategies, data feeds, and analyzers.
    cerebro = bt.Cerebro(stdstats=False, preload=True, runonce=True)
    cerebro.broker.setcash(cfg["starting_cash"])
    if cfg["commission"] and cfg["commission"] > 0:
        cerebro.broker.setcommission(commission=cfg["commission"])

    # -------------------------------------------------------------------------
    # Parse optional date filters (applied to CSV feed loading)
    # -------------------------------------------------------------------------
    fromdate = None
    todate = None
    if args.fromdate:
        try:
            fromdate = datetime.strptime(args.fromdate, "%Y-%m-%d").date()
        except ValueError:
            sys.exit("Error: --fromdate must be in YYYY-MM-DD format.")
    if args.todate:
        try:
            todate = datetime.strptime(args.todate, "%Y-%m-%d").date()
        except ValueError:
            sys.exit("Error: --todate must be in YYYY-MM-DD format.")

    # -------------------------------------------------------------------------
    # Add 10 CSV data feeds
    # -------------------------------------------------------------------------
    # The data_loader automatically detects OHLCV columns, aligns by date,
    # and applies fromdate/todate filtering.
    datas = add_10_csv_feeds(cerebro, data_dir, fromdate=fromdate, todate=todate)

    # -------------------------------------------------------------------------
    # Load and wrap user strategy
    # -------------------------------------------------------------------------
    # load_strategy_class dynamically imports strategies.<module_name>
    # and wraps it in COMP396Base to enforce trading rules (overspend guard,
    # slippage, final-day liquidation, etc.).
    StrategyClass = load_strategy_class(cfg["strategy"], cfg["strategy_class"])

    # Parse CLI-provided parameter overrides
    params = parse_param_args(args.param)
    strategy_name = args.strategy or cfg["strategy"]

    # Display active configuration for reproducibility
    print("\n=== Backtest Configuration ===")
    print(f"Strategy: {strategy_name}")
    print("Parameters (CLI overrides):")
    if params:
        for k, v in params.items():
            print(f"  {k}: {v}")
    else:
        print("  <using strategy defaults>")
    print("================================\n")

    # -------------------------------------------------------------------------
    # Build COMP396 broker configuration object
    # -------------------------------------------------------------------------
    # This structure carries simulation settings to all strategy instances.
    # It is passed via the special '_comp396' param merged into bt.Strategy params.
    params["_comp396"] = COMP396BrokerConfig(
        s_mult=float(cfg["s_mult"]),
        end_policy=str(cfg["end_policy"]).lower(),
        output_dir=str(output_dir),
        debug=debug_flag,
    )

    if debug_flag:
        print("Debug logging ENABLED (orders, fills, slippage, trade PnL)")

    # -------------------------------------------------------------------------
    # Register strategy and analyzers with Cerebro
    # -------------------------------------------------------------------------
    cerebro.addstrategy(StrategyClass, **params)
    cerebro.addanalyzer(OpenOpenPnL, _name="oopnl")       # frictionless open→open PnL
    cerebro.addanalyzer(PDRatio, _name="pd")              # PD ratio based on open→open
    cerebro.addanalyzer(Activity, _name="activity")       # % of active trading days
    cerebro.addanalyzer(RealizedPnL, _name="realpnl")     # trade-level realized PnL
    cerebro.addanalyzer(TruePortfolioPD, _name="truepd")  # PD ratio based on true equity
    
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades') # Trade Analyser Added 24Nov
    cerebro.addanalyzer(TradeTypeAnalyzer, _name='trade_types')     # MR/TF Analyser Added 26Jan

    # -------------------------------------------------------------------------
    # Execute backtest
    # -------------------------------------------------------------------------
    strat = cerebro.run(maxcpus=1)[0]  # single-threaded for deterministic runs

    # -------------------------------------------------------------------------
    # Collect analyzer outputs
    # -------------------------------------------------------------------------
    # These analyzers return structured dictionaries with per-instrument and
    # portfolio-level metrics, consumed later by plotting and reporting functions.
    oopnl = strat.analyzers.oopnl.get_analysis()
    pdres = strat.analyzers.pd.get_analysis()
    act = strat.analyzers.activity.get_analysis()
    realpnl = strat.analyzers.realpnl.get_analysis()
    truepd = strat.analyzers.truepd.get_analysis()

    # -------------------------------------------------------------------------
    # Summarize and persist results
    # -------------------------------------------------------------------------
    # The run_summary.json file provides a compact overview of final results
    # for automated grading, comparisons, or batch tests.
    summary = {
        "final_value": float(cerebro.broker.getvalue()),
        "bankrupt": bool(oopnl.get("bankrupt", False)),
        "bankrupt_date": (
            oopnl.get("bankrupt_date").isoformat()
            if oopnl.get("bankrupt_date") is not None
            else None
        ),
        "open_pnl_pd_ratio": pdres.get("portfolio", {}).get("pd_ratio"),
        "true_pd_ratio": truepd.get("pd_ratio") if truepd else None,
        "activity_pct": act.get("activity_pct"),
        "end_policy": cfg["end_policy"],
        "s_mult": cfg["s_mult"]
    }

    # Write summary JSON to output directory
    with (output_dir / "run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # -------------------------------------------------------------------------
    # Per-series PD metrics and plots
    # -------------------------------------------------------------------------
    # save_per_series_pd writes PD ratios for each instrument and portfolio
    # to a JSON table for further inspection or visual dashboards.
    save_per_series_pd(pdres, output_dir, truepd)

    # -------------------------------------------------------------------------
    # Plot generation (optional)
    # -------------------------------------------------------------------------
    # These plotting functions render various aspects of performance:
    #   - Cumulative PnL (open→open, realized, equity)
    #   - Per-series contributions
    #   - Portfolio-level dashboards and drawdowns
    # They use matplotlib and are saved as PNGs.
    if cfg["plot"]:
        save_true_equity_plot(truepd, output_dir)
        save_equity_plot(oopnl, pdres, act, output_dir, truepd)
        save_per_series_plots(oopnl, output_dir)
        save_combined_equity_dashboard(oopnl, realpnl, pdres, act, output_dir, truepd)
        save_equity_dashboard(oopnl, pdres, act, output_dir, truepd)
        save_all_series_equity(oopnl, output_dir)
        save_portfolio_underwater(oopnl, output_dir)
        save_realized_equity_plot(realpnl, output_dir)

    # Console summary output for quick inspection
    print(json.dumps(summary, indent=2))
    
    # Trade Analyser Added 24Nov
    trade_analysis = strat.analyzers.trades.get_analysis()
    print("=== Trade Analyzer Stats ===")
    if 'total' in trade_analysis:
        print(f"Total closed trades: {trade_analysis['total'].get('closed', 'N/A')}")
        print(f"Wins: {trade_analysis['won'].get('total', 'N/A') if 'won' in trade_analysis else 'N/A'}")
        print(f"Losses: {trade_analysis['lost'].get('total', 'N/A') if 'lost' in trade_analysis else 'N/A'}")
    else:
        print("Trade analysis data unavailable.")


    # Get trade type breakdown  Added Jan26 
    trade_types = strat.analyzers.trade_types.get_analysis()
    
    print("\n=== Trade Breakdown by Strategy Type ===")
    print(f"\nMean Reversion Trades Long:")
    print(f"  Total trades: {trade_types['mean_reversion_long']['total']}")
    print(f"  Wins: {trade_types['mean_reversion_long']['wins']}")
    print(f"  Losses: {trade_types['mean_reversion_long']['losses']}")
    print(f"  Total PnL: ${trade_types['mean_reversion_long']['pnl']:.2f}")
    if trade_types['mean_reversion_long']['total'] > 0:
        win_rate = (trade_types['mean_reversion_long']['wins'] / trade_types['mean_reversion_long']['total']) * 100
        print(f"  Win Rate: {win_rate:.1f}%")
    
    print(f"\nTrend Following Trades Long:")
    print(f"  Total trades: {trade_types['trend_following_long']['total']}")
    print(f"  Wins: {trade_types['trend_following_long']['wins']}")
    print(f"  Losses: {trade_types['trend_following_long']['losses']}")
    print(f"  Total PnL: ${trade_types['trend_following_long']['pnl']:.2f}")
    if trade_types['trend_following_long']['total'] > 0:
        win_rate = (trade_types['trend_following_long']['wins'] / trade_types['trend_following_long']['total']) * 100
        print(f"  Win Rate: {win_rate:.1f}%")
    
    print(f"\nMean Reversion Trades Short:")
    print(f"  Total trades: {trade_types['mean_reversion_short']['total']}")
    print(f"  Wins: {trade_types['mean_reversion_short']['wins']}")
    print(f"  Losses: {trade_types['mean_reversion_short']['losses']}")
    print(f"  Total PnL: ${trade_types['mean_reversion_short']['pnl']:.2f}")
    if trade_types['mean_reversion_short']['total'] > 0:
        win_rate = (trade_types['mean_reversion_short']['wins'] / trade_types['mean_reversion_short']['total']) * 100
        print(f"  Win Rate: {win_rate:.1f}%")
        
    print(f"\nTrend Following Trades Shorts:")
    print(f"  Total trades: {trade_types['trend_following_short']['total']}")
    print(f"  Wins: {trade_types['trend_following_short']['wins']}")
    print(f"  Losses: {trade_types['trend_following_short']['losses']}")
    print(f"  Total PnL: ${trade_types['trend_following_short']['pnl']:.2f}")
    if trade_types['trend_following_short']['total'] > 0:
        win_rate = (trade_types['trend_following_short']['wins'] / trade_types['trend_following_short']['total']) * 100
        print(f"  Win Rate: {win_rate:.1f}%")

# -----------------------------------------------------------------------------
# Entrypoint guard
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # If run without arguments, print framework version and exit
    if len(sys.argv) <= 1:
        from framework import __version__ as BT396_VERSION, __release_date__ as BT396_DATE
        print(f"{BT396_VERSION} ({BT396_DATE})")
    else:
        main()
