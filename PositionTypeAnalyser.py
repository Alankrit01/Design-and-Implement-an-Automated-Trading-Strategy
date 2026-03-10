"""
PositionTypeAnalyzer
====================
A self-contained Backtrader Analyzer that classifies every completed POSITION
(not every Backtrader trade-object) into one of the four strategy types:

    mean_reversion_long  |  mean_reversion_short
    trend_following_long |  trend_following_short

Usage in main.py
----------------

get_analysis() return structure
--------------------------------
{
    'mean_reversion_long':   PositionStats,
    'mean_reversion_short':  PositionStats,
    'trend_following_long':  PositionStats,
    'trend_following_short': PositionStats,
    'other':                 PositionStats,
    'positions':             [PositionRecord, ...]   # full per-position log
}

PositionStats fields  : total, wins, losses, breakeven, pnl, avg_pnl,
                        win_rate, avg_win, avg_loss, expectancy, profit_factor

PositionRecord fields : symbol, entry_type, entry_bar, entry_date,
                        entry_price, exit_bar, exit_date, exit_price,
                        size, side, pnl, won
"""

import backtrader as bt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

VALID_TYPES = {
    'mean_reversion_long',
    'mean_reversion_short',
    'trend_following_long',
    'trend_following_short',
}


@dataclass
class PositionRecord:
    """One complete round-trip position."""
    symbol:      str
    entry_type:  str
    entry_bar:   int
    entry_date:  str          # ISO date string
    entry_price: float
    exit_bar:    int
    exit_date:   str
    exit_price:  float
    size:        float        # absolute share count
    side:        int          # +1 long, -1 short
    pnl:         float        # in currency units
    won:         bool


@dataclass
class PositionStats:
    total:          int   = 0
    wins:           int   = 0
    losses:         int   = 0
    breakeven:      int   = 0
    pnl:            float = 0.0
    avg_pnl:        float = 0.0
    win_rate:       float = 0.0
    avg_win:        float = 0.0
    avg_loss:       float = 0.0
    expectancy:     float = 0.0
    profit_factor:  float = 0.0


def _compute_stats(records: List[PositionRecord]) -> PositionStats:
    s = PositionStats()
    if not records:
        return s

    s.total     = len(records)
    pnls        = [r.pnl for r in records]
    s.pnl       = sum(pnls)
    s.avg_pnl   = s.pnl / s.total

    wins  = [p for p in pnls if p > 0]
    loss  = [p for p in pnls if p < 0]
    be    = [p for p in pnls if p == 0]

    s.wins      = len(wins)
    s.losses    = len(loss)
    s.breakeven = len(be)
    s.win_rate  = (s.wins / s.total * 100.0) if s.total > 0 else 0.0
    s.avg_win   = (sum(wins) / len(wins))  if wins else 0.0
    s.avg_loss  = (sum(loss) / len(loss))  if loss else 0.0

    # Expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
    loss_rate    = s.losses / s.total if s.total > 0 else 0.0
    s.expectancy = (s.win_rate / 100.0) * s.avg_win + loss_rate * s.avg_loss

    # Profit factor = gross_profit / |gross_loss|  (inf if no losses)
    gross_profit = sum(wins)
    gross_loss   = abs(sum(loss))
    if gross_loss > 1e-9:
        s.profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        s.profit_factor = float('inf')
    else:
        s.profit_factor = 0.0

    return s


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class PositionTypeAnalyzer(bt.Analyzer):
    """
    Classifies every completed position by entry type.

    Reads directly from ChildStrategy.closed_trades at stop() — no broker
    callbacks, no timing dependencies on notify_order.  Each child appends
    one dict per closed position inside record_exit(), which is called
    synchronously from SuperBayesian.next(), so all data is guaranteed to
    be present by the time the backtest ends.
    """

    def start(self):
        self._records: List[PositionRecord] = []

    def notify_trade(self, trade):
        """
        Called by Backtrader when a trade closes.  This is the exact same hook
        that TradeTypeAnalyzer in main.py uses successfully.  We read entry_type
        from self.strategy.state[data] (set by SuperBayesian and cleared one bar
        later), and all PnL / price info from the trade object itself.
        """
        if not trade.isclosed:
            return



        data = trade.data
        sym  = data._name

        # Read entry_type from the top-level strategy state — same as TradeTypeAnalyzer
        entry_type = 'other'
        try:
            if hasattr(self.strategy, 'state') and data in self.strategy.state:
                et = self.strategy.state[data].get('entry_type', '')
                if et in VALID_TYPES:
                    entry_type = et
        except Exception:
            pass

        # trade.long is reliable even on closed trades (size is 0 when closed)
        side = 1 if trade.long else -1
        pnl  = float(trade.pnl)

        # dtopen/dtclose are Backtrader matplotlib float dates — use bt.num2date()
        try:
            entry_date = bt.num2date(trade.dtopen).strftime('%Y-%m-%d')
        except Exception:
            entry_date = 'unknown'
        try:
            exit_date = bt.num2date(trade.dtclose).strftime('%Y-%m-%d')
        except Exception:
            exit_date = 'unknown'

        entry_bar   = trade.baropen
        exit_bar    = trade.barclose
        entry_price = float(trade.price)

        # trade.size is 0 on closed trades and history is empty in this framework.
        # The strategy stores the actual share count in _exit_sizes[sym] at the
        # moment the exit signal fires (before the position is closed).
        size = 0.0
        try:
            size = float(self.strategy._exit_sizes.get(sym, 0))
        except Exception:
            pass

        # Exit price: derive algebraically from pnl = side * (exit - entry) * size
        if size > 0:
            exit_price = entry_price + (pnl / (side * size))
        else:
            exit_price = 0.0

        self._records.append(PositionRecord(
            symbol      = sym,
            entry_type  = entry_type,
            side        = side,
            size        = size,
            entry_price = entry_price,
            exit_price  = exit_price,
            pnl         = pnl,
            won         = pnl > 0,
            entry_bar   = entry_bar,
            entry_date  = entry_date,
            exit_bar    = exit_bar,
            exit_date   = exit_date,
        ))

    def stop(self):
        pass

    # ------------------------------------------------------------------
    def get_analysis(self):
        # Group records by entry_type
        grouped: dict = defaultdict(list)
        for rec in self._records:
            grouped[rec.entry_type].append(rec)

        # Build stats for every known type (even if empty)
        all_types = list(VALID_TYPES) + ['other']
        result = {}
        for t in all_types:
            result[t] = _compute_stats(grouped.get(t, []))

        # Full position log for downstream use
        result['positions'] = list(self._records)
        return result

def print_position_analysis(analysis: dict):
    """
    Print a formatted breakdown of positions by type.
    Call with the dict returned by PositionTypeAnalyzer.get_analysis().
    """
    LABELS = {
        'mean_reversion_long':   'Mean Reversion Long',
        'mean_reversion_short':  'Mean Reversion Short',
        'trend_following_long':  'Trend Following Long',
        'trend_following_short': 'Trend Following Short',
        'other':                 'Other / Unlabelled',
    }

    print("\n=== Position Breakdown by Strategy Type ===")

    for key, label in LABELS.items():
        s: PositionStats = analysis.get(key, PositionStats())
        print(f"\n{label}:")
        print(f"  Positions   : {s.total}")
        if s.total == 0:
            continue
        print(f"  Wins        : {s.wins}")
        print(f"  Losses      : {s.losses}")
        print(f"  Breakeven   : {s.breakeven}")
        print(f"  Win Rate    : {s.win_rate:.1f}%")
        print(f"  Total PnL   : ${s.pnl:,.2f}")
        print(f"  Avg PnL     : ${s.avg_pnl:,.2f}")
        print(f"  Avg Win     : ${s.avg_win:,.2f}")
        print(f"  Avg Loss    : ${s.avg_loss:,.2f}")
        print(f"  Expectancy  : ${s.expectancy:,.2f}")
        pf = f"{s.profit_factor:.2f}" if s.profit_factor != float('inf') else "∞"
        print(f"  Profit Fac. : {pf}")

    # Summary table
    positions: List[PositionRecord] = analysis.get('positions', [])
    total_pos = len(positions)
    total_pnl = sum(r.pnl for r in positions)
    total_wins = sum(1 for r in positions if r.won)
    print(f"\n--- Portfolio Total ---")
    print(f"  Total positions : {total_pos}")
    print(f"  Total wins      : {total_wins}")
    print(f"  Overall PnL     : ${total_pnl:,.2f}")
    if total_pos > 0:
        print(f"  Overall win rate: {total_wins / total_pos * 100:.1f}%")


def print_position_log(analysis: dict, max_rows: int = 50):
    """
    Print a compact per-position log sorted chronologically.
    Useful for debugging — set max_rows=0 to print all.
    """
    positions: List[PositionRecord] = analysis.get('positions', [])
    if not positions:
        print("No completed positions recorded.")
        return

    positions = sorted(positions, key=lambda r: r.entry_bar)
    if max_rows and len(positions) > max_rows:
        positions = positions[:max_rows]
        truncated = True
    else:
        truncated = False

    print(f"\n{'Symbol':<10} {'Type':<24} {'EntryDate':<12} "
          f"{'ExitDate':<12} {'PnL':>12}  W/L")
    print("-" * 110)
    for r in positions:
        side_str = "LONG" if r.side > 0 else "SHORT"
        wl       = "WIN"  if r.won  else ("LOSS" if r.pnl < 0 else "BE")
        print(
            f"{r.symbol:<10} {r.entry_type:<24} {r.entry_date:<12} "
            f"{r.exit_date:<12}"
            f"{r.pnl:>12,.2f}  {wl}"
        )

    if truncated:
        print(f"  ... (showing first {max_rows} of {len(analysis['positions'])} positions)")
