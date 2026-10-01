"""
Solana AI Quant Agent - Industrial-Grade Causal Backtest Engine
Implements strict next-open execution, M2M dynamic drawdown tracking,
Wilson 95% confidence intervals, and 1x normal vs 3x stress testing.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from config import StrategyConfig, SolanaFrictionConfig, PaperTradingConfig
from factors import compute_all_factors
from strategy import SolanaTrendAgent
from paper_broker import PaperBroker, PaperTradeRecord


def calculate_wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Computes Wilson 95% Score Confidence Interval for win rate."""
    if total <= 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / total
    denominator = 1.0 + (z**2 / total)
    centre_adjusted_probability = p_hat + (z**2 / (2 * total))
    adjusted_std_dev = math.sqrt((p_hat * (1 - p_hat) / total) + (z**2 / (4 * total**2)))
    
    lower_bound = (centre_adjusted_probability - z * adjusted_std_dev) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_std_dev) / denominator
    span = upper_bound - lower_bound
    return lower_bound, upper_bound, span


class SolanaBacktestEngine:
    """
    Backtesting engine enforcing causal execution:
    - Bar t close evaluates indicators and emits signal
    - Bar t+1 OPEN price executes the order with full Solana DEX friction
    """
    def __init__(
        self,
        strategy_config: StrategyConfig = None,
        friction_config: SolanaFrictionConfig = None,
        paper_config: PaperTradingConfig = None
    ):
        self.strat_cfg = strategy_config or StrategyConfig()
        self.frict_cfg = friction_config or SolanaFrictionConfig()
        self.paper_cfg = paper_config or PaperTradingConfig()

    def run_single_pass(
        self,
        df: pd.DataFrame,
        stress_mult: float = 1.0,
        sol_price: float = 140.0
    ) -> Dict[str, Any]:
        """
        Executes a single backtest pass across the historical DataFrame.
        """
        # 1. Compute Indicators
        data = compute_all_factors(df, self.strat_cfg)
        
        # 2. Initialize Agent & Broker
        agent = SolanaTrendAgent(self.strat_cfg, self.frict_cfg)
        broker = PaperBroker(self.paper_cfg, self.frict_cfg, stress_mult=stress_mult)

        pending_signal = None
        entry_bar_idx = 0

        # Loop through bars causally
        for i in range(len(data) - 1):
            curr_bar = data.iloc[i]
            next_bar = data.iloc[i + 1]
            timestamp_str = str(curr_bar['timestamp'])

            # ---------------------------------------------------------
            # Step A: Execute pending signal from previous bar on next OPEN
            # ---------------------------------------------------------
            if pending_signal is not None:
                exec_price = float(next_bar['open'])
                pool_liq = float(next_bar.get('liquidity_usd', 1000000.0))

                if pending_signal.action == 'BUY' and broker.position is None:
                    broker.execute_buy(
                        token_symbol="SOL",
                        token_mint=self.paper_cfg.target_token_mint,
                        price=exec_price,
                        trade_usd=self.strat_cfg.trade_size_usdc,
                        stop_loss=pending_signal.stop_loss,
                        take_profit=pending_signal.take_profit,
                        trailing_stop=pending_signal.trailing_stop,
                        sol_price=sol_price,
                        pool_liquidity=pool_liq,
                        timestamp=next_bar['timestamp']
                    )
                    entry_bar_idx = i + 1

                elif pending_signal.action == 'SELL' and broker.position is not None:
                    hold_bars = max((i + 1) - entry_bar_idx, 1)
                    broker.execute_sell(
                        price=exec_price,
                        reason=pending_signal.reason,
                        sol_price=sol_price,
                        pool_liquidity=pool_liq,
                        timestamp=next_bar['timestamp'],
                        hold_bars=hold_bars
                    )

                pending_signal = None

            # ---------------------------------------------------------
            # Step B: Update position tracking and M2M Equity on curr bar
            # ---------------------------------------------------------
            if broker.position is not None:
                # Update peak high for trailing stop
                if curr_bar['high'] > broker.position.highest_price:
                    broker.position.highest_price = float(curr_bar['high'])

            broker.update_m2m(
                current_price=float(curr_bar['close']),
                sol_price=sol_price,
                timestamp=timestamp_str
            )

            # ---------------------------------------------------------
            # Step C: Evaluate strategy at current bar CLOSE
            # ---------------------------------------------------------
            pos_dict = None
            if broker.position is not None:
                pos_dict = {
                    'size': broker.position.token_amount,
                    'entry_price': broker.position.entry_price,
                    'highest_price': broker.position.highest_price,
                    'stop_loss': broker.position.stop_loss,
                    'take_profit': broker.position.take_profit
                }

            signal = agent.evaluate_bar(curr_bar, pos_dict, sol_price_usd=sol_price)
            if signal.action in ('BUY', 'SELL'):
                pending_signal = signal

        # Close any open position at the final bar close
        if broker.position is not None:
            last_bar = data.iloc[-1]
            hold_bars = len(data) - entry_bar_idx
            broker.execute_sell(
                price=float(last_bar['close']),
                reason="END_OF_BACKTEST",
                sol_price=sol_price,
                pool_liquidity=float(last_bar.get('liquidity_usd', 1000000.0)),
                timestamp=last_bar['timestamp'],
                hold_bars=hold_bars
            )
            broker.update_m2m(float(last_bar['close']), sol_price, str(last_bar['timestamp']))

        # ---------------------------------------------------------
        # Step D: Performance & Statistical Audit (Deep Quant Metrics)
        # ---------------------------------------------------------
        trades = broker.trade_history
        total_trades = len(trades)
        final_equity = broker.get_portfolio_equity(data.iloc[-1]['close'], sol_price)
        net_profit_usd = final_equity - broker.initial_equity_usd
        return_pct = (net_profit_usd / broker.initial_equity_usd) * 100.0

        # Benchmark (Buy & Hold) Return
        start_price = float(data.iloc[0]['close'])
        end_price = float(data.iloc[-1]['close'])
        bench_return_pct = ((end_price - start_price) / start_price) * 100.0
        alpha_pct = return_pct - bench_return_pct

        # Annualized Sharpe, Sortino, Calmar
        if len(broker.equity_curve) > 5:
            eq_series = pd.Series([pt['equity'] for pt in broker.equity_curve])
            bar_rets = eq_series.pct_change().dropna()
            # 15m intervals: 96 bars/day * 365 days = 35040 bars/year
            ann_factor = math.sqrt(35040)
            if bar_rets.std() > 1e-9:
                sharpe_ratio = float((bar_rets.mean() / bar_rets.std()) * ann_factor)
            else:
                sharpe_ratio = 0.0

            downside_rets = bar_rets[bar_rets < 0]
            if len(downside_rets) > 0 and downside_rets.std() > 1e-9:
                sortino_ratio = float((bar_rets.mean() / downside_rets.std()) * ann_factor)
            else:
                sortino_ratio = max(sharpe_ratio, 0.0)

            max_dd_pct = broker.max_drawdown_pct * 100.0
            calmar_ratio = float(return_pct / max(max_dd_pct, 0.1)) if return_pct > 0 else 0.0
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            calmar_ratio = 0.0
            max_dd_pct = 0.0

        if total_trades > 0:
            winning_trades = [t for t in trades if t.net_pnl_usd > 0]
            losing_trades = [t for t in trades if t.net_pnl_usd <= 0]
            win_count = len(winning_trades)
            win_rate = (win_count / total_trades) * 100.0

            gross_gain = sum(t.net_pnl_usd for t in winning_trades)
            gross_loss = abs(sum(t.net_pnl_usd for t in losing_trades))
            profit_factor = (gross_gain / gross_loss) if gross_loss > 0 else (99.0 if gross_gain > 0 else 0.0)

            avg_win = (gross_gain / len(winning_trades)) if winning_trades else 0.0
            avg_loss = (gross_loss / len(losing_trades)) if losing_trades else 0.0
            win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else (99.0 if avg_win > 0 else 0.0)
            avg_hold_bars = sum(t.hold_bars for t in trades) / total_trades

            total_fees = sum(t.total_fees_usd for t in trades)
            total_slippage = sum(t.total_slippage_usd for t in trades)
            total_friction = sum(t.total_friction_usd for t in trades)

            # Top 3 Trade Concentration Check
            sorted_pnls = sorted([t.net_pnl_usd for t in trades], reverse=True)
            top3_sum = sum(sorted_pnls[:3])
            concentration_pct = (top3_sum / net_profit_usd * 100.0) if net_profit_usd > 0 else 100.0

            # Wilson 95% Confidence Interval
            w_low, w_high, w_span = calculate_wilson_interval(win_count, total_trades)
            
            # Reliability Grading
            if total_trades >= 30 and w_span <= 0.35:
                reliability = "A_RELIABLE"
            elif total_trades >= 20:
                reliability = "B_MARGINAL"
            else:
                reliability = "F_UNRELIABLE"
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            win_loss_ratio = 0.0
            avg_hold_bars = 0.0
            total_fees = 0.0
            total_slippage = 0.0
            total_friction = 0.0
            concentration_pct = 0.0
            w_low, w_high, w_span = 0.0, 0.0, 0.0
            reliability = "F_NO_TRADES"

        # ---------------------------------------------------------
        # Step E: Prepare Charting Payload (Candlesticks, Markers, Overlays)
        # ---------------------------------------------------------
        # K-line data for ECharts: [time, open, close, low, high, volume, zlema_fast, zlema_slow, supertrend]
        ohlcv_bars = []
        equity_curve_points = []
        drawdown_curve_points = []

        # Map timestamps to equity points
        m2m_map = {pt['timestamp']: pt for pt in broker.equity_curve}

        for i in range(len(data)):
            row = data.iloc[i]
            t_str = str(row['timestamp'])
            ohlcv_bars.append([
                t_str,
                float(row['open']),
                float(row['close']),
                float(row['low']),
                float(row['high']),
                float(row.get('volume_usd', 0.0)),
                float(row.get('zlema_fast', row['close'])),
                float(row.get('zlema_slow', row['close'])),
                float(row.get('supertrend', row['close']))
            ])

            # M2M equity & benchmark equity
            bench_eq = broker.initial_equity_usd * (float(row['close']) / start_price)
            pt = m2m_map.get(t_str)
            eq_val = pt['equity'] if pt else broker.initial_equity_usd
            dd_val = (pt['drawdown_pct'] * 100.0) if pt else 0.0

            equity_curve_points.append({
                "time": t_str,
                "strategy_equity": round(eq_val, 2),
                "benchmark_equity": round(bench_eq, 2)
            })
            drawdown_curve_points.append({
                "time": t_str,
                "drawdown_pct": round(dd_val, 2)
            })

        # Chart Markers for BUY and SELL execution on K-line
        chart_markers = []
        for t in trades:
            # BUY marker
            chart_markers.append({
                "name": f"BUY #{t.id}",
                "coord": [str(t.entry_time), float(t.entry_price)],
                "value": f"买入 #{t.id}",
                "itemStyle": { "color": "#10b981" },
                "trade_id": t.id,
                "type": "BUY",
                "price": t.entry_price
            })
            # SELL marker
            pnl_sign = "+" if t.net_pnl_usd >= 0 else ""
            chart_markers.append({
                "name": f"SELL #{t.id}",
                "coord": [str(t.exit_time), float(t.exit_price)],
                "value": f"卖出 #{t.id} ({pnl_sign}${t.net_pnl_usd:.1f})",
                "itemStyle": { "color": "#ef4444" },
                "trade_id": t.id,
                "type": "SELL",
                "price": t.exit_price,
                "net_pnl": t.net_pnl_usd,
                "reason": t.exit_reason
            })

        return {
            "stress_mult": stress_mult,
            "total_bars": len(data),
            "start_time": str(data.iloc[0]['timestamp']),
            "end_time": str(data.iloc[-1]['timestamp']),
            "initial_equity": broker.initial_equity_usd,
            "final_equity": final_equity,
            "net_profit_usd": net_profit_usd,
            "return_pct": return_pct,
            "bench_return_pct": bench_return_pct,
            "alpha_pct": alpha_pct,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "win_loss_ratio": win_loss_ratio,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_hold_bars": avg_hold_bars,
            "max_drawdown_pct": max_dd_pct,
            "max_drawdown_usd": broker.max_drawdown_usd,
            "total_fees_usd": total_fees,
            "total_slippage_usd": total_slippage,
            "total_friction_usd": total_friction,
            "top3_concentration_pct": concentration_pct,
            "wilson_ci_low": w_low * 100.0,
            "wilson_ci_high": w_high * 100.0,
            "wilson_ci_span": w_span * 100.0,
            "reliability": reliability,
            "trades": trades,
            "ohlcv_bars": ohlcv_bars,
            "chart_markers": chart_markers,
            "equity_curve_points": equity_curve_points,
            "drawdown_curve_points": drawdown_curve_points
        }

    def run_full_stress_audit(self, df: pd.DataFrame, sol_price: float = 140.0) -> Dict[str, Any]:
        """
        Runs both 1x Baseline and 3x Extreme Stress test to verify strategy anti-fragility.
        """
        res_1x = self.run_single_pass(df, stress_mult=1.0, sol_price=sol_price)
        res_3x = self.run_single_pass(df, stress_mult=3.0, sol_price=sol_price)

        passed_3x = (res_3x['net_profit_usd'] > 0 and res_3x['profit_factor'] >= 1.05)
        passed_lln = (res_1x['total_trades'] >= 25 and res_1x['reliability'] in ('A_RELIABLE', 'B_MARGINAL'))
        passed_concentration = (res_1x['top3_concentration_pct'] <= 50.0)

        overall_status = "PASSED" if (passed_3x and passed_lln and passed_concentration) else "REJECTED"

        return {
            "res_1x": res_1x,
            "res_3x": res_3x,
            "passed_3x": passed_3x,
            "passed_lln": passed_lln,
            "passed_concentration": passed_concentration,
            "overall_status": overall_status
        }
