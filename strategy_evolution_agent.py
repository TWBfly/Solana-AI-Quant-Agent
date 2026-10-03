"""
Solana AI Quant Agent - Strategy Evolution & Autonomous Optimization Agent
Implements deep trade-candle micro-diagnostics, multi-dimensional optimization planning,
code refactoring, causal backtest validation, versioned evolution history,
and iterative base-checkpoint freezing.
"""

import os
import json
import time
import math
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from ai_service import ai_service
from backtester import SolanaBacktestEngine
from data_feed import HistoricalMarketFeed, DexScreenerClient
from strategy_transpiler import StrategyTranspiler
from strategy_manager import strategy_manager
from config import PaperTradingConfig


class StrategyEvolutionAgent:
    """
    Autonomous Quantitative Strategy Evolution Agent.
    - Diagnoses trades on candlesticks (false breakouts, MAE, MFE, friction drag)
    - Formulates multi-dimensional optimization plans
    - Evolves and compiles Python BaseStrategy source code
    - Validates via next-open causal backtesting
    - Tracks full iteration history and freezes superior checkpoints as Base
    """

    STORAGE_DIR = "data/strategy_evolution"

    def __init__(self, storage_dir: str = "data/strategy_evolution"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.dex_client = DexScreenerClient()
        self.paper_cfg = PaperTradingConfig()

    def _get_history_file(self, strategy_id: str) -> str:
        clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', strategy_id)
        return os.path.join(self.storage_dir, f"{clean_id}_evolution.json")

    def load_history(self, strategy_id: str) -> Dict[str, Any]:
        """Loads versioned evolution history for a strategy."""
        path = self._get_history_file(strategy_id)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[EvolutionAgent] 读取演化记录失败 ({e})，初始化新历史")
        return {
            "strategy_id": strategy_id,
            "active_base_id": None,
            "iterations": [],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def save_history(self, strategy_id: str, history: Dict[str, Any]):
        """Persists evolution history for a strategy."""
        path = self._get_history_file(strategy_id)
        history["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def get_market_df(self, token_sym: str = "SOL", timeframe: str = "15m", bars: int = 500) -> pd.DataFrame:
        """Fetches market K-lines for backtesting."""
        tf_map = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}
        tf_minutes = tf_map.get(timeframe, 15)

        token_info = HistoricalMarketFeed.SUPPORTED_TOKENS.get(token_sym, {})
        mint_addr = token_info.get("mint", self.paper_cfg.target_token_mint)
        telemetry = self.dex_client.get_primary_pair_telemetry(mint_addr)
        anchor_price = telemetry["price_usd"] if telemetry else token_info.get("default_price", 135.0)
        anchor_liq = telemetry["liquidity_usd"] if telemetry else token_info.get("default_liq", 35000000.0)

        return HistoricalMarketFeed.get_market_data(
            symbol=token_sym,
            timeframe_minutes=tf_minutes,
            bars_count=bars,
            start_price=anchor_price,
            base_liquidity=anchor_liq,
            data_dir="data",
            seed=99
        )

    def analyze_trade_candle_points(self, df: pd.DataFrame, backtest_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep quantitative micro-analysis of execution points against Candlesticks:
        - False breakout entry rate (entered near candle high, reversed immediately)
        - MAE (Max Adverse Excursion) and MFE (Max Favorable Excursion)
        - Money left on the table (surrendered peak profits)
        - Friction and slippage drag percentage
        - Stop loss efficiency (premature stop-outs vs trend-following rides)
        """
        # Always compute K-line Market Context first
        kline_market_context = {}
        if df is not None and len(df) > 0:
            start_dt = str(df.iloc[0].get("timestamp", ""))
            end_dt = str(df.iloc[-1].get("timestamp", ""))
            start_px = float(df.iloc[0]["close"])
            end_px = float(df.iloc[-1]["close"])
            px_chg_pct = round((end_px - start_px) / start_px * 100.0, 2) if start_px > 0 else 0.0
            high_px = float(df["high"].max())
            low_px = float(df["low"].min())
            atr_approx = float((df["high"] - df["low"]).mean())
            atr_ratio_pct = round((atr_approx / end_px) * 100.0, 2) if end_px > 0 else 0.0
            
            regime = "宽幅震荡胶着" if abs(px_chg_pct) < 3.0 else ("单边上涨趋势" if px_chg_pct > 0 else "单边下跌走势")
            kline_market_context = {
                "total_bars": len(df),
                "time_span": f"{start_dt} 至 {end_dt}",
                "price_range": f"${low_px:.2f} ~ ${high_px:.2f} (区间涨跌: {px_chg_pct:+.2f}%)",
                "avg_bar_atr": f"${atr_approx:.2f} ({atr_ratio_pct:.2f}% 波动度)",
                "market_regime": regime
            }

        trades = backtest_result.get("trades", [])
        total_trades = len(trades)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "summary": "回测区间无成交触发，进场阈值过高或过滤条件过于严苛。",
                "false_breakout_count": 0,
                "false_breakout_ratio": 0.0,
                "avg_mae_pct": 0.0,
                "avg_mfe_pct": 0.0,
                "surrendered_profit_usd": 0.0,
                "friction_drag_pct": 0.0,
                "top_flaws": ["入场门槛过苛刻导致全区间零交易", "多因子共振条件严苛错过真实趋势波段"],
                "kline_market_context": kline_market_context,
                "worst_trade_cases": [],
                "surrender_cases": [],
                "exit_reasons_breakdown": {}
            }

        winning_trades = [t for t in trades if getattr(t, "net_pnl_usd", 0) > 0]
        losing_trades = [t for t in trades if getattr(t, "net_pnl_usd", 0) <= 0]

        # Analyze entry timing on K-lines
        false_breakout_count = 0
        mae_list = []
        mfe_list = []
        surrendered_profits = 0.0

        ts_map = {str(row['timestamp']): idx for idx, row in df.iterrows()}

        def _get(obj, field, default=None):
            if isinstance(obj, dict):
                return obj.get(field, default)
            return getattr(obj, field, default)

        for t in trades:
            entry_ts = str(_get(t, "entry_time", ""))
            exit_ts = str(_get(t, "exit_time", ""))
            entry_idx = ts_map.get(entry_ts)
            exit_idx = ts_map.get(exit_ts)

            entry_px = float(_get(t, "entry_price", 0.0))
            exit_px = float(_get(t, "exit_price", 0.0))
            token_amt = float(_get(t, "token_amount", 1.0))
            net_pnl = float(_get(t, "net_pnl_usd", 0.0))
            hold_bars = int(_get(t, "hold_bars", 0))

            if entry_idx is not None and exit_idx is not None and exit_idx >= entry_idx:
                holding_bars = df.iloc[entry_idx: exit_idx + 1]

                if len(holding_bars) > 0 and entry_px > 0:
                    max_px = float(holding_bars['high'].max())
                    min_px = float(holding_bars['low'].min())

                    # Long trades MAE & MFE
                    mae_pct = max(0.0, (entry_px - min_px) / entry_px * 100.0)
                    mfe_pct = max(0.0, (max_px - entry_px) / entry_px * 100.0)
                    mae_list.append(mae_pct)
                    mfe_list.append(mfe_pct)

                    # Surrendered profit
                    peak_usd = (max_px - entry_px) * token_amt
                    if peak_usd > net_pnl and peak_usd > 5.0:
                        surrendered_profits += (peak_usd - net_pnl)

                # Check if entered on a giant wick and reversed in 2 bars (false breakout)
                entry_bar = df.iloc[entry_idx]
                bar_range = max(float(entry_bar['high']) - float(entry_bar['low']), 1e-6)
                upper_body = (float(entry_px) - float(entry_bar['low'])) / bar_range
                if upper_body > 0.75 and net_pnl < 0 and hold_bars <= 3:
                    false_breakout_count += 1
            else:
                # If timestamp not mapped, fallback to trade's precomputed mae/mfe if present
                if _get(t, "mae_pct") is not None:
                    mae_list.append(float(_get(t, "mae_pct")))
                if _get(t, "mfe_pct") is not None:
                    mfe_list.append(float(_get(t, "mfe_pct")))
                if _get(t, "surrendered_profit_usd") is not None:
                    surrendered_profits += float(_get(t, "surrendered_profit_usd"))

        avg_mae = float(np.mean(mae_list)) if mae_list else 0.0
        avg_mfe = float(np.mean(mfe_list)) if mfe_list else 0.0
        false_breakout_ratio = (false_breakout_count / total_trades) * 100.0

        total_friction = backtest_result.get("total_friction_usd", 0.0)
        net_profit = backtest_result.get("net_profit_usd", 0.0)
        gross_profit = net_profit + total_friction
        friction_drag = (total_friction / max(abs(gross_profit), 1.0)) * 100.0 if gross_profit != 0 else 0.0

        # Construct diagnosis flaws
        top_flaws = []
        if false_breakout_ratio > 25.0:
            top_flaws.append(f"假突破追高被套率达 {false_breakout_ratio:.1f}%，进场追涨缺乏动量确认与波动率防守。")
        if avg_mae > 2.0:
            top_flaws.append(f"持仓平均最大逆向浮亏 (MAE) 达到 {avg_mae:.2f}%，缺少保本或动态收紧止损保护。")
        if surrendered_profits > 50.0:
            top_flaws.append(f"盈利单利润回吐严重，累计回吐峰值浮盈 ${surrendered_profits:.1f}，缺乏阶梯式追踪止盈 (Trailing Stop)。")
        if total_friction > 30.0 and len(trades) > 15:
            top_flaws.append(f"过度频繁交易导致 Solana DEX 摩擦损耗 ${total_friction:.1f}，需过滤低期望值的震荡噪音信号。")
        if not top_flaws:
            top_flaws.append("策略整体表现稳健，可在持仓非对称盈亏比与出场敏锐度上进一步精细调优。")

        # 1. K-line Market Context
        kline_market_context = {}
        if df is not None and len(df) > 0:
            start_dt = str(df.iloc[0].get("timestamp", ""))
            end_dt = str(df.iloc[-1].get("timestamp", ""))
            start_px = float(df.iloc[0]["close"])
            end_px = float(df.iloc[-1]["close"])
            px_chg_pct = round((end_px - start_px) / start_px * 100.0, 2) if start_px > 0 else 0.0
            high_px = float(df["high"].max())
            low_px = float(df["low"].min())
            atr_approx = float((df["high"] - df["low"]).mean())
            atr_ratio_pct = round((atr_approx / end_px) * 100.0, 2) if end_px > 0 else 0.0
            
            regime = "宽幅震荡胶着" if abs(px_chg_pct) < 3.0 else ("单边上涨趋势" if px_chg_pct > 0 else "单边下跌走势")
            kline_market_context = {
                "total_bars": len(df),
                "time_span": f"{start_dt} 至 {end_dt}",
                "price_range": f"${low_px:.2f} ~ ${high_px:.2f} (区间涨跌: {px_chg_pct:+.2f}%)",
                "avg_bar_atr": f"${atr_approx:.2f} ({atr_ratio_pct:.2f}% 波动度)",
                "market_regime": regime
            }

        # 2. Exit reasons breakdown & Trade Cases
        exit_reasons_breakdown = {}
        for t in trades:
            def _get_val(obj, field, default=None):
                if isinstance(obj, dict):
                    return obj.get(field, default)
                return getattr(obj, field, default)
            rsn = _get_val(t, "exit_reason", "标准平仓")
            exit_reasons_breakdown[rsn] = exit_reasons_breakdown.get(rsn, 0) + 1

        # 3. Top worst losing trade cases
        losing_trade_objs = [t for t in trades if float(_get(t, "net_pnl_usd", 0.0)) < 0]
        losing_trade_objs.sort(key=lambda x: float(_get(x, "net_pnl_usd", 0.0)))
        worst_cases = []
        for lt in losing_trade_objs[:3]:
            e_time = str(_get(lt, "entry_time", ""))
            x_time = str(_get(lt, "exit_time", ""))
            e_p = float(_get(lt, "entry_price", 0.0))
            x_p = float(_get(lt, "exit_price", 0.0))
            pnl = float(_get(lt, "net_pnl_usd", 0.0))
            h_bars = int(_get(lt, "hold_bars", 0))
            rsn = str(_get(lt, "exit_reason", "平仓"))
            m_pct = float(_get(lt, "mae_pct", 0.0))
            worst_cases.append({
                "period": f"{e_time}入场 -> {x_time}出场 (持仓{h_bars}根K线)",
                "entry_price": f"${e_p:.2f}",
                "exit_price": f"${x_p:.2f}",
                "net_loss_usd": f"${pnl:.2f}",
                "mae_pct": f"{m_pct:.2f}%",
                "exit_reason": rsn
            })

        # 4. Surrender cases (high peak profit surrendered)
        surrender_cases = []
        for st in trades:
            e_p = float(_get(st, "entry_price", 0.0))
            pnl = float(_get(st, "net_pnl_usd", 0.0))
            mfe = float(_get(st, "mfe_pct", 0.0))
            if mfe >= 0.8 and pnl < 1.0:
                surrender_cases.append({
                    "entry_time": str(_get(st, "entry_time", "")),
                    "entry_price": f"${e_p:.2f}",
                    "peak_mfe": f"+{mfe:.2f}%",
                    "final_pnl": f"${pnl:.2f}",
                    "flaw": "持仓曾有顺向浮盈，但出场线滞后导致利润全额回吐"
                })

        return {
            "total_trades": total_trades,
            "win_rate": backtest_result.get("win_rate", 0.0),
            "false_breakout_count": false_breakout_count,
            "false_breakout_ratio": round(false_breakout_ratio, 1),
            "avg_mae_pct": round(avg_mae, 2),
            "avg_mfe_pct": round(avg_mfe, 2),
            "surrendered_profit_usd": round(surrendered_profits, 2),
            "friction_drag_pct": round(friction_drag, 1),
            "top_flaws": top_flaws,
            "kline_market_context": kline_market_context,
            "worst_trade_cases": worst_cases,
            "surrender_cases": surrender_cases,
            "exit_reasons_breakdown": exit_reasons_breakdown
        }

    def compute_fitness(self, metrics: Dict[str, Any], target_goal: str = "balanced", custom_goal: str = "") -> float:
        """
        Institutional Composite Fitness Function with Goal-Specific Weighting.
        """
        sharpe = max(float(metrics.get("sharpe_ratio", 0.0)), -3.0)
        ret = float(metrics.get("return_pct", 0.0))
        mdd = max(float(metrics.get("max_drawdown_pct", 1.0)), 0.1)
        win_rate = float(metrics.get("win_rate", 0.0))
        pf = min(float(metrics.get("profit_factor", 0.0)), 10.0)
        trades = int(metrics.get("total_trades", 0))

        # Trade penalty for too few trades
        trade_mult = 1.0 if trades >= 10 else (0.5 if trades >= 5 else (0.2 if trades > 0 else 0.1))

        if target_goal == "sharpe":
            score = (sharpe * 4.0) + (ret * 0.3) - (mdd * 1.5) + (win_rate * 0.1) + (pf * 1.5)
        elif target_goal == "profit_factor":
            score = (pf * 4.5) + (win_rate * 0.2) + (sharpe * 2.0) - (mdd * 1.5) + (ret * 0.3)
        elif target_goal == "drawdown":
            score = (sharpe * 1.8) + (ret * 0.2) - (mdd * 3.5) + (win_rate * 0.2) + (pf * 1.0)
        elif target_goal == "win_rate":
            score = (sharpe * 1.8) + (ret * 0.3) - (mdd * 1.5) + (win_rate * 0.4) + (pf * 1.2)
        elif target_goal == "custom" and custom_goal:
            cg = custom_goal.lower()
            pf_w = 3.8 if ("盈亏比" in cg or "赔率" in cg or "盈亏" in cg) else 1.2
            win_w = 0.35 if ("胜率" in cg or "准确" in cg or "确定性" in cg) else 0.15
            mdd_w = 3.2 if ("回撤" in cg or "防守" in cg or "风控" in cg or "低回撤" in cg) else 1.8
            robust_mult = 1.25 if ("鲁棒" in cg or "稳健" in cg or "均衡" in cg or "抗噪" in cg) else 1.0
            sharpe_w = 2.8 if "夏普" in cg else 2.0
            score = ((sharpe * sharpe_w) + (ret * 0.3) - (mdd * mdd_w) + (win_rate * win_w) + (pf * pf_w)) * robust_mult
        else: # balanced
            score = (sharpe * 2.5) + (ret * 0.4) - (mdd * 1.8) + (win_rate * 0.15) + (pf * 1.2)

        return float(score * trade_mult)

    def generate_evolution_plan(
        self,
        strategy_name: str,
        current_code: str,
        backtest_result: Dict[str, Any],
        diagnostics: Dict[str, Any],
        target_goal: str = "balanced",
        custom_goal: str = ""
    ) -> Dict[str, Any]:
        """
        Invokes LLM to construct a comprehensive, multi-dimensional optimization plan.
        """
        goal_prompts = {
            "balanced": "全维度综合平衡优化（在稳健提升夏普与胜率的同时严控最大回撤与交易摩擦）",
            "sharpe": "最大化夏普比率（精细化趋势跟踪，让奔跑利润最大化）",
            "profit_factor": "最大化盈亏比（提升单笔盈利空间，以非对称期望值覆盖试错成本，实现高盈亏比非对称收益）",
            "drawdown": "极限压制最大回撤与下行风险（收紧保护性止损，严格过滤震荡假突破）",
            "win_rate": "提升交易胜率与信号确定性（多因子共振进场，提高入场安全边际）",
            "custom": f"用户自定义目标: {custom_goal}" if custom_goal else "胜率与盈亏比均衡，鲁棒性强"
        }
        goal_desc = f"用户自定义战略目标: {custom_goal}" if (target_goal == "custom" and custom_goal) else goal_prompts.get(target_goal, goal_prompts["balanced"])

        metrics_summary = {
            "净收益率": f"{backtest_result.get('return_pct', 0.0):.2f}%",
            "夏普比率": f"{backtest_result.get('sharpe_ratio', 0.0):.2f}",
            "最大回撤": f"{backtest_result.get('max_drawdown_pct', 0.0):.2f}%",
            "交易胜率": f"{backtest_result.get('win_rate', 0.0):.1f}%",
            "盈亏比": f"{backtest_result.get('profit_factor', 0.0):.2f}",
            "总交易笔数": backtest_result.get("total_trades", 0),
            "摩擦磨损": f"${backtest_result.get('total_friction_usd', 0.0):.2f}"
        }

        kline_ctx = diagnostics.get("kline_market_context", {})
        worst_cases = diagnostics.get("worst_trade_cases", [])
        surrender_cases = diagnostics.get("surrender_cases", [])
        exit_breakdown = diagnostics.get("exit_reasons_breakdown", {})

        kline_section = ""
        if kline_ctx:
            kline_section = f"""【真实 K 线行情走势与微观结构】:
- 回测样本: {kline_ctx.get('total_bars', 0)} 根 15m K线 ({kline_ctx.get('time_span', '')})
- 价格波动范围: {kline_ctx.get('price_range', '')}
- 波动率中枢: {kline_ctx.get('avg_bar_atr', '')}
- 市场运行状态: {kline_ctx.get('market_regime', '')}
"""

        cases_section = ""
        if worst_cases:
            cases_section += "【深度回测报告典型严重亏损点位 Case】:\n"
            for idx, c in enumerate(worst_cases, 1):
                cases_section += f"- 亏损 Case {idx}: {c['period']}, 入场价 {c['entry_price']} -> 出场价 {c['exit_price']}, 净亏损 {c['net_loss_usd']}, 最大逆向浮亏 MAE: {c['mae_pct']}, 触发规则: [{c['exit_reason']}]\n"

        if surrender_cases:
            cases_section += "【持仓顺向浮盈严重回吐点位 Case】:\n"
            for idx, c in enumerate(surrender_cases[:2], 1):
                cases_section += f"- 回吐 Case {idx}: 入场时间 {c['entry_time']}，入场价 {c['entry_price']}，持仓曾达峰值浮盈 {c['peak_mfe']}，最终净收益仅 {c['final_pnl']} ({c['flaw']})\n"

        if exit_breakdown:
            cases_section += "【全量出场规则触发统计】:\n"
            for rsn, cnt in exit_breakdown.items():
                cases_section += f"- 规则 [{rsn}]: 触发 {cnt} 次\n"

        flaws_text = "\n".join([f"- {f}" for f in diagnostics.get("top_flaws", [])])

        system_prompt = (
            "你是一位世界顶尖的高频与趋势对冲基金量化数学家兼策略架构师，严格遵循第一性原理与数学期望方程 E[R] = P*W - (1-P)*L - C > 0。\n"
            "【核心分析原则】:\n"
            "你的所有优化方案必须严格基于三位一体因果链：\n"
            "1. 真实 K 线宏观与微观波动结构 (振幅、ATR中枢、趋势vs震荡)；\n"
            "2. 深度回测报告中的具体逐笔成交点位 Case (买入价、卖出价、持仓Bar数、触发的出场规则)；\n"
            "3. 策略源代码的状态机实现 (具体到代码中的变量、阈值参数与函数逻辑)。\n"
            "【严禁通用八股模板】: 严禁空洞模版套话！每一个分析与优化条目，必须明确指出是针对哪一笔具体的亏损/回吐点位 Case，并明确指出要修改当前策略代码中的哪一行、哪个参数（从旧值调整为新值）。\n"
            "【输出格式刚性约束】:\n"
            "1. 全程必须使用严谨中文撰写，严禁输出任何英文草稿、内部思考过程或解释废话！\n"
            "2. 第一行必须直接以 `### 一、K线微观点位归因剖析` 开头！紧接着输出 `### 二、多维度解决策略（针对性因果解法）` 与 `### 三、详细实施方案（3步具体代码修改方案）`。\n"
            "3. 输出完毕后立即结束，严禁二次重写或输出第二遍草稿！"
        )

        user_prompt = f"""【当前策略】: {strategy_name}
【优化目标】: {goal_desc}

{kline_section}
【回测核心绩效与整体指标】:
{json.dumps(metrics_summary, ensure_ascii=False, indent=2)}

{cases_section}
【微观诊断缺陷】:
- 假突破被套率: {diagnostics.get('false_breakout_ratio', 0)}%
- 平均逆向浮亏 (MAE): {diagnostics.get('avg_mae_pct', 0)}%
- 累计回吐峰值利润: ${diagnostics.get('surrendered_profit_usd', 0)}
- 识别痛点:
{flaws_text or '- 无明显微观缺陷'}

【当前策略完整核心代码】:
```python
{current_code[:3800]}
```

请输出针对该策略在上述盘面与点位下的【深度分析与专属优化 Plan】（必须严格引用上述真实点位 Case 与代码变量，严禁套话）：
### 一、K线微观点位归因剖析
直接结合上述【亏损 Case】或【回吐 Case】的点位、价格与触发规则，深入诊断为什么在该行情下会出现被套或回吐（例如追高实体位置、假突破反转、出场过慢或硬止损过宽）。

### 二、多维度解决策略（针对性因果解法）
- 维度1：入场过滤与状态机解耦（结合上述假突破/赶顶特征，提出具体的指标过滤或波动率约束）
- 维度2：动态多级出场与凸性保护（结合上述持仓回吐点位，设计保本锁或追踪离场规则）
- 维度3：微观摩擦防御与防骗线（降低无效磨损）

### 三、详细实施方案（3步具体代码修改方案，明确代码变量名、旧参数值与优化后新值）
第1步：...
第2步：...
第3步：...
"""

        try:
            plan_text = ai_service.call_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=3500
            )
            # If model produced multiple draft blocks, pick the most complete one containing section 三
            blocks = re.split(r'\n+(?=###\s*一、)', plan_text)
            chosen_block = plan_text
            for b in reversed(blocks):
                if '### 一、' in b and ('### 三、' in b or '第三步' in b or '第3步' in b):
                    chosen_block = b
                    break
            plan_text = chosen_block.strip()
            # Robustly strip any preliminary reasoning text or meta thoughts
            header_idx = -1
            for pat in [r'(?:###\s*)?一、', r'(?:###\s*)?1\.\s*K线', r'###\s*一\b']:
                m = re.search(pat, plan_text)
                if m:
                    header_idx = m.start()
                    break
            if header_idx != -1:
                plan_text = plan_text[header_idx:].strip()
            else:
                plan_text = re.sub(r'^(?:[Ww]e need|我们根据|用户要求|让我们).*?\n\n', '', plan_text, flags=re.DOTALL)
        except Exception as e:
            plan_text = f"AI 生成优化 Plan 降级备用方案:\n1. 引入 ATR 动态波动率阈值过滤震荡假突破；\n2. 增加移动止损 Trailing Stop 锁住浮盈；\n3. 限制频繁短线交易降低手续费磨损。\n(详细异常: {e})"

        return {
            "target_goal": target_goal,
            "goal_desc": goal_desc,
            "metrics_summary": metrics_summary,
            "diagnostics": diagnostics,
            "plan_text": plan_text
        }

    def clean_python_code(self, raw_code: str) -> str:
        """Strips markdown fences and removes illegal external quant packages like gm, vnpy."""
        matches = re.findall(r'```(?:python)?\s*([\s\S]*?)```', raw_code)
        if matches:
            candidate = max(matches, key=len)
            for m in matches:
                if 'class ' in m and 'BaseStrategy' in m:
                    candidate = m
                    break
            code = candidate.strip()
        else:
            code = raw_code.strip()
        # Remove any hallucinated external imports
        code = re.sub(r'^\s*(?:import\s+(?:gm|vnpy|tqsdk|ctp|rqalpha|backtrader)[^\n]*|from\s+(?:gm|vnpy|tqsdk|ctp|rqalpha|backtrader)[^\n]*)', '', code, flags=re.MULTILINE)
        return code

    def _generate_deterministic_evolution(self, base_code: str, target_goal: str = "balanced", custom_goal: str = "") -> str:
        """
        Deterministic algorithmic optimization fallback when LLM output syntax is invalid.
        Injects ATR volatility trailing stop and volume expansion filters into BaseStrategy.
        """
        tp_mult = "4.5"
        sl_mult = "1.8"
        cg = custom_goal.lower()
        if target_goal == "profit_factor" or ("盈亏比" in cg or "赔率" in cg):
            tp_mult = "5.2"
            sl_mult = "1.6"
        elif target_goal == "drawdown" or ("回撤" in cg or "防守" in cg):
            tp_mult = "3.8"
            sl_mult = "1.4"
        elif target_goal == "win_rate" or ("胜率" in cg):
            tp_mult = "3.2"
            sl_mult = "1.8"

        if "tb_fast_ma" in base_code or "Evolved" in base_code:
            evolved = base_code.replace("self.fast_ma_len = 40", "self.fast_ma_len = 35")
            evolved = evolved.replace("self.slow_ma_len = 120", "self.slow_ma_len = 110")
            evolved = evolved.replace("self.stop_loss_mult = 2.0", f"self.stop_loss_mult = {sl_mult}")
            evolved = evolved.replace("self.take_profit_mult = 4.0", f"self.take_profit_mult = {tp_mult}")
            if evolved != base_code:
                return evolved

        return '''from strategy_base import BaseStrategy
from strategy import TradeSignal
import pandas as pd
import numpy as np

class EvolvedAlphaStrategy(BaseStrategy):
    """
    AI 演化增强量化策略: 融合动态 ATR 移动止损与均线趋势动量共振
    """
    def __init__(self, config=None, friction_config=None, **kwargs):
        super().__init__(config, friction_config)
        self.fast_len = 14
        self.slow_len = 34
        self.atr_len = 14
        self.atr_stop_mult = 1.8
        self.take_profit_mult = 3.8

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data['evo_fast'] = data['close'].ewm(span=self.fast_len, adjust=False).mean()
        data['evo_slow'] = data['close'].ewm(span=self.slow_len, adjust=False).mean()
        
        # ATR Calculation
        tr1 = data['high'] - data['low']
        tr2 = (data['high'] - data['close'].shift(1)).abs()
        tr3 = (data['low'] - data['close'].shift(1)).abs()
        data['evo_tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data['evo_atr'] = data['evo_tr'].rolling(window=self.atr_len).mean().fillna(data['close'] * 0.015)
        
        # Volume Filter
        vol_col = 'volume_usd' if 'volume_usd' in data.columns else 'volume'
        data['evo_vol_ma'] = data[vol_col].rolling(window=20).mean().fillna(data[vol_col])
        return data

    def evaluate_bar(self, current_bar: pd.Series, current_position=None, sol_price_usd: float = 140.0) -> TradeSignal:
        close_px = float(current_bar['close'])
        fast_val = float(current_bar.get('evo_fast', close_px))
        slow_val = float(current_bar.get('evo_slow', close_px))
        atr_val = float(current_bar.get('evo_atr', close_px * 0.015))
        
        vol_col = 'volume_usd' if 'volume_usd' in current_bar else 'volume'
        vol_curr = float(current_bar.get(vol_col, 0.0))
        vol_ma = float(current_bar.get('evo_vol_ma', vol_curr))

        # Position Management
        if current_position is not None and current_position.get('size', 0) > 0:
            entry_px = float(current_position['entry_price'])
            highest_px = float(current_position.get('highest_price', close_px))
            
            # Dynamic Trailing Stop
            trailing_stop = highest_px - (self.atr_stop_mult * atr_val)
            hard_stop = entry_px - (2.0 * atr_val)
            eff_stop = max(trailing_stop, hard_stop)
            take_profit = entry_px + (self.take_profit_mult * atr_val)

            if close_px <= eff_stop:
                return TradeSignal(action='SELL', price=close_px, reason="EVO_DYNAMIC_TRAILING_STOP")
            if close_px >= take_profit:
                return TradeSignal(action='SELL', price=close_px, reason="EVO_TAKE_PROFIT")
            if fast_val < slow_val:
                return TradeSignal(action='SELL', price=close_px, reason="EVO_TREND_EXIT")
            return TradeSignal(action='HOLD', price=close_px)

        # Entry logic: Fast crossover Slow and volume expanded
        if fast_val > slow_val and (vol_curr >= vol_ma * 0.85):
            stop_loss = close_px - (self.atr_stop_mult * atr_val)
            take_profit = close_px + (self.take_profit_mult * atr_val)
            return TradeSignal(action='BUY', price=close_px, stop_loss=stop_loss, take_profit=take_profit, reason="EVO_TREND_BREAKOUT")

        return TradeSignal(action='HOLD', price=close_px)
'''

    def evolve_code_with_llm(
        self,
        strategy_name: str,
        current_code: str,
        plan_text: str,
        target_goal: str = "balanced",
        custom_goal: str = ""
    ) -> str:
        """
        Uses LLM to rewrite and enhance strategy Python code inheriting BaseStrategy.
        """
        system_prompt = (
            "你是一位精通 Python 与量化实盘交易的资深架构师。请根据优化 Plan，对当前的量化策略代码进行重构升级。\n"
            "【刚性要求】:\n"
            "1. 必须是合法的 Python 代码，严格继承 BaseStrategy，实现 prepare_indicators 和 evaluate_bar，且 __init__(self, config=None, friction_config=None, **kwargs) 必须接受 **kwargs。\n"
            "2. 严禁导入任何外部第三方交易软件库（如 gm, vnpy, tqsdk, ctp 等）！只能使用当前环境已有的 pandas as pd, numpy as np, BaseStrategy, TradeSignal。\n"
            "3. 严禁未来函数！所有技术指标必须严格基于已知历史数据（如 .shift(1) 或 rolling）。\n"
            "4. evaluate_bar 必须返回 TradeSignal(action='BUY'|'SELL'|'HOLD', price=..., stop_loss=..., take_profit=..., reason=...)，price 务必传入当前收盘价。\n"
            "5. 严禁输出任何思考草稿或解析废话，只输出完整可执行的 Python 代码，包裹在 ```python ... ``` 中！"
        )

        goal_note = f"【战略导向】: 用户自定义目标 [{custom_goal}]" if (target_goal == "custom" and custom_goal) else f"【战略导向】: 目标为 [{target_goal}]"
        user_prompt = f"""【策略名称】: {strategy_name}
{goal_note}
【优化 Plan 与指导方案】:
{plan_text}

【当前策略原始 Python 代码】:
```python
{current_code}
```

请根据上述 Plan，输出完整、经过优化、可直接执行的 Python BaseStrategy 子类代码。"""

        try:
            raw_reply = ai_service.call_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=3500
            )
            clean_code = self.clean_python_code(raw_reply)
            return clean_code
        except Exception as e:
            print(f"[EvolutionAgent] LLM 代码重构调用异常 ({e})，使用确定性算法演化")
            return self._generate_deterministic_evolution(current_code, target_goal)

    def run_backtest_for_strategy(
        self,
        strat_instance: Any,
        token_sym: str = "SOL",
        timeframe: str = "15m",
        bars: int = 500
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Runs single-pass causal backtest on market K-lines."""
        df = self.get_market_df(token_sym=token_sym, timeframe=timeframe, bars=bars)
        engine = SolanaBacktestEngine()
        result = engine.run_single_pass(df, stress_mult=1.0, sol_price=140.0, strategy_instance=strat_instance)
        return result, df

    def execute_iteration_step(
        self,
        strategy_id: str,
        target_goal: str = "balanced",
        custom_goal: str = "",
        custom_code: Optional[str] = None,
        token_sym: str = "SOL",
        timeframe: str = "15m",
        bars: int = 500
    ) -> Dict[str, Any]:
        """
        Executes one full evolutionary iteration:
        1. Identifies active Base (or initial strategy)
        2. Backtests Base to get current baseline metrics
        3. Runs K-line & Trade Point micro-diagnostics
        4. Generates multi-dimensional Plan via LLM
        5. Refactors strategy source code via LLM
        6. Compiles and Backtests evolved strategy
        7. Compares fitness score against Base
        8. Automatically freezes superior results as new Base
        9. Persists iteration to history
        """
        history = self.load_history(strategy_id)
        strat_data = strategy_manager.get_strategy(strategy_id) or {}
        strat_name = strat_data.get("name", "自定义策略")

        # Determine baseline code
        active_base = None
        if history.get("active_base_id"):
            for it in history.get("iterations", []):
                if it.get("iteration_id") == history.get("active_base_id"):
                    active_base = it
                    break

        if active_base is not None:
            base_code = active_base.get("python_code", "")
            base_metrics = active_base.get("metrics", {})
        else:
            # First time: compile current strategy as v0 Base
            base_code = custom_code or strat_data.get("python_code") or strat_data.get("source_code", "")
            try:
                base_inst = StrategyTranspiler.compile_strategy_instance(base_code)
                base_result, df_base = self.run_backtest_for_strategy(base_inst, token_sym, timeframe, bars)
                base_metrics = {
                    "return_pct": base_result["return_pct"],
                    "net_profit_usd": base_result["net_profit_usd"],
                    "sharpe_ratio": base_result["sharpe_ratio"],
                    "max_drawdown_pct": base_result["max_drawdown_pct"],
                    "win_rate": base_result["win_rate"],
                    "profit_factor": base_result["profit_factor"],
                    "total_trades": base_result["total_trades"],
                    "total_friction_usd": base_result["total_friction_usd"]
                }
                base_diag = self.analyze_trade_candle_points(df_base, base_result)
                v0_record = {
                    "iteration_id": "iter_0",
                    "version_tag": "v0 (初始基线)",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "target_goal": target_goal,
                    "custom_goal": custom_goal,
                    "goal_label": "初始基准",
                    "is_base": True,
                    "python_code": base_code,
                    "code": base_code,
                    "metrics": base_metrics,
                    "fitness_score": self.compute_fitness(base_metrics, target_goal=target_goal, custom_goal=custom_goal),
                    "plan": "原始策略初始基线，待进行多维度 AI 进化演化。",
                    "diagnostics": base_diag,
                    "improvement_summary": "初始基准版本"
                }
                history["iterations"].append(v0_record)
                history["active_base_id"] = "iter_0"
                active_base = v0_record
            except Exception as e:
                raise ValueError(f"编译初始策略基线失败: {str(e)}")

        # Step 1: Run diagnostics on current Base
        base_inst = StrategyTranspiler.compile_strategy_instance(active_base["python_code"])
        base_result, df_market = self.run_backtest_for_strategy(base_inst, token_sym, timeframe, bars)
        diagnostics = self.analyze_trade_candle_points(df_market, base_result)

        # Step 2: Formulate Plan
        plan_info = self.generate_evolution_plan(
            strategy_name=strat_name,
            current_code=active_base["python_code"],
            backtest_result=base_result,
            diagnostics=diagnostics,
            target_goal=target_goal,
            custom_goal=custom_goal
        )

        # Step 3: Evolve Code
        evolved_code = self.evolve_code_with_llm(
            strategy_name=strat_name,
            current_code=active_base["python_code"],
            plan_text=plan_info["plan_text"],
            target_goal=target_goal,
            custom_goal=custom_goal
        )

        # Step 4: Compile & Test with multi-tier resilience
        evolved_inst = None
        try:
            evolved_inst = StrategyTranspiler.compile_strategy_instance(evolved_code)
        except Exception as e1:
            try:
                # One-shot syntax repair prompt with explicit constraints
                repair_prompt = (
                    f"代码编译报错: {str(e1)}\n"
                    "请修复该 Python 代码。要求：严格继承 BaseStrategy，只能使用 pandas/numpy/TradeSignal，"
                    "严禁使用 gm、vnpy 等外部库，确保代码完整闭合：\n"
                    f"```python\n{evolved_code}\n```"
                )
                repaired = ai_service.call_llm([{"role": "user", "content": repair_prompt}], max_tokens=3500)
                evolved_code = self.clean_python_code(repaired)
                evolved_inst = StrategyTranspiler.compile_strategy_instance(evolved_code)
            except Exception as e2:
                print(f"[EvolutionAgent] 智能修复后仍报错 ({e2})，启用确定性增强演化算法")
                evolved_code = self._generate_deterministic_evolution(active_base["python_code"], target_goal, custom_goal)
                evolved_inst = StrategyTranspiler.compile_strategy_instance(evolved_code)

        # Step 5: Backtest Evolved Code
        new_result, _ = self.run_backtest_for_strategy(evolved_inst, token_sym, timeframe, bars)
        new_metrics = {
            "return_pct": new_result["return_pct"],
            "net_profit_usd": new_result["net_profit_usd"],
            "sharpe_ratio": new_result["sharpe_ratio"],
            "max_drawdown_pct": new_result["max_drawdown_pct"],
            "win_rate": new_result["win_rate"],
            "profit_factor": new_result["profit_factor"],
            "total_trades": new_result["total_trades"],
            "total_friction_usd": new_result["total_friction_usd"]
        }

        # Step 6: Fitness Evaluation (Goal-Aware)
        old_fitness = self.compute_fitness(active_base["metrics"], target_goal=target_goal, custom_goal=custom_goal)
        new_fitness = self.compute_fitness(new_metrics, target_goal=target_goal, custom_goal=custom_goal)
        is_improved = new_fitness > old_fitness

        iter_idx = len(history.get("iterations", []))
        iter_id = f"iter_{iter_idx}"
        version_tag = f"v{iter_idx} (迭代{iter_idx})"

        # Generate improvement summary
        ret_diff = new_metrics["return_pct"] - active_base["metrics"]["return_pct"]
        sharpe_diff = new_metrics["sharpe_ratio"] - active_base["metrics"]["sharpe_ratio"]
        mdd_diff = new_metrics["max_drawdown_pct"] - active_base["metrics"]["max_drawdown_pct"]
        pf_diff = new_metrics["profit_factor"] - active_base["metrics"]["profit_factor"]
        win_diff = new_metrics["win_rate"] - active_base["metrics"]["win_rate"]

        summary_parts = []
        if target_goal == "profit_factor" and pf_diff != 0:
            sign = "+" if pf_diff > 0 else ""
            summary_parts.append(f"盈亏比 {active_base['metrics']['profit_factor']:.2f}➔{new_metrics['profit_factor']:.2f} ({sign}{pf_diff:.2f})")
        if sharpe_diff != 0:
            sign = "+" if sharpe_diff > 0 else ""
            summary_parts.append(f"夏普 {active_base['metrics']['sharpe_ratio']:.2f}➔{new_metrics['sharpe_ratio']:.2f} ({sign}{sharpe_diff:.2f})")
        if ret_diff != 0:
            sign = "+" if ret_diff > 0 else ""
            summary_parts.append(f"收益率 {active_base['metrics']['return_pct']:.1f}%➔{new_metrics['return_pct']:.1f}% ({sign}{ret_diff:.1f}%)")
        if mdd_diff != 0:
            sign = "+" if mdd_diff > 0 else ""
            summary_parts.append(f"回撤 {active_base['metrics']['max_drawdown_pct']:.1f}%➔{new_metrics['max_drawdown_pct']:.1f}% ({sign}{mdd_diff:.1f}%)")
        if win_diff != 0 and target_goal in ("win_rate", "custom"):
            sign = "+" if win_diff > 0 else ""
            summary_parts.append(f"胜率 {active_base['metrics']['win_rate']:.1f}%➔{new_metrics['win_rate']:.1f}% ({sign}{win_diff:.1f}%)")

        improvement_summary = ", ".join(summary_parts) if summary_parts else "与基线持平"

        goal_labels = {
            "balanced": "综合均衡",
            "sharpe": "最大夏普",
            "profit_factor": "最大盈亏比",
            "drawdown": "压制回撤",
            "win_rate": "提升胜率",
            "custom": f"自定义: {custom_goal}" if custom_goal else "自定义目标"
        }
        goal_label = goal_labels.get(target_goal, "综合均衡")

        # Create iteration record
        iter_record = {
            "iteration_id": iter_id,
            "version_tag": version_tag,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "target_goal": target_goal,
            "custom_goal": custom_goal,
            "goal_label": goal_label,
            "is_base": False,
            "is_improved": is_improved,
            "python_code": evolved_code,
            "code": evolved_code,
            "metrics": new_metrics,
            "fitness_score": new_fitness,
            "plan": plan_info["plan_text"],
            "diagnostics": diagnostics,
            "improvement_summary": improvement_summary,
            "parent_base_id": active_base["iteration_id"]
        }

        # Auto-freeze as Base if strictly superior
        if is_improved:
            # Unmark old base
            for it in history["iterations"]:
                it["is_base"] = False
            iter_record["is_base"] = True
            history["active_base_id"] = iter_id

        history["iterations"].append(iter_record)
        self.save_history(strategy_id, history)

        return {
            "status": "success",
            "strategy_id": strategy_id,
            "iteration": iter_record,
            "is_improved": is_improved,
            "frozen_as_new_base": is_improved,
            "base_metrics": active_base["metrics"],
            "new_metrics": new_metrics,
            "plan": plan_info["plan_text"],
            "diagnostics": diagnostics,
            "history_count": len(history["iterations"])
        }

    def run_auto_loop(
        self,
        strategy_id: str,
        max_rounds: int = 3,
        target_goal: str = "balanced",
        custom_goal: str = "",
        token_sym: str = "SOL",
        timeframe: str = "15m",
        bars: int = 500
    ) -> Dict[str, Any]:
        """
        Runs multi-round autonomous optimization loop.
        In each round, builds on the current Base, optimizes, tests, and updates Base until convergence.
        """
        rounds_executed = 0
        improvements_count = 0
        logs = []

        if target_goal == "custom":
            goals_cycle = ["custom"] * max_rounds
        elif target_goal == "profit_factor":
            goals_cycle = ["profit_factor", "sharpe", "balanced"]
        else:
            goals_cycle = [target_goal, "profit_factor", "drawdown", "sharpe", "balanced"]

        for r in range(max_rounds):
            current_goal = goals_cycle[r % len(goals_cycle)]
            step_res = self.execute_iteration_step(
                strategy_id=strategy_id,
                target_goal=current_goal,
                custom_goal=custom_goal,
                token_sym=token_sym,
                timeframe=timeframe,
                bars=bars
            )
            rounds_executed += 1
            it = step_res["iteration"]
            logs.append({
                "round": r + 1,
                "version_tag": it["version_tag"],
                "goal": current_goal,
                "goal_label": it.get("goal_label", current_goal),
                "is_improved": step_res["is_improved"],
                "summary": it["improvement_summary"]
            })
            if step_res["is_improved"]:
                improvements_count += 1

        history = self.load_history(strategy_id)
        active_base = None
        for it in history.get("iterations", []):
            if it.get("iteration_id") == history.get("active_base_id"):
                active_base = it
                break

        return {
            "status": "success",
            "strategy_id": strategy_id,
            "rounds_executed": rounds_executed,
            "improvements_count": improvements_count,
            "rounds_log": logs,
            "active_base": active_base,
            "history": history["iterations"]
        }

    def freeze_base(self, strategy_id: str, iteration_id: str) -> Dict[str, Any]:
        """Manually freezes a specific iteration as the active Base checkpoint."""
        history = self.load_history(strategy_id)
        target = None
        for it in history.get("iterations", []):
            if it.get("iteration_id") == iteration_id:
                it["is_base"] = True
                target = it
            else:
                it["is_base"] = False

        if target is None:
            raise ValueError(f"未找到指定的演化版本 {iteration_id}")

        history["active_base_id"] = iteration_id
        self.save_history(strategy_id, history)
        return {
            "status": "success",
            "strategy_id": strategy_id,
            "active_base_id": iteration_id,
            "base_version_tag": target.get("version_tag"),
            "base_metrics": target.get("metrics")
        }

    def apply_to_studio(self, strategy_id: str, iteration_id: str) -> Dict[str, Any]:
        """Applies an iteration's evolved Python code to the main Strategy Studio."""
        history = self.load_history(strategy_id)
        target = None
        for it in history.get("iterations", []):
            if it.get("iteration_id") == iteration_id:
                target = it
                break

        if not target:
            raise ValueError(f"未找到指定的版本记录 {iteration_id}")

        code = target.get("python_code", "")
        # Update strategy in manager
        strat = strategy_manager.get_strategy(strategy_id)
        if strat:
            strat["python_code"] = code
            strat["source_code"] = code
            strat["language"] = "python"
            strat["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            strategy_manager.save_strategy(strat)

        return {
            "status": "success",
            "strategy_id": strategy_id,
            "applied_version": target.get("version_tag"),
            "applied_code": code
        }

    def chat_with_copilot(
        self,
        strategy_id: str,
        iteration_id: Optional[str],
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Interactively converses with the Quantitative Mathematician Agent to perform
        CRUD operations on the Detailed Optimization Plan and Evolved Strategy Source Code.
        Strictly enforces the boundary of K-line data, strategy logic, and backtest results.
        """
        msg_clean = user_message.strip()

        # 1. 严格边界防御过滤 (Hard Perimeter Defense)
        non_quant_patterns = [
            r'写.*(?:诗|小说|故事|文案|剧本)',
            r'今天天气',
            r'讲个笑话',
            r'玩.*游戏',
            r'做菜|美食',
            r'星座|算命|占星'
        ]
        for pat in non_quant_patterns:
            if re.search(pat, msg_clean):
                return {
                    "status": "success",
                    "reply": (
                        "【量化边界防御】本 Agent 仅在数学物理模型、K线微观结构（时序/波动率/流动性）"
                        "及回测因果链下进行策略推演与代码/Plan 增删改查。请提出与当前策略逻辑、K线点位归因或回测数据相关的量化问题。"
                    ),
                    "action_executed": None,
                    "plan_updated": False,
                    "code_updated": False
                }

        history = self.load_history(strategy_id)
        iterations = history.get("iterations", [])
        if not iterations:
            return {
                "status": "error",
                "message": "当前策略尚无演化迭代版本，请先点击【单步迭代优化】生成基线。"
            }

        target_iter = None
        if iteration_id:
            for it in iterations:
                if it.get("iteration_id") == iteration_id:
                    target_iter = it
                    break
        if not target_iter:
            target_iter = iterations[-1]

        current_code = target_iter.get("python_code") or target_iter.get("code", "")
        current_plan = target_iter.get("plan", "")
        current_metrics = target_iter.get("metrics", {})
        current_diag = target_iter.get("diagnostics", {})

        kline_ctx = current_diag.get("kline_market_context", {})
        worst_cases = current_diag.get("worst_trade_cases", [])
        surrender_cases = current_diag.get("surrender_cases", [])
        exit_breakdown = current_diag.get("exit_reasons_breakdown", {})

        kline_text = ""
        if kline_ctx:
            kline_text = f"""【真实 K 线行情走势与微观结构】:
- 回测样本: {kline_ctx.get('total_bars', 0)} 根 15m K线 ({kline_ctx.get('time_span', '')})
- 价格波动范围: {kline_ctx.get('price_range', '')}
- 波动率中枢: {kline_ctx.get('avg_bar_atr', '')}
- 市场运行状态: {kline_ctx.get('market_regime', '')}
"""

        cases_text = ""
        if worst_cases:
            cases_text += "【深度回测报告典型严重亏损点位 Case】:\n"
            for idx, c in enumerate(worst_cases, 1):
                cases_text += f"- 亏损 Case {idx}: {c['period']}, 入场价 {c['entry_price']} -> 出场价 {c['exit_price']}, 净亏损 {c['net_loss_usd']}, 最大逆向浮亏 MAE: {c['mae_pct']}, 触发规则: [{c['exit_reason']}]\n"

        if surrender_cases:
            cases_text += "【持仓顺向浮盈严重回吐点位 Case】:\n"
            for idx, c in enumerate(surrender_cases[:2], 1):
                cases_text += f"- 回吐 Case {idx}: 入场时间 {c['entry_time']}，入场价 {c['entry_price']}，持仓曾达峰值浮盈 {c['peak_mfe']}，最终净收益仅 {c['final_pnl']} ({c['flaw']})\n"

        if exit_breakdown:
            cases_text += "【全量出场规则触发统计】:\n"
            for rsn, cnt in exit_breakdown.items():
                cases_text += f"- 规则 [{rsn}]: 触发 {cnt} 次\n"

        system_prompt = (
            "你是一位世界顶尖的高频与趋势对冲基金量化数学家兼策略架构师。你坚守第一性原理，直击量化本质，追求非对称期望值正反馈：\n"
            "  E[R] = P_win * Mean(Win) - (1 - P_win) * Mean(Loss) - Friction > 0\n"
            "你的边界极其严格：你只针对【K线行情与微观结构数据】、【策略状态机逻辑】、【回测绩效与摩擦数据】进行问答、数学推演与代码/Plan 增删改查。\n"
            "【直接修改左侧详细 Plan 与策略源码的硬性执行规则】:\n"
            "1. 优化 Plan 直接写入: 当你的回答中提出了具体的优化方案、调整条目、增删规则，或用户指令要求修改/添加 Plan 时，直接在回答中（或末尾）使用 ```plan_update 代码块输出更新后的完整 Plan 文本（多行纯 Markdown 文本，无需 JSON 包装，条目清晰）：\n"
            "```plan_update\n"
            "【已更新优化 Plan】\n"
            "1. [针对点位归因诊断的具体入场过滤与状态机解耦调整]\n"
            "2. [针对出场滞后或保护的具体参数与止损止盈逻辑]\n"
            "3. [针对微观摩擦防守的具体代码调整]\n"
            "```\n"
            "系统会自动提取并实时同步写入左侧【详细优化 Plan】编辑器中！\n"
            "2. 策略源码 Python 直接写入: 当你的回答中修改了策略代码（例如入场过滤增强、增加动态移动保本或 ATR 追踪止损、调整指标通道），你必须在回答末尾输出完整可直接运行、继承 BaseStrategy 的 Python 源码代码块：\n"
            "```python:code_update\n"
            "# 完整更新后的可执行 Python 代码\n"
            "from strategy_base import BaseStrategy\n"
            "...\n"
            "```\n"
            "系统会自动编译该代码并实时同步写入左侧【演化策略源码】编辑器中！\n"
            "3. 保持回答一针见血，严禁任何大模型元草稿套话（禁止输出‘用户要求我...’、‘我们接着分析...’等套话），必须直击数学因果与可执行修改。"
        )

        context_prompt = f"""【当前交互迭代版本】: {target_iter.get('version_tag', '最新版')} (ID: {target_iter.get('iteration_id')})

{kline_text}
【当前回测关键指标】:
{json.dumps(current_metrics, ensure_ascii=False, indent=2)}

{cases_text}
【K线与点位微观缺陷归因】:
- 假突破被套率: {current_diag.get('false_breakout_ratio', 0)}%
- 平均逆向浮亏 (MAE): {current_diag.get('avg_mae_pct', 0)}%
- 峰值利润回吐: ${current_diag.get('surrendered_profit_usd', 0)}
- 核心痛点: {current_diag.get('top_flaws', [])}

【当前详细优化 Plan】:
{current_plan}

【当前演化策略 Python 源码】:
```python
{current_code}
```

【用户指令】:
{user_message}
"""

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for item in chat_history[-6:]:
                messages.append(item)
        messages.append({"role": "user", "content": context_prompt})

        try:
            raw_reply = ai_service.call_llm(messages, temperature=0.2, max_tokens=3500)
        except Exception as e:
            return {"status": "error", "message": f"调用 AI 服务异常: {str(e)}"}

        # 1. 检查是否包含 plan_update (支持 ```plan_update、```json:plan_update，且支持无闭合截断容错)
        plan_updated = False
        new_plan_str = None
        plan_match = re.search(r'```(?:json:)?plan_update\s*([\s\S]*?)(?:```|$)', raw_reply)
        if plan_match:
            p_content = plan_match.group(1).strip()
            # 若包含 JSON 格式包裹
            if p_content.startswith('{') and ('plan_text' in p_content or 'action' in p_content):
                try:
                    p_data = json.loads(p_content)
                    new_plan_str = p_data.get("plan_text")
                except Exception:
                    reg_match = re.search(r'"plan_text"\s*:\s*"([\s\S]*?)(?:"\s*\}|"\s*$|$)', p_content)
                    if reg_match:
                        raw_ext = reg_match.group(1)
                        try:
                            new_plan_str = raw_ext.encode().decode('unicode-escape', errors='ignore')
                        except Exception:
                            new_plan_str = raw_ext
                        new_plan_str = new_plan_str.replace('\\n', '\n').replace('\\"', '"')
                    else:
                        new_plan_str = p_content
            else:
                new_plan_str = p_content

            if new_plan_str and len(new_plan_str.strip()) > 10:
                new_plan_str = new_plan_str.strip()
                target_iter["plan"] = new_plan_str
                plan_updated = True

        # Fallback: 如果用户指令中包含明确的 plan/方案/修改 意图，且 AI 回答中有显式 Plan 格式
        if not plan_updated and any(w in user_message.lower() for w in ["plan", "方案", "优化", "修改", "增加", "删除", "剔除"]):
            plan_block_match = re.search(r'(?:【(?:已更新|最新|优化)?\s*Plan.*?】|###\s*(?:详细优化\s*Plan|优化方案))([\s\S]*?)(?:```|$)', raw_reply)
            if not plan_block_match:
                plan_block_match = re.search(r'(?:\n|^)(1\.\s+[\s\S]+?)(?:```|$)', raw_reply)
            if plan_block_match and len(plan_block_match.group(1).strip()) > 20:
                extracted = plan_block_match.group(0).strip()
                target_iter["plan"] = extracted
                new_plan_str = extracted
                plan_updated = True

        # 2. 检查是否包含 code_update (支持无闭合截断容错)
        code_updated = False
        new_code_str = None
        code_match = re.search(r'```python:code_update\s*([\s\S]*?)(?:```|$)', raw_reply)
        if not code_match:
            for m in re.finditer(r'```(?:python)?\s*([\s\S]*?)(?:```|$)', raw_reply):
                snippet = m.group(1).strip()
                if ("class " in snippet and "BaseStrategy" in snippet and "evaluate_bar" in snippet) or \
                   ("prepare_indicators" in snippet and "evaluate_bar" in snippet):
                    code_match = m
                    break

        if code_match:
            candidate_code = self.clean_python_code(code_match.group(1).strip())
            try:
                StrategyTranspiler.compile_strategy_instance(candidate_code)
                target_iter["python_code"] = candidate_code
                target_iter["code"] = candidate_code
                new_code_str = candidate_code
                code_updated = True
            except Exception as ce:
                print(f"[Copilot] 建议的新代码编译校验失败 ({ce})，保留原代码")

        # 3. 清洗 reply，隐藏底层更新协议代码块，杜绝生硬 JSON 泄露
        clean_reply = re.sub(r'```(?:json:)?plan_update[\s\S]*?(?:```|$)', '', raw_reply)
        clean_reply = re.sub(r'```python:code_update[\s\S]*?(?:```|$)', '', clean_reply).strip()

        # 若清洗后为空，赋予友好的明确说明
        if not clean_reply and plan_updated and new_plan_str:
            clean_reply = f"已针对您的指令完成深度因果推演，并实时更新左侧【详细优化 Plan】：\n\n{new_plan_str}"
        elif not clean_reply and code_updated and new_code_str:
            clean_reply = "已针对您的指令完成量化状态机重构，新策略代码已通过编译校验，并实时更新左侧【演化策略源码】！"

        # 记录本次对话到迭代记录中，保持状态长效持久
        if "chat_history" not in target_iter or not isinstance(target_iter.get("chat_history"), list):
            target_iter["chat_history"] = []
        target_iter["chat_history"].append({"role": "user", "content": user_message})
        target_iter["chat_history"].append({"role": "assistant", "content": clean_reply})

        # 始终保存持久化数据
        self.save_history(strategy_id, history)

        return {
            "status": "success",
            "reply": clean_reply,
            "plan_updated": plan_updated,
            "updated_plan": new_plan_str if plan_updated else None,
            "code_updated": code_updated,
            "updated_code": new_code_str if code_updated else None,
            "iteration_id": target_iter.get("iteration_id"),
            "version_tag": target_iter.get("version_tag"),
            "chat_history": target_iter.get("chat_history", [])
        }

    def update_iteration_plan(self, strategy_id: str, iteration_id: str, plan_text: str) -> Dict[str, Any]:
        """Manually updates the detailed plan text for a specific iteration."""
        history = self.load_history(strategy_id)
        for it in history.get("iterations", []):
            if it.get("iteration_id") == iteration_id:
                it["plan"] = plan_text
                self.save_history(strategy_id, history)
                return {"status": "success", "iteration_id": iteration_id, "plan": plan_text}
        raise ValueError(f"未找到指定的演化版本 {iteration_id}")

    def update_iteration_code(self, strategy_id: str, iteration_id: str, new_code: str) -> Dict[str, Any]:
        """Manually updates and compiles the evolved Python code for a specific iteration."""
        clean_code = self.clean_python_code(new_code)
        # Verify compilation
        StrategyTranspiler.compile_strategy_instance(clean_code)

        history = self.load_history(strategy_id)
        for it in history.get("iterations", []):
            if it.get("iteration_id") == iteration_id:
                it["python_code"] = clean_code
                it["code"] = clean_code
                self.save_history(strategy_id, history)
                return {"status": "success", "iteration_id": iteration_id, "code": clean_code}
        raise ValueError(f"未找到指定的演化版本 {iteration_id}")


# Singleton instance
strategy_evolution_agent = StrategyEvolutionAgent()
