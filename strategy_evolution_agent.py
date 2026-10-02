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
                "top_flaws": ["入场门槛过苛刻导致零交易", "未有效捕捉到趋势波段"]
            }

        winning_trades = [t for t in trades if getattr(t, "net_pnl_usd", 0) > 0]
        losing_trades = [t for t in trades if getattr(t, "net_pnl_usd", 0) <= 0]

        # Analyze entry timing on K-lines
        false_breakout_count = 0
        mae_list = []
        mfe_list = []
        surrendered_profits = 0.0

        ts_map = {str(row['timestamp']): idx for idx, row in df.iterrows()}

        for t in trades:
            def _get(field, default=None):
                if isinstance(t, dict):
                    return t.get(field, default)
                return getattr(t, field, default)

            entry_ts = str(_get("entry_time", ""))
            exit_ts = str(_get("exit_time", ""))
            entry_idx = ts_map.get(entry_ts)
            exit_idx = ts_map.get(exit_ts)

            entry_px = float(_get("entry_price", 0.0))
            exit_px = float(_get("exit_price", 0.0))
            token_amt = float(_get("token_amount", 1.0))
            net_pnl = float(_get("net_pnl_usd", 0.0))
            hold_bars = int(_get("hold_bars", 0))

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
                if _get("mae_pct") is not None:
                    mae_list.append(float(_get("mae_pct")))
                if _get("mfe_pct") is not None:
                    mfe_list.append(float(_get("mfe_pct")))
                if _get("surrendered_profit_usd") is not None:
                    surrendered_profits += float(_get("surrendered_profit_usd"))

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

        return {
            "total_trades": total_trades,
            "win_rate": backtest_result.get("win_rate", 0.0),
            "false_breakout_count": false_breakout_count,
            "false_breakout_ratio": round(false_breakout_ratio, 1),
            "avg_mae_pct": round(avg_mae, 2),
            "avg_mfe_pct": round(avg_mfe, 2),
            "surrendered_profit_usd": round(surrendered_profits, 2),
            "friction_drag_pct": round(friction_drag, 1),
            "top_flaws": top_flaws
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

        flaws_text = "\n".join([f"- {f}" for f in diagnostics.get("top_flaws", [])])

        system_prompt = (
            "你是一位世界顶尖的高频与趋势对冲基金量化架构师，擅长根据回测微观成交点位与K线走势，"
            "制定严密、可实装、无未来函数的量化策略演化优化方案。输出必须只用中文，言简意赅，逻辑清晰。"
        )

        user_prompt = f"""【当前策略名称】: {strategy_name}
【优化战略目标】: {goal_desc}

【当前回测关键指标】:
{json.dumps(metrics_summary, ensure_ascii=False, indent=2)}

【K线与点位微观缺陷归因】:
- 假突破被套率: {diagnostics.get('false_breakout_ratio', 0)}%
- 平均最大逆向浮亏 (MAE): {diagnostics.get('avg_mae_pct', 0)}%
- 累计回吐峰值利润: ${diagnostics.get('surrendered_profit_usd', 0)}
- 识别的核心痛点:
{flaws_text}

【当前策略核心代码片段】:
```python
{current_code[:1200]}
```

请输出包含以下维度的深度分析与详细优化 Plan：
1. 【K线点位归因剖析】: 深入分析进场点位、假突破、持仓浮亏与出场滞后的核心成因（2-3句话，一针见血）。
2. 【多维度解决策略】: 
   - 维度1：入场过滤（如 ATR 波动率过滤、均线斜率、成交量放量确认）
   - 维度2：动态出场与保本（如 ATR 移动止盈、梯级保本机制）
   - 维度3：摩擦与假突破防御
3. 【详细实施 Plan】: 列出具体的 3 步代码修改方案（明确具体指标参数，避免模棱两可）。
"""

        try:
            plan_text = ai_service.call_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1800
            )
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


# Singleton instance
strategy_evolution_agent = StrategyEvolutionAgent()
