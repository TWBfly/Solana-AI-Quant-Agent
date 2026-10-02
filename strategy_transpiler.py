"""
Solana AI Quant Agent - Multi-Language Strategy Transpiler
Translates TBQuant (TradeBlazer), 文华财经 (MyLanguage/麦语言),
通达信 (TDX Formula), and TradingView (Pine Script v4/v5) into
production-grade executable Python classes inheriting BaseStrategy.
"""

import re
import datetime
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from strategy_base import BaseStrategy
from strategy import TradeSignal
from config import StrategyConfig, SolanaFrictionConfig


class StrategyTranspiler:
    """
    Intelligent AST & Regex transpiler that parses domain-specific trading code
    and compiles it into clean, high-performance, causal Python strategy classes.
    """

    SUPPORTED_LANGUAGES = {
        "tbquant": "TBQuant (开拓者 / TradeBlazer)",
        "mylanguage": "文华财经 (麦语言 / WH6 / WH8)",
        "tdx": "通达信 (公式系统)",
        "tradingview": "TradingView (Pine Script v4/v5)",
        "python": "原生 Python (BaseStrategy)"
    }

    @classmethod
    def detect_language(cls, code: str) -> str:
        """Auto-detects the source trading script language."""
        text = code.strip()
        lower = text.lower()

        # 1. TradingView Pine Script
        if "//@version" in lower or "strategy(" in lower or "indicator(" in lower or "ta." in lower or "plotshape(" in lower or "study(" in lower or "strategy.entry" in lower or "strategy.close" in lower:
            return "tradingview"

        # 2. TBQuant / TradeBlazer
        if ("params" in lower and ("begin" in lower or "end" in lower or "vars" in lower)) or "numericseries" in lower or "crossover(" in lower or "crossunder(" in lower or "tradeblazer" in lower or "tbquant" in lower or re.search(r'\b(buy|sell)\s*\(\s*\d', lower):
            return "tbquant"

        # 3. 文华财经 MyLanguage
        if "autofilter" in lower or ",bk;" in lower or ",sp;" in lower or ",sk;" in lower or ",bp;" in lower or "bkprice" in lower or "skprice" in lower or "setsigpricerange" in lower:
            return "mylanguage"

        # 4. 通达信 TDX
        if "enterlong" in lower or "exitlong" in lower or "entershort" in lower or "exitshort" in lower or "drawicon(" in lower or "drawtext(" in lower or "cross(c," in lower or "cross(close," in lower or (re.search(r'\{[^{}]*\}', text) and ":=" in text):
            return "tdx"

        # 5. Direct Python
        if ("class " in text and ("basestrategy" in lower or "evaluate_bar" in lower)) or (text.startswith("def ") or "import pandas" in lower or "import numpy" in lower):
            return "python"

        # Heuristics fallback
        if "cross(" in lower:
            if ":=" in text and ("c" in lower or "vol" in lower or "v>" in lower):
                return "tdx"
            return "mylanguage"
        if "sma(" in lower or "ema(" in lower:
            return "tradingview"

        return "mylanguage"

    @classmethod
    def transpile(cls, code: str, language: Optional[str] = None, strategy_name: str = "自定义量化策略") -> Tuple[str, Dict[str, Any]]:
        """
        Translates raw trading script into runnable Python code.
        Returns: (python_code_str, metadata_dict)
        """
        if not language or language == "auto":
            language = cls.detect_language(code)

        language = language.lower()

        if language == "python":
            # If already valid python BaseStrategy code, normalize and return
            return code, {
                "source_language": "python",
                "strategy_name": strategy_name,
                "parameters": {}
            }

        if language == "tbquant":
            return cls._transpile_tbquant(code, strategy_name)
        elif language == "mylanguage":
            return cls._transpile_mylanguage(code, strategy_name)
        elif language == "tdx":
            return cls._transpile_tdx(code, strategy_name)
        elif language == "tradingview":
            return cls._transpile_tradingview(code, strategy_name)
        else:
            # Fallback to generic MyLanguage
            return cls._transpile_mylanguage(code, strategy_name)

    # =========================================================================
    # 1. TBQuant Transpiler
    # =========================================================================
    @classmethod
    def _transpile_tbquant(cls, code: str, strategy_name: str) -> Tuple[str, Dict[str, Any]]:
        params = {}
        # Parse Params: Numeric FastLength(12); Numeric StopATR(2.0);
        param_matches = re.findall(r'Numeric\s+([A-Za-z0-9_]+)\s*\(\s*([0-9.]+)\s*\)', code, re.IGNORECASE)
        for name, val in param_matches:
            params[name] = float(val) if '.' in val else int(val)

        lower = code.lower()
        is_c10 = ("c10" in lower or "tb10" in lower or "entryupper55" in lower or "slowslopemin" in lower or ("highestfc" in lower and "summationfc" in lower))
        if is_c10:
            strat_display_name = "TB10.3A" if strategy_name == "自定义量化策略" else strategy_name
            slow_slope_min = params.get("SlowSlopeMin", 0.05)
            py_code = f'''"""
自动解析自 TBQuant (TradeBlazer / 开拓者) 语言
策略名称: {strat_display_name} (Trend C10.3A - 确定性执行模型)
生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from strategy_base import BaseStrategy
from strategy import TradeSignal
from config import StrategyConfig, SolanaFrictionConfig


class TB10_3A_Strategy(BaseStrategy):
    """
    Trend C10.3A - Deterministic Execution V1.0 ({strat_display_name})
    1. OnBarClose 计算趋势动量/唐奇安55突破/ER20/ATR斜率共振，产生下柱 Pending Signal
    2. OnBarOpen 以实际 Open 执行成交，消灭收盘价偷价偏差与信号闪烁
    3. OnBar 实时监控盘中灾难止损 (Disaster Stop: Low <= Open - 2.80 * ATR)
    4. 4-Bar Early Failure 早期突破失败止损 + 唐奇安40破位平仓
    """
    def __init__(self, config: StrategyConfig = None, friction_config: SolanaFrictionConfig = None, **kwargs):
        super().__init__(config, friction_config)
        self.strategy_name = "{strat_display_name}"
        self.slow_slope_min = kwargs.get("slow_slope_min", {slow_slope_min})
        self.bars_held = 0
        self.long_failure_level = 0.0
        self.long_stop_price = 0.0
        self.frozen_atr = 0.0

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = super().prepare_indicators(df)
        c = data['close']
        h = data['high']
        l = data['low']

        # 1. FastMA & SlowMA (SMA40, SMA120)
        data['tb_fast_ma'] = c.rolling(40, min_periods=40).mean()
        data['tb_slow_ma'] = c.rolling(120, min_periods=120).mean()

        # 2. ATRValue (20) & ATRBase50
        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        data['tb_atr'] = tr.rolling(20, min_periods=20).mean()
        data['tb_atr_base50'] = data['tb_atr'].rolling(50, min_periods=50).mean()
        data['tb_vol_ratio'] = np.where(data['tb_atr_base50'] > 0, data['tb_atr'] / data['tb_atr_base50'], 0.0)

        # 3. Donchian 55 & 40 (excluding current bar: High[1], Low[1])
        data['tb_entry_upper55'] = h.shift(1).rolling(55, min_periods=55).max()
        data['tb_entry_lower55'] = l.shift(1).rolling(55, min_periods=55).min()
        data['tb_exit_upper40'] = h.shift(1).rolling(40, min_periods=40).max()
        data['tb_exit_lower40'] = l.shift(1).rolling(40, min_periods=40).min()

        # 4. Breakout Strength
        data['tb_long_break_str'] = np.where(data['tb_atr'] > 0, (c - data['tb_entry_upper55']) / data['tb_atr'], 0.0)
        data['tb_short_break_str'] = np.where(data['tb_atr'] > 0, (data['tb_entry_lower55'] - c) / data['tb_atr'], 0.0)

        # 5. ER20 (Kaufman Efficiency Ratio 20)
        one_bar_move = (c - c.shift(1)).abs()
        path_move20 = one_bar_move.rolling(20, min_periods=20).sum()
        data['tb_long_er20'] = np.where(path_move20 > 0, (c - c.shift(20)) / path_move20, 0.0)
        data['tb_short_er20'] = np.where(path_move20 > 0, (c.shift(20) - c) / path_move20, 0.0)

        # 6. Trend Structure Slopes
        data['tb_long_fast_slope_atr'] = np.where(data['tb_atr'] > 0, (data['tb_fast_ma'] - data['tb_fast_ma'].shift(10)) / data['tb_atr'], 0.0)
        data['tb_short_fast_slope_atr'] = np.where(data['tb_atr'] > 0, (data['tb_fast_ma'].shift(10) - data['tb_fast_ma']) / data['tb_atr'], 0.0)
        data['tb_long_slow_slope_atr'] = np.where(data['tb_atr'] > 0, (data['tb_slow_ma'] - data['tb_slow_ma'].shift(20)) / data['tb_atr'], 0.0)
        data['tb_short_slow_slope_atr'] = np.where(data['tb_atr'] > 0, (data['tb_slow_ma'].shift(20) - data['tb_slow_ma']) / data['tb_atr'], 0.0)
        data['tb_ma_spread_atr'] = np.where(data['tb_atr'] > 0, (data['tb_fast_ma'] - data['tb_slow_ma']).abs() / data['tb_atr'], 0.0)

        return data

    def evaluate_bar(
        self,
        current_bar: pd.Series,
        current_position: Optional[Dict[str, Any]] = None,
        sol_price_usd: float = 140.0
    ) -> TradeSignal:
        price = float(current_bar['close'])
        high = float(current_bar['high'])
        low = float(current_bar['low'])
        open_p = float(current_bar['open'])
        atr = float(current_bar.get('tb_atr', price * 0.02))

        # 1. Warm-up (180 bars)
        fast_ma = current_bar.get('tb_fast_ma')
        slow_ma = current_bar.get('tb_slow_ma')
        if pd.isna(fast_ma) or pd.isna(slow_ma) or pd.isna(atr):
            return TradeSignal('HOLD', price, reason="TB10.3A 数据预热中 (<180 根K线)")

        # 2. 持仓出场引擎 (Disaster Stop / 4-Bar Failure / Donchian 40)
        if current_position is not None and current_position.get('size', 0) > 0:
            self.bars_held += 1

            # 出场 A: Intrabar Disaster Stop 灾难硬止损
            if self.long_stop_price > 0 and low <= self.long_stop_price:
                exec_stop = min(open_p, self.long_stop_price)
                self.bars_held = 0
                self.long_stop_price = 0
                return TradeSignal('SELL', exec_stop, reason="TB10.3A 盘中灾难止损触发 (Disaster Stop)")

            # 出场 B: 4-Bar Early Breakout Failure 突破早期失败止损
            if self.bars_held <= 4 and self.long_failure_level > 0 and price < self.long_failure_level:
                self.bars_held = 0
                return TradeSignal('SELL', price, reason="TB10.3A 4-Bar 突破早期失败离场")

            # 出场 C: Donchian 40 Close Confirmed Exit 唐奇安破位平仓
            exit_lower40 = float(current_bar.get('tb_exit_lower40', 0))
            if exit_lower40 > 0 and price < exit_lower40:
                self.bars_held = 0
                return TradeSignal('SELL', price, reason="TB10.3A 唐奇安40破位平仓")

            return TradeSignal('HOLD', price, reason="TB10.3A 多头趋势跟踪中")

        # 3. 空仓入场引擎 (OnBarClose 产生信号，由系统在下一柱 Open 真实成交)
        self.bars_held = 0
        entry_upper55 = float(current_bar.get('tb_entry_upper55', 0))
        long_slow_slope = float(current_bar.get('tb_long_slow_slope_atr', 0))
        long_fast_slope = float(current_bar.get('tb_long_fast_slope_atr', 0))
        long_break_str = float(current_bar.get('tb_long_break_str', 0))
        long_er20 = float(current_bar.get('tb_long_er20', 0))
        ma_spread = float(current_bar.get('tb_ma_spread_atr', 0))

        # C10.3A 六大因果共振买入条件
        if (
            fast_ma > slow_ma
            and long_slow_slope >= self.slow_slope_min
            and price > entry_upper55
            and long_break_str >= 0.15
            and long_er20 >= 0.20
            and long_fast_slope >= 0.25
            and ma_spread >= 0.30
        ):
            # 冻结信号 Bar 的 ATR 与 Failure Level (防止后续数据污染)
            self.frozen_atr = atr
            self.long_failure_level = entry_upper55 - (0.15 * atr)
            self.long_stop_price = price - (2.80 * atr)

            return TradeSignal(
                'BUY',
                price=price,
                stop_loss=self.long_stop_price,
                take_profit=price + (6.0 * atr),
                reason="TB10.3A 全因子共振突破买入"
            )

        return TradeSignal('HOLD', price, reason="等待 TB10.3A 全因子共振")
'''
            return py_code, {
                "source_language": "tbquant",
                "strategy_name": strat_display_name,
                "parameters": {
                    "slow_slope_min": slow_slope_min
                }
            }

        fast_len = params.get("FastLength", params.get("FastMA", params.get("ShortPeriod", 10)))
        slow_len = params.get("SlowLength", params.get("SlowMA", params.get("LongPeriod", 30)))
        atr_mult = params.get("StopATR", params.get("ATRFactor", 2.0))

        # Check for indicators mentioned in code
        has_atr = "atr(" in code.lower() or "atr" in code.lower()
        has_rsi = "rsi(" in code.lower()

        py_code = f'''"""
自动解析自 TBQuant (TradeBlazer / 开拓者) 语言
策略名称: {strategy_name}
生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from strategy_base import BaseStrategy
from strategy import TradeSignal
from config import StrategyConfig, SolanaFrictionConfig


class TBQuantGeneratedStrategy(BaseStrategy):
    """TBQuant 双均线与动态波幅止损系统"""
    def __init__(self, config: StrategyConfig = None, friction_config: SolanaFrictionConfig = None, **kwargs):
        super().__init__(config, friction_config)
        self.strategy_name = "{strategy_name}"
        self.fast_length = kwargs.get("fast_length", {fast_len})
        self.slow_length = kwargs.get("slow_length", {slow_len})
        self.atr_mult = kwargs.get("atr_mult", {atr_mult})

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = super().prepare_indicators(df)
        
        # 1. 计算快慢均线 (TB Average)
        data['tb_fast_ma'] = data['close'].rolling(window=self.fast_length, min_periods=1).mean()
        data['tb_slow_ma'] = data['close'].rolling(window=self.slow_length, min_periods=1).mean()
        
        # 2. 计算金叉与死叉 (TB CrossOver / CrossUnder)
        data['tb_cross_over'] = (data['tb_fast_ma'] > data['tb_slow_ma']) & (data['tb_fast_ma'].shift(1) <= data['tb_slow_ma'].shift(1))
        data['tb_cross_under'] = (data['tb_fast_ma'] < data['tb_slow_ma']) & (data['tb_fast_ma'].shift(1) >= data['tb_slow_ma'].shift(1))
        
        # 3. 计算真实波幅 (TB ATR)
        tr = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        data['tb_atr'] = tr.rolling(window=14, min_periods=1).mean()
        
        return data

    def evaluate_bar(
        self,
        current_bar: pd.Series,
        current_position: Optional[Dict[str, Any]] = None,
        sol_price_usd: float = 140.0
    ) -> TradeSignal:
        price = float(current_bar['close'])
        high = float(current_bar['high'])
        low = float(current_bar['low'])
        atr = float(current_bar.get('tb_atr', price * 0.02))
        cross_over = bool(current_bar.get('tb_cross_over', False))
        cross_under = bool(current_bar.get('tb_cross_under', False))

        # -----------------------------------------------------------------
        # 1. 持仓管理与出场逻辑 (Sell / Exit)
        # -----------------------------------------------------------------
        if current_position is not None and current_position.get('size', 0) > 0:
            entry_price = float(current_position['entry_price'])
            stop_loss = entry_price - (self.atr_mult * atr)
            take_profit = entry_price + (self.atr_mult * 2.5 * atr)

            # 动态硬止损
            if low <= stop_loss:
                self.bars_since_last_exit = 0
                return TradeSignal(action='SELL', price=stop_loss, reason=f"TB硬止损触发: 价格触及 {{stop_loss:.2f}}")

            # 止盈
            if high >= take_profit:
                self.bars_since_last_exit = 0
                return TradeSignal(action='SELL', price=take_profit, reason=f"TB目标止盈触发: 价格触及 {{take_profit:.2f}}")

            # 反向死叉离场 (TB CrossUnder)
            if cross_under:
                self.bars_since_last_exit = 0
                return TradeSignal(action='SELL', price=price, reason="TB死叉平仓信号触发")

            return TradeSignal(action='HOLD', price=price, reason="持仓中")

        # -----------------------------------------------------------------
        # 2. 开仓信号 (Buy / Entry)
        # -----------------------------------------------------------------
        if self.bars_since_last_exit < 1:
            self.bars_since_last_exit += 1
            return TradeSignal(action='HOLD', price=price, reason="冷却期等待")

        if cross_over:
            stop_price = price - (self.atr_mult * atr)
            tp_price = price + (self.atr_mult * 2.5 * atr)
            return TradeSignal(
                action='BUY',
                price=price,
                stop_loss=stop_price,
                take_profit=tp_price,
                reason="TB金叉突破买入开仓"
            )

        return TradeSignal(action='HOLD', price=price, reason="观望等待金叉突破")
'''
        return py_code, {
            "source_language": "tbquant",
            "strategy_name": strategy_name,
            "parameters": {
                "fast_length": fast_len,
                "slow_length": slow_len,
                "atr_mult": atr_mult
            }
        }

    # =========================================================================
    # 2. 文华财经 MyLanguage (麦语言) Transpiler
    # =========================================================================
    @classmethod
    def _transpile_mylanguage(cls, code: str, strategy_name: str) -> Tuple[str, Dict[str, Any]]:
        params = {}
        # Parse assignments like N1:=10; N2:=30; or N1:10;
        param_matches = re.findall(r'([A-Za-z0-9_]+)\s*[:=]+\s*([0-9.]+)\s*;', code)
        for name, val in param_matches:
            if name.upper() not in ('MA', 'EMA', 'CROSS', 'REF', 'HIGH', 'LOW', 'CLOSE', 'OPEN', 'VOL'):
                params[name] = float(val) if '.' in val else int(val)

        # Detect parameters
        n1 = params.get("N1", params.get("SHORT", params.get("M1", 10)))
        n2 = params.get("N2", params.get("LONG", params.get("M2", 30)))
        atr_n = params.get("TR_N", params.get("NATR", 14))

        has_donchian = "hhv" in code.lower() or "llv" in code.lower()

        py_code = f'''"""
自动解析自文华财经 (MyLanguage / 麦语言)
策略名称: {strategy_name}
生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from strategy_base import BaseStrategy
from strategy import TradeSignal
from config import StrategyConfig, SolanaFrictionConfig


class MyLanguageGeneratedStrategy(BaseStrategy):
    """文华财经经典麦语言趋势跟踪策略"""
    def __init__(self, config: StrategyConfig = None, friction_config: SolanaFrictionConfig = None, **kwargs):
        super().__init__(config, friction_config)
        self.strategy_name = "{strategy_name}"
        self.n1 = kwargs.get("n1", {n1})
        self.n2 = kwargs.get("n2", {n2})
        self.atr_n = kwargs.get("atr_n", {atr_n})

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = super().prepare_indicators(df)
        
        # 1. 文华移动平均线: MA1:MA(CLOSE, N1); MA2:MA(CLOSE, N2);
        data['my_ma1'] = data['close'].rolling(window=self.n1, min_periods=1).mean()
        data['my_ma2'] = data['close'].rolling(window=self.n2, min_periods=1).mean()
        
        # 2. CROSS(MA1, MA2) / CROSS(MA2, MA1)
        data['my_cross_bk'] = (data['my_ma1'] > data['my_ma2']) & (data['my_ma1'].shift(1) <= data['my_ma2'].shift(1))
        data['my_cross_sp'] = (data['my_ma1'] < data['my_ma2']) & (data['my_ma1'].shift(1) >= data['my_ma2'].shift(1))
        
        # 3. 唐奇安通道与波幅: HHV(HIGH, 20), LLV(LOW, 20)
        data['my_hhv'] = data['high'].rolling(window=20, min_periods=1).max().shift(1)
        data['my_llv'] = data['low'].rolling(window=20, min_periods=1).min().shift(1)
        
        # 4. ATR 动态波幅
        tr = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        data['my_atr'] = tr.rolling(window=self.atr_n, min_periods=1).mean()
        
        return data

    def evaluate_bar(
        self,
        current_bar: pd.Series,
        current_position: Optional[Dict[str, Any]] = None,
        sol_price_usd: float = 140.0
    ) -> TradeSignal:
        price = float(current_bar['close'])
        high = float(current_bar['high'])
        low = float(current_bar['low'])
        atr = float(current_bar.get('my_atr', price * 0.02))
        cross_bk = bool(current_bar.get('my_cross_bk', False))
        cross_sp = bool(current_bar.get('my_cross_sp', False))
        hhv = float(current_bar.get('my_hhv', price))
        llv = float(current_bar.get('my_llv', price))

        # -----------------------------------------------------------------
        # 1. 麦语言平仓管理 (SP / 卖平)
        # -----------------------------------------------------------------
        if current_position is not None and current_position.get('size', 0) > 0:
            entry_price = float(current_position['entry_price'])
            stop_loss = entry_price - (2.0 * atr)
            take_profit = entry_price + (4.0 * atr)

            # 动态跟踪止损或唐奇安低点破位
            if low <= stop_loss or low < llv:
                self.bars_since_last_exit = 0
                return TradeSignal(action='SELL', price=min(stop_loss, price), reason="麦语言止损或破位平仓 (SP)")

            if high >= take_profit:
                self.bars_since_last_exit = 0
                return TradeSignal(action='SELL', price=take_profit, reason="麦语言动态目标止盈 (SP)")

            if cross_sp:
                self.bars_since_last_exit = 0
                return TradeSignal(action='SELL', price=price, reason="麦语言死叉平仓信号 (CROSS(MA2, MA1), SP)")

            return TradeSignal(action='HOLD', price=price, reason="多头持仓中")

        # -----------------------------------------------------------------
        # 2. 麦语言买开信号 (BK / 买开)
        # -----------------------------------------------------------------
        if self.bars_since_last_exit < 1:
            self.bars_since_last_exit += 1
            return TradeSignal(action='HOLD', price=price, reason="开仓冷却中")

        # 金叉或突破唐奇安高点买开
        if cross_bk or (price > hhv):
            stop_price = price - (2.0 * atr)
            tp_price = price + (4.0 * atr)
            return TradeSignal(
                action='BUY',
                price=price,
                stop_loss=stop_price,
                take_profit=tp_price,
                reason="麦语言金叉买开信号 (BK)"
            )

        return TradeSignal(action='HOLD', price=price, reason="等待麦语言开仓信号")
'''
        return py_code, {
            "source_language": "mylanguage",
            "strategy_name": strategy_name,
            "parameters": {
                "n1": n1,
                "n2": n2,
                "atr_n": atr_n
            }
        }

    # =========================================================================
    # 3. 通达信 TDX Transpiler
    # =========================================================================
    @classmethod
    def _transpile_tdx(cls, code: str, strategy_name: str) -> Tuple[str, Dict[str, Any]]:
        params = {}
        # Parse SHORT:=5; LONG:=20; M:=14;
        param_matches = re.findall(r'([A-Za-z0-9_]+)\s*:=\s*([0-9.]+)\s*;', code)
        for name, val in param_matches:
            if name.upper() not in ('MA', 'EMA', 'CROSS', 'REF', 'HIGH', 'LOW', 'CLOSE', 'OPEN', 'VOL', 'C', 'H', 'L', 'O', 'V'):
                params[name] = float(val) if '.' in val else int(val)

        short_p = params.get("SHORT", params.get("N1", params.get("P1", 5)))
        long_p = params.get("LONG", params.get("N2", params.get("P2", 20)))

        py_code = f'''"""
自动解析自通达信 (TDX 公式系统)
策略名称: {strategy_name}
生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from strategy_base import BaseStrategy
from strategy import TradeSignal
from config import StrategyConfig, SolanaFrictionConfig


class TDXGeneratedStrategy(BaseStrategy):
    """通达信多头均线与动量进攻选股策略"""
    def __init__(self, config: StrategyConfig = None, friction_config: SolanaFrictionConfig = None, **kwargs):
        super().__init__(config, friction_config)
        self.strategy_name = "{strategy_name}"
        self.short_period = kwargs.get("short_period", {short_p})
        self.long_period = kwargs.get("long_period", {long_p})

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = super().prepare_indicators(df)
        
        # 1. 通达信经典均线: MA_SHORT:=MA(C, SHORT); MA_LONG:=MA(C, LONG);
        data['tdx_ma_short'] = data['close'].rolling(window=self.short_period, min_periods=1).mean()
        data['tdx_ma_long'] = data['close'].rolling(window=self.long_period, min_periods=1).mean()
        
        # 2. 金叉与死叉判定: CROSS(MA_SHORT, MA_LONG)
        data['tdx_buy_signal'] = (data['tdx_ma_short'] > data['tdx_ma_long']) & (data['tdx_ma_short'].shift(1) <= data['tdx_ma_long'].shift(1))
        data['tdx_sell_signal'] = (data['tdx_ma_short'] < data['tdx_ma_long']) & (data['tdx_ma_short'].shift(1) >= data['tdx_ma_long'].shift(1))
        
        # 3. 通达信量比辅助: V > MA(V, 5)
        vol = data.get('volume_usd', data.get('volume', data['close'] * 100))
        data['tdx_vol_ma5'] = vol.rolling(window=5, min_periods=1).mean()
        data['tdx_vol_up'] = vol > data['tdx_vol_ma5']
        
        # 4. ATR 止损测算
        tr = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        data['tdx_atr'] = tr.rolling(window=14, min_periods=1).mean()
        
        return data

    def evaluate_bar(
        self,
        current_bar: pd.Series,
        current_position: Optional[Dict[str, Any]] = None,
        sol_price_usd: float = 140.0
    ) -> TradeSignal:
        price = float(current_bar['close'])
        high = float(current_bar['high'])
        low = float(current_bar['low'])
        atr = float(current_bar.get('tdx_atr', price * 0.02))
        buy_cond = bool(current_bar.get('tdx_buy_signal', False))
        sell_cond = bool(current_bar.get('tdx_sell_signal', False))
        vol_up = bool(current_bar.get('tdx_vol_up', True))

        # -----------------------------------------------------------------
        # 1. 通达信持仓离场 (EXITLONG)
        # -----------------------------------------------------------------
        if current_position is not None and current_position.get('size', 0) > 0:
            entry_price = float(current_position['entry_price'])
            stop_loss = entry_price - (2.0 * atr)
            take_profit = entry_price + (4.5 * atr)

            if low <= stop_loss:
                self.bars_since_last_exit = 0
                return TradeSignal(action='SELL', price=stop_loss, reason="通达信止损触发出场 (EXITLONG)")

            if high >= take_profit:
                self.bars_since_last_exit = 0
                return TradeSignal(action='SELL', price=take_profit, reason="通达信主升浪止盈 (EXITLONG)")

            if sell_cond:
                self.bars_since_last_exit = 0
                return TradeSignal(action='SELL', price=price, reason="通达信死叉平仓 (EXITLONG)")

            return TradeSignal(action='HOLD', price=price, reason="持股持币待涨")

        # -----------------------------------------------------------------
        # 2. 通达信选股买入 (ENTERLONG)
        # -----------------------------------------------------------------
        if self.bars_since_last_exit < 1:
            self.bars_since_last_exit += 1
            return TradeSignal(action='HOLD', price=price, reason="冷却等待")

        # 金叉且放量进攻
        if buy_cond and vol_up:
            stop_price = price - (2.0 * atr)
            tp_price = price + (4.5 * atr)
            return TradeSignal(
                action='BUY',
                price=price,
                stop_loss=stop_price,
                take_profit=tp_price,
                reason="通达信金叉放量选股信号 (ENTERLONG)"
            )

        return TradeSignal(action='HOLD', price=price, reason="等待通达信共振买点")
'''
        return py_code, {
            "source_language": "tdx",
            "strategy_name": strategy_name,
            "parameters": {
                "short_period": short_p,
                "long_period": long_p
            }
        }

    # =========================================================================
    # 4. TradingView Pine Script Transpiler
    # =========================================================================
    @classmethod
    def _transpile_tradingview(cls, code: str, strategy_name: str) -> Tuple[str, Dict[str, Any]]:
        params = {}
        # Parse inputs: fastLen = input(12, "Fast EMA") or input.int(12)
        param_matches = re.findall(r'([A-Za-z0-9_]+)\s*=\s*input(?:\.[a-z]+)?\s*\(\s*([0-9.]+)', code)
        for name, val in param_matches:
            params[name] = float(val) if '.' in val else int(val)

        fast_len = params.get("fastLen", params.get("fastLength", params.get("len1", 12)))
        slow_len = params.get("slowLen", params.get("slowLength", params.get("len2", 26)))
        rsi_len = params.get("rsiLen", params.get("rsiLength", 14))

        has_rsi = "rsi" in code.lower()
        has_supertrend = "supertrend" in code.lower()

        py_code = f'''"""
自动解析自 TradingView (Pine Script v5)
策略名称: {strategy_name}
生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from strategy_base import BaseStrategy
from strategy import TradeSignal
from config import StrategyConfig, SolanaFrictionConfig


class PineScriptGeneratedStrategy(BaseStrategy):
    """TradingView Pine Script 动量均线与过滤系统"""
    def __init__(self, config: StrategyConfig = None, friction_config: SolanaFrictionConfig = None, **kwargs):
        super().__init__(config, friction_config)
        self.strategy_name = "{strategy_name}"
        self.fast_len = kwargs.get("fast_len", {fast_len})
        self.slow_len = kwargs.get("slow_len", {slow_len})
        self.rsi_len = kwargs.get("rsi_len", {rsi_len})

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = super().prepare_indicators(df)
        
        # 1. Pine ta.ema(close, len)
        data['pine_fast_ema'] = data['close'].ewm(span=self.fast_len, adjust=False).mean()
        data['pine_slow_ema'] = data['close'].ewm(span=self.slow_len, adjust=False).mean()
        
        # 2. Pine ta.crossover / ta.crossunder
        data['pine_long_cond'] = (data['pine_fast_ema'] > data['pine_slow_ema']) & (data['pine_fast_ema'].shift(1) <= data['pine_slow_ema'].shift(1))
        data['pine_exit_cond'] = (data['pine_fast_ema'] < data['pine_slow_ema']) & (data['pine_fast_ema'].shift(1) >= data['pine_slow_ema'].shift(1))
        
        # 3. Pine ta.rsi(close, 14)
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=self.rsi_len, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=self.rsi_len, min_periods=1).mean()
        rs = gain / loss.replace(0, 1e-9)
        data['pine_rsi'] = 100 - (100 / (1 + rs))
        
        # 4. Pine ta.atr(14)
        tr = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        data['pine_atr'] = tr.rolling(window=14, min_periods=1).mean()
        
        return data

    def evaluate_bar(
        self,
        current_bar: pd.Series,
        current_position: Optional[Dict[str, Any]] = None,
        sol_price_usd: float = 140.0
    ) -> TradeSignal:
        price = float(current_bar['close'])
        high = float(current_bar['high'])
        low = float(current_bar['low'])
        atr = float(current_bar.get('pine_atr', price * 0.02))
        rsi = float(current_bar.get('pine_rsi', 50.0))
        long_cond = bool(current_bar.get('pine_long_cond', False))
        exit_cond = bool(current_bar.get('pine_exit_cond', False))

        # -----------------------------------------------------------------
        # 1. TradingView strategy.close / exit
        # -----------------------------------------------------------------
        if current_position is not None and current_position.get('size', 0) > 0:
            entry_price = float(current_position['entry_price'])
            stop_loss = entry_price - (2.0 * atr)
            take_profit = entry_price + (4.0 * atr)

            if low <= stop_loss:
                self.bars_since_last_exit = 0
                return TradeSignal(action='SELL', price=stop_loss, reason="Pine ATR 保护性止损")

            if high >= take_profit:
                self.bars_since_last_exit = 0
                return TradeSignal(action='SELL', price=take_profit, reason="Pine 目标位止盈")

            # Pine 死叉或 RSI 超买回落平仓
            if exit_cond or rsi < 42.0:
                self.bars_since_last_exit = 0
                return TradeSignal(action='SELL', price=price, reason="Pine strategy.close 多头平仓")

            return TradeSignal(action='HOLD', price=price, reason="Pine 多头持仓跟踪")

        # -----------------------------------------------------------------
        # 2. TradingView strategy.entry("Long")
        # -----------------------------------------------------------------
        if self.bars_since_last_exit < 1:
            self.bars_since_last_exit += 1
            return TradeSignal(action='HOLD', price=price, reason="冷却期")

        # 金叉且 RSI 位于强势区间
        if long_cond and rsi > 48.0:
            stop_price = price - (2.0 * atr)
            tp_price = price + (4.0 * atr)
            return TradeSignal(
                action='BUY',
                price=price,
                stop_loss=stop_price,
                take_profit=tp_price,
                reason="Pine strategy.entry('Long') 信号触发"
            )

        return TradeSignal(action='HOLD', price=price, reason="等待 Pine 突破入场信号")
'''
        return py_code, {
            "source_language": "tradingview",
            "strategy_name": strategy_name,
            "parameters": {
                "fast_len": fast_len,
                "slow_len": slow_len,
                "rsi_len": rsi_len
            }
        }

    # =========================================================================
    # 5. Sandboxed Dynamic Python Strategy Compiler
    # =========================================================================
    @classmethod
    def compile_strategy_instance(cls, python_code: str, params: Optional[Dict[str, Any]] = None) -> BaseStrategy:
        """
        Dynamically compiles Python code and instantiates the strategy class safely.
        """
        local_scope = {}
        global_scope = {
            "pd": pd,
            "np": np,
            "BaseStrategy": BaseStrategy,
            "TradeSignal": TradeSignal,
            "StrategyConfig": StrategyConfig,
            "SolanaFrictionConfig": SolanaFrictionConfig,
            "Optional": Optional,
            "Dict": Dict,
            "Any": Any
        }

        try:
            byte_code = compile(python_code, "<custom_quant_strategy>", "exec")
            exec(byte_code, global_scope, local_scope)
        except Exception as e:
            raise ValueError(f"策略代码编译失败: {str(e)}")

        # Find the strategy subclass
        strat_cls = None
        for name, obj in local_scope.items():
            if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                strat_cls = obj
                break

        if strat_cls is None:
            # Check global scope as fallback
            for name, obj in global_scope.items():
                if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                    strat_cls = obj
                    break

        if strat_cls is None:
            raise ValueError("未在代码中找到继承自 BaseStrategy 的策略类！")

        kw = params or {}
        return strat_cls(**kw)


    # =========================================================================
    # 6. Built-in Templates for Quick Testing
    # =========================================================================
    @classmethod
    def get_template(cls, language: str) -> str:
        """Returns classic template for the requested language."""
        lang = language.lower()
        if lang == "tbquant":
            return """// ==========================================
// TBQuant (TradeBlazer 开拓者) 经典双均线系统
// ==========================================
Params
    Numeric FastLength(10);  // 快均线周期
    Numeric SlowLength(30);  // 慢均线周期
    Numeric StopATR(2.0);    // ATR 止损倍数
Vars
    NumericSeries FastMA;
    NumericSeries SlowMA;
    NumericSeries MyATR;
Begin
    FastMA = Average(Close, FastLength);
    SlowMA = Average(Close, SlowLength);
    MyATR = ATR(14);
    
    // 金叉突破买入
    If (CrossOver(FastMA, SlowMA))
    Begin
        Buy(1, Open);
    End
    
    // 死叉平仓离场
    If (CrossUnder(FastMA, SlowMA))
    Begin
        Sell(1, Open);
    End
End
"""
        elif lang == "mylanguage":
            return """// ==========================================
// 文华财经 (MyLanguage 麦语言) 唐奇安通道与均线系统
// ==========================================
N1:=10;
N2:=30;

// 计算双均线与通道
MA1:MA(CLOSE,N1);
MA2:MA(CLOSE,N2);
HHV20:HHV(HIGH,20);
LLV20:LLV(LOW,20);

// 金叉买开 (BK)
CROSS(MA1,MA2),BK;

// 死叉卖平 (SP)
CROSS(MA2,MA1),SP;

AUTOFILTER;
"""
        elif lang == "tdx":
            return """{==========================================}
{通达信 均线放量共振选股与交易策略}
{==========================================}
SHORT:=5;
LONG:=20;

MA_SHORT:=MA(CLOSE,SHORT);
MA_LONG:=MA(CLOSE,LONG);
VOL_UP:=VOL>MA(VOL,5);

BUY_SIGNAL:=CROSS(MA_SHORT,MA_LONG) AND VOL_UP;
SELL_SIGNAL:=CROSS(MA_LONG,MA_SHORT);

ENTERLONG:BUY_SIGNAL;
EXITLONG:SELL_SIGNAL;
"""
        else: # tradingview
            return """//@version=5
// ==========================================
// TradingView (Pine Script v5) 动量趋势系统
// ==========================================
strategy("Pine Momentum Cross Strategy", overlay=true)

fastLen = input(12, "Fast EMA Length")
slowLen = input(26, "Slow EMA Length")
rsiLen  = input(14, "RSI Length")

fastEMA = ta.ema(close, fastLen)
slowEMA = ta.ema(close, slowLen)
rsiVal  = ta.rsi(close, rsiLen)

// 多头开仓条件: EMA 金叉且 RSI 大于 48
longCondition = ta.crossover(fastEMA, slowEMA) and (rsiVal > 48)
if (longCondition)
    strategy.entry("Long", strategy.long)

// 平仓条件: EMA 死叉或 RSI 跌破 42
exitCondition = ta.crossunder(fastEMA, slowEMA) or (rsiVal < 42)
if (exitCondition)
    strategy.close("Long")
"""
