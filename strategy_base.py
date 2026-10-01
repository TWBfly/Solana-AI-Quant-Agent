"""
Solana AI Quant Agent - Strategy Base Class & Custom Strategy Template
Defines the standard protocol for authoring quantitative strategies on Solana DEX.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
from strategy import TradeSignal
from config import StrategyConfig, SolanaFrictionConfig
from friction import SolanaFrictionModel


class BaseStrategy(ABC):
    """
    Abstract Base Class for all Solana quantitative strategies.
    
    To implement a custom strategy:
    1. Inherit from BaseStrategy.
    2. (Optional) Override `prepare_indicators(df)` to compute custom factors.
    3. Implement `evaluate_bar(current_bar, current_position, sol_price_usd) -> TradeSignal`.
    """
    def __init__(self, config: StrategyConfig = None, friction_config: SolanaFrictionConfig = None):
        self.config = config or StrategyConfig()
        self.friction_config = friction_config or SolanaFrictionConfig()
        self.friction_model = SolanaFrictionModel(self.friction_config)
        self.bars_since_last_exit = 10

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates indicators and features across the historical DataFrame.
        Override to add custom machine learning factors, RSI, MACD, etc.
        Must strictly avoid future lookahead bias (e.g. use .shift(1) where appropriate).
        """
        from factors import compute_all_factors
        return compute_all_factors(df, self.config)

    @abstractmethod
    def evaluate_bar(
        self,
        current_bar: pd.Series,
        current_position: Optional[Dict[str, Any]] = None,
        sol_price_usd: float = 140.0
    ) -> TradeSignal:
        """
        Evaluates the latest closed bar and generates a TradeSignal.
        
        Args:
            current_bar: pd.Series of the current bar (open, high, low, close, volume, indicators).
            current_position: Dict containing active position info or None:
                              {'size': float, 'entry_price': float, 'highest_price': float, 'stop_loss': float}
            sol_price_usd: Current price of SOL in USD for gas fee calculation.
            
        Returns:
            TradeSignal(action='BUY'|'SELL'|'HOLD', price=float, stop_loss=float, take_profit=float, reason=str)
        """
        pass


class CustomRSIMomentumStrategy(BaseStrategy):
    """
    Example of a custom user-defined strategy:
    Combines RSI Momentum filter with SuperTrend trend-following on Solana.
    """
    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = super().prepare_indicators(df)
        
        # Calculate 14-period RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1e-9)
        data['rsi'] = 100 - (100 / (1 + rs))
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
        atr = float(current_bar.get('atr', price * 0.02))
        rsi = float(current_bar.get('rsi', 50.0))
        trend_dir = int(current_bar.get('trend_dir', 1))

        # 1. Manage Active Position
        if current_position is not None and current_position.get('size', 0) > 0:
            entry_price = float(current_position['entry_price'])
            hard_stop = entry_price - (2.0 * atr)
            take_profit = entry_price + (4.0 * atr)

            if low <= hard_stop:
                self.bars_since_last_exit = 0
                return TradeSignal(action='SELL', price=hard_stop, reason=f"CUSTOM_RSI_HARD_STOP: {low:.2f} <= {hard_stop:.2f}")

            if high >= take_profit:
                self.bars_since_last_exit = 0
                return TradeSignal(action='SELL', price=take_profit, reason=f"CUSTOM_RSI_TAKE_PROFIT: {high:.2f} >= {take_profit:.2f}")

            if trend_dir == -1 or rsi < 45.0:
                self.bars_since_last_exit = 0
                return TradeSignal(action='SELL', price=price, reason=f"CUSTOM_RSI_EXIT: Trend flipped or RSI dropped below 45 ({rsi:.1f})")

            return TradeSignal(action='HOLD', price=price, reason="CUSTOM_RSI_HOLD")

        # 2. Evaluate Entry
        self.bars_since_last_exit += 1
        if self.bars_since_last_exit < 4:
            return TradeSignal(action='HOLD', price=price, reason="COOLDOWN")

        # Entry logic: Bullish trend, RSI between 52 and 68 (momentum expansion, not overbought)
        if trend_dir == 1 and (52.0 <= rsi <= 68.0):
            stop_loss = price - (2.0 * atr)
            take_profit = price + (4.0 * atr)
            return TradeSignal(
                action='BUY',
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=0.85,
                reason=f"CUSTOM_RSI_ENTRY: SuperTrend Bullish + RSI Momentum ({rsi:.1f})"
            )

        return TradeSignal(action='HOLD', price=price, reason="NO_SETUP")
