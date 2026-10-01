"""
Solana AI Quant Agent - Solana Trend & Liquidity Strategy Agent
Evaluates on-chain metrics, DEX order flow, and technical factors to produce
actionable BUY/SELL signals equipped with dynamic ATR Chandelier exits and hard stops.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
from config import StrategyConfig, SolanaFrictionConfig
from friction import SolanaFrictionModel


@dataclass
class TradeSignal:
    action: str                        # 'BUY', 'SELL', 'HOLD'
    price: float                       # Current trigger price
    stop_loss: Optional[float] = None  # Dynamic initial hard stop
    trailing_stop: Optional[float] = None # Dynamic trailing chandelier stop
    take_profit: Optional[float] = None   # Target take profit level
    confidence: float = 1.0            # Signal confidence score (0.0 to 1.0)
    reason: str = ""                   # Diagnostic audit rationale


class SolanaTrendAgent:
    """
    AI Quant Agent combining on-chain DEX liquidity, order flow,
    and Zero-Lag trend following with dynamic risk gates.
    """
    def __init__(self, config: StrategyConfig = None, friction_config: SolanaFrictionConfig = None):
        self.config = config or StrategyConfig()
        self.friction_model = SolanaFrictionModel(friction_config or SolanaFrictionConfig())
        self.bars_since_last_exit = 10

    def evaluate_bar(
        self,
        current_bar: pd.Series,
        current_position: Optional[Dict[str, Any]] = None,
        sol_price_usd: float = 140.0
    ) -> TradeSignal:
        """
        Evaluates the latest closed bar and generates causal trading signals.
        Strictly prevents lookahead bias.
        """
        price = float(current_bar['close'])
        high = float(current_bar['high'])
        low = float(current_bar['low'])
        atr = float(current_bar.get('atr', price * 0.02))
        z_fast = float(current_bar.get('zlema_fast', price))
        z_slow = float(current_bar.get('zlema_slow', price))
        z_slope = float(current_bar.get('zlema_slope', 0.0))
        trend_dir = int(current_bar.get('trend_dir', 1))
        st_line = float(current_bar.get('supertrend', price))
        liquidity = float(current_bar.get('liquidity_usd', 1000000.0))
        buy_ratio = float(current_bar.get('buy_ratio', 0.50))
        vol_ratio = float(current_bar.get('vol_ratio', 1.0))
        highest_20 = float(current_bar.get('highest_20', price))

        # -------------------------------------------------------------
        # 1. Manage Active Long Position (Exit Conditions)
        # -------------------------------------------------------------
        if current_position is not None and current_position.get('size', 0) > 0:
            entry_price = float(current_position['entry_price'])
            highest_since_entry = float(current_position.get('highest_price', high))
            highest_since_entry = max(highest_since_entry, high)

            hard_stop = entry_price - (self.config.stop_loss_atr_mult * atr)
            take_profit = entry_price + (self.config.take_profit_target_rr * atr)

            # Check Hard Stop Trigger (Disaster Protection)
            if low <= hard_stop:
                self.bars_since_last_exit = 0
                reason = f"HARD_STOP_TRIGGERED: Low {low:.2f} <= Hard Stop {hard_stop:.2f} (Entry: {entry_price:.2f})"
                return TradeSignal(action='SELL', price=hard_stop, reason=reason)

            # Check Take-Profit Milestone (Parabolic Extension)
            if high >= take_profit:
                self.bars_since_last_exit = 0
                reason = f"TAKE_PROFIT_TRIGGERED: High {high:.2f} >= Target {take_profit:.2f}"
                return TradeSignal(action='SELL', price=take_profit, reason=reason)

            # Check Trend Breakdown: SuperTrend flips to Bearish
            # This allows the trend to ride full wave without premature noise stop-outs
            if trend_dir == -1:
                self.bars_since_last_exit = 0
                reason = f"SUPERTREND_FLIP: SuperTrend turned Bearish at {price:.2f}"
                return TradeSignal(action='SELL', price=price, reason=reason)

            # Position maintained
            return TradeSignal(
                action='HOLD',
                price=price,
                trailing_stop=st_line,
                stop_loss=hard_stop,
                take_profit=take_profit,
                reason=f"HOLDING: PnL {((price - entry_price) / entry_price) * 100:+.2f}% | ST: {st_line:.2f}"
            )

        # -------------------------------------------------------------
        # 2. Evaluate Potential Long Entry (No Active Position)
        # -------------------------------------------------------------
        self.bars_since_last_exit += 1

        # Gate 1: Liquidity Filter
        if liquidity < self.config.min_pool_liquidity_usd:
            return TradeSignal(action='HOLD', price=price, reason=f"GATE_REJECT: Liquidity ${liquidity:,.0f} < ${self.config.min_pool_liquidity_usd:,.0f}")

        # Gate 2: Cooldown Filter (avoid whipsaw right after stop)
        if self.bars_since_last_exit < 4:
            return TradeSignal(action='HOLD', price=price, reason=f"COOLDOWN_ACTIVE: {self.bars_since_last_exit}/4 bars")

        cross_up = bool(current_bar.get('cross_up', False))
        highest_20 = float(current_bar.get('highest_20', price))
        breakout = (price >= highest_20 * 0.999)

        # Gate 3: Order Flow Filter (Buyers must hold active interest)
        if buy_ratio < 0.48:
            return TradeSignal(action='HOLD', price=price, reason=f"GATE_REJECT: Buy Ratio {buy_ratio:.2f} < 0.48")

        # Trend & Volume Momentum Setup:
        # Require clear separation between Fast and Slow ZLEMA (avoid flat chop)
        strong_trend = (z_fast > z_slow * 1.004) and (trend_dir == 1)
        trigger_active = (cross_up or breakout)
        volume_momentum = (vol_ratio >= 0.95)

        if strong_trend and trigger_active and volume_momentum:
            # Dynamic Stop Loss & Take Profit
            stop_loss = price - (self.config.stop_loss_atr_mult * atr)
            take_profit = price + (self.config.take_profit_target_rr * atr)
            trailing_stop = st_line

            # Gate 4: Friction Gate (Expected Gain >= 3.0x Total Friction)
            sim_exec = self.friction_model.simulate_execution(
                side='BUY',
                requested_price=price,
                trade_usd=self.config.trade_size_usdc,
                sol_price_usd=sol_price_usd,
                pool_liquidity_usd=liquidity
            )
            round_trip_friction_usd = sim_exec.total_friction_usd * 2.0
            expected_gain_usd = (take_profit - price) / price * self.config.trade_size_usdc

            if expected_gain_usd < (round_trip_friction_usd * self.config.friction_gate_multiple):
                return TradeSignal(
                    action='HOLD',
                    price=price,
                    reason=f"FRICTION_GATE_REJECT: Expected Gain ${expected_gain_usd:.2f} < {self.config.friction_gate_multiple}x Friction ${round_trip_friction_usd:.2f}"
                )

            trigger_name = "GoldenCross" if cross_up else "DonchianBreakout"
            reason = f"BUY_ENTRY: {trigger_name} + SuperTrend UP + VolRatio {vol_ratio:.2f}"
            return TradeSignal(
                action='BUY',
                price=price,
                stop_loss=stop_loss,
                trailing_stop=trailing_stop,
                take_profit=take_profit,
                confidence=0.92,
                reason=reason
            )

        return TradeSignal(action='HOLD', price=price, reason="NO_SETUP: Waiting for strong trend breakout")
