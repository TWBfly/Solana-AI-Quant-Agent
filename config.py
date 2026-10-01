"""
Solana AI Quant Agent - System Configuration
Centralized settings for Solana DEX trading, friction parameters, risk limits, and paper broker.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class SolanaFrictionConfig:
    """
    Realistic Solana DEX friction parameters.
    Consistent with live Mainnet-Beta execution via Jupiter / Raydium / Orca.
    """
    # Solana Base Transaction Fee (5000 lamports = 0.000005 SOL per signature)
    base_tx_fee_sol: float = 0.000010  # ~2 signatures (compute budget + swap)
    
    # Priority Fee / Compute Budget (Micro-lamports per Compute Unit)
    # Typical dynamic priority fee during normal Solana network conditions: ~50,000 - 150,000 micro-lamports
    # Equivalent to ~0.0005 - 0.0015 SOL per swap transaction
    priority_fee_sol: float = 0.000800
    
    # Optional Jito MEV Tip (solves landing rate under congestion)
    jito_tip_sol: float = 0.000500
    
    # DEX Liquidity Pool Fee:
    # Raydium Standard AMM: 0.25% (0.0025)
    # Orca Whirlpool / Raydium CLMM: typically 0.05% - 0.30%
    dex_pool_fee_rate: float = 0.0025
    
    # Jupiter Aggregator Platform Fee (Default 0% on standard swaps)
    jupiter_platform_fee_rate: float = 0.0000
    
    # Dynamic Slippage Base (Execution Latency & Block Drift)
    base_slippage_rate: float = 0.0010  # 10 bps minimum execution delay slippage
    
    # AMM Price Impact Multiplier (Impact = Trade_Size / Pool_Liquidity * impact_scale)
    # In a constant product x*y=k pool, impact for small-medium trades scales linearly
    price_impact_scale: float = 1.0
    
    # Max Slippage Tolerance (Jupiter max slippage parameter)
    max_slippage_tolerance: float = 0.025  # 2.5% max tolerated before tx reverts


@dataclass
class StrategyConfig:
    """Quantitative Strategy & Risk Management Parameters"""
    # Trend Strategy Factors
    zlema_fast_period: int = 9
    zlema_slow_period: int = 21
    supertrend_period: int = 10
    supertrend_multiplier: float = 2.5
    atr_period: int = 14
    
    # Solana DEX Filters
    min_pool_liquidity_usd: float = 50000.0   # Reject illiquid pools with < $50k liquidity
    min_volume_24h_usd: float = 100000.0      # Minimum 24h trading volume
    min_buy_sell_ratio: float = 0.90          # Order flow filter: minimum buy/sell ratio
    
    # Friction Gate
    # Expected profit space must be >= friction_multiple * total_friction_cost
    friction_gate_multiple: float = 3.0
    
    # Risk Control
    stop_loss_atr_mult: float = 2.0           # Hard stop-loss = Entry - 2.0 * ATR
    trailing_stop_atr_mult: float = 1.8       # Chandelier trailing stop = Highest - 1.8 * ATR
    take_profit_target_rr: float = 3.2        # Take-profit target risk-reward ratio
    max_drawdown_limit: float = 0.15          # Max portfolio drawdown circuit breaker (15%)
    max_position_pct: float = 0.30            # Max 30% capital per trade
    trade_size_usdc: float = 1000.0           # Default nominal trade size in USDC


@dataclass
class PaperTradingConfig:
    """Virtual Paper Account Configuration"""
    initial_cash_usdc: float = 10000.0        # Initial USDC paper capital
    initial_sol_balance: float = 5.0          # Initial SOL balance for gas fees
    target_pair: str = "SOL/USDC"
    target_token_mint: str = "So11111111111111111111111111111111111111112"
    usdc_token_mint: str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    poll_interval_sec: float = 5.0            # Real-time polling frequency in seconds
