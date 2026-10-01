"""
Solana AI Quant Agent - Quantitative Factor Engine
Implements Zero-Lag EMA (ZLEMA), SuperTrend, ATR Volatility,
and On-Chain Order Flow / DEX Liquidity Factors.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR)."""
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def calculate_zlema(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Zero-Lag Exponential Moving Average (ZLEMA).
    Removes standard EMA lag by shifting price back by half the cycle period.
    """
    lag = int((period - 1) / 2)
    shifted = series.shift(lag).fillna(series)
    zlema_input = 2 * series - shifted
    return zlema_input.ewm(span=period, adjust=False).mean()


def calculate_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 2.5
) -> Tuple[pd.Series, pd.Series]:
    """
    SuperTrend indicator.
    Returns: (supertrend_line, trend_direction) where 1 = Bullish, -1 = Bearish.
    """
    hl2 = (df['high'] + df['low']) / 2.0
    atr = calculate_atr(df, period)
    
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)
    
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    trend = pd.Series(1, index=df.index)
    supertrend = pd.Series(0.0, index=df.index)
    
    for i in range(1, len(df)):
        # Upper band adjustment
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or df['close'].iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]
            
        # Lower band adjustment
        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or df['close'].iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]
            
        # Trend flip
        if df['close'].iloc[i] > final_upper.iloc[i - 1]:
            trend.iloc[i] = 1
        elif df['close'].iloc[i] < final_lower.iloc[i - 1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]
            
        supertrend.iloc[i] = final_lower.iloc[i] if trend.iloc[i] == 1 else final_upper.iloc[i]
        
    return supertrend, trend


def compute_all_factors(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Computes all strategy indicators and factors on the dataset.
    Preserves strict causality: no lookahead bias.
    """
    res = df.copy()
    
    # 1. ZLEMA Trend Filters
    res['zlema_fast'] = calculate_zlema(res['close'], config.zlema_fast_period)
    res['zlema_slow'] = calculate_zlema(res['close'], config.zlema_slow_period)
    
    # 2. SuperTrend
    st_line, st_dir = calculate_supertrend(res, config.supertrend_period, config.supertrend_multiplier)
    res['supertrend'] = st_line
    res['trend_dir'] = st_dir
    
    # 3. Volatility
    res['atr'] = calculate_atr(res, config.atr_period)
    
    # 4. Volume Velocity
    vol_ma = res['volume_usd'].rolling(window=20, min_periods=1).mean()
    res['vol_ratio'] = res['volume_usd'] / vol_ma.replace(0, 1.0)
    
    # 5. Order flow moving average (if present)
    if 'buy_ratio' in res.columns:
        res['buy_ratio_ma'] = res['buy_ratio'].rolling(window=5, min_periods=1).mean()
    else:
        res['buy_ratio_ma'] = 0.50

    # 6. Trend Momentum & Breakout Filters (Shifted by 1 to prevent lookahead)
    res['highest_20'] = res['high'].shift(1).rolling(20, min_periods=5).max()
    res['zlema_slope'] = res['zlema_fast'] - res['zlema_fast'].shift(2)
    res['ema_macro'] = res['close'].ewm(span=80, adjust=False).mean()
    
    # State transitions: fresh Golden Cross or 20-bar breakout
    res['cross_up'] = (res['zlema_fast'] > res['zlema_slow']) & (res['zlema_fast'].shift(1) <= res['zlema_slow'].shift(1))
    res['breakout'] = (res['close'] > res['highest_20']) & (res['close'].shift(1) <= res['highest_20'])

    return res
