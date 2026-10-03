"""
Solana AI Quant Agent - On-Chain Data Feed & Historical Market Engine
Fetches live Solana DEX data from DexScreener (liquidity, price, order flow)
and supplies historical / synthetic multi-regime OHLCV data for backtesting.
"""

import os
import time
import math
import random
import logging
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger("SolanaDataFeed")


class DexScreenerClient:
    """
    Direct interface to DexScreener's public API for Solana DEX telemetry.
    Fetches real-time price, liquidity depth, 24h/1h volume, and buy/sell transaction counts.
    """
    BASE_URL = "https://api.dexscreener.com/latest/dex"
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    def __init__(self, timeout: int = 8):
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)
        self.timeout = timeout

    def get_token_pairs(self, token_address: str) -> List[Dict[str, Any]]:
        """Fetch all DEX pairs for a Solana token mint address."""
        url = f"{self.BASE_URL}/tokens/{token_address}"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                pairs = data.get("pairs") or []
                # Filter only Solana chain pairs
                solana_pairs = [p for p in pairs if p.get("chainId") == "solana"]
                return solana_pairs
            else:
                logger.warning(f"DexScreener returned status {resp.status_code}")
                return []
        except Exception as e:
            logger.warning(f"Failed to query DexScreener for token {token_address}: {e}")
            return []

    def get_primary_pair_telemetry(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Extracts structured quantitative telemetry for the deepest liquidity pair.
        Returns:
            - symbol, dex_id, pair_address
            - price_usd
            - liquidity_usd
            - volume_24h, volume_1h, volume_5m
            - buys_1h, sells_1h, buy_sell_ratio_1h
            - price_change_1h, price_change_24h
        """
        pairs = self.get_token_pairs(token_address)
        if not pairs:
            return None

        # Sort by liquidity USD descending to find the primary market pool
        pairs.sort(key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0.0), reverse=True)
        primary = pairs[0]

        price_usd = float(primary.get("priceUsd") or 0.0)
        liq_usd = float((primary.get("liquidity") or {}).get("usd") or 0.0)
        volume = primary.get("volume") or {}
        txns = primary.get("txns") or {}
        price_change = primary.get("priceChange") or {}

        txns_1h = txns.get("h1") or {}
        buys_1h = int(txns_1h.get("buys") or 0)
        sells_1h = int(txns_1h.get("sells") or 0)
        total_txns_1h = buys_1h + sells_1h
        buy_sell_ratio = (buys_1h / max(sells_1h, 1)) if sells_1h > 0 else 1.0
        net_buy_imbalance = (buys_1h - sells_1h) / max(total_txns_1h, 1)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": primary.get("baseToken", {}).get("symbol", "SOL"),
            "dex_id": primary.get("dexId", "orca"),
            "pair_address": primary.get("pairAddress", ""),
            "price_usd": price_usd,
            "liquidity_usd": liq_usd,
            "volume_24h_usd": float(volume.get("h24") or 0.0),
            "volume_1h_usd": float(volume.get("h1") or 0.0),
            "volume_5m_usd": float(volume.get("m5") or 0.0),
            "buys_1h": buys_1h,
            "sells_1h": sells_1h,
            "buy_sell_ratio_1h": buy_sell_ratio,
            "net_buy_imbalance_1h": net_buy_imbalance,
            "price_change_1h": float(price_change.get("h1") or 0.0),
            "price_change_24h": float(price_change.get("h24") or 0.0)
        }


class HistoricalMarketFeed:
    """
    Generates and manages multi-regime Solana 15m/1h OHLCV data.
    Simulates authentic crypto micro-structure:
    1. Trend expansion (Hyper-Bull & Panic-Crash)
    2. Liquidity pool depth fluctuation
    3. Order flow imbalance (Net Buy/Sell volume)
    4. Volatility regime clustering (GARCH-like ATR swings)
    """

    SUPPORTED_TOKENS = {
        "SOL": {
            "name": "Solana",
            "symbol": "SOL",
            "mint": "So11111111111111111111111111111111111111112",
            "pair": "SOL/USDC",
            "category": "bluechip",
            "default_price": 135.0,
            "default_liq": 35000000.0
        },
        "BTC": {
            "name": "Bitcoin (cbBTC)",
            "symbol": "BTC",
            "mint": "cbbtcf3aa214zHAbiAZJayMMjNUutRrwBp8JTag6Fst",
            "pair": "BTC/USDC",
            "category": "bluechip",
            "default_price": 64500.0,
            "default_liq": 85000000.0
        },
        "ETH": {
            "name": "Ethereum (WETH)",
            "symbol": "ETH",
            "mint": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",
            "pair": "ETH/USDC",
            "category": "bluechip",
            "default_price": 2650.0,
            "default_liq": 55000000.0
        },
        "JUP": {
            "name": "Jupiter",
            "symbol": "JUP",
            "mint": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
            "pair": "JUP/USDC",
            "category": "defi",
            "default_price": 0.85,
            "default_liq": 12000000.0
        },
        "RAY": {
            "name": "Raydium",
            "symbol": "RAY",
            "mint": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
            "pair": "RAY/USDC",
            "category": "defi",
            "default_price": 1.95,
            "default_liq": 9500000.0
        },
        "BONK": {
            "name": "Bonk",
            "symbol": "BONK",
            "mint": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
            "pair": "BONK/USDC",
            "category": "meme",
            "default_price": 0.000022,
            "default_liq": 14000000.0
        },
        "WIF": {
            "name": "dogwifhat",
            "symbol": "WIF",
            "mint": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
            "pair": "WIF/USDC",
            "category": "meme",
            "default_price": 1.80,
            "default_liq": 22000000.0
        },
        "POPCAT": {
            "name": "Popcat",
            "symbol": "POPCAT",
            "mint": "7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr",
            "pair": "POPCAT/USDC",
            "category": "meme",
            "default_price": 1.25,
            "default_liq": 18000000.0
        },
        "PYTH": {
            "name": "Pyth Network",
            "symbol": "PYTH",
            "mint": "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3",
            "pair": "PYTH/USDC",
            "category": "infra",
            "default_price": 0.35,
            "default_liq": 8500000.0
        },
        "JTO": {
            "name": "Jito",
            "symbol": "JTO",
            "mint": "jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL",
            "pair": "JTO/USDC",
            "category": "infra",
            "default_price": 2.45,
            "default_liq": 11000000.0
        },
        "DRIFT": {
            "name": "Drift Protocol",
            "symbol": "DRIFT",
            "mint": "DriFtupJYLTosbwoN8koMbEYSx54aFAVLddWsbksjwg7",
            "pair": "DRIFT/USDC",
            "category": "defi",
            "default_price": 0.52,
            "default_liq": 7000000.0
        },
        "ORCA": {
            "name": "Orca",
            "symbol": "ORCA",
            "mint": "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE",
            "pair": "ORCA/USDC",
            "category": "defi",
            "default_price": 2.10,
            "default_liq": 6000000.0
        },
        "KMNO": {
            "name": "Kamino Finance",
            "symbol": "KMNO",
            "mint": "KMNo3nJsBXfcpJTVhZcXLW7RmTwTt4GVFE7suUBo9sS",
            "pair": "KMNO/USDC",
            "category": "defi",
            "default_price": 0.12,
            "default_liq": 5500000.0
        },
        "ME": {
            "name": "Magic Eden",
            "symbol": "ME",
            "mint": "MEFNBXixE4v3nM9Z7U4v796QyJz4aL6N8X7tP9v2YyR",
            "pair": "ME/USDC",
            "category": "infra",
            "default_price": 1.15,
            "default_liq": 8000000.0
        },
        "TNSR": {
            "name": "Tensor",
            "symbol": "TNSR",
            "mint": "TNSRxcUxoT9xBG3de7PiJyTDYu7kskLqcpddxnEJAS6",
            "pair": "TNSR/USDC",
            "category": "infra",
            "default_price": 0.65,
            "default_liq": 4500000.0
        },
        "RENDER": {
            "name": "Render Network",
            "symbol": "RENDER",
            "mint": "rndrizKT3MK1iimdxRdWabcF7Zg7AR5T4nud4EkHBof",
            "pair": "RENDER/USDC",
            "category": "infra",
            "default_price": 5.80,
            "default_liq": 16000000.0
        },
        "HNT": {
            "name": "Helium",
            "symbol": "HNT",
            "mint": "hntyVP6YFm1Hg25TN9WGLqM12b8TQmcknKrdu1oxWux",
            "pair": "HNT/USDC",
            "category": "infra",
            "default_price": 6.20,
            "default_liq": 9000000.0
        },
        "MOBILE": {
            "name": "Helium Mobile",
            "symbol": "MOBILE",
            "mint": "mb1eu7TzEc71KxDpsmsKoucSSuuoGLv1drys1oP2jh6",
            "pair": "MOBILE/USDC",
            "category": "infra",
            "default_price": 0.0011,
            "default_liq": 4000000.0
        },
        "BOME": {
            "name": "BOOK OF MEME",
            "symbol": "BOME",
            "mint": "ukHH6c7mMyiWCf1b9pnWe25TSpkDDt3H5pQZgZ74J82",
            "pair": "BOME/USDC",
            "category": "meme",
            "default_price": 0.0075,
            "default_liq": 8500000.0
        },
        "MEW": {
            "name": "cat in a dogs world",
            "symbol": "MEW",
            "mint": "MEW1gQWJ3nEXg2qgERiKu7FAFj79PHvQVREQUzScPP5",
            "pair": "MEW/USDC",
            "category": "meme",
            "default_price": 0.0065,
            "default_liq": 10500000.0
        },
        "SLERF": {
            "name": "Slerf",
            "symbol": "SLERF",
            "mint": "713rhdnbK2A4qS4G8fG47p5qfFfG2y8hY9R7Qy8eJ1wE",
            "pair": "SLERF/USDC",
            "category": "meme",
            "default_price": 0.18,
            "default_liq": 4200000.0
        },
        "MYRO": {
            "name": "Myro",
            "symbol": "MYRO",
            "mint": "HhJpBhRRn4g56VsyAbT8DLmDAbWP3FsMm9zFZBQXoDAJ",
            "pair": "MYRO/USDC",
            "category": "meme",
            "default_price": 0.095,
            "default_liq": 3800000.0
        },
        "WEN": {
            "name": "Wen",
            "symbol": "WEN",
            "mint": "WENWENvqqNya429ubCdXr7oHootJbJxWCcjiHrSFnTn",
            "pair": "WEN/USDC",
            "category": "meme",
            "default_price": 0.000095,
            "default_liq": 5200000.0
        },
        "SAMO": {
            "name": "Samoyedcoin",
            "symbol": "SAMO",
            "mint": "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",
            "pair": "SAMO/USDC",
            "category": "meme",
            "default_price": 0.0092,
            "default_liq": 3500000.0
        },
        "MOTHER": {
            "name": "Mother Iggy",
            "symbol": "MOTHER",
            "mint": "3S8qX1MsMqRbiwKg2cQnv7noGoMCWPSCUEquwuAApump",
            "pair": "MOTHER/USDC",
            "category": "meme",
            "default_price": 0.065,
            "default_liq": 4800000.0
        },
        "FIDA": {
            "name": "Bonfida",
            "symbol": "FIDA",
            "mint": "EchesyfXePKdLtoiZSL8pBe8Myagyy8ZRqsACNCFGnvp",
            "pair": "FIDA/USDC",
            "category": "infra",
            "default_price": 0.22,
            "default_liq": 3600000.0
        },
        "GOAT": {
            "name": "Goatseus Maximus",
            "symbol": "GOAT",
            "mint": "CzLSujWBLFsSjncfkh59rUFqvafWcY5tzedWJSuBg9R3",
            "pair": "GOAT/USDC",
            "category": "meme",
            "default_price": 0.65,
            "default_liq": 12500000.0
        },
        "ACT": {
            "name": "Act I : Prophecy",
            "symbol": "ACT",
            "mint": "GJAFwWjJ3vnTsrQVabjBVK2TYB1YtRCQXRDfDgqSpump",
            "pair": "ACT/USDC",
            "category": "meme",
            "default_price": 0.42,
            "default_liq": 9500000.0
        },
        "PNUT": {
            "name": "Peanut the Squirrel",
            "symbol": "PNUT",
            "mint": "2qEHjDLDLbuBgRYvsxhc5RefwhHyJCPGtU9w2KUSpump",
            "pair": "PNUT/USDC",
            "category": "meme",
            "default_price": 0.95,
            "default_liq": 15000000.0
        },
        "MOODENG": {
            "name": "Moo Deng",
            "symbol": "MOODENG",
            "mint": "ED5nyyWEzpPPiWimP8vYm7sD7TD3LAt3Q3gRTWHzPJBY",
            "pair": "MOODENG/USDC",
            "category": "meme",
            "default_price": 0.28,
            "default_liq": 7500000.0
        }
    }

    @staticmethod
    def generate_solana_market_data(
        bars_count: int = 1500,
        start_price: float = None,
        base_liquidity: float = None,
        timeframe_minutes: int = 15,
        seed: int = 42,
        symbol: str = "SOL"
    ) -> pd.DataFrame:
        """
        Creates a realistic, high-frequency Solana historical dataset.
        Includes bull trends, bear selloffs, whipsaw chop, and volume surges.
        """
        np.random.seed(seed)
        random.seed(seed)

        # Lookup token defaults if not specified
        token_info = HistoricalMarketFeed.SUPPORTED_TOKENS.get(symbol.upper(), {})
        if start_price is None:
            start_price = token_info.get("default_price", 135.0)
        if base_liquidity is None:
            base_liquidity = token_info.get("default_liq", 25000000.0)

        start_time = datetime(2026, 1, 1, 0, 0, 0)
        timestamps = [start_time + timedelta(minutes=i * timeframe_minutes) for i in range(bars_count)]

        prices = [start_price]
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        liquidities = []
        net_buy_ratios = []

        current_price = start_price
        regime = "TREND_UP"
        regime_duration = 0

        for i in range(bars_count):
            regime_duration += 1
            if regime_duration > np.random.randint(60, 180):
                # Switch market regimes
                regime = np.random.choice(["TREND_UP", "TREND_DOWN", "CHOP_WHIPSAW", "VOLATILITY_EXPANSION"],
                                          p=[0.35, 0.25, 0.25, 0.15])
                regime_duration = 0

            # Regime-dependent drift and volatility
            if regime == "TREND_UP":
                drift = 0.0006
                vol = 0.0055
            elif regime == "TREND_DOWN":
                drift = -0.0008
                vol = 0.0075
            elif regime == "CHOP_WHIPSAW":
                drift = 0.0000
                vol = 0.0040
            else:  # VOLATILITY_EXPANSION
                drift = 0.0002
                vol = 0.0120

            # Price simulation (GBM with jump process)
            ret = drift + vol * np.random.normal()
            # 2% probability of flash dump or pump
            if np.random.rand() < 0.02:
                ret += np.random.choice([-0.025, 0.025])

            open_p = current_price
            min_floor = max(start_price * 0.05, 1e-8)
            close_p = max(open_p * (1.0 + ret), min_floor)
            intra_vol = abs(ret) + vol * np.random.uniform(0.5, 1.8)
            high_p = max(open_p, close_p) * (1.0 + intra_vol * np.random.uniform(0.3, 0.9))
            low_p = max(min(open_p, close_p) * (1.0 - intra_vol * np.random.uniform(0.3, 0.9)), min_floor)

            # Volume & liquidity dynamics
            base_vol = 80000.0 * (1.0 + abs(ret) * 15.0)
            vol_usd = base_vol * np.random.uniform(0.7, 1.6)
            pool_liq = base_liquidity * (0.85 + 0.3 * (close_p / max(start_price, 1e-8)))

            # Order flow: in uptrends, buy volume dominates; in downtrends, sell volume dominates
            buy_bias = 0.5 + (0.35 if ret > 0 else -0.35)
            buy_bias = min(max(buy_bias + np.random.normal(0, 0.08), 0.15), 0.85)

            decimals = 8 if start_price < 0.001 else (5 if start_price < 2.0 else 4)
            opens.append(round(open_p, decimals))
            highs.append(round(high_p, decimals))
            lows.append(round(low_p, decimals))
            closes.append(round(close_p, decimals))
            volumes.append(round(vol_usd, 2))
            liquidities.append(round(pool_liq, 2))
            net_buy_ratios.append(round(buy_bias, 4))

            current_price = close_p

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume_usd": volumes,
            "liquidity_usd": liquidities,
            "buy_ratio": net_buy_ratios
        })
        return df

    @classmethod
    def get_market_data(
        cls,
        symbol: str = "SOL",
        timeframe_minutes: int = 15,
        bars_count: int = 35000,
        start_price: float = None,
        base_liquidity: float = None,
        data_dir: str = "data",
        seed: int = 99
    ) -> pd.DataFrame:
        """
        Hybrid market data loader:
        1. Priority 1: High-Performance SQLite WAL Time-Series Database (data/market.db)
        2. Priority 2: Local CSV data file (data/{SYMBOL}_{TIMEFRAME}.csv)
        3. Priority 3: Calibrated multi-regime Solana stochastic generator.
        """
        tf_str = "15m"
        if timeframe_minutes == 5:
            tf_str = "5m"
        elif timeframe_minutes == 60:
            tf_str = "1h"
        elif timeframe_minutes == 240:
            tf_str = "4h"

        sym = symbol.upper()

        # -------------------------------------------------------------
        # 1. Query SQLite WAL Database
        # -------------------------------------------------------------
        try:
            from market_db import market_db
            df_db = market_db.get_klines(symbol=sym, timeframe=tf_str, limit=bars_count)
            if not df_db.empty and len(df_db) >= bars_count:
                logger.info(f"Loaded {len(df_db)} real market bars from SQLite DB for {sym} ({tf_str})")
                return df_db
        except Exception as e:
            logger.warning(f"MarketDB query failed for {sym}: {e}")

        # -------------------------------------------------------------
        # 2. Check Local CSV File
        # -------------------------------------------------------------
        file_path = os.path.join(data_dir, f"{sym}_{tf_str}.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                required_cols = {"timestamp", "open", "high", "low", "close"}
                if required_cols.issubset(set(df.columns)):
                    if len(df) >= bars_count:
                        df = df.iloc[-bars_count:].reset_index(drop=True)
                    if "volume_usd" not in df.columns:
                        df["volume_usd"] = df.get("volume", 100000.0)
                    if "liquidity_usd" not in df.columns:
                        token_info = cls.SUPPORTED_TOKENS.get(sym, {})
                        df["liquidity_usd"] = token_info.get("default_liq", 25000000.0)
                    if "buy_ratio" not in df.columns:
                        df["buy_ratio"] = 0.50
                    return df
            except Exception as e:
                logger.warning(f"Failed to read local dataset {file_path}: {e}")

        # -------------------------------------------------------------
        # 3. Fallback to Calibrated Multi-Regime Generator
        # -------------------------------------------------------------
        return cls.generate_solana_market_data(
            bars_count=bars_count,
            start_price=start_price,
            base_liquidity=base_liquidity,
            timeframe_minutes=timeframe_minutes,
            seed=seed,
            symbol=symbol
        )

