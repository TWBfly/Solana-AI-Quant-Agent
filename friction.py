"""
Solana AI Quant Agent - Realistic Friction & Execution Cost Model
Implements Solana on-chain gas fees, priority fees, Jito MEV tips,
DEX pool fees (Raydium/Orca), and dynamic AMM price-impact slippage.
"""

from dataclasses import dataclass
from typing import Tuple
from config import SolanaFrictionConfig


class ExecutionRoute:
    ROUTE_A_CEX = "ROUTE_A_CEX"      # 路径 A: 中心化交易所 (Binance / OKX CLOB)
    ROUTE_B_DEX = "ROUTE_B_DEX"      # 路径 B: 链上聚合器 / DEX API (Jupiter v6 / Raydium)
    ROUTE_C_HFT = "ROUTE_C_HFT"      # 路径 C: 链上极速直连 (Jito MEV Bundles / Yellowstone gRPC)


@dataclass
class ExecutionResult:
    """Detailed breakdown of a simulated trade execution across CEX, DEX, or HFT routes."""
    side: str                          # 'BUY' or 'SELL'
    requested_price: float             # Reference price (e.g., Open or Mid)
    execution_price: float             # Final executed price including slippage
    nominal_usd: float                 # Nominal trade value in USD
    token_amount: float                # Amount of token traded
    base_network_fee_usd: float        # Base tx signature fee (SOL -> USD)
    priority_fee_usd: float            # Compute budget priority fee (SOL -> USD)
    jito_tip_usd: float                # Optional Jito MEV tip
    dex_protocol_fee_usd: float        # DEX AMM fee or CEX maker/taker fee
    slippage_cost_usd: float           # Dollar cost incurred via slippage
    slippage_rate: float               # Effective slippage percentage (e.g., 0.0015 = 15 bps)
    total_friction_usd: float          # All-in total friction cost (Fees + Slippage)
    stress_multiplier: float = 1.0     # 1.0 = baseline, 3.0 = stress test
    route: str = "ROUTE_B_DEX"         # Execution route identifier
    latency_ms: float = 460.0          # Estimated execution latency
    engine_name: str = "Jupiter v6 DEX Router"
    atomic_protection: bool = False    # True if Jito MEV atomic bundle guarantees zero loss on failure
    execution_status: str = "CONFIRMED"


class SolanaFrictionModel:
    """
    Computes exact, realistic execution costs and slippage for Solana DEX trades.
    Guarantees no zero-friction illusions in backtesting or paper trading.
    """
    def __init__(self, config: SolanaFrictionConfig = None):
        self.config = config or SolanaFrictionConfig()

    def calculate_network_fee_usd(self, sol_price_usd: float, stress_mult: float = 1.0) -> Tuple[float, float, float]:
        """
        Computes on-chain Solana fees in USD.
        Returns: (base_fee_usd, priority_fee_usd, jito_tip_usd)
        """
        safe_sol_price = max(sol_price_usd, 1.0)
        base_fee = self.config.base_tx_fee_sol * safe_sol_price * stress_mult
        priority_fee = self.config.priority_fee_sol * safe_sol_price * stress_mult
        jito_tip = self.config.jito_tip_sol * safe_sol_price * stress_mult
        return base_fee, priority_fee, jito_tip

    def calculate_slippage_rate(
        self,
        trade_usd: float,
        pool_liquidity_usd: float,
        volatility_bps: float = 5.0,
        stress_mult: float = 1.0
    ) -> float:
        """
        Calculates dynamic slippage based on AMM Constant-Product formula + latency drift.
        Impact = (Trade_Size / Available_Side_Liquidity)
        Total Slippage = Base Slippage + AMM Impact + Volatility Jitter
        """
        # Effective single-side liquidity is roughly pool_liquidity / 2
        effective_liquidity = max(pool_liquidity_usd * 0.5, 10000.0)
        amm_price_impact = (trade_usd / effective_liquidity) * self.config.price_impact_scale
        
        # Volatility latency drift (e.g., block confirmation delay of 400ms-1200ms)
        latency_drift = (volatility_bps / 10000.0)
        
        base_rate = self.config.base_slippage_rate
        total_slippage = (base_rate + amm_price_impact + latency_drift) * stress_mult
        
        # Clip to max slippage tolerance
        max_allowed = self.config.max_slippage_tolerance * (1.5 if stress_mult > 1.0 else 1.0)
        return min(total_slippage, max_allowed)

    def simulate_execution(
        self,
        side: str,
        requested_price: float,
        trade_usd: float,
        sol_price_usd: float,
        pool_liquidity_usd: float = 1000000.0,
        volatility_bps: float = 5.0,
        stress_mult: float = 1.0,
        route: str = ExecutionRoute.ROUTE_B_DEX
    ) -> ExecutionResult:
        """
        Simulates execution of a swap/trade across Three Pathways:
        - ROUTE_A_CEX: Binance/OKX CLOB (Ultra-low 18ms latency, 0.04% fee, 1.2 bps spread, 0 on-chain Gas)
        - ROUTE_B_DEX: Jupiter v6 / Raydium CLMM (460ms latency, 0.25% LP fee, dynamic AMM impact, Solana Gas)
        - ROUTE_C_HFT: Yellowstone gRPC + Jito MEV Bundles (165ms latency, 0.25% fee + Jito Tip, 0.8 bps slippage, ATOMIC REVERT SAFETY)
        """
        side = side.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side: {side}. Must be 'BUY' or 'SELL'.")

        if route == ExecutionRoute.ROUTE_A_CEX:
            # 路径 A: CEX 内存撮合
            latency_ms = 18.5
            engine_name = "Binance / OKX CLOB (Memory Matching Engine)"
            base_fee_usd = 0.0
            priority_fee_usd = 0.0
            jito_tip_usd = 0.0
            total_network_fee_usd = 0.0
            dex_protocol_fee_usd = trade_usd * 0.0004 * stress_mult  # 0.04% Taker fee
            slippage_rate = 0.00012 * stress_mult                    # 1.2 bps spread
            atomic_protection = False
            execution_status = "FILLED"
        elif route == ExecutionRoute.ROUTE_C_HFT:
            # 路径 C: Jito MEV Bundles 直达 Leader
            latency_ms = 165.0
            engine_name = "Yellowstone gRPC + Jito MEV Bundles (Anti-Sandwich)"
            safe_sol = max(sol_price_usd, 1.0)
            base_fee_usd = self.config.base_tx_fee_sol * safe_sol * stress_mult
            priority_fee_usd = self.config.priority_fee_sol * safe_sol * stress_mult
            jito_tip_usd = 0.0001 * safe_sol * stress_mult           # 0.0001 SOL Jito MEV Tip
            total_network_fee_usd = base_fee_usd + priority_fee_usd + jito_tip_usd
            dex_protocol_fee_usd = trade_usd * 0.0025 * stress_mult  # 0.25% LP fee
            slippage_rate = 0.00008 * stress_mult                    # 0.8 bps (Top of Slot)
            atomic_protection = True                                 # All-or-none bundle protects against loss
            execution_status = "BUNDLE_LANDED"
        else:
            # 路径 B: 默认链上 DEX 聚合
            route = ExecutionRoute.ROUTE_B_DEX
            latency_ms = 460.0
            engine_name = "Jupiter v6 DEX Router / Raydium CLMM"
            base_fee_usd, priority_fee_usd, jito_tip_usd = self.calculate_network_fee_usd(
                sol_price_usd=sol_price_usd,
                stress_mult=stress_mult
            )
            total_network_fee_usd = base_fee_usd + priority_fee_usd + jito_tip_usd
            effective_dex_fee_rate = self.config.dex_pool_fee_rate * stress_mult
            dex_protocol_fee_usd = trade_usd * effective_dex_fee_rate
            slippage_rate = self.calculate_slippage_rate(
                trade_usd=trade_usd,
                pool_liquidity_usd=pool_liquidity_usd,
                volatility_bps=volatility_bps,
                stress_mult=stress_mult
            )
            atomic_protection = False
            execution_status = "CONFIRMED"

        # Fill Price
        if side == "BUY":
            execution_price = requested_price * (1.0 + slippage_rate)
            token_amount = (trade_usd - dex_protocol_fee_usd) / execution_price
        else:
            execution_price = requested_price * (1.0 - slippage_rate)
            token_amount = trade_usd / requested_price

        # Dollar value lost to slippage compared to theoretical mid-price fill
        theoretical_value = trade_usd
        actual_fill_value = token_amount * execution_price if side == "SELL" else trade_usd / (1.0 + slippage_rate)
        slippage_cost_usd = abs(theoretical_value - actual_fill_value)

        # Total all-in friction
        total_friction_usd = total_network_fee_usd + dex_protocol_fee_usd + slippage_cost_usd

        return ExecutionResult(
            side=side,
            requested_price=requested_price,
            execution_price=execution_price,
            nominal_usd=trade_usd,
            token_amount=token_amount,
            base_network_fee_usd=base_fee_usd,
            priority_fee_usd=priority_fee_usd,
            jito_tip_usd=jito_tip_usd,
            dex_protocol_fee_usd=dex_protocol_fee_usd,
            slippage_cost_usd=slippage_cost_usd,
            slippage_rate=slippage_rate,
            total_friction_usd=total_friction_usd,
            stress_multiplier=stress_mult,
            route=route,
            latency_ms=latency_ms,
            engine_name=engine_name,
            atomic_protection=atomic_protection,
            execution_status=execution_status
        )
