"""
Solana AI Quant Agent - Virtual Paper Trading Broker (虚拟盘核心账本与路由模拟器)
Provides a production-grade virtual trading environment with 100% adherence
to Solana on-chain economics: gas, priority fees, DEX LP fees, and slippage.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from config import PaperTradingConfig, SolanaFrictionConfig
from friction import SolanaFrictionModel, ExecutionResult, ExecutionRoute


@dataclass
class PaperPosition:
    token_symbol: str
    token_mint: str
    token_amount: float
    entry_price: float
    entry_time: datetime
    highest_price: float
    stop_loss: float
    take_profit: float
    trailing_stop: float
    nominal_entry_usd: float
    entry_network_fee_usd: float
    entry_dex_fee_usd: float
    entry_slippage_usd: float
    entry_friction_usd: float
    route: str = "ROUTE_B_DEX"
    latency_ms: float = 460.0
    engine_name: str = "Jupiter v6 DEX Router"


@dataclass
class PaperTradeRecord:
    id: int
    side: str
    token_symbol: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    token_amount: float
    gross_pnl_usd: float
    net_pnl_usd: float
    total_fees_usd: float
    total_slippage_usd: float
    total_friction_usd: float
    return_pct: float
    hold_bars: int
    exit_reason: str
    route: str = "ROUTE_B_DEX"
    latency_ms: float = 460.0
    engine_name: str = "Jupiter v6 DEX Router"


class PaperBroker:
    """
    Virtual Broker maintaining full portfolio state, simulating Jupiter swaps,
    and recording all causal metrics for live paper trading and backtesting.
    """
    def __init__(
        self,
        paper_config: PaperTradingConfig = None,
        friction_config: SolanaFrictionConfig = None,
        stress_mult: float = 1.0
    ):
        self.paper_cfg = paper_config or PaperTradingConfig()
        self.friction_cfg = friction_config or SolanaFrictionConfig()
        self.stress_mult = stress_mult
        self.friction_model = SolanaFrictionModel(self.friction_cfg)

        # Execution Route (ROUTE_A_CEX, ROUTE_B_DEX, ROUTE_C_HFT)
        self.active_route: str = ExecutionRoute.ROUTE_B_DEX

        # Account Balances
        self.cash_usdc = self.paper_cfg.initial_cash_usdc
        self.sol_balance = self.paper_cfg.initial_sol_balance
        self.initial_equity_usd = self.cash_usdc + (self.sol_balance * 140.0)

        # Active Position
        self.position: Optional[PaperPosition] = None
        
        # M2M Tracking
        self.peak_equity = self.initial_equity_usd
        self.max_drawdown_usd = 0.0
        self.max_drawdown_pct = 0.0
        
        # Historical Ledger
        self.trade_history: List[PaperTradeRecord] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.trade_counter = 0

    def set_execution_route(self, route: str) -> str:
        """Sets the active execution route."""
        valid_routes = [ExecutionRoute.ROUTE_A_CEX, ExecutionRoute.ROUTE_B_DEX, ExecutionRoute.ROUTE_C_HFT]
        if route in valid_routes:
            self.active_route = route
        return self.active_route

    def get_portfolio_equity(self, current_price: float, sol_price: float = 140.0) -> float:
        """Computes instantaneous Mark-to-Market (M2M) portfolio equity."""
        position_value = (self.position.token_amount * current_price) if self.position else 0.0
        gas_sol_value = self.sol_balance * sol_price
        return self.cash_usdc + position_value + gas_sol_value

    def update_m2m(self, current_price: float, sol_price: float = 140.0, timestamp: str = "") -> float:
        """Updates M2M equity, high watermark, and current drawdown."""
        equity = self.get_portfolio_equity(current_price, sol_price)
        if equity > self.peak_equity:
            self.peak_equity = equity

        drawdown_usd = self.peak_equity - equity
        drawdown_pct = (drawdown_usd / self.peak_equity) if self.peak_equity > 0 else 0.0

        if drawdown_pct > self.max_drawdown_pct:
            self.max_drawdown_pct = drawdown_pct
            self.max_drawdown_usd = drawdown_usd

        self.equity_curve.append({
            "timestamp": timestamp or datetime.utcnow().isoformat(),
            "equity": equity,
            "cash": self.cash_usdc,
            "drawdown_pct": drawdown_pct
        })
        return equity

    def execute_buy(
        self,
        token_symbol: str,
        token_mint: str,
        price: float,
        trade_usd: float,
        stop_loss: float,
        take_profit: float,
        trailing_stop: float,
        sol_price: float = 140.0,
        pool_liquidity: float = 1000000.0,
        timestamp: Optional[datetime] = None,
        route: Optional[str] = None
    ) -> Optional[ExecutionResult]:
        """
        Executes a paper BUY order across selected route (CEX, DEX, or HFT).
        Deducts nominal cash, network gas fees (from SOL), DEX fees, and slippage.
        """
        if self.position is not None:
            return None  # Single-position model

        available_cash = self.cash_usdc
        if available_cash < trade_usd:
            trade_usd = available_cash * 0.95  # Clip to available cash

        if trade_usd < 50.0:
            return None  # Insufficient funds

        target_route = route or self.active_route

        # Simulate execution with chosen route friction
        exec_res = self.friction_model.simulate_execution(
            side="BUY",
            requested_price=price,
            trade_usd=trade_usd,
            sol_price_usd=sol_price,
            pool_liquidity_usd=pool_liquidity,
            stress_mult=self.stress_mult,
            route=target_route
        )

        # Gas fee deducted from SOL balance
        if exec_res.base_network_fee_usd > 0 or exec_res.priority_fee_usd > 0 or exec_res.jito_tip_usd > 0:
            gas_sol = (exec_res.base_network_fee_usd + exec_res.priority_fee_usd + exec_res.jito_tip_usd) / sol_price
            self.sol_balance = max(self.sol_balance - gas_sol, 0.0)

        # Cash balance deducted by trade nominal
        self.cash_usdc -= trade_usd

        # Open Position
        entry_network_fee = exec_res.base_network_fee_usd + exec_res.priority_fee_usd + exec_res.jito_tip_usd
        self.position = PaperPosition(
            token_symbol=token_symbol,
            token_mint=token_mint,
            token_amount=exec_res.token_amount,
            entry_price=exec_res.execution_price,
            entry_time=timestamp or datetime.utcnow(),
            highest_price=exec_res.execution_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            nominal_entry_usd=trade_usd,
            entry_network_fee_usd=entry_network_fee,
            entry_dex_fee_usd=exec_res.dex_protocol_fee_usd,
            entry_slippage_usd=exec_res.slippage_cost_usd,
            entry_friction_usd=exec_res.total_friction_usd,
            route=exec_res.route,
            latency_ms=exec_res.latency_ms,
            engine_name=exec_res.engine_name
        )
        return exec_res

    def execute_sell(
        self,
        price: float,
        reason: str,
        sol_price: float = 140.0,
        pool_liquidity: float = 1000000.0,
        timestamp: Optional[datetime] = None,
        hold_bars: int = 1,
        route: Optional[str] = None
    ) -> Optional[PaperTradeRecord]:
        """
        Executes a paper SELL order across selected route (CEX, DEX, or HFT).
        Deducts exit friction, calculates exact Net PnL, and logs to trade ledger.
        """
        if self.position is None:
            return None

        pos = self.position
        nominal_exit_usd = pos.token_amount * price
        target_route = route or pos.route or self.active_route

        # Simulate exit friction
        exec_res = self.friction_model.simulate_execution(
            side="SELL",
            requested_price=price,
            trade_usd=nominal_exit_usd,
            sol_price_usd=sol_price,
            pool_liquidity_usd=pool_liquidity,
            stress_mult=self.stress_mult,
            route=target_route
        )

        # Deduct gas from SOL balance
        if exec_res.base_network_fee_usd > 0 or exec_res.priority_fee_usd > 0 or exec_res.jito_tip_usd > 0:
            gas_sol = (exec_res.base_network_fee_usd + exec_res.priority_fee_usd + exec_res.jito_tip_usd) / sol_price
            self.sol_balance = max(self.sol_balance - gas_sol, 0.0)

        # Net cash proceeds after DEX fee and execution slippage
        net_exit_cash = (pos.token_amount * exec_res.execution_price) - exec_res.dex_protocol_fee_usd
        self.cash_usdc += net_exit_cash

        # Total Friction across Entry + Exit
        exit_network_fee = exec_res.base_network_fee_usd + exec_res.priority_fee_usd + exec_res.jito_tip_usd
        total_fees = (pos.entry_network_fee_usd + pos.entry_dex_fee_usd) + (exit_network_fee + exec_res.dex_protocol_fee_usd)
        total_slippage = pos.entry_slippage_usd + exec_res.slippage_cost_usd
        total_friction = total_fees + total_slippage

        gross_pnl_usd = (price - pos.entry_price) * pos.token_amount
        net_pnl_usd = net_exit_cash - pos.nominal_entry_usd
        return_pct = (net_pnl_usd / pos.nominal_entry_usd) * 100.0

        self.trade_counter += 1
        record = PaperTradeRecord(
            id=self.trade_counter,
            side="LONG",
            token_symbol=pos.token_symbol,
            entry_time=str(pos.entry_time),
            exit_time=str(timestamp or datetime.utcnow()),
            entry_price=pos.entry_price,
            exit_price=exec_res.execution_price,
            token_amount=pos.token_amount,
            gross_pnl_usd=gross_pnl_usd,
            net_pnl_usd=net_pnl_usd,
            total_fees_usd=total_fees,
            total_slippage_usd=total_slippage,
            total_friction_usd=total_friction,
            return_pct=return_pct,
            hold_bars=hold_bars,
            exit_reason=reason,
            route=exec_res.route,
            latency_ms=exec_res.latency_ms,
            engine_name=exec_res.engine_name
        )

        self.trade_history.append(record)
        self.position = None
        return record
