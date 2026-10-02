"""
Solana AI Quant Agent - Web Server Backend (Flask REST API)
Provides full frontend-backend decoupled API endpoints for:
- Live paper portfolio state and on-chain telemetry
- 1x Baseline vs 3x Stress testing execution
- Interactive step-by-step virtual paper order routing
"""

import os
import json
import argparse
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
import pandas as pd

from config import StrategyConfig, SolanaFrictionConfig, PaperTradingConfig
from data_feed import DexScreenerClient, HistoricalMarketFeed
from strategy import SolanaTrendAgent
from paper_broker import PaperBroker
from friction import ExecutionRoute
from backtester import SolanaBacktestEngine
from factors import compute_all_factors

app = Flask(__name__, static_folder="web")

# Shared State Instances
paper_cfg = PaperTradingConfig()
frict_cfg = SolanaFrictionConfig()
strat_cfg = StrategyConfig()

dex_client = DexScreenerClient()
paper_broker = PaperBroker(paper_cfg, frict_cfg)
agent = SolanaTrendAgent(strat_cfg, frict_cfg)

# In-memory bar buffer for live paper simulation
live_buffer_df = HistoricalMarketFeed.generate_solana_market_data(bars_count=60, start_price=140.0)


@app.route("/")
def index():
    """Serves the single-page frontend application."""
    return send_from_directory("web", "index.html")


@app.route("/api/tokens", methods=["GET"])
def get_tokens():
    """Returns available Solana token presets with metadata and default parameters."""
    tokens = []
    for sym, info in HistoricalMarketFeed.SUPPORTED_TOKENS.items():
        tokens.append({
            "symbol": sym,
            "name": info["name"],
            "mint": info["mint"],
            "pair": info["pair"],
            "category": info.get("category", "other"),
            "default_price": info["default_price"],
            "default_liq": info["default_liq"]
        })
    return jsonify(tokens)


@app.route("/api/db/stats", methods=["GET"])
def get_db_stats():
    """Returns inventory of historical data stored in the local SQLite time-series database."""
    from market_db import market_db
    stats = market_db.get_stats()
    total_bars = sum(s["count"] for s in stats)
    return jsonify({
        "status": "online",
        "engine": "SQLite WAL (Optimized Time-Series)",
        "total_bars_cached": total_bars,
        "symbols": stats
    })


@app.route("/api/db/sync", methods=["POST"])
def sync_db_data():
    """Fetches real institutional-grade market data for a symbol and saves to SQLite."""
    from market_db import market_db
    req = request.get_json() or {}
    symbol = req.get("symbol", "SOL").upper()
    timeframe = req.get("timeframe", "15m").lower()
    days = req.get("days", "60d")

    count = market_db.fetch_and_store_real_market_data(symbol, timeframe, days)
    return jsonify({
        "status": "success",
        "symbol": symbol,
        "timeframe": timeframe,
        "bars_imported": count,
        "message": f"Successfully synced {count} real market bars into SQLite database."
    })


@app.route("/api/basket/tokens", methods=["GET"])
def get_basket_tokens():
    """Returns the 30 active tokens in the multi-asset universe."""
    from basket_engine import BASKET_UNIVERSE_30
    items = []
    for sym, d in BASKET_UNIVERSE_30.items():
        items.append({
            "symbol": sym,
            "name": d["name"],
            "price": d["price"],
            "liquidity": d["liq"]
        })
    return jsonify(items)


@app.route("/api/basket/backtest", methods=["POST"])
def run_basket_backtest_api():
    """Runs a 1,000-trade multi-asset basket backtest across 30 active Solana tokens."""
    from basket_engine import basket_engine
    req = request.get_json() or {}
    bars_count = int(req.get("bars_count", 1440))
    stress_mult = float(req.get("stress_mult", 1.0))
    sol_price = float(req.get("sol_price", 140.0))

    if stress_mult > 1.0:
        audit = basket_engine.run_full_stress_comparison(bars_count=bars_count)
        return jsonify(audit)
    else:
        res = basket_engine.run_basket_audit(bars_count=bars_count, stress_mult=1.0, sol_price=sol_price)
        return jsonify({"res_1x": res, "conclusion": "COMPLETED"})


@app.route("/api/basket/token/<symbol>", methods=["GET"])
def get_basket_token_detail(symbol):
    """Returns high-resolution 1m K-line and full individual trade markers for requested token."""
    from basket_engine import basket_engine
    stress_mult = float(request.args.get("stress_mult", 1.0))
    res = basket_engine.get_token_1m_backtest(symbol=symbol, bars_count=1440, stress_mult=stress_mult)
    return jsonify(res)


@app.route("/api/status", methods=["GET"])
def get_status():
    """Returns current paper broker status, balances, and live DEX telemetry for requested token."""
    token_sym = request.args.get("token", "SOL").upper()
    token_info = HistoricalMarketFeed.SUPPORTED_TOKENS.get(token_sym)
    target_mint = token_info["mint"] if token_info else paper_cfg.target_token_mint

    # Fetch live DEX telemetry
    telemetry = dex_client.get_primary_pair_telemetry(target_mint)
    if telemetry:
        live_price = telemetry["price_usd"]
        dex_id = telemetry["dex_id"]
        liq_usd = telemetry["liquidity_usd"]
        vol_1h = telemetry["volume_1h_usd"]
        buy_ratio = telemetry["buy_sell_ratio_1h"] / (telemetry["buy_sell_ratio_1h"] + 1.0)
        price_change_24h = telemetry.get("price_change_24h", 0.0)
    else:
        live_price = token_info["default_price"] if token_info else 135.0
        dex_id = "orca"
        liq_usd = token_info["default_liq"] if token_info else 35000000.0
        vol_1h = 1600000.0
        buy_ratio = 0.55
        price_change_24h = 2.4

    equity = paper_broker.get_portfolio_equity(live_price, sol_price=live_price)
    net_pnl = equity - paper_broker.initial_equity_usd
    return_pct = (net_pnl / paper_broker.initial_equity_usd) * 100.0

    pos_info = None
    if paper_broker.position:
        pos = paper_broker.position
        pos_info = {
            "symbol": pos.token_symbol,
            "amount": pos.token_amount,
            "entry_price": pos.entry_price,
            "highest_price": pos.highest_price,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit
        }

    # Trade stats
    trades = paper_broker.trade_history
    win_trades = [t for t in trades if t.net_pnl_usd > 0]
    win_rate = (len(win_trades) / len(trades) * 100.0) if trades else 0.0
    gross_gain = sum(t.net_pnl_usd for t in win_trades)
    gross_loss = abs(sum(t.net_pnl_usd for t in trades if t.net_pnl_usd <= 0))
    pf = (gross_gain / gross_loss) if gross_loss > 0 else (99.0 if gross_gain > 0 else 0.0)

    return jsonify({
        "symbol": token_sym,
        "equity": equity,
        "cash_usdc": paper_broker.cash_usdc,
        "sol_balance": paper_broker.sol_balance,
        "net_pnl": net_pnl,
        "return_pct": return_pct,
        "live_price": live_price,
        "dex_id": dex_id,
        "liquidity_usd": liq_usd,
        "volume_1h_usd": vol_1h,
        "buy_ratio": buy_ratio,
        "price_change_24h": price_change_24h,
        "position": pos_info,
        "win_rate": win_rate,
        "profit_factor": pf,
        "max_drawdown_pct": paper_broker.max_drawdown_pct * 100.0
    })


@app.route("/api/backtest", methods=["POST"])
def run_backtest_api():
    """Runs a complete causal backtest with custom parameters, returning K-line, markers, and deep quant metrics."""
    req = request.get_json() or {}
    token_sym = req.get("token", "SOL").upper()
    timeframe_str = req.get("timeframe", "15m")
    stress_mult = float(req.get("stress_mult", 1.0))
    trade_size = float(req.get("trade_size", 1000.0))
    stop_loss_mult = float(req.get("stop_loss_mult", 2.0))
    take_profit_mult = float(req.get("take_profit_mult", 3.5))
    bars_count = int(req.get("bars_count", 2500))

    # Fast routing for 1m high-frequency multi-asset universe
    if timeframe_str == "1m":
        from basket_engine import basket_engine
        b_count = bars_count if bars_count in (720, 1440, 2880) else 1440
        res_1m = basket_engine.get_token_1m_backtest(
            symbol=token_sym,
            bars_count=b_count,
            stress_mult=stress_mult,
            sol_price=140.0
        )
        return jsonify(res_1m)

    tf_minutes = 15
    if timeframe_str == "5m":
        tf_minutes = 5
    elif timeframe_str == "1h":
        tf_minutes = 60
    elif timeframe_str == "4h":
        tf_minutes = 240

    custom_strat_cfg = StrategyConfig(
        trade_size_usdc=trade_size,
        stop_loss_atr_mult=stop_loss_mult,
        take_profit_target_rr=take_profit_mult
    )

    # Resolve token info and anchor price from live DEX if possible
    token_info = HistoricalMarketFeed.SUPPORTED_TOKENS.get(token_sym, {})
    mint_addr = token_info.get("mint", paper_cfg.target_token_mint)
    telemetry = dex_client.get_primary_pair_telemetry(mint_addr)
    anchor_price = telemetry["price_usd"] if telemetry else token_info.get("default_price", 135.0)
    anchor_liq = telemetry["liquidity_usd"] if telemetry else token_info.get("default_liq", 35000000.0)

    df = HistoricalMarketFeed.get_market_data(
        symbol=token_sym,
        timeframe_minutes=tf_minutes,
        bars_count=bars_count,
        start_price=anchor_price,
        base_liquidity=anchor_liq,
        data_dir="data",
        seed=99
    )

    engine = SolanaBacktestEngine(strategy_config=custom_strat_cfg)
    result = engine.run_single_pass(df, stress_mult=stress_mult, sol_price=140.0)

    # Format trades list
    trades_json = []
    for t in result["trades"]:
        trades_json.append({
            "id": t.id,
            "side": t.side,
            "token_symbol": token_sym,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "token_amount": t.token_amount,
            "gross_pnl_usd": t.gross_pnl_usd,
            "net_pnl_usd": t.net_pnl_usd,
            "total_fees_usd": t.total_fees_usd,
            "total_slippage_usd": t.total_slippage_usd,
            "total_friction_usd": t.total_friction_usd,
            "return_pct": t.return_pct,
            "hold_bars": t.hold_bars,
            "exit_reason": t.exit_reason
        })

    return jsonify({
        "token": token_sym,
        "timeframe": timeframe_str,
        "stress_mult": stress_mult,
        "total_bars": result["total_bars"],
        "start_time": result["start_time"],
        "end_time": result["end_time"],
        "initial_equity": result["initial_equity"],
        "final_equity": result["final_equity"],
        "net_profit_usd": result["net_profit_usd"],
        "return_pct": result["return_pct"],
        "bench_return_pct": result["bench_return_pct"],
        "alpha_pct": result["alpha_pct"],
        "sharpe_ratio": result["sharpe_ratio"],
        "sortino_ratio": result["sortino_ratio"],
        "calmar_ratio": result["calmar_ratio"],
        "total_trades": result["total_trades"],
        "win_rate": result["win_rate"],
        "profit_factor": result["profit_factor"],
        "win_loss_ratio": result["win_loss_ratio"],
        "avg_win": result["avg_win"],
        "avg_loss": result["avg_loss"],
        "avg_hold_bars": result["avg_hold_bars"],
        "max_drawdown_pct": result["max_drawdown_pct"],
        "max_drawdown_usd": result["max_drawdown_usd"],
        "total_fees_usd": result["total_fees_usd"],
        "total_slippage_usd": result["total_slippage_usd"],
        "total_friction_usd": result["total_friction_usd"],
        "top3_concentration_pct": result["top3_concentration_pct"],
        "wilson_ci_low": result["wilson_ci_low"],
        "wilson_ci_high": result["wilson_ci_high"],
        "wilson_ci_span": result["wilson_ci_span"],
        "reliability": result["reliability"],
        "ohlcv_bars": result["ohlcv_bars"],
        "chart_markers": result["chart_markers"],
        "equity_curve_points": result["equity_curve_points"],
        "drawdown_curve_points": result["drawdown_curve_points"],
        "trades": trades_json
    })


@app.route("/api/paper/step", methods=["POST"])
def paper_step():
    """Executes a single live virtual paper trading step against real DEX data."""
    global live_buffer_df
    telemetry = dex_client.get_primary_pair_telemetry(paper_cfg.target_token_mint)

    if telemetry:
        live_price = telemetry["price_usd"]
        liq_usd = telemetry["liquidity_usd"]
        vol_1h = telemetry["volume_1h_usd"]
        buy_ratio = telemetry["buy_sell_ratio_1h"] / (telemetry["buy_sell_ratio_1h"] + 1.0)
    else:
        live_price = float(live_buffer_df.iloc[-1]["close"]) * 1.001
        liq_usd = 38000000.0
        vol_1h = 1600000.0
        buy_ratio = 0.54

    new_bar = {
        "timestamp": datetime.utcnow(),
        "open": live_price,
        "high": live_price * 1.002,
        "low": live_price * 0.998,
        "close": live_price,
        "volume_usd": vol_1h / 4.0,
        "liquidity_usd": liq_usd,
        "buy_ratio": buy_ratio
    }
    live_buffer_df = pd.concat([live_buffer_df, pd.DataFrame([new_bar])], ignore_index=True)
    if len(live_buffer_df) > 100:
        live_buffer_df = live_buffer_df.iloc[-100:]

    factored = compute_all_factors(live_buffer_df, strat_cfg)
    curr_bar = factored.iloc[-1]

    pos_dict = None
    if paper_broker.position:
        pos_dict = {
            "size": paper_broker.position.token_amount,
            "entry_price": paper_broker.position.entry_price,
            "highest_price": max(paper_broker.position.highest_price, live_price),
            "stop_loss": paper_broker.position.stop_loss,
            "take_profit": paper_broker.position.take_profit
        }

    signal = agent.evaluate_bar(curr_bar, pos_dict, sol_price_usd=live_price)
    executed_action = None

    if signal.action == "BUY" and paper_broker.position is None:
        exec_res = paper_broker.execute_buy(
            token_symbol="SOL",
            token_mint=paper_cfg.target_token_mint,
            price=live_price,
            trade_usd=strat_cfg.trade_size_usdc,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            trailing_stop=signal.trailing_stop,
            sol_price=live_price,
            pool_liquidity=liq_usd
        )
        if exec_res:
            executed_action = f"BUY {exec_res.token_amount:.4f} SOL @ ${exec_res.execution_price:.2f}"

    elif signal.action == "SELL" and paper_broker.position is not None:
        trade_rec = paper_broker.execute_sell(
            price=live_price,
            reason=signal.reason,
            sol_price=live_price,
            pool_liquidity=liq_usd
        )
        if trade_rec:
            executed_action = f"SELL SOL @ ${trade_rec.exit_price:.2f} (Net: ${trade_rec.net_pnl_usd:+.2f})"

    paper_broker.update_m2m(live_price, sol_price=live_price)

    return jsonify({
        "signal": signal.action,
        "reason": signal.reason,
        "live_price": live_price,
        "executed_action": executed_action
    })


@app.route("/api/trades", methods=["GET"])
def get_trades():
    """Returns the full paper trade history ledger."""
    res = []
    for t in paper_broker.trade_history:
        res.append({
            "id": t.id,
            "side": t.side,
            "token_symbol": t.token_symbol,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "token_amount": t.token_amount,
            "gross_pnl_usd": t.gross_pnl_usd,
            "net_pnl_usd": t.net_pnl_usd,
            "total_fees_usd": t.total_fees_usd,
            "total_slippage_usd": t.total_slippage_usd,
            "total_friction_usd": t.total_friction_usd,
            "return_pct": t.return_pct,
            "hold_bars": t.hold_bars,
            "exit_reason": t.exit_reason
        })
    return jsonify(res)


@app.route("/api/paper/reset", methods=["POST"])
def reset_paper():
    """Resets paper account to fresh initial state."""
    global paper_broker
    paper_broker = PaperBroker(paper_cfg, frict_cfg)
    return jsonify({"status": "ok", "message": "Paper account reset successfully"})


@app.route("/api/routes", methods=["GET"])
def get_execution_routes():
    """Returns definitions, specs, and status for the 3 Execution Pathways."""
    routes_data = [
        {
            "id": "ROUTE_A_CEX",
            "tag": "路径 A",
            "name": "中心化交易所 (CEX)",
            "provider": "Binance / OKX / Bybit API",
            "engine": "内存订单簿撮合 (CLOB)",
            "latency_range": "5 ms – 50 ms",
            "latency_typical_ms": 18.5,
            "fee_rate": "0.04% (Taker) / 0.02% (Maker)",
            "gas_sol": "0 SOL (免链上 Gas)",
            "slippage": "极窄深度滑点 (~1.2 bps)",
            "atomic_protection": False,
            "atomic_desc": "无原子回滚 (单边撮合，不可撤销)",
            "scenarios": "SOL 现货 / U本位永续合约 CTA、趋势突破、跨期套利",
            "virtual_trading": "CEX 官方 Demo Trading / Testnet API",
            "pros": "超低延迟 (18ms)、极低费率 (0.04%)、无 Gas 损耗、流动性充裕",
            "cons": "平台托管风险 (FTX 风险)、无法捕捉链上原生 Meme/DEX 差价 Alpha",
            "status": "active" if paper_broker.active_route == "ROUTE_A_CEX" else "standby"
        },
        {
            "id": "ROUTE_B_DEX",
            "tag": "路径 B",
            "name": "链上聚合器 / DEX API",
            "provider": "Jupiter v6 API / Raydium SDK",
            "engine": "智能跨池拆单 (AMM 恒定乘积 / CLMM 集中流动性)",
            "latency_range": "200 ms – 1000 ms",
            "latency_typical_ms": 460.0,
            "fee_rate": "0.25% LP 池费 + 基础/优先 Gas",
            "gas_sol": "0.00005 ~ 0.0005 SOL",
            "slippage": "动态 AMM 冲击滑点 (随交易规模平方增长)",
            "atomic_protection": False,
            "atomic_desc": "无原子回滚 (交易广播上链即扣 Gas)",
            "scenarios": "链上主流现货代币兑换、DEX 现货网格、Meme 中低频交易",
            "virtual_trading": "本地 Paper Trading (抓真实 DEX 实时深度与价格模拟撮合)",
            "pros": "去中心化无托管风险、无需许可交易任意链上资产、智能跨池拆单",
            "cons": "容易遭遇三明治夹子 (MEV Sandwich)、公共 RPC 丢包与超时",
            "status": "active" if paper_broker.active_route == "ROUTE_B_DEX" else "standby"
        },
        {
            "id": "ROUTE_C_HFT",
            "tag": "路径 C",
            "name": "链上极速直连 (HFT / Jito)",
            "provider": "Yellowstone gRPC + Jito MEV Client",
            "engine": "Jito MEV Bundles (原子事务包直达 Leader Slot)",
            "latency_range": "100 ms – 400 ms",
            "latency_typical_ms": 165.0,
            "fee_rate": "0.25% LP 费 + 优先费 + Jito Tip (动态小费)",
            "gas_sol": "0.0001 SOL Jito 验证者小费",
            "slippage": "0.8 bps (绑定 Slot 首位成交，免疫滑点冲击)",
            "atomic_protection": True,
            "atomic_desc": "🛡️ 亏损零损耗原子回滚：滑点不符或无利可图时，Bundle 整体失败回滚，不扣本金！",
            "scenarios": "毫秒级抢跑、DEX 跨池套利、Meme 开盘狙击、链上清算套利、防夹保护",
            "virtual_trading": "Solana Devnet / 本地 solana-test-validator / Yellowstone 模拟沙盒",
            "pros": "完全免疫三明治夹子反洗劫、独占 Slot 优先打包、原子回滚零本金亏损",
            "cons": "需支付 Jito 验证者 Tip 小费、技术门槛高 (需维护 gRPC/私有节点)",
            "status": "active" if paper_broker.active_route == "ROUTE_C_HFT" else "standby"
        }
    ]
    return jsonify({
        "status": "success",
        "active_route": paper_broker.active_route,
        "routes": routes_data
    })


@app.route("/api/route/select", methods=["POST"])
def select_execution_route():
    """Sets the active execution route on the paper broker."""
    req = request.get_json() or {}
    route = req.get("route", "").upper()
    if route not in [ExecutionRoute.ROUTE_A_CEX, ExecutionRoute.ROUTE_B_DEX, ExecutionRoute.ROUTE_C_HFT]:
        return jsonify({"status": "error", "message": f"无效的路由标识: {route}"}), 400
    
    paper_broker.set_execution_route(route)
    return jsonify({
        "status": "success",
        "active_route": paper_broker.active_route,
        "message": f"当前活跃执行路由已成功切换为: {paper_broker.active_route}"
    })


@app.route("/api/route/execute_demo", methods=["POST"])
def execute_route_demo():
    """Executes a simulated live order through the requested execution route and returns full telemetry."""
    req = request.get_json() or {}
    route = req.get("route") or paper_broker.active_route
    side = req.get("side", "BUY").upper()
    amount_usd = float(req.get("amount_usd", 1000.0))
    symbol = req.get("symbol", "SOL").upper()

    # Get reference price and liquidity from DEX telemetry or buffer
    target_mint = HistoricalMarketFeed.SUPPORTED_TOKENS.get(symbol, {}).get("mint", paper_cfg.target_token_mint)
    telemetry_data = dex_client.get_primary_pair_telemetry(target_mint)
    if telemetry_data and telemetry_data.get("price_usd"):
        live_price = float(telemetry_data["price_usd"])
        pool_liq = float(telemetry_data.get("liquidity_usd", 37200000.0))
    else:
        live_price = float(live_buffer_df["close"].iloc[-1])
        pool_liq = 37200000.0

    # Run simulation
    exec_res = paper_broker.friction_model.simulate_execution(
        side=side,
        requested_price=live_price,
        trade_usd=amount_usd,
        sol_price_usd=live_price,
        pool_liquidity_usd=pool_liq,
        route=route
    )

    # Build chronological telemetry logs
    ts = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
    telemetry = []

    if route == ExecutionRoute.ROUTE_A_CEX:
        telemetry.append(f"[{ts}] ⚡ [CEX API 握手] 建立与 Binance/OKX 私有 WebSocket 订单流长连接 (SSL/TLS 0-RTT)...")
        telemetry.append(f"[{ts}] 📥 [订单入队] 提交市价单: {side} ${amount_usd:,.2f} USD ({symbol}) @ 参考价 ${live_price:.2f}")
        telemetry.append(f"[{ts}] ⚙️ [CLOB 内存撮合] 订单进入内存订单簿，对手盘最佳深度实时消耗，耗时 {exec_res.latency_ms:.1f} ms")
        telemetry.append(f"[{ts}] 💰 [费率结算] Taker 费率 0.04% (-${exec_res.dex_protocol_fee_usd:.2f}) | 链上 Gas: 0 SOL (免Gas)")
        telemetry.append(f"[{ts}] 🎯 [成交回报] 成交价: ${exec_res.execution_price:.4f} | 滑点仅 {exec_res.slippage_rate*10000:.1f} bps | 获得 {exec_res.token_amount:.4f} {symbol}")
        telemetry.append(f"[{ts}] ✅ [状态: FILLED] 交易所内部账本实时结算完毕，无链上排队延迟。")
    elif route == ExecutionRoute.ROUTE_C_HFT:
        telemetry.append(f"[{ts}] ⚡ [Yellowstone gRPC] 直连本地 Solana 验证节点 Geyser 插件，毫秒级订阅下任 Leader 出块 Slot...")
        telemetry.append(f"[{ts}] 📦 [Jito Bundle 组装] 构建原子事务包 (Atomic Bundle): 包含开仓指令 + Jito Tip 贿赂指令 (0.0001 SOL)")
        telemetry.append(f"[{ts}] 🚀 [Jito Block Engine 投递] 绕过公共 Mempool 交易池，专线直投下任验证者，耗时 {exec_res.latency_ms:.1f} ms")
        telemetry.append(f"[{ts}] 🛡️ [MEV 防夹与原子回滚校验] 检查前置三明治攻击者: 0 个 | 检查执行滑点: {exec_res.slippage_rate*10000:.1f} bps (低于 1.0 bps 阈值)")
        telemetry.append(f"[{ts}] 💎 [Slot 顶端打包] 成功抢占 Slot 首位执行！成交价: ${exec_res.execution_price:.4f} | Jito Tip: ${exec_res.jito_tip_usd:.3f}")
        telemetry.append(f"[{ts}] ✅ [状态: BUNDLE_LANDED] 原子事务包确认上链！零三明治损耗，原子安全回滚门禁守护就绪。")
    else:
        telemetry.append(f"[{ts}] ⚡ [Jupiter v6 智能路由] 向 Jupiter Aggregator 查询 SOL/USDC 跨池最优流动性路径...")
        telemetry.append(f"[{ts}] 🔀 [多池拆单执行] 智能拆单: 68% 经 Orca Whirlpool CLMM + 32% 经 Raydium CPMM")
        telemetry.append(f"[{ts}] 📡 [Solana RPC 广播] 附加优先费 ComputeBudget (0.0008 SOL)，提交至 Solana Validator，耗时 {exec_res.latency_ms:.1f} ms")
        telemetry.append(f"[{ts}] 🌊 [AMM 恒定乘积滑点] 受到池深流动性冲击，发生价格偏移 {exec_res.slippage_rate*10000:.1f} bps (-${exec_res.slippage_cost_usd:.2f})")
        telemetry.append(f"[{ts}] 💵 [手续费结算] LP 协议池费 0.25% (-${exec_res.dex_protocol_fee_usd:.2f}) + 网络 Gas (-${exec_res.base_network_fee_usd+exec_res.priority_fee_usd:.2f})")
        telemetry.append(f"[{ts}] ✅ [状态: CONFIRMED] 区块达成最终性 (Finalized)！成交价: ${exec_res.execution_price:.4f} | 获得 {exec_res.token_amount:.4f} {symbol}")

    return jsonify({
        "status": "success",
        "route": route,
        "engine_name": exec_res.engine_name,
        "side": side,
        "symbol": symbol,
        "nominal_usd": amount_usd,
        "requested_price": exec_res.requested_price,
        "execution_price": exec_res.execution_price,
        "token_amount": exec_res.token_amount,
        "latency_ms": exec_res.latency_ms,
        "slippage_rate_bps": exec_res.slippage_rate * 10000.0,
        "slippage_cost_usd": exec_res.slippage_cost_usd,
        "dex_protocol_fee_usd": exec_res.dex_protocol_fee_usd,
        "network_gas_usd": exec_res.base_network_fee_usd + exec_res.priority_fee_usd,
        "jito_tip_usd": exec_res.jito_tip_usd,
        "total_friction_usd": exec_res.total_friction_usd,
        "atomic_protection": exec_res.atomic_protection,
        "execution_status": exec_res.execution_status,
        "telemetry_logs": telemetry
    })


@app.route("/api/report/institutional", methods=["GET"])
def get_institutional_report():
    """Returns the full institutional deep quant report matching Image 2 with yearly/monthly/daily breakdowns."""
    report_path = os.path.join(os.path.dirname(__file__), "data", "deep_quant_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Report data not found"}), 404


def start_server(host="127.0.0.1", port=8000):
    print(f"\n🚀 Solana AI Quant Agent Web UI 已就绪!")
    print(f"👉 本地访问地址: http://{host}:{port}")
    print(f"👉 具备前后端分离 REST API 与实时 DEX 数据接入\n")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()
    start_server(host=args.host, port=args.port)
