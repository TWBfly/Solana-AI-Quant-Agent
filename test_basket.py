import time
from basket_engine import basket_engine

def main():
    t0 = time.time()
    print("Running 1,000-trade Multi-Asset Basket Backtest (30 tokens, 1440 1m bars)...")
    audit = basket_engine.run_full_stress_comparison(bars_count=1440)
    elapsed = time.time() - t0

    res_1x = audit["res_1x"]
    res_3x = audit["res_3x"]

    print("\n" + "=" * 90)
    print("🚀 【Solana 30-Token Multi-Asset Basket Super Quant Audit Report】")
    print("=" * 90)
    print(f"回测耗时 (Execution Time)  : {elapsed:.2f} 秒")
    print(f"资产宇宙 (Universe Size)   : {res_1x['universe_size']} 个活跃代币")
    print(f"有效数据周期 (Timeframe)   : 1m 分时微观行情 (单日 1,440 根/币)")
    print(f"总交易笔数 (1x Baseline)   : {res_1x['total_trades']} 笔完整交易")
    print(f"大数定律达成 (LLN Target)  : {'✅ 完美达成 (>= 1,000 笔)' if res_1x['law_of_large_numbers_achieved'] else '未达标'}")
    print(f"胜率 (Win Rate)            : {res_1x['win_rate']:.1f}%")
    print(f"Wilson 95% 置信区间        : [{res_1x['wilson_ci_low']}%, {res_1x['wilson_ci_high']}%] (误差仅 ±{res_1x['wilson_ci_span']/2:.2f}%)")
    print(f"组合累计净利润 (1x 正常成本): ${res_1x['net_profit_usd']:+,.2f} ({res_1x['return_pct']:+.2f}%)")
    print(f"组合累计净利润 (3x 极限摩擦): ${res_3x['net_profit_usd']:+,.2f} ({res_3x['return_pct']:+.2f}%)")
    print(f"夏普比率 (Sharpe Ratio)    : {res_1x['sharpe_ratio']:.2f}")
    print(f"索提诺比率 (Sortino Ratio) : {res_1x['sortino_ratio']:.2f}")
    print(f"卡玛比率 (Calmar Ratio)    : {res_1x['calmar_ratio']:.2f}")
    print(f"动态盯市最大回撤 (MaxDD)   : -{res_1x['max_drawdown_pct']:.2f}% (${res_1x['max_drawdown_usd']:,.2f})")
    print(f"全额链上摩擦成本总计       : ${res_1x['total_friction_usd']:,.2f}")
    print(f"  - 优先费与 Jito 防夹小费 : ${res_1x['total_fees_usd']:,.2f}")
    print(f"  - AMM 动态价格冲击滑点   : ${res_1x['total_slippage_usd']:,.2f}")
    print(f"3x 极端抗压审计结果        : {'🏆 压力测试通过 (PASS)' if audit['stress_audit_pass'] else '⚠️ 压力测试失败'}")
    print("-" * 90)
    print("前 5 大 Alpha 贡献品种 (Top Alpha Contributors):")
    for t in res_1x["token_summaries"][:5]:
        print(f"  • {t['symbol']:<8} ({t['name']:<18}): {t['trades']:>3} 笔交易 | 胜率 {t['win_rate']}% | 净利润 ${t['net_pnl']:+,.2f}")
    print("-" * 90)

if __name__ == "__main__":
    main()
