# Solana AI Quant Agent (基于 Solana 的 AI 量化交易与自动执行系统)

**Solana AI Quant Agent** 是专为 Solana 生态设计的工业级 AI 量化交易与虚拟盘执行系统。系统深度整合链上实时数据、DEX 流动性池深度、高阶动量与趋势策略因子，模拟 **Jupiter 路由执行**，并具备严苛的**全额摩擦成本因果核算与风险控制**。

---

## 🌟 核心架构与功能特性

### 1. 真实 Solana DEX 摩擦成本模型 (`friction.py`)
坚决杜绝零摩擦幻觉，在虚拟盘与回测中 100% 还原 Solana 主网真实链上交易磨损：
- **Solana 基础网络手续费 (Base Gas)**: 默认 5,000 lamports / 签名（单笔交易含计算预算指令约 0.000010 SOL）。
- **优先计算费用 (Compute Budget Priority Fee)**: 动态优先费（默认 0.000800 SOL / 笔），模拟主网拥堵情况下的交易加速。
- **Jito MEV Tip**: 模拟 Jito 捆绑交易小费（默认 0.000500 SOL / 笔），保障极端波动下的交易上链率。
- **DEX 流动性池协议费 (DEX LP Fee)**: 真实扣除 0.25% (Raydium / Orca 常用池标准费率)。
- **动态 AMM 恒定乘积滑点模型**:
  $$\text{Slippage} = \text{BaseSlippage} + \frac{\text{TradeSize}}{\text{PoolLiquidity} \times 0.5} + \text{LatencyDrift}$$
- **1x 正常成本 vs 3x 极限压力测试**: 支持将网络费、DEX 手续费与滑点按 3.0 倍放大，审计策略在恶劣网络拥堵与流动性踩踏下的抗崩溃能力。

### 2. 链上实时数据与高频分时引擎 (`data_feed.py`)
- **DexScreener Live API**: 直连 Solana 链上池（Orca / Raydium / Meteora），实时获取代币价格、池流动性、24h/1h 成交量、1h 买单/卖单笔数及净买入流倾向。
- **多机制历史分时行情生成器**: 涵盖单边暴涨 (Hyper-Bull)、恐慌深跌 (Panic-Crash)、宽幅洗盘 (Whipsaw Range) 与波动率聚集机制，支持 15m/1h 分时级别评估。

### 3. 多因子 AI 趋势策略智能体 (`strategy.py` & `factors.py`)
- **零滞后指数平滑移动平均线 (ZLEMA)**: 消除一阶 EMA 滞后，快速捕捉 Solana 动量拐点。
- **SuperTrend 自适应波带**: 结合 ATR 动态跟踪趋势方向，杜绝单柱微观噪声洗出。
- **链上订单流与流动性门禁 (Liquidity & Order Flow Gate)**: 剔除池深度低于 $50,000 与买盘严重不足的陷阱池。
- **摩擦空间门禁 (Friction Space Gate)**: 要求期望获利空间 $\ge 3.0 \times$ 双边总摩擦成本，坚决杜绝微利高频磨损。

### 4. 虚拟盘核心撮合与盯市账本 (`paper_broker.py`)
- **双资产真实记账**: 分别管理 USDC 交易现金本金与 SOL 链上 Gas 账户，撮合后按链上实际消耗扣除 SOL。
- **逐柱动态盯市 (Mark-to-Market Equity)**: 实时跟踪未平仓浮动盈亏、历史最高权益 (High Watermark) 与真实回撤 (MaxDD)。
- **全生命周期成交流水**: 记录买入价、卖出价、持仓柱数、净盈亏 (Net PnL)、手续费细项、滑点细项与退出诊断原因。

### 5. 严格因果时序回测与大数定律审计 (`backtester.py`)
- **严格因果撮合 (Next-Open Fill)**: 第 $t$ 根 Bar 收盘产生信号 $\to$ 强制在第 $t+1$ 根 Bar 开盘价撮合成交，彻底杜绝未来函数。
- **大数定律统计置信度 (Wilson 95% CI)**: 自动计算胜率置信区间与可靠性分级（A_RELIABLE / B_MARGINAL / F_UNRELIABLE）。
- **利润集中度审计**: 检验前三大盈利单占比，识别并否决依赖少数单边行情的幸存者偏差。

---

## 📂 项目结构

```
Solana/
├── config.py           # 系统参数配置（DEX 手续费、优先费、滑点、风控比例）
├── friction.py         # Solana DEX 真实摩擦成本模型（Gas/Priority/Jito/AMM 滑点）
├── data_feed.py        # DexScreener 链上实时数据接口与历史分时引擎
├── factors.py          # 策略因子计算库（ZLEMA, SuperTrend, ATR, 订单流净比）
├── strategy.py         # Solana 趋势与流动性策略智能体（信号生成与动态出场）
├── paper_broker.py     # 虚拟盘撮合经纪人（记账本、盯市权益曲线、订单生命周期）
├── backtester.py       # 严格因果回测引擎与 1x vs 3x 双轨极限压力测试
├── runner.py           # 命令行统一运行器（支持 self-check、backtest、live paper）
└── README.md           # 系统架构与使用说明文档
```

---

## 🚀 快速上手与操作指南

在终端进入项目根目录：
```bash
cd /Users/tang/PycharmProjects/pythonProject/Solana
```

### 1. 运行核心系统自检 (Runnable Self-Check)
验证摩擦模型、3x 压测倍率、虚拟盘记账逻辑及因果回测引擎：
```bash
python3 runner.py test
```
*预期输出：所有 4 项断言自检全部通过。*

### 2. 执行工业级回测与 1x vs 3x 极限压力测试
在 2,500 根 15m 分时 K 线上运行严格因果回测，扣除全额摩擦并输出标准审计报告：
```bash
python3 runner.py backtest
```

### 3. 运行实时虚拟盘 (Live Paper Trading)
连接 DexScreener 实时抓取 Solana 链上主网代币（默认 SOL/USDC）行情与流动性，执行模拟路由与实时记账：
```bash
# 运行 5 次实时轮询示例（可自定义轮询次数与间隔秒数）
python3 runner.py paper --loops 5 --interval 3.0
```

---

## 📊 策略风控规则与核算准则
1. **胜率与收益 100% 由全额净盈亏 (Net PnL) 驱动**，严禁使用毛利掩盖交易摩擦。
2. **最大回撤 (MaxDD)** 从包含盘中浮动盈亏的逐柱盯市权益曲线动态计算。
3. **单笔仓位限制**: 默认单次 nominal trade size 为 $1,000 ~ $2,500 USDC（占总资金 10%~25%），杜绝单注过重。
4. **硬性止损保护**: 采用动态 ATR 硬止损与 SuperTrend 方向反转出场，最大化截断左尾风险，放飞右尾趋势。
