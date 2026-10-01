# 本地高频/长周期真实历史数据存放目录 (Data Lake)

本系统支持**自动优先加载本地真实历史行情数据**。只要本目录下存在对应命名格式的 CSV 文件，系统在回测时将直接读取本地真实数据，无需任何网络等待。

## 1. 文件命名规范
- 格式：`{SYMBOL}_{TIMEFRAME}.csv`
- 示例：
  - `data/SOL_15m.csv`
  - `data/BTC_15m.csv`
  - `data/ETH_1h.csv`
  - `data/WIF_15m.csv`

## 2. CSV 列名标准 (标准 OHLCV 格式)
| 列名 | 含义 | 示例 | 必须/可选 |
| :--- | :--- | :--- | :--- |
| `timestamp` | 时间戳/日期时间 | `2026-01-01 00:00:00` | **必须** |
| `open` | 开盘价 | `135.25` | **必须** |
| `high` | 最高价 | `137.50` | **必须** |
| `low` | 最低价 | `134.80` | **必须** |
| `close` | 收盘价 | `136.90` | **必须** |
| `volume_usd` | 成交额 (USD) | `1580000` | 可选（若无自动兼容 `volume`） |
| `liquidity_usd` | DEX 池子流动性深度 | `35000000` | 可选（若无自动采用该币种默认主池深度） |
| `buy_ratio` | 净买单占比 | `0.55` | 可选（若无默认 0.50） |

## 3. 免费真实历史数据获取渠道
1. **Binance Data Vision (官方全免费历史归档)**:
   - 网址: `https://data.binance.vision/`
   - 包含 BTC, ETH, SOL 自 2017 年以来的逐月/逐日 1m/5m/15m/1h K 线 CSV 压缩包。
2. **CoinGecko / GeckoTerminal (Solana DEX 原生数据)**:
   - 网址: `https://www.geckoterminal.com/`
   - 支持下载 Raydium / Orca 原生代币池的真实历史蜡烛图。
3. **Yahoo Finance / TradingView 导出**:
   - 可以在 TradingView 上导出任意标的的 K 线 CSV，重命名放入本目录即可直接回测。
