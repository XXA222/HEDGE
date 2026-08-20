# HPRL Freqtrade Multi-Timeframe V3 - Source Improvements

状态：`SOURCE_REPAIRED_MTF_TESTS_DEFERRED`

本版本只完成源码架构整改。未执行 HPRL 训练、30 组正式 HEDGE/Freqtrade 回测、收益比较或性能验收。

## 1. 核心变化

V2.x 的 5 个算法可以分别在 1m/5m/15m/1h/8h/1d 上运行，但每次 Strategy 只消费当前 timeframe。V3 保留原来的 5 x 6 = 30 组矩阵，同时让每个 base timeframe 消费 Freqtrade informative 高周期：

| Base | Policy input timeframes |
|---|---|
| 1m | 1m + 5m + 15m + 1h + 8h + 1d |
| 5m | 5m + 15m + 1h + 8h + 1d |
| 15m | 15m + 1h + 8h + 1d |
| 1h | 1h + 8h + 1d |
| 8h | 8h + 1d |
| 1d | 1d |

Strategy 通过 `informative_pairs()` 注册高周期，并通过 Freqtrade `DataProvider.get_pair_dataframe()` 获取 OHLCV。旧 `HPRL_ETH_MULTI_TF/runner.py` 仍不参与正式收益计算。

## 2. 严格 closed-candle 对齐

Freqtrade OHLCV 的 `date` 表示 candle open time。V3 的每一个 base decision 使用 base candle close 作为决策时间：

`decision_time = base_open + base_duration`

informative candle 只有满足：

`informative_open + informative_duration <= decision_time`

才可以进入 observation。

实现使用 NumPy `searchsorted(..., side="right") - 1` 查找最后一根已经闭合的 source candle。没有使用对未来数据不透明的普通 `resample + ffill`。

如果最后一根 informative candle 相对当前 decision 已经老化达到一个完整 source timeframe，则该 source 被判定 stale 并 fail-closed。不会无限 forward-fill 历史高周期数据。

原始 OHLCV 缺口现在与 Freqtrade 正式 backtesting/DataProvider 保持同一语义：使用 Freqtrade no-action candle 补齐后再进行严格时间轴检查。这样训练阶段与正式 Strategy 不会因 `fill_up_missing` 设置不同而产生隐蔽的数据分布差异。

## 3. MTF feature layout

每个 timeframe 仍使用同一套 causal OHLCV features，并带 timeframe prefix，例如：

- `1m__logret_1`
- `5m__rsi14`
- `1h__atr14_pct`
- `1d__ema_spread_55`

每个 informative timeframe 额外加入：

- `<tf>__age_frac`

`age_frac` 表示从最后一根已闭合 informative candle 的 close 到当前 base decision 的时间，占 source timeframe 的比例，范围严格为 `[0, 1)`。

这让 agent 知道高周期信息距离下一次更新还有多远，同时不泄漏未收盘 candle。

## 4. 内存策略

没有把 6 个宽 DataFrame 直接 `merge` 成百万行级 pandas 表。

`align_multi_timeframe_features()`：

1. 逐 timeframe 计算 feature frame；
2. 通过 integer timestamp/searchsorted 生成 source row index；
3. 直接写入一个最终 `float32` NumPy observation matrix；
4. 当前 timeframe block 写完后释放 feature DataFrame 引用。

因此 1m 两年历史的主要新增内存为最终 float32 MTF feature matrix，而不是多份 float64 pandas merge 副本。

## 5. startup / informative warmup

`startup_candle_count` 提升到 96。

这不仅覆盖 base timeframe 的 EMA55 / rolling24 等 warmup，也会被 Freqtrade DataProvider 用于 informative historical loading，使 1d 等高周期在正式 backtest 起点之前拥有足够的历史。

正式 HEDGE replay 仍从 Freqtrade trim 掉 startup rows 后开始。HPRL shadow env 在同一个 base row 边界 reset 为 flat，避免 startup candle 改变正式起始账户状态。

## 6. Training / Strategy feature parity

训练和 Strategy 推理共用：

- `build_feature_frame()`
- `align_multi_timeframe_features()`
- 同一 `input_timeframes_for(base)`
- 同一 closed-candle visibility rule
- 同一 feature names/order

训练阶段会读取 base timeframe 及其全部 informative timeframes。每个 source 在 train start 前额外保留 96 根自身 timeframe 的 warmup candle，再在 base decision timeline 上对齐。

## 7. Checkpoint / stale-artifact 防护

模型 metadata 升级到 `hprl-freqtrade-model-v5-mtf`。

runtime contract 现在绑定：

- model / algorithm / Strategy class
- base timeframe
- input timeframes
- feature version
- exact feature names/order
- MTF alignment contract
- action/cost/reward/model spec
- suite semantic source hashes
- repository HPRL implementation hashes
- DataProvider / Strategy interface / history loader / backtesting / HEDGE Strategy contract / HPRL adapter hashes

旧 V2.x checkpoint 不允许静默加载。需要重新训练后才会生成新的 MTF artifact manifest。

## 8. 30 组矩阵没有删除

V3 仍保留：

`5 algorithms x 6 base timeframes = 30 formal tasks`

差别是低周期 task 不再是孤立单周期模型，而是具有高周期 informative context 的真实 MTF model。

## 9. 正式执行权仍属于 HEDGE

HPRL Strategy 只产生 canonical：

- `hedge_long_score`
- `hedge_short_score`
- `hedge_long_exposure_scale`
- `hedge_short_exposure_scale`
- `hedge_confidence`
- `hedge_risk_scale`
- `hedge_allow_new_risk`
- nullable target fields / regime / reason / model version

HPRL shadow env 不拥有正式成交、手续费、funding、钱包余额、强平或最终 PnL。

正式结果仍只能来自：

`python -m freqtrade hedge-backtesting`

## 10. 仍然存在的架构边界

当前 Freqtrade historical lifecycle 仍然是 Strategy 先分析 dataframe，随后 HEDGE replay 执行。因此 HPRL env 中的 position/equity state 仍是 policy-state shadow，不是 HEDGE replay 的 authoritative state。

V3 修复的是多时间周期数据输入、closed-candle causal alignment、训练/推理一致性和 artifact provenance；它没有假装已经完成逐 bar 的 HEDGE-state -> HPRL -> planner -> fill -> next-HPRL-state 闭环。

真正的下一阶段闭环需要修改 HEDGE replay orchestration，使 policy evaluation 进入 replay loop 本身。
