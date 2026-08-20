# HPRL Freqtrade Multi-Timeframe V3

这是 HEDGE 仓库的 HPRL -> Freqtrade Strategy -> HEDGE execution 源码套件。

本版本重点是 **Freqtrade 原生 informative 多时间周期输入**。状态为：

`SOURCE_REPAIRED_MTF_TESTS_DEFERRED`

当前交付未执行训练、30 组正式 backtesting 或性能验收。

## 多周期结构

V3 保留 5 个 native HPRL 算法：

- Fast-TD3
- Fast-DSAC
- SimBa-SAC
- XQC
- ReBRAC-v2

并保留 6 个 base timeframe：

- 1m
- 5m
- 15m
- 1h
- 8h
- 1d

每个 base timeframe 自动消费自身和所有更高 timeframe：

```text
1m  -> 1m + 5m + 15m + 1h + 8h + 1d
5m  -> 5m + 15m + 1h + 8h + 1d
15m -> 15m + 1h + 8h + 1d
1h  -> 1h + 8h + 1d
8h  -> 8h + 1d
1d  -> 1d
```

所以未来仍可执行 5 x 6 = 30 个正式 task，但这些 task 已经不是原来的 isolated single-timeframe policy。

## Freqtrade 接入方式

Strategy 实现 `informative_pairs()`，高周期 OHLCV 由：

```python
self.dp.get_pair_dataframe(pair=pair, timeframe=timeframe)
```

提供。

多周期数据不会普通 forward-fill。feature alignment 只接受已经闭合的 informative candle：

```text
informative_open + informative_duration <= base_candle_close
```

超过一个 source timeframe 没有可用的已闭合 informative candle 会 fail-closed。

原始 OHLCV 的缺口处理与 Freqtrade 正式历史加载保持一致：由 Freqtrade 使用 no-action candle 补齐，再进入严格连续时间轴和 closed-candle 对齐；训练与正式 Strategy 不再采用两套不同的缺口语义。

## 文件

- `suite_specs.py`：模型、risk/action/cost、timeframe hierarchy 和 MTF alignment contract
- `features.py`：单周期 causal features + closed-candle MTF alignment
- `artifact_contract.py`：checkpoint/runtime/source fingerprint
- `prepare_models.py`：native HPRL MTF 模型准备/训练入口
- `strategies/hprl_mtf_v3_base.py`：真实 Freqtrade IStrategy bridge
- `strategies/hprl_*_eth.py`：5 个真实 Strategy classes
- `configs/*.json`：5 个 HEDGE/Freqtrade config
- `run_suite.py`：未来执行 30 组 formal `hedge-backtesting`
- `run_all.ps1`：PowerShell 5.1 future full-run entry point
- `Install-HPRL-Freqtrade-MTF-V3.ps1`：源码安装器，不执行训练或回测
- `SOURCE_IMPROVEMENTS.md`：本次源码设计与改进明细

## 安装行为

安装器只把本套件放入 HEDGE repo root 下：

```text
<HEDGE>\HPRL_FREQTRADE_MTF_V3
```

不会修改 `freqtrade/hedge/hprl` native 实现，不会 `pip install -e`，不会启动训练，不会启动 backtest。

如果目标目录已经存在，安装器会先将旧目录压缩备份到用户 `Downloads`，然后再替换。

## CPU 默认

V3 的 `run_all.ps1`、`run_suite.py`、`prepare_models.py` 和 Strategy 推理默认 device 均为 CPU。只有显式指定时才使用其他设备。

## 旧 checkpoint

V2.x checkpoint 不兼容 V3 MTF feature contract。V3 会通过 metadata schema、input timeframe list、feature layout、alignment contract 和 source fingerprint 拒绝旧 artifact。

## 后续运行

等源码整改确认后，再进行训练和正式 30 组 HEDGE/Freqtrade backtesting。正式收益证据只认 `hedge-backtesting` 输出，不认 legacy runner 的自定义 evaluator。
