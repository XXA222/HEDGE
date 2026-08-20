# HPRL ETH 过去两年多周期回测套件

目标仓库：`XXA222/HEDGE`

本套件直接复用仓库 `freqtrade.hedge.hprl` 的算法注册器、tiered action codec、
`VectorizedHedgeEnv`、replay buffer、`OfflineTrainer` 与 `evaluate_trading`，不另造一套
与 HPRL 无关的“伪 RL”回测器。

## 覆盖矩阵

模型（每个模型一个策略文件）：

- Fast-TD3
- Fast-DSAC
- SimBa-SAC
- XQC
- ReBRAC-v2

周期：

- 1m
- 5m
- 15m
- 1h
- 8h
- 1d

共 30 个独立 `模型 × 周期` 任务。默认每个任务做 2 个时间顺序 fold。

## 固定数据区间

UTC `[2024-08-19 00:00:00, 2026-08-19 00:00:00)`，恰好两年。
使用 Binance USD-M `ETHUSDT` 永续 K 线；资金费率通过 Binance funding history
接口拉取，并映射到实际结算前一根 bar。特征只使用决策时点及以前的数据，
`forward_return[t]` 只在动作选择后进入环境。

## 运行方式

将整个 `HPRL_ETH_MULTI_TF` 目录解压到 HEDGE 仓库根目录，然后在仓库 Python
环境中执行：

```bash
python HPRL_ETH_MULTI_TF/runner.py run-all --repo-root . --device auto --budget balanced --folds 2
```

更快的连通性测试可把 `balanced` 改成 `fast`；更长训练改成 `deep`。

## 不会因单个任务失败而中断

主控使用**独立子进程**运行每个数据准备任务和每个 `模型 × 周期` 回测任务。
某个策略导入失败、CUDA OOM、某个周期数据失败、worker 返回非零状态，都会被写入
日志/失败清单，然后继续下一个任务。

每个 worker 内部还会对 fold 做第二层异常隔离；一个 fold 失败不阻止该任务尝试后续 fold。

## 最终结果包

运行结束后会在 `hprl_results/` 下生成：

`HPRL_ETH_2Y_RESULTS_<UTC时间戳>.zip`

该 ZIP 包含：

- `summary.csv` / `summary.json`
- 每个任务的 `metrics.json`
- 抽样后的权益/仓位曲线 CSV
- 每个 prepare/worker 的 stdout/stderr 日志
- `failures.json`
- `run_manifest.json`
- 本次实际使用的策略/runner 源码快照
- 数据区间、条数和 SHA256 指纹

**不会把 1m 两年原始行情塞进结果 ZIP**，以免结果包过大。原始/预处理行情保留在
`user_data/hprl_eth_cache/`，结果 ZIP 记录哈希，可复现核对。

把最终 `HPRL_ETH_2Y_RESULTS_*.zip` 上传给 ChatGPT，即可进一步判断各模型收益、
回撤、成本、稳定性与周期适配性。

## 重要说明

- 这是研究/回测工具，不是收益保证。
- 默认 `compile_mode=off`，优先保证跨机器可运行；确认稳定后可加
  `--compile-mode auto`。
- 1m 两年数据量很大，首次下载和最终逐 bar 样本外推理是整套任务中最重的部分。
- Binance 公共接口若在本机网络不可访问，相应周期会记为失败，但主控仍会继续。


## 已有数据自动识别

新版 runner 会优先扫描 `<HEDGE>\user_data\data` 下已经下载的 ETH 历史数据，
支持 Freqtrade 常见的 Feather/Parquet/JSON/CSV 格式，并自动匹配
1m/5m/15m/1h/8h/1d。只有某个周期找不到可用本地数据时才访问 Binance 补数据。

Funding 文件也会自动识别；如果 OHLCV 已存在但 funding 无法找到且网络不可用，
该周期仍继续回测，funding 置零，并在 metadata 中明确标记 `missing_zero_fallback`。
