# HEDGE high-intensity deep validation

Run from an elevated PowerShell only if the local environment requires it. The runner isolates
each phase in a subprocess, records `PASS/FAIL/TIMEOUT/ERROR/SKIPPED`, continues after failures,
and creates a ZIP containing the logs, reports, checklist and scripts.

```powershell
Set-Location 'D:\Program Files\HEDGE'
& '.\.venv\Scripts\python.exe' tools\run_hedge_deep_validation.py `
  --project-root 'D:\Program Files\HEDGE' `
  --checklist 'C:\Users\QX\Downloads\HEDGE_统一功能测试验证主清单_88dded9 (1).md' `
  --profile maximum `
  --python 'D:\Program Files\HEDGE\.venv\Scripts\python.exe' `
  --data-root 'D:\Program Files\HEDGE\artifacts\eth-two-year-deep' `
  --require-data `
  --phase-timeout 7200 `
  --training-timeout 21600 `
  --risk-timesteps 100000 `
  --risk-rows 20000 `
  --hprl-iterations 10000 `
  --hprl-replay-iterations 50000 `
  --performance-cycles 10000
```

The equivalent wrapper command is:

```powershell
Set-Location 'D:\Program Files\HEDGE'
& '.\tools\Run-HEDGE-DeepValidation.ps1' `
  -Profile maximum `
  -Checklist 'C:\Users\QX\Downloads\HEDGE_统一功能测试验证主清单_88dded9 (1).md' `
  -DataRoot 'D:\Program Files\HEDGE\artifacts\eth-two-year-deep' `
  -RequireData `
  -PhaseTimeout 7200 `
  -TrainingTimeout 21600 `
  -RiskTimesteps 100000 `
  -RiskRows 20000 `
  -HprlIterations 10000 `
  -HprlReplayIterations 50000 `
  -PerformanceCycles 10000
```

The command returns exit code `0` only when no executed phase is failed, timed out or errored.
Even with a non-zero exit code, the later phases have already been attempted and the ZIP is
still produced. The archive is written beside its timestamped evidence directory under
`D:\Program Files\HEDGE\artifacts\deep-validation\`.

The runner never enables exchange writes and never reads API credentials. Production Binance,
PostgreSQL, 24/72-hour soak, and other external/manual gates remain explicitly skipped unless
you provide and run those checks separately.
