# HEDGE + HPRL local integration

This installation keeps the current Hedge workspace as the only Freqtrade mainline and integrates HPRL as a code submodule.

## Layout

- `freqtrade/hedge/hprl/` — HPRL 2.5.2 research/runtime package.
- `freqtrade/hedge/production/` — HPRL production adapters, closed-loop runtime, recovery, and acceptance components.
- `freqtrade/hedge/memory_lifecycle.py` — shared memory lifecycle support required by HPRL.
- `config_examples/hprl*.json` — CPU/automatic and GPU examples.
- `tools/*hprl*.py` and `scripts/Run-HPRL*.ps1` — HPRL validation and operations entry points.
- `tests/hedge/hprl/` and `tests/hedge/production/` — imported HPRL verification suites.

The HPRL repository is itself a full Freqtrade fork. Its duplicate Freqtrade tree was intentionally not nested or overlaid. In particular, its older copies of Hedge configuration, CLI, optimization, and core runtime files were not allowed to replace the newer Hedge mainline.

## Local runtime

Run commands from `D:\Program Files\HEDGE`:

```powershell
.\.venv\Scripts\python.exe -m freqtrade --version
.\.venv\Scripts\python.exe -m freqtrade.hedge.hprl inspect
.\.venv\Scripts\python.exe -m freqtrade.hedge.hprl compat --project-root .
.\scripts\Verify-HEDGE-Local.ps1
```

HPRL core adds no dependency version authority; it reuses the Torch stack already installed for Hedge/FreqAI RL. PostgreSQL acceptance is optional and uses `requirements-hprl-postgres.txt`.

The copied virtual environment uses its local-source `.pth` authority for `D:\Program Files\HEDGE` and `D:\Program Files\HEDGE\ft_client`. This keeps imports bound to the new project without installing a second Freqtrade package copy into `site-packages`. The two source projects remain unchanged.

## Safety notes

- `python -m freqtrade.hedge.hprl inspect` reports `live_order_write: false`.
- Use `config_examples/hprl.example.json` for initial local/CPU checks.
- The production layer is present for integration testing, but exchange writes must still pass Hedge's existing production gates and explicit runtime configuration.
