# HPRL GPU runtimes

The merged project supports both native Windows GPU training and the mature Linux GPU dependency layer from the original HPRL Docker image. Both runtimes load the same merged source installed at `D:\Program Files\HEDGE`.

## Native Windows runtime

- Python: `D:\Program Files\HEDGE\.venv\Scripts\python.exe`
- PyTorch: `2.13.0+cu130`
- CUDA runtime: `13.0`
- Device: NVIDIA GeForce RTX 5070 Laptop GPU
- Compute capability: `12.0`
- PyTorch architecture support: `sm_120`

Run from `D:\Program Files\HEDGE`:

```powershell
.\.venv\Scripts\python.exe -m freqtrade.hedge.hprl device --device cuda
.\.venv\Scripts\python.exe -m freqtrade.hedge.hprl train-smoke --device cuda --algorithm fast_td3 --mixed-precision
.\scripts\Test-HEDGE-Windows-GPU.ps1
```

The reproducible installer is `scripts\Install-HEDGE-Windows-GPU.ps1`. It downloads the pinned official Windows wheel, verifies its published size and SHA-256, installs it into `.venv`, and verifies CUDA plus `sm_120`.

## Docker reference runtime

- Container: `HEDGE-HPRL-GPU`
- Image: `freqtrade-hedge:hprl-gpu-20260812-222754`
- Host source: `D:\Program Files\HEDGE`
- Container source: `/opt/freqtrade-hedge`
- Container Python: `/opt/hedge-venv/bin/python`
- PyTorch: `2.13.0+cu130`

The Docker image remains a known-good isolation and regression baseline. Its Linux shared libraries are kept inside the container; Windows uses the matching official Windows CUDA wheel in `.venv`.

## Commands

Run from `D:\Program Files\HEDGE`:

```powershell
.\scripts\Start-HEDGE-HPRL-GPU.ps1
.\scripts\Start-HEDGE-HPRL-GPU.ps1 inspect
.\scripts\Start-HEDGE-HPRL-GPU.ps1 train-smoke --device cuda --algorithm fast_td3 --mixed-precision
.\scripts\Test-HEDGE-HPRL-GPU.ps1
```

The launcher verifies the exact mature GPU image and the merged-source mount before executing HPRL. If the named container does not exist, it creates it from the local image with NVIDIA GPU access. It does not download or rebuild dependencies.

## Verified hardware paths

Both Windows and Docker runtimes were checked against the NVIDIA GeForce RTX 5070 Laptop GPU:

- CUDA device resolution: `cuda:0`
- Compute capability: `12.0`
- PyTorch architecture support: `sm_120`
- CUDA tensor smoke: passed
- `fast_td3` mixed-precision gradient-update smoke: passed

