"""Freqtrade bot and source-authority identity."""

from hashlib import sha256
from pathlib import Path


_resolved_path = Path(__file__).resolve()
_source_root = _resolved_path.parent.parent
_source_root_parts = {part.casefold() for part in _resolved_path.parts}
_source_markers = (
    _source_root / "pyproject.toml",
    _source_root / "freqtrade" / "hedge" / "contracts",
)
if "site-packages" in _source_root_parts or not all(marker.exists() for marker in _source_markers):
    raise RuntimeError(
        "Refusing to start from a non-source-authoritative freqtrade copy; "
        "run the clean-mainline checkout instead"
    )

__source_root__ = str(_source_root)
__source_authority_fingerprint__ = sha256(__source_root__.casefold().encode("utf-8")).hexdigest()[
    :12
]
__version__ = "2026.8-dev"

if "dev" in __version__:
    try:
        import subprocess  # noqa: S404, RUF100

        freqtrade_basedir = Path(__file__).parent

        __version__ = (
            __version__
            + "-"
            + subprocess.check_output(
                ["git", "log", '--format="%h"', "-n 1"],
                stderr=subprocess.DEVNULL,
                cwd=freqtrade_basedir,
            )
            .decode("utf-8")
            .rstrip()
            .strip('"')
        )

    except Exception:  # pragma: no cover
        # git not available, keep the source identity below.
        try:
            # Try Fallback to freqtrade_commit file (created by CI while building docker image)
            versionfile = Path("./freqtrade_commit")
            if versionfile.is_file():
                __version__ = f"docker-{__version__}-{versionfile.read_text()[:8]}"
        except Exception:  # noqa: S110
            pass

__version__ = f"{__version__}-src{__source_authority_fingerprint__}"
