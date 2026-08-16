"""Source-safe isolated test environment for HPRL container acceptance.

This module creates an ephemeral virtual environment outside the project tree, bridges
the selected runtime interpreter's site-packages read-only through a ``.pth`` file, and
installs only missing test tools into the disposable environment.
It never installs the project itself and attests source Python bytes before and after.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import tomllib
import xml.etree.ElementTree as ET

_SPEC_NAME = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
_TEST_PREFIXES = ("pytest", "time-machine")
_IMPORT_BY_DISTRIBUTION = {
    "pytest": "pytest",
    "pytest-asyncio": "pytest_asyncio",
    "pytest-cov": "pytest_cov",
    "pytest-mock": "pytest_mock",
    "pytest-random-order": "random_order",
    "pytest-timeout": "pytest_timeout",
    "pytest-xdist": "xdist",
    "time-machine": "time_machine",
}


def _distribution_name(spec: str) -> str:
    match = _SPEC_NAME.match(spec)
    if match is None:
        raise ValueError(f"invalid dependency spec: {spec!r}")
    return match.group(1).lower().replace("_", "-")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_python_files(root: str | Path) -> tuple[Path, ...]:
    base = Path(root).resolve()
    rows: list[Path] = []
    for prefix in ("freqtrade", "tests/hedge", "tools"):
        folder = base / prefix
        if not folder.exists():
            continue
        rows.extend(
            path
            for path in folder.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    return tuple(sorted(set(rows)))


def source_tree_sha256(root: str | Path) -> str:
    base = Path(root).resolve()
    digest = sha256()
    for path in source_python_files(base):
        digest.update(path.relative_to(base).as_posix().encode("utf-8") + b"\0")
        digest.update(_file_sha256(path).encode("ascii") + b"\0")
    return digest.hexdigest()


def read_test_specs(root: str | Path) -> tuple[str, ...]:
    base = Path(root).resolve()
    payload = tomllib.loads((base / "pyproject.toml").read_text(encoding="utf-8"))
    optional = payload.get("project", {}).get("optional-dependencies", {})
    develop = tuple(str(item) for item in optional.get("develop", ()))
    selected = tuple(
        spec
        for spec in develop
        if _distribution_name(spec).startswith(_TEST_PREFIXES)
    )
    if "pytest" not in {_distribution_name(item) for item in selected}:
        raise ValueError("pyproject develop dependencies do not declare pytest")
    return selected


def _python_in_environment(environment_root: Path) -> Path:
    windows = environment_root / "Scripts" / "python.exe"
    posix = environment_root / "bin" / "python"
    if windows.is_file():
        return windows
    return posix


def _run(
    command: list[str],
    *,
    cwd: str | Path,
    timeout_seconds: int,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
    )


def _available_in_python(python: Path, import_name: str, *, timeout_seconds: int = 30) -> bool:
    code = (
        "import importlib.util,sys; "
        f"sys.exit(0 if importlib.util.find_spec({import_name!r}) is not None else 1)"
    )
    result = _run(
        [str(python), "-c", code],
        cwd=tempfile.gettempdir(),
        timeout_seconds=timeout_seconds,
    )
    return result.returncode == 0



def _bridge_runtime_site_packages(
    runtime: Path,
    env_python: Path,
) -> tuple[list[str], str | None]:
    runtime_probe = _run(
        [str(runtime), "-c", "import json,site; print(json.dumps(site.getsitepackages()))"],
        cwd=tempfile.gettempdir(),
        timeout_seconds=30,
    )
    target_probe = _run(
        [str(env_python), "-c", "import json,site; print(json.dumps(site.getsitepackages()))"],
        cwd=tempfile.gettempdir(),
        timeout_seconds=30,
    )
    if runtime_probe.returncode != 0 or target_probe.returncode != 0:
        return [], "VENV_SITE_PATH_PROBE_FAILED"
    try:
        runtime_sites = [
            str(Path(item).resolve())
            for item in json.loads(runtime_probe.stdout.strip())
            if Path(item).is_dir()
        ]
        target_sites = [Path(item) for item in json.loads(target_probe.stdout.strip())]
    except (TypeError, ValueError, json.JSONDecodeError):
        return [], "VENV_SITE_PATH_PROBE_FAILED"
    if not target_sites:
        return runtime_sites, "VENV_SITE_PATH_PROBE_FAILED"
    bridge = target_sites[0] / "hprl_runtime_site_packages.pth"
    bridge.write_text("".join(item + "\n" for item in runtime_sites), encoding="utf-8")
    return runtime_sites, None


def _missing_test_specs(env_python: Path, specs: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for spec in specs:
        distribution = _distribution_name(spec)
        import_name = _IMPORT_BY_DISTRIBUTION.get(distribution, distribution.replace("-", "_"))
        if not _available_in_python(env_python, import_name):
            missing.append(spec)
    return missing


def _install_test_specs(
    env_python: Path,
    specs: list[str],
    *,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return _run(
        [
            str(env_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            *specs,
        ],
        cwd=tempfile.gettempdir(),
        timeout_seconds=timeout_seconds,
        env=env,
    )


def _resolve_test_dependencies(
    env_python: Path,
    specs: tuple[str, ...],
    *,
    install_missing: bool,
    timeout_seconds: int,
) -> tuple[list[str], list[str], str | None]:
    missing = _missing_test_specs(env_python, specs)
    if not missing:
        return [], [], None
    if not install_missing:
        return [], [], "TEST_DEPENDENCIES_MISSING"
    install = _install_test_specs(
        env_python,
        missing,
        timeout_seconds=timeout_seconds,
    )
    lines = install.stdout.splitlines()
    if install.returncode != 0:
        return [], lines, f"TEST_DEPENDENCY_INSTALL_EXIT_NONZERO:{install.returncode}"
    return missing, lines, None

@dataclass(frozen=True, slots=True)
class IsolatedTestEnvironmentReport:
    source_root: str
    runtime_python: str
    environment_root: str
    environment_python: str
    requested_specs: tuple[str, ...]
    installed_specs: tuple[str, ...]
    pytest_available: bool
    source_sha256_before: str
    source_sha256_after: str
    source_unchanged: bool
    bootstrap_output_tail: tuple[str, ...]
    runtime_site_paths: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.pytest_available and self.source_unchanged and not self.reasons


def create_isolated_test_environment(
    root: str | Path,
    *,
    runtime_python: str | Path,
    environment_root: str | Path,
    install_missing: bool = True,
    timeout_seconds: int = 1200,
) -> IsolatedTestEnvironmentReport:
    base = Path(root).resolve()
    runtime = Path(runtime_python)
    target = Path(environment_root).resolve()
    if not runtime.is_file():
        raise FileNotFoundError(f"runtime Python does not exist: {runtime}")
    if target == base or base in target.parents:
        raise ValueError("isolated test environment must live outside the source tree")
    before = source_tree_sha256(base)
    specs = read_test_specs(base)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_lines: list[str] = []
    reasons: list[str] = []

    result = _run(
        [str(runtime), "-m", "venv", str(target)],
        cwd=tempfile.gettempdir(),
        timeout_seconds=timeout_seconds,
    )
    bootstrap_lines.extend(result.stdout.splitlines())
    if result.returncode != 0:
        reasons.append(f"VENV_CREATE_EXIT_NONZERO:{result.returncode}")
    env_python = _python_in_environment(target)
    if not reasons and not env_python.is_file():
        reasons.append("VENV_PYTHON_MISSING")

    runtime_site_paths: list[str] = []
    if not reasons:
        runtime_site_paths, bridge_error = _bridge_runtime_site_packages(runtime, env_python)
        if bridge_error:
            reasons.append(bridge_error)

    installed: list[str] = []
    if not reasons:
        installed, install_lines, install_error = _resolve_test_dependencies(
            env_python,
            specs,
            install_missing=install_missing,
            timeout_seconds=timeout_seconds,
        )
        bootstrap_lines.extend(install_lines)
        if install_error:
            reasons.append(install_error)

    pytest_available = bool(
        not reasons
        and env_python.is_file()
        and _available_in_python(env_python, "pytest")
    )
    if not pytest_available:
        reasons.append("PYTEST_UNAVAILABLE")
    after = source_tree_sha256(base)
    unchanged = before == after
    if not unchanged:
        reasons.append("SOURCE_TREE_CHANGED_DURING_TEST_BOOTSTRAP")
    return IsolatedTestEnvironmentReport(
        source_root=str(base),
        runtime_python=str(runtime),
        environment_root=str(target),
        environment_python=str(env_python),
        requested_specs=specs,
        installed_specs=tuple(installed),
        pytest_available=pytest_available,
        source_sha256_before=before,
        source_sha256_after=after,
        source_unchanged=unchanged,
        bootstrap_output_tail=tuple(bootstrap_lines[-80:]),
        runtime_site_paths=tuple(runtime_site_paths),
        reasons=tuple(dict.fromkeys(reasons)),
    )


@dataclass(frozen=True, slots=True)
class IsolatedPytestReport:
    targets: tuple[str, ...]
    tests: int
    failures: int
    errors: int
    skipped: int
    duration_seconds: float
    returncode: int
    minimum_tests: int
    junit_sha256: str
    source_sha256_before: str
    source_sha256_after: str
    source_unchanged: bool
    stdout_tail: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.returncode == 0
            and self.failures == 0
            and self.errors == 0
            and self.tests >= self.minimum_tests > 0
            and self.source_unchanged
        )


def _junit_counts(path: Path) -> tuple[int, int, int, int, float]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = failures = errors = skipped = 0
    duration = 0.0
    for suite in suites:
        tests += int(suite.attrib.get("tests", "0"))
        failures += int(suite.attrib.get("failures", "0"))
        errors += int(suite.attrib.get("errors", "0"))
        skipped += int(suite.attrib.get("skipped", "0"))
        try:
            duration += float(suite.attrib.get("time", "0"))
        except ValueError:
            pass
    return tests, failures, errors, skipped, duration


def run_isolated_pytest(
    root: str | Path,
    *,
    environment_python: str | Path,
    targets: Iterable[str],
    minimum_tests: int,
    junit_path: str | Path,
    timeout_seconds: int = 7200,
    use_xdist: bool = False,
) -> IsolatedPytestReport:
    base = Path(root).resolve()
    python = Path(environment_python)
    selected = tuple(str(item) for item in targets)
    if not selected:
        raise ValueError("at least one pytest target is required")
    if minimum_tests <= 0:
        raise ValueError("minimum_tests must be positive")
    if not python.is_file():
        raise FileNotFoundError(f"isolated Python does not exist: {python}")
    junit = Path(junit_path).resolve()
    if base == junit.parent or base in junit.parents:
        raise ValueError("JUnit output must live outside the source tree")
    junit.parent.mkdir(parents=True, exist_ok=True)
    before = source_tree_sha256(base)
    temp_root = Path(tempfile.mkdtemp(prefix="hprl-priority5-pytest-"))
    try:
        command = [
            str(python),
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "--basetemp",
            str(temp_root / "tmp"),
            "--junitxml",
            str(junit),
        ]
        if use_xdist and _available_in_python(python, "xdist"):
            command.extend(["-n", "auto"])
        command.extend(selected)
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        inherited = os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        env["PYTHONPATH"] = str(base) + inherited
        completed = _run(command, cwd=base, timeout_seconds=timeout_seconds, env=env)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    if junit.is_file():
        tests, failures, errors, skipped, duration = _junit_counts(junit)
        junit_hash = _file_sha256(junit)
    else:
        tests, failures, errors, skipped, duration = (0, 0, 1, 0, 0.0)
        junit_hash = "0" * 64
    after = source_tree_sha256(base)
    return IsolatedPytestReport(
        targets=selected,
        tests=tests,
        failures=failures,
        errors=errors,
        skipped=skipped,
        duration_seconds=duration,
        returncode=completed.returncode,
        minimum_tests=minimum_tests,
        junit_sha256=junit_hash,
        source_sha256_before=before,
        source_sha256_after=after,
        source_unchanged=before == after,
        stdout_tail=tuple(completed.stdout.splitlines()[-120:]),
    )


def report_json(report: object) -> str:
    from dataclasses import asdict

    return json.dumps(asdict(report), sort_keys=True, indent=2, default=str) + "\n"
