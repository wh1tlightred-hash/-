"""pytest 根配置：确保项目根目录在 sys.path，便于 `from src import ...`。"""
import sys
import shutil
from uuid import uuid4
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    """避免 Windows 上残留的 pytest 临时目录权限损坏影响后续运行。"""
    if config.option.basetemp is None:
        run_dir = ROOT / ".pytest_runs"
        run_dir.mkdir(exist_ok=True)
        base = run_dir / uuid4().hex
        config.option.basetemp = str(base)
        config._project_basetemp = base


def pytest_unconfigure(config):
    base = getattr(config, "_project_basetemp", None)
    if base is not None:
        shutil.rmtree(base, ignore_errors=True)
