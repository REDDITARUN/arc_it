"""Configuration loading with automatic Mac/H100 adaptation."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from arc_it.utils.device import get_device


def load_config(
    config_path: str = "configs/default.yaml",
    override_path: Optional[str] = None,
    auto_adapt: bool = True,
) -> Dict[str, Any]:
    """Load configuration with optional overrides and auto-adaptation.

    Args:
        config_path: Path to base config YAML.
        override_path: Path to override YAML (e.g., mac_dev.yaml).
        auto_adapt: If True and no override specified, auto-detect
                     Mac vs CUDA and apply appropriate overrides.

    Returns:
        Merged configuration dictionary.
    """
    base_path = Path(config_path)
    if not base_path.exists():
        base_path = Path(__file__).parent.parent.parent / config_path
    with open(base_path, "r") as f:
        config = yaml.safe_load(f)

    if override_path is not None:
        override_file = Path(override_path)
        if not override_file.exists():
            override_file = Path(__file__).parent.parent.parent / override_path
        with open(override_file, "r") as f:
            overrides = yaml.safe_load(f)
        config = _deep_merge(config, overrides)
    elif auto_adapt:
        device = get_device()
        if device.type != "cuda":
            mac_config = Path(__file__).parent.parent.parent / "configs" / "mac_dev.yaml"
            if mac_config.exists():
                with open(mac_config, "r") as f:
                    overrides = yaml.safe_load(f)
                config = _deep_merge(config, overrides)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
