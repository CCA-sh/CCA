"""Trust store for .mcp.json server configs — ~/.cca/mcp-trust.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

TRUST_FILE = Path.home() / ".cca" / "mcp-trust.json"


def _fingerprint(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def _load() -> dict:
    if TRUST_FILE.exists():
        try:
            with open(TRUST_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save(data: dict) -> None:
    TRUST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRUST_FILE, "w") as f:
        json.dump(data, f, indent=2)
    TRUST_FILE.chmod(0o600)


def is_trusted(config_path: str, server_name: str, cfg: dict) -> bool:
    return _load().get(str(config_path), {}).get(server_name) == _fingerprint(cfg)


def record_trust(config_path: str, server_name: str, cfg: dict) -> None:
    data = _load()
    data.setdefault(str(config_path), {})[server_name] = _fingerprint(cfg)
    _save(data)


def revoke_trust(config_path: str, server_name: str) -> bool:
    data = _load()
    entry = data.get(str(config_path), {})
    if server_name not in entry:
        return False
    del entry[server_name]
    _save(data)
    return True
