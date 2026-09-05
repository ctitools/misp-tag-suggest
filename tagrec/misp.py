"""Shared MISP client construction without credential logging."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import urllib3
from dotenv import load_dotenv
from pymisp import PyMISP


def env_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True, slots=True)
class MISPSettings:
    url: str
    key: str
    verify_tls: bool


def load_misp_settings(env_path: str | Path = ".env") -> MISPSettings:
    load_dotenv(Path(env_path))
    url = os.environ.get("MISP_BASE_URL")
    key = os.environ.get("MISP_API_KEY")
    verify_tls = env_bool(os.environ.get("MISP_VERIFY_SSL"), default=True)
    if not url or not key:
        raise RuntimeError(f"MISP_BASE_URL or MISP_API_KEY missing from {env_path}")
    if not verify_tls:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return MISPSettings(url.rstrip("/"), key, verify_tls)


def create_misp_client(env_path: str | Path = ".env") -> PyMISP:
    settings = load_misp_settings(env_path)
    return PyMISP(settings.url, settings.key, ssl=settings.verify_tls)
