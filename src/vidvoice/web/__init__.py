"""Local web interface for vidvoice."""

from __future__ import annotations

from .app import create_app, serve

__all__ = ["create_app", "serve"]
