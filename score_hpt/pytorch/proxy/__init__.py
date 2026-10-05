"""Differentiable supervision backends for weakly supervised velocity training."""

from .objective import build_proxy_objective

__all__ = ["build_proxy_objective"]
