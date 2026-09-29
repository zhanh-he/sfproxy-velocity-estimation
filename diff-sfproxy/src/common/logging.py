"""Logging helpers for """

from __future__ import annotations

import logging
from typing import Mapping, Optional

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    """A rank-aware logger compatible with Lightning distributed utilities."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__(logger=logging.getLogger(name), extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        if not self.isEnabledFor(level):
            return

        msg, kwargs = self.process(msg, kwargs)
        current_rank = getattr(rank_zero_only, "rank", None)
        if current_rank is None:
            raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")

        msg = rank_prefixed_message(msg, current_rank)
        if self.rank_zero_only:
            if current_rank == 0:
                self.logger.log(level, msg, *args, **kwargs)
            return

        if rank is None or current_rank == rank:
            self.logger.log(level, msg, *args, **kwargs)
