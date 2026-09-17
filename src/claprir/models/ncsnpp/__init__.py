"""Vendored NCSN++ backbone.

Source: https://github.com/sp-uhh/buddy (``networks/ncsnpp.py`` and
``networks/ncsnpp_utils/``), itself derived from Google Research's
score_sde_pytorch (Apache-2.0).  Copied verbatim except for import paths so
that this repository has no dependency on an external checkout.

``utils.py`` and the ``op/`` CUDA extension are deliberately not vendored:
the former imports buddy's SDE package, and the latter is only reachable when
``fir=True``, which this project never enables.
"""
from .backbone import NCSNpp, NCSNppTime, get_window

__all__ = ["NCSNpp", "NCSNppTime", "get_window"]
