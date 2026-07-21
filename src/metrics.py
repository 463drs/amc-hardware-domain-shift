"""Metric helpers shared by training and, later, the cross-domain evaluation script.

Deliberately torch-free: the SNR bucketing is pure arithmetic, so a standalone eval script can
import it without pulling in torch. Keeping the bucket-edge rule here (rather than inline in
validate()) is what stops the two from binning SNR differently.
"""

from __future__ import annotations

import math

# RadioML SNRs lie on a 2 dB grid; per-SNR validation accuracy is bucketed to this width.
SNR_BIN_WIDTH_DB = 2


def snr_bucket(snr: float, width: int = SNR_BIN_WIDTH_DB) -> int:
    """Lower edge (dB) of the `width`-dB SNR bin containing `snr`.

    Single source of the bucket-edge rule so validate() and any later cross-domain eval script
    bin SNR identically instead of duplicating the arithmetic.
    """
    return int(math.floor(snr / width) * width)
