#!/usr/bin/env python3
"""Lightweight Binance UM fundingRate parser for the dual-cycle study.

Reads data/binance_vision/funding/<SYMBOL>/<SYMBOL>-fundingRate-YYYY-MM.zip
(sha256-verified at download) and emits the exact schema the frozen engine
expects (research_trend_baseline.load_funding output):
    slot_min   settlement minute (UTC epoch-minute, calc_time snapped to the
               nearest hour — observed jitter is a few milliseconds)
    rate       funding rate at settlement
    settle_px  last 1m close at/before the minute preceding settlement
               (same convention as the OKX stage; last price, not mark —
               stated simplification, identical on both legs of comparison)

Sign convention downstream is the engine's: positive rate -> long pays
(Binance and OKX share this convention). Settlement timing comes from the
file's calc_time, NOT from an assumed 00/08/16 grid, so any 4h-interval
episodes are handled automatically; interval values are reported for the
trust check.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUND_ROOT = PROJECT_ROOT / "data" / "binance_vision" / "funding"


def load_funding_binance(b_symbol: str, m1: pd.DataFrame) -> pd.DataFrame:
    """Funding schedule with settle price from 1m closes (engine schema)."""
    frames = []
    for zp in sorted((FUND_ROOT / b_symbol).glob(f"{b_symbol}-fundingRate-*.zip")):
        with zipfile.ZipFile(zp) as z:
            with z.open(z.namelist()[0]) as f:
                frames.append(pd.read_csv(io.BytesIO(f.read())))
    fr = pd.concat(frames, ignore_index=True)
    fr = fr.drop_duplicates("calc_time").sort_values("calc_time").reset_index(drop=True)
    # snap ms-jittered calc_time to the nearest hour -> epoch minute
    fr["slot_min"] = ((fr["calc_time"] / 3_600_000).round().astype("int64") * 60)
    fr["rate"] = pd.to_numeric(fr["last_funding_rate"])
    px = m1.set_index("min_utc")["close"]
    idx = px.index.searchsorted(fr["slot_min"].to_numpy() - 1, side="right") - 1
    ok = idx >= 0
    fr = fr.loc[ok].copy()
    fr["settle_px"] = px.to_numpy()[idx[ok]]
    fr["interval_h"] = pd.to_numeric(fr["funding_interval_hours"])
    return fr[["slot_min", "rate", "settle_px", "interval_h"]].reset_index(drop=True)
