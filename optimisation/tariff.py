"""Tariff adapter.

The optimiser never hard-codes a tariff. It asks a ``Tariff`` object two separate things:

A. **Billing** – ``bill(hourly_kwh, timestamps)``: what the household pays for this 24-hour profile.
B. **Scheduling signal** – ``scheduling_rates(timestamps)``: the ₹/kWh price per hour that the MILP
   sees. Only an hour-dependent signal can make *when* energy is used matter for cost.

``is_time_varying()`` reports whether B changes across hours. If it does not (flat or slab), the
cost term in the objective is constant, shifting cannot change the bill, and the engine reports
``cost_responsive = False`` instead of claiming cost savings.

No official tariff lives in this module. ``illustrative_tou_tariff()`` reproduces the Review-3
illustrative scenario and is labelled as such; it is for tests and synthetic validation only.
The official tariff will be supplied as a dict/JSON (``tariff_from_dict``) once confirmed.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ILLUSTRATIVE_LABEL = "ILLUSTRATIVE TARIFF – NOT OFFICIAL – TEST/SYNTHETIC VALIDATION ONLY"


def _clock_hours(timestamps) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(timestamps)).hour.to_numpy()


@dataclass
class Tariff(ABC):
    label: str = ILLUSTRATIVE_LABEL
    is_official: bool = False
    source: str = ""
    effective_from: str | None = None
    effective_to: str | None = None
    fixed_charge_per_horizon: float = 0.0
    currency: str = "INR"
    tariff_type: str = field(init=False, default="abstract")

    def __post_init__(self):
        if self.is_official and not self.source:
            raise ValueError("An official tariff must cite its source document")
        if self.fixed_charge_per_horizon < 0:
            raise ValueError("fixed_charge_per_horizon must be >= 0")

    @abstractmethod
    def scheduling_rates(self, timestamps) -> np.ndarray:
        """₹/kWh per hour seen by the optimiser (B)."""

    @abstractmethod
    def energy_charge(self, hourly_kwh: np.ndarray, timestamps) -> float:
        """Energy charge for the 24-hour profile (part of A)."""

    def bill(self, hourly_kwh, timestamps) -> float:
        """Total bill for the horizon: energy charge + fixed charge (A)."""
        return float(self.energy_charge(np.asarray(hourly_kwh, float), timestamps) + self.fixed_charge_per_horizon)

    def is_time_varying(self, timestamps) -> bool:
        rates = self.scheduling_rates(timestamps)
        return bool(np.ptp(rates) > 1e-12)

    def describe(self) -> dict[str, Any]:
        return {"tariff_type": self.tariff_type, "label": self.label, "is_official": self.is_official,
                "source": self.source, "effective_from": self.effective_from, "effective_to": self.effective_to,
                "fixed_charge_per_horizon": self.fixed_charge_per_horizon, "currency": self.currency}


@dataclass
class FlatTariff(Tariff):
    rate: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.tariff_type = "flat"
        if self.rate < 0:
            raise ValueError("rate must be >= 0")

    def scheduling_rates(self, timestamps) -> np.ndarray:
        return np.full(len(_clock_hours(timestamps)), float(self.rate))

    def energy_charge(self, hourly_kwh, timestamps) -> float:
        return float(np.sum(hourly_kwh) * self.rate)

    def describe(self):
        return {**super().describe(), "rate": self.rate}


@dataclass
class TouTariff(Tariff):
    """Time-of-use: one ₹/kWh rate per clock hour (24 values, index = clock hour)."""

    hourly_rates: list[float] = field(default_factory=list)
    period_names: list[str] | None = None

    def __post_init__(self):
        super().__post_init__()
        self.tariff_type = "tou"
        if len(self.hourly_rates) != 24:
            raise ValueError("TouTariff needs exactly 24 hourly rates (index = clock hour)")
        if min(self.hourly_rates) < 0:
            raise ValueError("Rates must be >= 0")

    @classmethod
    def from_periods(cls, periods: list[dict[str, Any]], **meta) -> "TouTariff":
        """periods: [{"name":..., "start_hour":h0, "end_hour":h1 (exclusive, may wrap), "rate":r}, ...]."""
        rates: list[float | None] = [None] * 24
        names: list[str | None] = [None] * 24
        for p in periods:
            h = p["start_hour"] % 24
            end = p["end_hour"] % 24
            while True:
                if rates[h] is not None:
                    raise ValueError(f"ToU periods overlap at hour {h}")
                rates[h], names[h] = float(p["rate"]), p.get("name", "")
                h = (h + 1) % 24
                if h == end:
                    break
        if any(r is None for r in rates):
            raise ValueError("ToU periods must cover all 24 hours")
        return cls(hourly_rates=rates, period_names=names, **meta)

    def scheduling_rates(self, timestamps) -> np.ndarray:
        return np.asarray(self.hourly_rates, float)[_clock_hours(timestamps)]

    def energy_charge(self, hourly_kwh, timestamps) -> float:
        return float(np.sum(hourly_kwh * self.scheduling_rates(timestamps)))

    def describe(self):
        return {**super().describe(), "hourly_rates": list(self.hourly_rates), "period_names": self.period_names}


@dataclass
class SlabTariff(Tariff):
    """Consumption-slab tariff over a billing cycle.

    ``slabs``: ordered [{"upto_kwh": 100, "rate": 0.0}, {"upto_kwh": 200, "rate": 2.25}, ..., {"upto_kwh": None, "rate": x}]
    ``telescopic``: True  → each slab's rate applies only to the energy inside that slab.
                    False → the whole cycle's energy is charged at the rate of the slab it ends in.
    ``prior_cycle_kwh``: energy already consumed in the current billing cycle before this horizon.

    The horizon's energy charge is the *increment* in the cycle bill caused by this day's energy.
    The bill depends only on total kWh, not on *when* it is used, so shifting appliances inside the
    day cannot change it → ``is_time_varying()`` is False and cost savings are never claimed.
    """

    slabs: list[dict[str, Any]] = field(default_factory=list)
    telescopic: bool = True
    prior_cycle_kwh: float = 0.0
    billing_cycle_days: int | None = None

    def __post_init__(self):
        super().__post_init__()
        self.tariff_type = "slab"
        if not self.slabs:
            raise ValueError("SlabTariff needs at least one slab")
        limits = [s["upto_kwh"] for s in self.slabs[:-1]]
        if any(l is None for l in limits) or limits != sorted(limits):
            raise ValueError("Slab upper limits must be increasing; only the last may be None")
        if self.prior_cycle_kwh < 0:
            raise ValueError("prior_cycle_kwh must be >= 0")

    def cycle_charge(self, cycle_kwh: float) -> float:
        if self.telescopic:
            cost, lower = 0.0, 0.0
            for s in self.slabs:
                upper = np.inf if s["upto_kwh"] is None else float(s["upto_kwh"])
                if cycle_kwh > lower:
                    cost += (min(cycle_kwh, upper) - lower) * float(s["rate"])
                lower = upper
            return float(cost)
        for s in self.slabs:
            if s["upto_kwh"] is None or cycle_kwh <= s["upto_kwh"]:
                return float(cycle_kwh * s["rate"])
        return float(cycle_kwh * self.slabs[-1]["rate"])

    def energy_charge(self, hourly_kwh, timestamps) -> float:
        total = float(np.sum(hourly_kwh))
        return self.cycle_charge(self.prior_cycle_kwh + total) - self.cycle_charge(self.prior_cycle_kwh)

    def scheduling_rates(self, timestamps) -> np.ndarray:
        # Hour-independent signal: the average marginal rate at the current cycle position.
        n = len(_clock_hours(timestamps))
        probe = 1.0
        marginal = self.cycle_charge(self.prior_cycle_kwh + probe) - self.cycle_charge(self.prior_cycle_kwh)
        return np.full(n, marginal / probe)

    def describe(self):
        return {**super().describe(), "slabs": self.slabs, "telescopic": self.telescopic,
                "prior_cycle_kwh": self.prior_cycle_kwh, "billing_cycle_days": self.billing_cycle_days}


# --------------------------------------------------------------------------- factories
_META_KEYS = {"label", "is_official", "source", "effective_from", "effective_to", "fixed_charge_per_horizon", "currency"}


def tariff_from_dict(config: dict[str, Any]) -> Tariff:
    """Build a tariff from the JSON format documented in TARIFF_INTERFACE.md."""
    kind = config["tariff_type"].lower()
    meta = {k: v for k, v in config.items() if k in _META_KEYS}
    if kind == "flat":
        return FlatTariff(rate=config["rate"], **meta)
    if kind == "tou":
        if "hourly_rates" in config:
            return TouTariff(hourly_rates=config["hourly_rates"], period_names=config.get("period_names"), **meta)
        return TouTariff.from_periods(config["periods"], **meta)
    if kind == "slab":
        return SlabTariff(slabs=config["slabs"], telescopic=config.get("telescopic", True),
                          prior_cycle_kwh=config.get("prior_cycle_kwh", 0.0),
                          billing_cycle_days=config.get("billing_cycle_days"), **meta)
    raise ValueError(f"Unknown tariff_type: {kind}")


def tariff_from_json(path: str | Path) -> Tariff:
    return tariff_from_dict(json.loads(Path(path).read_text()))


def illustrative_tou_tariff() -> TouTariff:
    """The Review-3 illustrative ToU scenario (4.0 / 6.0 / 9.0 / 4.5 ₹/kWh). NOT OFFICIAL."""
    return TouTariff.from_periods([
        {"name": "Off-peak", "start_hour": 0, "end_hour": 6, "rate": 4.0},
        {"name": "Normal", "start_hour": 6, "end_hour": 18, "rate": 6.0},
        {"name": "Peak", "start_hour": 18, "end_hour": 22, "rate": 9.0},
        {"name": "Off-peak (late)", "start_hour": 22, "end_hour": 0, "rate": 4.5},
    ], label=ILLUSTRATIVE_LABEL + " (Review-3 Indian ToU scenario)")
