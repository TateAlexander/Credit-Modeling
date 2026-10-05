"""Finite-portfolio credit risk models. All probabilities refer to 12 months."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from io import StringIO
import csv
import hashlib
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

MODEL_NAMES = ("Independent", "Vasicek", "CreditRisk+")
MAX_LOANS = 2000


@dataclass(frozen=True)
class Settings:
    trials: int = 10000
    confidence: float = 0.99
    rho: float = 0.15
    factor_cv: float = 0.75
    systematic_weight: float = 0.75
    macro_share: float = 0.5
    pd_multiplier: float = 1.5
    lgd_add: float = 0.10
    loss_budget_pct: float = 0.20
    seed: int = 42
    as_of: str = "2026-10-02"

    def validate(self):
        for key, value in asdict(self).items():
            if key == "as_of":
                try:
                    date.fromisoformat(value)
                except (TypeError, ValueError):
                    raise ValueError("as_of must be a valid YYYY-MM-DD date.") from None
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number.")
        if not isinstance(self.trials, int) or not 1000 <= self.trials <= 50000:
            raise ValueError("trials must be an integer from 1,000 to 50,000.")
        if not isinstance(self.seed, int) or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be an integer from 0 to 4,294,967,295.")
        bounds = {"confidence": (0.90, 0.999), "rho": (0, 0.95),
                  "factor_cv": (0, 2), "systematic_weight": (0, 1),
                  "macro_share": (0, 1), "pd_multiplier": (1, 5),
                  "lgd_add": (0, 1), "loss_budget_pct": (0, 1)}
        for key, (lo, hi) in bounds.items():
            if not lo <= getattr(self, key) <= hi:
                raise ValueError(f"{key} must lie between {lo} and {hi}.")


def load_portfolio(text: str, as_of: str) -> tuple[pd.DataFrame, list[str]]:
    """Strict CSV validation; never silently impute credit assumptions."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > 2_000_000:
        raise ValueError("CSV must be text smaller than 2 MB.")
    try:
        rows = list(csv.reader(StringIO(text.lstrip("\ufeff")), strict=True))
    except csv.Error as exc:
        raise ValueError(f"Invalid CSV: {exc}") from None
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        raise ValueError("CSV needs a header and at least one loan.")
    header = [h.strip().lower() for h in rows[0]]
    if len(set(header)) != len(header) or "" in header:
        raise ValueError("CSV column names must be unique and nonempty.")
    if any(len(row) != len(header) for row in rows[1:]):
        raise ValueError("Every CSV row must have the same number of fields as the header.")
    required = {"loan_id", "ead", "pd_12m", "lgd"}
    missing = required - set(header)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if len(rows) - 1 > MAX_LOANS:
        raise ValueError(f"This local version supports at most {MAX_LOANS:,} borrowers.")
    frame = pd.DataFrame(rows[1:], columns=header).apply(lambda s: s.str.strip())
    if (frame.loan_id == "").any() or frame.loan_id.duplicated().any():
        raise ValueError("loan_id must be nonempty and unique; use one row per borrower.")
    if (frame.loan_id.str.len() > 120).any():
        raise ValueError("loan_id must be at most 120 characters.")
    if "borrower_id" in frame:
        if (frame.borrower_id == "").any() or frame.borrower_id.duplicated().any():
            raise ValueError("Use one row per borrower: aggregate multiple loans before upload.")
    for name in ["ead", "pd_12m", "lgd"]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
        bad = ~np.isfinite(frame[name].to_numpy(dtype=float))
        if bad.any():
            raise ValueError(f"{name} has missing or nonnumeric values at data row {np.flatnonzero(bad)[0]+1}.")
    if (frame.ead < 0).any() or (frame.ead > 1e9).any() or frame.ead.sum() <= 0:
        raise ValueError("ead must be between 0 and $1 billion, with a positive portfolio total.")
    for name in ["pd_12m", "lgd"]:
        if ((frame[name] < 0) | (frame[name] > 1)).any():
            raise ValueError(f"{name} must be a decimal from 0 to 1 (12% = 0.12).")
    warnings = []
    if "sector" not in frame:
        frame["sector"] = "Whole portfolio"
        warnings.append("No sector column: CreditRisk+ uses a single shared sector.")
    frame["sector"] = frame.sector.replace("", "Unassigned")
    if (frame.sector.str.len() > 120).any():
        raise ValueError("sector labels must be at most 120 characters.")
    if "days_past_due" in frame:
        numeric = pd.to_numeric(frame.days_past_due.where(frame.days_past_due != ""), errors="coerce")
        bad = (frame.days_past_due != "") & (~np.isfinite(numeric) | (numeric < 0) | (numeric % 1 != 0))
        if bad.any():
            raise ValueError("days_past_due must be a nonnegative integer or blank.")
        frame["days_past_due"] = numeric
    else:
        frame["days_past_due"] = np.nan
    if frame.days_past_due.isna().any():
        warnings.append("Some DPD values are unavailable; delinquency rates use only accounts with known DPD.")
    if "origination_date" in frame:
        raw = frame.origination_date.where(frame.origination_date != "")
        dates = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
        if (raw.notna() & dates.isna()).any() or (dates > pd.Timestamp(as_of)).any():
            raise ValueError("origination_date must be YYYY-MM-DD, on or before the as-of date, or blank.")
        frame["origination_date"] = dates
    else:
        frame["origination_date"] = pd.NaT
    if frame.pd_12m.max() > 0.10:
        warnings.append("Some PDs exceed 10%. CreditRisk+ uses a Poisson count approximation; repeated default counts can materially distort high-PD loan results. Favor the bounded Vasicek results for these loans.")
    if frame.pd_12m.eq(1).any():
        warnings.append("PD=1 is treated as certain default by the Bernoulli models. Already defaulted accounts should be assessed separately for recovery risk.")
    return frame, warnings


def demo_csv(n: int = 400, seed: int = 8) -> str:
    rng = np.random.default_rng(seed)
    sectors = rng.choice(["North branch", "Central branch", "South branch"], n, p=[0.3, 0.45, 0.25])
    p = np.clip(rng.beta(2.0, 18, n) + (sectors == "South branch") * 0.025, 0.005, 0.40)
    dpd = rng.choice([0, 15, 35, 65, 95], n, p=[0.77, 0.10, 0.07, 0.04, 0.02])
    p = np.clip(p + (dpd >= 30) * 0.12, 0, 0.55)
    origin = pd.Timestamp("2026-10-01") - pd.to_timedelta(rng.integers(30, 1000, n), unit="D")
    return pd.DataFrame({"loan_id": [f"DEMO-{i+1:04d}" for i in range(n)],
        "ead": np.round(rng.uniform(3500, 24000, n), 2), "pd_12m": np.round(p, 4),
        "lgd": np.round(rng.uniform(0.35, 0.75, n), 4), "sector": sectors,
        "days_past_due": dpd, "origination_date": origin.strftime("%Y-%m-%d")}).to_csv(index=False)


def simulate(frame: pd.DataFrame, settings: Settings, model: str) -> np.ndarray:
    """Chunked simulation with local RNG and reproducible, separate model streams."""
    settings.validate()
    if model not in MODEL_NAMES:
        raise ValueError("Unknown model.")
    p = frame.pd_12m.to_numpy(dtype=float)
    severity = (frame.ead * frame.lgd).to_numpy(dtype=float)
    sectors, labels = pd.factorize(frame.sector, sort=True)
    rng = np.random.default_rng(np.random.SeedSequence([settings.seed, MODEL_NAMES.index(model)]))
    losses = np.empty(settings.trials)
    threshold = norm.ppf(p)
    for start in range(0, settings.trials, 256):
        size = min(256, settings.trials - start)
        if model == "Independent":
            counts = rng.random((size, len(p))) < p
        elif model == "Vasicek":
            z = rng.standard_normal((size, 1))
            conditional = norm.cdf((threshold - np.sqrt(settings.rho) * z) / np.sqrt(1-settings.rho))
            counts = rng.random((size, len(p))) < conditional
        else:
            if settings.factor_cv == 0:
                factor = np.ones((size, len(p)))
            else:
                variance = settings.factor_cv**2
                macro = rng.gamma(1/variance, variance, (size, 1))
                sector = rng.gamma(1/variance, variance, (size, len(labels)))
                factor = ((1-settings.systematic_weight) + settings.systematic_weight *
                    (settings.macro_share * macro + (1-settings.macro_share) * sector[:, sectors]))
            # Original compound-Poisson approximation: counts can exceed one.
            counts = rng.poisson(p * factor)
        losses[start:start+size] = np.sum(counts * severity, axis=1)
    return losses


def summarize(losses: np.ndarray, alpha: float, analytic_el: float, budget: float) -> dict:
    var = float(np.quantile(losses, alpha, method="inverted_cdf"))
    # Fractional allocation at the quantile handles discrete atom mass correctly.
    es = var + float(np.maximum(losses-var, 0).sum()) / (len(losses)*(1-alpha))
    return {"expected_loss": analytic_el, "simulated_mean": float(losses.mean()),
            "mean_mc_se": float(losses.std(ddof=1)/np.sqrt(len(losses))),
            "loss_std": float(losses.std(ddof=1)), "var": var, "es": es,
            "capital_buffer": max(0.0, var-analytic_el),
            "budget_exceedance": float(np.mean(losses > budget)),
            "simulated_max": float(losses.max())}


def creditrisk_variance(frame: pd.DataFrame, settings: Settings) -> float:
    """Exact variance of the implemented compound-Poisson/gamma mixture."""
    severity = frame.ead * frame.lgd
    el = frame.pd_12m * severity
    sums = el.groupby(frame.sector).sum()
    systematic = settings.factor_cv**2 * settings.systematic_weight**2 * (
        settings.macro_share**2 * el.sum()**2 + (1-settings.macro_share)**2 * (sums**2).sum())
    return float((frame.pd_12m * severity**2).sum() + systematic)


def analyze(csv_text: str, settings: Settings) -> dict:
    settings.validate()
    frame, warnings = load_portfolio(csv_text, settings.as_of)
    tail_count = settings.trials * (1-settings.confidence)
    if tail_count < 100:
        warnings.append(f"Only about {tail_count:.0f} tail simulations support this quantile. VaR and ES may be unstable; increase trials or reduce confidence.")
    total = float(frame.ead.sum())
    frame["expected_loss"] = frame.ead * frame.lgd * frame.pd_12m
    stress = frame.copy()
    stress["pd_12m"] = np.minimum(1, frame.pd_12m * settings.pd_multiplier)
    stress["lgd"] = np.minimum(1, frame.lgd + settings.lgd_add)
    stress["expected_loss"] = stress.ead * stress.lgd * stress.pd_12m
    base_el = float(frame.expected_loss.sum())
    stress_el = float(stress.expected_loss.sum())
    histograms, results = {}, []
    budget = total * settings.loss_budget_pct
    for name in MODEL_NAMES:
        losses = simulate(frame, settings, name)
        stressed = simulate(stress, settings, name)
        edges = np.linspace(0, max(float(losses.max()), float(stressed.max()), 1)*1.001, 45)
        histograms[name] = {"centers": ((edges[:-1]+edges[1:])/2).tolist(),
                            "base": (np.histogram(losses, edges)[0]/settings.trials).tolist(),
                            "stress": (np.histogram(stressed, edges)[0]/settings.trials).tolist()}
        results.append({"model": name, "base": summarize(losses, settings.confidence, base_el, budget),
                        "stress": summarize(stressed, settings.confidence, stress_el, budget)})
    if stress.pd_12m.max() > 0.1 and frame.pd_12m.max() <= 0.1:
        warnings.append("Stressed PDs exceed 10%; CreditRisk+ approximation limitations apply to the stress scenario.")
    if any(r["base"]["simulated_max"] > float((frame.ead*frame.lgd).sum()) or
           r["stress"]["simulated_max"] > float((stress.ead*stress.lgd).sum()) for r in results if r["model"] == "CreditRisk+"):
        warnings.append("CreditRisk+ produced losses above the physical maximum loss. This is possible under its uncapped Poisson approximation.")
    groups = frame.groupby("sector", sort=True).agg(accounts=("loan_id", "count"),
                ead=("ead", "sum"), expected_loss=("expected_loss", "sum"))
    groups["loss_share"] = groups.expected_loss/base_el if base_el else 0.0
    accounts = frame[["loan_id", "sector", "ead", "pd_12m", "lgd", "expected_loss"]].copy()
    accounts["stress_expected_loss"] = stress.expected_loss
    accounts["loss_share"] = accounts.expected_loss/base_el if base_el else 0.0
    accounts = accounts.sort_values("expected_loss", ascending=False)
    known = frame[frame.days_past_due.notna()]
    delinquency = []
    for cutoff in [30, 60, 90]:
        late = known[known.days_past_due >= cutoff]
        delinquency.append({"dpd": cutoff, "accounts": len(late),
            "account_rate": len(late)/len(known) if len(known) else None,
            "ead_rate": float(late.ead.sum()/known.ead.sum()) if known.ead.sum() else None})
    vintages = []
    dated = frame[frame.origination_date.notna()].copy()
    if len(dated):
        dated["vintage"] = dated.origination_date.dt.to_period("Q").astype(str)
        for label, group in dated.groupby("vintage", sort=True):
            obs = group[group.days_past_due.notna()]
            vintages.append({"vintage": label, "accounts": len(group), "ead": float(group.ead.sum()),
                "dpd_known": len(obs), "dpd30_rate": float((obs.days_past_due >= 30).mean()) if len(obs) else None,
                "modeled_loss_rate": float(group.expected_loss.sum()/group.ead.sum()) if group.ead.sum() else None})
    top_n = max(1, math.ceil(len(frame)*0.1))
    return {"settings": asdict(settings), "input_sha256": hashlib.sha256(csv_text.encode()).hexdigest(),
        "warnings": warnings, "overview": {"accounts": len(frame), "ead": total,
            "expected_loss": base_el, "loss_rate": base_el/total,
            "expected_defaults": float(frame.pd_12m.sum()),
            "exposure_weighted_pd": float((frame.pd_12m*frame.ead).sum()/total),
            "stress_expected_loss": stress_el, "loss_budget": budget,
            "top_decile_el_share": float(accounts.expected_loss.iloc[:top_n].sum()/base_el) if base_el else 0,
            "known_dpd_accounts": len(known), "missing_vintage_accounts": int(frame.origination_date.isna().sum())},
        "models": results, "histograms": histograms, "sectors": groups.reset_index().to_dict("records"),
        "accounts": accounts.to_dict("records"), "delinquency": delinquency, "vintages": vintages,
        "diagnostics": {"creditrisk_theoretical_std": math.sqrt(creditrisk_variance(frame, settings)),
            "independent_theoretical_std": float(np.sqrt(((frame.ead*frame.lgd)**2 * frame.pd_12m*(1-frame.pd_12m)).sum()))}}


def safe_csv(rows: list[dict]) -> str:
    """Protect exported identifier cells from spreadsheet formula interpretation."""
    if not rows:
        return ""
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "'"+v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v
                         for key, v in row.items()})
    return stream.getvalue()
