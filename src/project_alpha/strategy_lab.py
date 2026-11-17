"""Torneo delle strategie di allocazione (pre-registrato in ``docs/TORNEO.md``).

Ogni strategia è una funzione che, dato lo storico delle chiusure fino al
giorno della decisione, restituisce i pesi target (long-only, somma ≤ 1; il
resto è liquidità). Il motore comune applica a tutte le stesse regole:

- decisioni sull'ultima chiusura di ogni mese;
- pesi effettivi alla chiusura del giorno di borsa successivo;
- deriva dei pesi con i prezzi tra un ribilanciamento e l'altro;
- costi proporzionali al controvalore scambiato.

Le regole e i parametri non si modificano guardando i risultati: una
variante è una nuova strategia e una nuova prova.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

from project_alpha.evaluation import TRADING_DAYS, deflated_sharpe_ratio, performance_summary

CORE_WEIGHTS: dict[str, float] = {
    "SPY": 0.20,
    "QQQ": 0.10,
    "IWM": 0.05,
    "EFA": 0.15,
    "EEM": 0.10,
    "IEF": 0.15,
    "LQD": 0.10,
    "TLT": 0.05,
    "GLD": 0.05,
    "VNQ": 0.05,
}
UNIVERSE = tuple(CORE_WEIGHTS)
LOOKBACK_12M = 252
VOL_LOOKBACK = 63
SUBPERIODS = (
    ("2017-01-01", "2019-12-31"),
    ("2020-01-01", "2022-12-31"),
    ("2023-01-01", "2100-12-31"),
)

Weights = dict[str, float]
Strategy = Callable[[pd.DataFrame], Weights]


# ----------------------------------------------------------------------
# Strategie (parametri fissi, vedi docs/TORNEO.md)
# ----------------------------------------------------------------------


def _return_12m(history: pd.DataFrame) -> pd.Series:
    return history.iloc[-1] / history.iloc[-1 - LOOKBACK_12M] - 1.0


def attuale_vol_target(history: pd.DataFrame, target: float = 0.08) -> Weights:
    returns = history[list(UNIVERSE)].pct_change().iloc[-VOL_LOOKBACK:]
    core = sum(returns[s] * w for s, w in CORE_WEIGHTS.items())
    vol = float(core.std(ddof=1) * sqrt(TRADING_DAYS))
    exposure = 1.0 if vol <= 0 else min(1.0, target / vol)
    return {s: w * exposure for s, w in CORE_WEIGHTS.items()}


def core_100(history: pd.DataFrame) -> Weights:
    return dict(CORE_WEIGHTS)


def pesi_uguali(history: pd.DataFrame) -> Weights:
    return {s: 1.0 / len(UNIVERSE) for s in UNIVERSE}


def momentum_assoluto_12m(history: pd.DataFrame) -> Weights:
    momentum = _return_12m(history)
    return {s: w for s, w in CORE_WEIGHTS.items() if momentum[s] > 0}


def media_mobile_10m(history: pd.DataFrame) -> Weights:
    monthly = history.groupby(history.index.to_period("M")).tail(1)
    # L'ultima riga è il giorno della decisione (fine mese).
    window = monthly.iloc[-10:]
    last = monthly.iloc[-1]
    return {s: w for s, w in CORE_WEIGHTS.items() if last[s] > window[s].mean()}


def momentum_relativo_top3(history: pd.DataFrame) -> Weights:
    momentum = _return_12m(history)[list(UNIVERSE)].sort_values(ascending=False)
    return {s: 1.0 / 3.0 for s in momentum.index[:3]}


def dual_momentum(history: pd.DataFrame) -> Weights:
    momentum = _return_12m(history)
    if momentum["SPY"] > 0:
        return {"SPY": 1.0} if momentum["SPY"] >= momentum["EFA"] else {"EFA": 1.0}
    return {"IEF": 1.0}


def inverso_volatilita(history: pd.DataFrame) -> Weights:
    returns = history[list(UNIVERSE)].pct_change().iloc[-VOL_LOOKBACK:]
    inverse = 1.0 / returns.std(ddof=1)
    weights = inverse / inverse.sum()
    return {s: float(w) for s, w in weights.items()}


def portafoglio_permanente(history: pd.DataFrame) -> Weights:
    return {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25}


LOOKBACK_20D = 20


def _return_20d(history: pd.DataFrame) -> pd.Series:
    return history.iloc[-1] / history.iloc[-1 - LOOKBACK_20D] - 1.0


def direzione_20g(history: pd.DataFrame) -> Weights:
    momentum = _return_20d(history)
    return {s: w for s, w in CORE_WEIGHTS.items() if momentum[s] > 0}


def inversione_20g(history: pd.DataFrame) -> Weights:
    momentum = _return_20d(history)
    return {s: w for s, w in CORE_WEIGHTS.items() if momentum[s] < 0}


def migliori3_20g(history: pd.DataFrame) -> Weights:
    ranked = _return_20d(history)[list(UNIVERSE)].sort_values(ascending=False)
    return {s: 1.0 / 3.0 for s in ranked.index[:3]}


def peggiori3_20g(history: pd.DataFrame) -> Weights:
    ranked = _return_20d(history)[list(UNIVERSE)].sort_values(ascending=True)
    return {s: 1.0 / 3.0 for s in ranked.index[:3]}


REFERENCE = "R_attuale_vol_target"
STRATEGIES: dict[str, Strategy] = {
    REFERENCE: attuale_vol_target,
    "C1_core_100": core_100,
    "C2_pesi_uguali": pesi_uguali,
    "C3_momentum_assoluto_12m": momentum_assoluto_12m,
    "C4_media_mobile_10m": media_mobile_10m,
    "C5_momentum_relativo_top3": momentum_relativo_top3,
    "C6_dual_momentum": dual_momentum,
    "C7_inverso_volatilita": inverso_volatilita,
    "C8_portafoglio_permanente": portafoglio_permanente,
}


STRATEGIES_20D: dict[str, Strategy] = {
    "R_attuale_vol_target": attuale_vol_target,
    "D1_direzione_20g": direzione_20g,
    "D2_inversione_20g": inversione_20g,
    "D3_migliori3_20g": migliori3_20g,
    "D4_peggiori3_20g": peggiori3_20g,
}

# ----------------------------------------------------------------------
# Motore comune
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LabResult:
    returns: pd.Series
    weights: pd.DataFrame  # pesi decisi per ogni data di decisione
    turnover: pd.Series  # turnover per giorno di ribilanciamento


def load_closes(bars: pd.DataFrame) -> pd.DataFrame:
    data = bars.copy()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    closes = data.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
    closes = closes[list(UNIVERSE)].sort_index().dropna()
    closes.index = closes.index.tz_convert("America/New_York").tz_localize(None).normalize()
    return closes


def decision_dates(
    index: pd.DatetimeIndex, *, min_history: int = LOOKBACK_12M + 1
) -> list[pd.Timestamp]:
    """Ultimo giorno di borsa di ogni mese con almeno ``min_history`` giorni di storia."""
    positions = pd.Series(np.arange(len(index)), index=index)
    month_ends = positions.groupby(index.to_period("M")).max()
    return [index[i] for i in month_ends if i >= min_history and i < len(index) - 1]


def day20_dates(
    index: pd.DatetimeIndex, *, min_history: int = LOOKBACK_12M + 1
) -> list[pd.Timestamp]:
    """Ultima seduta con giorno del mese ≤ 20, per ogni mese (torneo 2)."""
    positions = pd.Series(np.arange(len(index)), index=index)
    eligible = positions[index.day <= 20]
    picks = eligible.groupby(eligible.index.to_period("M")).max()
    return [index[i] for i in picks if i >= min_history and i < len(index) - 1]


def run_strategy(
    closes: pd.DataFrame,
    strategy: Strategy,
    *,
    start: pd.Timestamp,
    cost_per_side: float = 0.001,
    cash_rate_annual: float = 0.0,
    decision_rule: Callable[[pd.DatetimeIndex], list[pd.Timestamp]] | None = None,
) -> LabResult:
    """Simula la strategia dal giorno ``start`` (escluso) alla fine dei dati."""
    asset_returns = closes.pct_change().fillna(0.0)
    daily_cash = (1.0 + cash_rate_annual) ** (1.0 / TRADING_DAYS) - 1.0
    decisions = set((decision_rule or decision_dates)(closes.index))
    index = closes.index

    holdings = pd.Series(0.0, index=closes.columns)  # pesi correnti (sull'equity)
    pending: Weights | None = None
    out_returns: dict[pd.Timestamp, float] = {}
    turnover: dict[pd.Timestamp, float] = {}
    decided: dict[pd.Timestamp, Weights] = {}

    for position, day in enumerate(index):
        if day <= start:
            if day == start and day in decisions:
                pending = strategy(closes.iloc[: position + 1])
                decided[day] = pending
            continue

        # Rendimento del giorno con i pesi di ieri (la liquidità rende daily_cash).
        cash_weight = 1.0 - holdings.sum()
        gross = float((holdings * asset_returns.loc[day]).sum()) + cash_weight * daily_cash
        growth = 1.0 + gross
        # Deriva dei pesi con i prezzi.
        holdings = holdings * (1.0 + asset_returns.loc[day]) / growth
        cost = 0.0

        # Ribilanciamento deciso ieri: eseguito alla chiusura di oggi.
        if pending is not None:
            target = pd.Series(pending, dtype=float).reindex(closes.columns).fillna(0.0)
            traded = float((target - holdings).abs().sum())
            cost = traded * cost_per_side
            turnover[day] = traded
            holdings = target
            pending = None

        out_returns[day] = (1.0 + gross) * (1.0 - cost) - 1.0

        if day in decisions:
            pending = strategy(closes.iloc[: position + 1])
            decided[day] = pending

    weights = pd.DataFrame(decided).T.reindex(columns=closes.columns).fillna(0.0)
    return LabResult(
        returns=pd.Series(out_returns, dtype=float),
        weights=weights,
        turnover=pd.Series(turnover, dtype=float),
    )


# ----------------------------------------------------------------------
# Valutazione
# ----------------------------------------------------------------------


def _sharpe(values: np.ndarray) -> float:
    std = values.std(ddof=1)
    return float(values.mean() / std * sqrt(TRADING_DAYS)) if std > 0 else float("nan")


def bootstrap_sharpe_difference(
    candidate: pd.Series,
    reference: pd.Series,
    *,
    block: int = 21,
    samples: int = 5000,
    seed: int = 42,
) -> float:
    """p-value unilaterale che lo Sharpe del candidato NON superi quello del riferimento.

    Bootstrap a blocchi circolari sui rendimenti giornalieri appaiati.
    """
    joined = pd.concat([candidate, reference], axis=1, join="inner").to_numpy()
    n = len(joined)
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / block))
    worse = 0

    for _ in range(samples):
        starts = rng.integers(0, n, size=blocks)
        rows = ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        sample = joined[rows]
        if _sharpe(sample[:, 0]) - _sharpe(sample[:, 1]) <= 0:
            worse += 1

    return (worse + 1) / (samples + 1)


def evaluate_tournament(
    results: Mapping[str, LabResult],
    *,
    reference: str = REFERENCE,
    samples: int = 5000,
    total_trials: int | None = None,
) -> pd.DataFrame:
    """Tabella del torneo con i criteri pre-registrati.

    ``total_trials``: prove complessive sugli stessi dati (correzione di
    Bonferroni e DSR); di default i candidati di questo torneo.
    """
    candidates = [name for name in results if name != reference]
    trials = total_trials or len(candidates)
    alpha = 0.05 / max(trials, 1)
    ref = results[reference].returns
    ref_summary = performance_summary(ref)
    daily_sharpes = [r.returns.mean() / r.returns.std(ddof=1) for r in results.values()]
    trials_std = float(np.std(daily_sharpes, ddof=1))

    rows = []
    for name, result in results.items():
        returns = result.returns
        summary = performance_summary(returns)
        years = len(returns) / TRADING_DAYS
        row: dict[str, object] = {
            "strategia": name,
            "cagr": summary["cagr"],
            "volatilita": summary["annualized_volatility"],
            "sharpe": summary["annualized_sharpe"],
            "max_drawdown": summary["max_drawdown"],
            "turnover_annuo": float(result.turnover.sum() / years),
            "esposizione_media": float(result.weights.sum(axis=1).mean()),
            "peso_max": float(result.weights.max().max()),
            "dsr": deflated_sharpe_ratio(returns, trials=trials + 1, trials_sharpe_std=trials_std),
        }

        wins = 0
        for index, (start, end) in enumerate(SUBPERIODS, start=1):
            part = returns.loc[start:end]
            ref_part = ref.loc[start:end]
            sharpe_part = _sharpe(part.to_numpy())
            row[f"sharpe_periodo_{index}"] = sharpe_part
            wins += int(sharpe_part > _sharpe(ref_part.to_numpy()))
        row["sottoperiodi_vinti"] = wins

        if name == reference:
            row.update({"p_vs_riferimento": None, "esito": "riferimento"})
        else:
            p_value = bootstrap_sharpe_difference(returns, ref, samples=samples)
            better = summary["annualized_sharpe"] > ref_summary["annualized_sharpe"]
            dd_ok = summary["max_drawdown"] >= ref_summary["max_drawdown"]
            robust = wins >= 2
            row["p_vs_riferimento"] = p_value
            if better and p_value <= alpha and robust and dd_ok:
                row["esito"] = "BATTE IL RIFERIMENTO"
            elif better and robust and dd_ok:
                row["esito"] = "promettente"
            else:
                row["esito"] = "scartata"
        rows.append(row)

    table = pd.DataFrame(rows).set_index("strategia")
    return table


TOURNAMENTS = {
    # nome: (strategie, regola delle date, prove complessive sugli stessi dati)
    "1": (STRATEGIES, decision_dates, 8),
    "2": (STRATEGIES_20D, day20_dates, 12),
}


def run_tournament(
    bars: pd.DataFrame,
    *,
    cost_per_side: float = 0.001,
    cash_rate_annual: float = 0.0,
    samples: int = 5000,
    tournament: str = "1",
) -> tuple[pd.DataFrame, dict[str, LabResult]]:
    strategies, rule, trials = TOURNAMENTS[tournament]
    closes = load_closes(bars)
    start = rule(closes.index)[0]
    results = {
        name: run_strategy(
            closes,
            strategy,
            start=start,
            cost_per_side=cost_per_side,
            cash_rate_annual=cash_rate_annual,
            decision_rule=rule,
        )
        for name, strategy in strategies.items()
    }
    return evaluate_tournament(results, samples=samples, total_trials=trials), results
