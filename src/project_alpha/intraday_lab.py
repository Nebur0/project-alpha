"""Laboratorio intraday pre-registrato (``docs/LAB_INTRADAY.md``).

Domanda: con candele da 15 minuti si riesce a prevedere la direzione delle
2 candele successive abbastanza bene da guadagnare dopo i costi?

Regole comuni: segnale alla chiusura di una candela, entrata a quel prezzo,
uscita alla chiusura di 2 candele dopo (o all'ultima candela del giorno),
nessuna posizione overnight, una posizione alla volta per simbolo, costo
fisso per lato. Il giudizio si fa solo fuori campione (dal 2022).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd

MARKET_TZ = "America/New_York"
HOLD_BARS = 2
OPENING_RANGE_BARS = 2
OOS_START = "2022-01-01"
YEARS = (("2022", "2022"), ("2023", "2023"), ("2024", "2024"), ("2025", "2100"))
TRADING_DAYS = 252
_NORMAL = NormalDist()


@dataclass(frozen=True)
class Day:
    date: pd.Timestamp
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


# ----------------------------------------------------------------------
# Dati
# ----------------------------------------------------------------------


def load_sessions(frame: pd.DataFrame, symbol: str) -> list[Day]:
    """Candele della seduta regolare (9:30–16:00 NY) di ``symbol``, divise per giorno."""
    data = frame.loc[frame["symbol"].astype(str).str.upper() == symbol.upper()].copy()
    stamps = pd.to_datetime(data["timestamp"], utc=True).dt.tz_convert(MARKET_TZ)
    minutes = stamps.dt.hour * 60 + stamps.dt.minute
    data = data.assign(ts=stamps).loc[(minutes >= 9 * 60 + 30) & (minutes < 16 * 60)]
    data = data.sort_values("ts")
    days = []
    for date, group in data.groupby(data["ts"].dt.date):
        if len(group) < OPENING_RANGE_BARS + HOLD_BARS + 1:
            continue
        days.append(
            Day(
                date=pd.Timestamp(date),
                open=group["open"].to_numpy(float),
                high=group["high"].to_numpy(float),
                low=group["low"].to_numpy(float),
                close=group["close"].to_numpy(float),
                volume=group["volume"].to_numpy(float),
            )
        )
    return days


# ----------------------------------------------------------------------
# Regole (segnale per candela: +1 long, -1 short, 0 niente)
# ----------------------------------------------------------------------

Signal = Callable[[Day], np.ndarray]


def momentum_1(day: Day) -> np.ndarray:
    return np.sign(day.close - day.open)


def inversione_1(day: Day) -> np.ndarray:
    return -np.sign(day.close - day.open)


def engulfing(day: Day) -> np.ndarray:
    signal = np.zeros(len(day.close))
    body = day.close - day.open
    top = np.maximum(day.open, day.close)
    bottom = np.minimum(day.open, day.close)
    for i in range(1, len(body)):
        covers = (
            top[i] >= top[i - 1] and bottom[i] <= bottom[i - 1] and abs(body[i]) > abs(body[i - 1])
        )
        if covers and body[i] > 0 > body[i - 1]:
            signal[i] = 1
        elif covers and body[i] < 0 < body[i - 1]:
            signal[i] = -1
    return signal


def simulate_day(
    day: Day, signal: np.ndarray, *, cost: float, hold: int | None = HOLD_BARS
) -> list[float]:
    """Rendimenti netti delle operazioni del giorno (frazioni del capitale per operazione)."""
    trades = []
    last = len(day.close) - 1
    i = 0
    while i < last:
        side = signal[i]
        if side == 0 or np.isnan(side):
            i += 1
            continue
        exit_bar = last if hold is None else min(i + hold, last)
        gross = side * (day.close[exit_bar] / day.close[i] - 1.0)
        trades.append(gross - 2 * cost)
        i = exit_bar
    return trades


def opening_range_trades(day: Day, *, cost: float) -> list[float]:
    high = day.high[:OPENING_RANGE_BARS].max()
    low = day.low[:OPENING_RANGE_BARS].min()
    last = len(day.close) - 1
    for i in range(OPENING_RANGE_BARS, last):
        side = 1 if day.close[i] > high else -1 if day.close[i] < low else 0
        if side:
            return [side * (day.close[last] / day.close[i] - 1.0) - 2 * cost]
    return []


# ----------------------------------------------------------------------
# Modello statistico: regressione logistica, addestramento mensile
# ----------------------------------------------------------------------

LAGS = 8


def _features(days: list[Day]) -> pd.DataFrame:
    """Una riga per candela: caratteristiche delle ultime 8 candele e obiettivo."""
    rows = []
    previous_close = None
    history_close: list[float] = []
    history_range: list[float] = []
    history_volume: list[float] = []
    for d_index, day in enumerate(days):
        last = len(day.close) - 1
        for i in range(len(day.close)):
            reference = day.close[i - 1] if i > 0 else (previous_close or day.open[0])
            history_close.append(day.close[i] / reference - 1.0)
            history_range.append((day.high[i] - day.low[i]) / day.close[i])
            history_volume.append(day.volume[i])
            if len(history_close) < LAGS + 26:
                continue
            returns = history_close[-LAGS:]
            ranges = history_range[-LAGS:]
            mean_volume = np.mean(history_volume[-26:]) or 1.0
            target = (
                np.sign(day.close[min(i + HOLD_BARS, last)] / day.close[i] - 1.0)
                if i < last
                else np.nan
            )
            rows.append(
                [
                    d_index,
                    i,
                    *returns,
                    *ranges,
                    history_volume[-1] / mean_volume,
                    i / 25.0,
                    (i / 25.0) ** 2,
                    target,
                ]
            )
        previous_close = day.close[-1]
    columns = (
        ["day", "bar"]
        + [f"r{k}" for k in range(LAGS)]
        + [f"hl{k}" for k in range(LAGS)]
        + ["vol", "t", "t2", "y"]
    )
    return pd.DataFrame(rows, columns=columns)


def _fit_logistic(
    x: np.ndarray, y: np.ndarray, *, l2: float = 1.0, iterations: int = 25
) -> np.ndarray:
    """Regressione logistica (Newton-Raphson con penalità L2), y in {0, 1}."""
    design = np.column_stack([np.ones(len(x)), x])
    weights = np.zeros(design.shape[1])
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-design @ weights))
        gradient = design.T @ (y - p) - penalty @ weights
        hessian = (design * (p * (1 - p))[:, None]).T @ design + penalty
        step = np.linalg.solve(hessian, gradient)
        weights += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return weights


def logistic_signals(days: list[Day], *, train_months: int = 12) -> dict[int, np.ndarray]:
    """Segnali del modello per giorno (indice → array), addestrando ogni mese sul passato."""
    table = _features(days)
    feature_columns = [c for c in table.columns if c not in ("day", "bar", "y")]
    dates = pd.Series([days[i].date for i in table["day"]], index=table.index)
    months = dates.dt.to_period("M")
    signals: dict[int, np.ndarray] = {i: np.zeros(len(d.close)) for i, d in enumerate(days)}

    for month in sorted(months.unique()):
        start = month - train_months
        train = table[(months >= start) & (months < month) & table["y"].notna() & (table["y"] != 0)]
        if months[(months >= start) & (months < month)].nunique() < train_months:
            continue
        test = table[months == month]
        x = train[feature_columns].to_numpy(float)
        mean, std = x.mean(axis=0), x.std(axis=0)
        std[std == 0] = 1.0
        weights = _fit_logistic((x - mean) / std, (train["y"].to_numpy() > 0).astype(float))
        xt = (test[feature_columns].to_numpy(float) - mean) / std
        prob = 1.0 / (1.0 + np.exp(-(np.column_stack([np.ones(len(xt)), xt]) @ weights)))
        side = np.where(prob > 0.55, 1.0, np.where(prob < 0.45, -1.0, 0.0))
        for (day_index, bar), value in zip(test[["day", "bar"]].to_numpy(int), side, strict=True):
            signals[day_index][bar] = value
    return signals


# ----------------------------------------------------------------------
# Valutazione
# ----------------------------------------------------------------------

RULES = ("L1_momentum_1", "L2_inversione_1", "L3_engulfing", "L4_rottura_apertura", "L5_logistica")


def daily_pnl(
    days: list[Day], rule: str, *, cost: float, model: dict[int, np.ndarray] | None = None
) -> pd.DataFrame:
    rows = []
    for index, day in enumerate(days):
        if rule == "L1_momentum_1":
            trades = simulate_day(day, momentum_1(day), cost=cost)
        elif rule == "L2_inversione_1":
            trades = simulate_day(day, inversione_1(day), cost=cost)
        elif rule == "L3_engulfing":
            trades = simulate_day(day, engulfing(day), cost=cost)
        elif rule == "L4_rottura_apertura":
            trades = opening_range_trades(day, cost=cost)
        elif rule == "L5_logistica":
            trades = simulate_day(day, model[index], cost=cost)
        else:
            raise ValueError(rule)
        wins = sum(t > 0 for t in trades)
        rows.append(
            {"date": day.date, "pnl": float(sum(trades)), "trades": len(trades), "wins": wins}
        )
    return pd.DataFrame(rows).set_index("date")


def newey_west_pvalue(values: np.ndarray, lags: int = 5) -> tuple[float, float]:
    """t-stat della media (errori Newey-West) e p-value unilaterale (media > 0)."""
    n = len(values)
    centered = values - values.mean()
    variance = centered @ centered / n
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        variance += 2 * weight * (centered[lag:] @ centered[:-lag]) / n
    se = sqrt(max(variance, 1e-30) / n)
    t = values.mean() / se
    return t, 1 - _NORMAL.cdf(t)


def evaluate(
    frame: pd.DataFrame, symbols: tuple[str, ...] = ("SPY", "QQQ"), *, cost: float = 0.0001
) -> pd.DataFrame:
    trials = len(RULES) * len(symbols)
    alpha = 0.05 / trials
    rows = []
    for symbol in symbols:
        days = load_sessions(frame, symbol)
        model = logistic_signals(days)
        for rule in RULES:
            pnl = daily_pnl(days, rule, cost=cost, model=model).loc[OOS_START:]
            stressed = daily_pnl(days, rule, cost=2 * cost, model=model).loc[OOS_START:]
            values = pnl["pnl"].to_numpy()
            t, p = newey_west_pvalue(values)
            years_positive = sum(pnl.loc[a:b, "pnl"].sum() > 0 for a, b in YEARS)
            trades = int(pnl["trades"].sum())
            mean_bps = values.mean() * 1e4
            passed = p <= alpha and years_positive >= 3 and stressed["pnl"].mean() > 0
            rows.append(
                {
                    "simbolo": symbol,
                    "regola": rule,
                    "giorni": len(pnl),
                    "operazioni_al_giorno": trades / max(len(pnl), 1),
                    "operazioni_vinte": pnl["wins"].sum() / max(trades, 1),
                    "media_giornaliera_bps": mean_bps,
                    "euro_al_giorno_per_1000": mean_bps / 10.0,
                    "totale_periodo_per_1000": values.sum() * 1000,
                    "sharpe_annuo": values.mean() / values.std(ddof=1) * sqrt(TRADING_DAYS)
                    if values.std() > 0
                    else 0.0,
                    "t_newey_west": t,
                    "p_value": p,
                    "anni_positivi": f"{years_positive}/{len(YEARS)}",
                    "media_bps_costi_doppi": stressed["pnl"].mean() * 1e4,
                    "esito": "FUNZIONA" if passed else "scartata",
                }
            )
    table = pd.DataFrame(rows)
    table.attrs["alpha"] = alpha
    return table
