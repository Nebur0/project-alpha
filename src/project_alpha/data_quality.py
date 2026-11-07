"""Controllo qualità dei dati di mercato giornalieri.

Errori (i dati non vanno usati così come sono):
- simboli richiesti assenti;
- righe duplicate (stesso simbolo e timestamp);
- prezzi mancanti o non positivi;
- ``high < low``;
- simboli che finiscono prima degli altri (dati disallineati).

Avvisi (da controllare, non bloccano):
- open/close fuori dall'intervallo low-high;
- date presenti per altri simboli ma mancanti per questo;
- rendimenti giornalieri estremi (possibili split non aggiustati);
- volumi nulli;
- simboli che iniziano dopo gli altri.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

PRICE_COLUMNS = ("open", "high", "low", "close")
EXTREME_RETURN = 0.40


@dataclass(frozen=True)
class DataIssue:
    severity: str  # "error" | "warning"
    code: str
    symbol: str | None
    detail: str


@dataclass(frozen=True)
class DataQualityReport:
    issues: tuple[DataIssue, ...]
    symbols: pd.DataFrame = field(repr=False)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> tuple[DataIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[DataIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def summary(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [
                {
                    "severity": issue.severity,
                    "code": issue.code,
                    "symbol": issue.symbol,
                    "detail": issue.detail,
                }
                for issue in self.issues
            ],
            "symbols": self.symbols.to_dict(orient="records"),
        }


def check_bars(
    frame: pd.DataFrame,
    symbols: Sequence[str] | None = None,
    *,
    extreme_return: float = EXTREME_RETURN,
) -> DataQualityReport:
    """Controlla un DataFrame di barre giornaliere (formato di ``fetch``)."""
    required = {"symbol", "timestamp", *PRICE_COLUMNS}
    missing_columns = required.difference(frame.columns)

    if missing_columns:
        raise ValueError(f"Colonne mancanti: {sorted(missing_columns)}")

    data = frame.copy()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data["_date"] = data["timestamp"].dt.tz_convert("America/New_York").dt.date

    issues: list[DataIssue] = []

    def add(severity: str, code: str, symbol: str | None, detail: str) -> None:
        issues.append(DataIssue(severity, code, symbol, detail))

    wanted = (
        [symbol.strip().upper() for symbol in symbols]
        if symbols is not None
        else sorted(data["symbol"].unique())
    )
    present = set(data["symbol"].unique())

    for symbol in wanted:
        if symbol not in present:
            add("error", "MISSING_SYMBOL", symbol, "Simbolo assente nei dati.")

    data = data.loc[data["symbol"].isin(wanted)]

    if data.empty:
        return DataQualityReport(tuple(issues), pd.DataFrame())

    all_dates = set(data["_date"].unique())
    global_end = data["_date"].max()
    global_start = data["_date"].min()
    rows: list[dict[str, object]] = []

    for symbol, group in data.groupby("symbol", sort=True):
        group = group.sort_values("timestamp")
        duplicates = int(group.duplicated(["_date"]).sum())
        prices = group[list(PRICE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        missing_prices = int(prices.isna().any(axis=1).sum())
        non_positive = int((prices <= 0).any(axis=1).sum())
        high_below_low = int((prices["high"] < prices["low"]).sum())
        outside_range = int(
            (
                (prices["close"] > prices["high"] * (1 + 1e-9))
                | (prices["close"] < prices["low"] * (1 - 1e-9))
                | (prices["open"] > prices["high"] * (1 + 1e-9))
                | (prices["open"] < prices["low"] * (1 - 1e-9))
            ).sum()
        )
        returns = prices["close"].pct_change().abs()
        extreme = int((returns > extreme_return).sum())
        zero_volume = (
            int((pd.to_numeric(group["volume"], errors="coerce") <= 0).sum())
            if "volume" in group.columns
            else 0
        )
        start, end = group["_date"].min(), group["_date"].max()
        expected = {day for day in all_dates if start <= day <= end}
        missing_dates = len(expected - set(group["_date"]))

        rows.append(
            {
                "symbol": symbol,
                "rows": int(len(group)),
                "start": str(start),
                "end": str(end),
                "duplicates": duplicates,
                "missing_prices": missing_prices,
                "non_positive_prices": non_positive,
                "high_below_low": high_below_low,
                "outside_high_low": outside_range,
                "missing_dates": missing_dates,
                "extreme_returns": extreme,
                "zero_volume_days": zero_volume,
            }
        )

        if duplicates:
            add("error", "DUPLICATES", symbol, f"{duplicates} date duplicate.")
        if missing_prices:
            add("error", "MISSING_PRICES", symbol, f"{missing_prices} righe con prezzi mancanti.")
        if non_positive:
            add("error", "NON_POSITIVE_PRICES", symbol, f"{non_positive} righe con prezzi <= 0.")
        if high_below_low:
            add("error", "HIGH_BELOW_LOW", symbol, f"{high_below_low} righe con high < low.")
        if end < global_end:
            add(
                "error",
                "ENDS_EARLY",
                symbol,
                f"Finisce il {end}, altri simboli arrivano al {global_end}.",
            )
        if start > global_start:
            add(
                "warning",
                "STARTS_LATE",
                symbol,
                f"Inizia il {start}, altri simboli dal {global_start}.",
            )
        if outside_range:
            add(
                "warning",
                "OUTSIDE_HIGH_LOW",
                symbol,
                f"{outside_range} righe con open/close fuori da low-high.",
            )
        if missing_dates:
            add(
                "warning",
                "MISSING_DATES",
                symbol,
                f"{missing_dates} date presenti per altri simboli ma non per questo.",
            )
        if extreme:
            add(
                "warning",
                "EXTREME_RETURNS",
                symbol,
                f"{extreme} rendimenti giornalieri oltre {extreme_return:.0%} "
                "(controllare split/aggiustamenti).",
            )
        if zero_volume:
            add("warning", "ZERO_VOLUME", symbol, f"{zero_volume} giorni con volume nullo.")

    return DataQualityReport(tuple(issues), pd.DataFrame(rows))
