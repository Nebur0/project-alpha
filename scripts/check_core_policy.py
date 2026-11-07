"""Verifica storica (una sola, non un'ottimizzazione) del core di paper trading.

Uso: python scripts/check_core_policy.py data/raw/bars_etf.csv
Scrive reports/policy/core_bilanciato_v1/historical_check.json.
Approssimazione: pesi ribilanciati ogni giorno, vol targeting con isteresi
5 punti, costo 10 bps sui cambi di esposizione, stop simulato con uscita il
giorno successivo e nessuna ripartenza.
"""

import sys
from pathlib import Path

import pandas as pd

from project_alpha.allocation import run_policy_portfolio
from project_alpha.io_utils import write_json_atomic
from project_alpha.paper import load_paper_config
from project_alpha.research import file_sha256

REPO = Path(__file__).resolve().parents[1]


def drawdown(returns: pd.Series) -> pd.Series:
    equity = (1 + returns).cumprod()
    return equity / equity.cummax() - 1


def main(data_path: str) -> None:
    config = load_paper_config(REPO / "configs" / "paper_core_v1.toml")
    bars = pd.read_csv(data_path, parse_dates=["timestamp"])
    closes = bars.pivot(index="timestamp", columns="symbol", values="close")
    returns = closes[config.symbols].pct_change().dropna()
    sleeves = {symbol: returns[symbol] for symbol in config.symbols}

    variants = {}
    for name, vol in (("core_100", None), ("core_vol_target", config.vol_target)):
        result = run_policy_portfolio(sleeves, dict(config.weights), vol_target=vol)
        dd = drawdown(result.returns)
        variants[name] = {
            **result.metrics,
            "worst_drawdown_date": dd.idxmin().date().isoformat(),
            "worst_drawdown_by_year": dd.groupby(dd.index.year).min().round(4).to_dict(),
        }

        if vol is not None:
            stopped = result.returns.copy()
            trigger = dd[dd <= -config.stop_drawdown].index.min()
            if pd.notna(trigger):
                stopped.loc[stopped.index > trigger] = 0.0
            variants["core_vol_target_with_stop"] = {
                "stop_date": None if pd.isna(trigger) else trigger.date().isoformat(),
                "max_drawdown": float(drawdown(stopped).min()),
                "note": "Dopo lo stop il sistema resta in cash finché il proprietario non lo sblocca.",
            }

    write_json_atomic(
        REPO / "reports" / "policy" / "core_bilanciato_v1" / "historical_check.json",
        {
            "config": "configs/paper_core_v1.toml",
            "data_sha256": file_sha256(data_path),
            "start": returns.index.min().date().isoformat(),
            "end": returns.index.max().date().isoformat(),
            "trials": 1,
            "variants": variants,
        },
    )


if __name__ == "__main__":
    main(sys.argv[1])
