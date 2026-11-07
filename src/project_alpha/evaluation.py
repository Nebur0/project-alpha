"""Valutazione standard di una strategia rispetto al suo benchmark.

Risponde alla domanda "la strategia aggiunge qualcosa rispetto a un
portafoglio passivo con lo stesso rischio?", che i confronti con il
benchmark al 100% non possono chiarire quando l'esposizione media è bassa.

Contenuto:

- confronto con un benchmark **statico a pari esposizione** (e a pari beta);
- decomposizione del rendimento lordo in esposizione, timing e selezione;
- regressioni CAPM e Treynor-Mazuy con errori standard Newey-West (HAC);
- rendimento degli asset nei giorni "dentro" e "fuori" (ON/OFF);
- test di permutazione del timing (shift circolari: stessa esposizione,
  stessa durata delle posizioni, momento casuale);
- Probabilistic e Deflated Sharpe Ratio, Minimum Track Record Length
  (Bailey e López de Prado).

Non dipende da scipy/statsmodels: usa numpy e ``statistics.NormalDist``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0
EULER_MASCHERONI = 0.5772156649015329
_NORMAL = NormalDist()


# --------------------------------------------------------------------------
# Statistiche di base
# --------------------------------------------------------------------------


def _as_array(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError("La serie dei rendimenti deve essere monodimensionale.")

    if array.size < 2:
        raise ValueError("Servono almeno due rendimenti.")

    if not np.isfinite(array).all():
        raise ValueError("La serie dei rendimenti contiene valori non finiti.")

    return array


def performance_summary(
    returns: pd.Series | np.ndarray | list[float],
) -> dict[str, float | None]:
    """CAGR, volatilità, Sharpe (rf=0) e max drawdown di rendimenti giornalieri."""
    values = _as_array(returns)
    equity = np.cumprod(1.0 + values)
    years = (values.size - 1) / TRADING_DAYS
    final = float(equity[-1])
    std = float(values.std(ddof=1))

    return {
        "total_return": final - 1.0,
        "cagr": (final ** (1.0 / years) - 1.0) if years > 0 and final > 0 else None,
        "annualized_volatility": std * sqrt(TRADING_DAYS),
        "annualized_sharpe": (
            float(values.mean()) / std * sqrt(TRADING_DAYS) if std > 0 else None
        ),
        "max_drawdown": float((equity / np.maximum.accumulate(equity) - 1.0).min()),
    }


# --------------------------------------------------------------------------
# Regressioni con errori HAC
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RegressionResult:
    coefficients: tuple[float, ...]
    standard_errors: tuple[float, ...]
    t_statistics: tuple[float, ...]
    r_squared: float
    observations: int
    hac_lags: int

    @property
    def alpha_daily(self) -> float:
        return self.coefficients[0]

    @property
    def alpha_annual(self) -> float:
        return self.coefficients[0] * TRADING_DAYS

    @property
    def alpha_t(self) -> float:
        return self.t_statistics[0]


def ols_newey_west(
    y: np.ndarray,
    regressors: np.ndarray,
    *,
    lags: int = 10,
) -> RegressionResult:
    """Minimi quadrati con intercetta ed errori Newey-West (kernel di Bartlett)."""
    target = _as_array(y)
    design = np.asarray(regressors, dtype=float)

    if design.ndim == 1:
        design = design[:, None]

    if design.shape[0] != target.size:
        raise ValueError("Regressori e variabile dipendente hanno lunghezze diverse.")

    if lags < 0:
        raise ValueError("Il numero di lag HAC deve essere non negativo.")

    design = np.column_stack([np.ones(target.size), design])

    if target.size <= design.shape[1]:
        raise ValueError("Osservazioni insufficienti per la regressione.")

    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coefficients
    bread = np.linalg.pinv(design.T @ design)
    scores = design * residuals[:, None]
    meat = scores.T @ scores

    for lag in range(1, min(lags, target.size - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)

    covariance = bread @ meat @ bread
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    t_statistics = np.divide(
        coefficients,
        standard_errors,
        out=np.full_like(coefficients, np.nan),
        where=standard_errors > 0,
    )
    total = float(((target - target.mean()) ** 2).sum())
    r_squared = 1.0 - float((residuals**2).sum()) / total if total > 0 else 0.0

    return RegressionResult(
        coefficients=tuple(float(value) for value in coefficients),
        standard_errors=tuple(float(value) for value in standard_errors),
        t_statistics=tuple(float(value) for value in t_statistics),
        r_squared=r_squared,
        observations=int(target.size),
        hac_lags=int(lags),
    )


# --------------------------------------------------------------------------
# Confronti a pari esposizione e decomposizione
# --------------------------------------------------------------------------


def exposure_matched_comparison(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    gross_exposure: np.ndarray,
    *,
    hac_lags: int = 10,
) -> dict[str, dict[str, float | None]]:
    """Strategia contro benchmark al 100%, a pari esposizione e a pari beta."""
    strategy = _as_array(strategy_returns)
    benchmark = _as_array(benchmark_returns)
    exposure = np.asarray(gross_exposure, dtype=float)

    if not (strategy.size == benchmark.size == exposure.size):
        raise ValueError("Le serie devono avere la stessa lunghezza.")

    mean_exposure = float(exposure.mean())
    capm = ols_newey_west(strategy, benchmark, lags=hac_lags)
    beta = capm.coefficients[1]

    rows = {
        "strategy": performance_summary(strategy),
        "benchmark": performance_summary(benchmark),
        "benchmark_static_exposure": performance_summary(mean_exposure * benchmark),
        "benchmark_daily_exposure": performance_summary(exposure * benchmark),
        "benchmark_static_beta": performance_summary(beta * benchmark),
    }
    rows["benchmark_static_exposure"]["exposure"] = mean_exposure
    rows["benchmark_static_beta"]["exposure"] = beta

    return rows


def _position_matrix(
    curves: Mapping[str, pd.DataFrame],
    column: str,
) -> np.ndarray:
    return np.column_stack(
        [curve[column].to_numpy(dtype=float) for curve in curves.values()]
    )


def decompose_gross_returns(
    positions: np.ndarray,
    asset_returns: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Scompone il rendimento lordo medio annuo (aritmetico) in tre parti.

    - esposizione: esposizione media x rendimento medio del benchmark;
    - timing: quanto l'esposizione totale è alta nei giorni buoni del
      benchmark (covarianza esposizione/benchmark);
    - selezione: quanto i titoli scelti, a parità di esposizione,
      rendono più del paniere.

    La somma delle tre parti è il rendimento lordo medio annuo.
    """
    position_matrix = np.asarray(positions, dtype=float)
    return_matrix = np.asarray(asset_returns, dtype=float)
    weight_vector = np.asarray(weights, dtype=float)

    if position_matrix.shape != return_matrix.shape:
        raise ValueError("Posizioni e rendimenti hanno forme diverse.")

    if weight_vector.shape != (position_matrix.shape[1],):
        raise ValueError("Il vettore dei pesi non corrisponde agli asset.")

    gross = (position_matrix * return_matrix) @ weight_vector
    exposure = position_matrix @ weight_vector
    benchmark = return_matrix @ weight_vector

    exposure_part = float(exposure.mean() * benchmark.mean())
    timing_part = float((exposure * benchmark).mean()) - exposure_part
    selection_part = float(gross.mean()) - float((exposure * benchmark).mean())

    return {
        "exposure_component": exposure_part * TRADING_DAYS,
        "timing_component": timing_part * TRADING_DAYS,
        "selection_component": selection_part * TRADING_DAYS,
        "gross_return_arithmetic": float(gross.mean()) * TRADING_DAYS,
    }


def on_off_returns(
    curves: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Rendimento medio annuo di ogni asset nei giorni in posizione e fuori."""
    rows = []

    for symbol, curve in curves.items():
        position = curve["executed_position"].to_numpy(dtype=float)
        asset = curve["asset_return"].to_numpy(dtype=float)
        on = asset[position > 0]
        off = asset[position <= 0]

        rows.append(
            {
                "symbol": symbol,
                "exposure": float(position.mean()),
                "on_days": int(on.size),
                "off_days": int(off.size),
                "on_return_annual": float(on.mean() * TRADING_DAYS) if on.size else None,
                "off_return_annual": float(off.mean() * TRADING_DAYS) if off.size else None,
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Test di permutazione
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PermutationTestResult:
    statistic: str
    actual: float
    permutation_median: float
    permutation_p95: float
    p_value: float
    permutations: int
    seed: int


def timing_permutation_test(
    positions: np.ndarray,
    asset_returns: np.ndarray,
    weights: np.ndarray,
    *,
    permutations: int = 5_000,
    seed: int = 0,
) -> dict[str, PermutationTestResult]:
    """Confronta il timing reale con timing casuali della stessa struttura.

    Ogni colonna di posizioni viene spostata circolarmente di un offset
    casuale: esposizione, numero e durata delle posizioni restano uguali,
    cambia solo *quando* si è dentro. Il p-value è la quota di permutazioni
    almeno buone quanto la realtà (con correzione +1).
    Le statistiche sono lorde (senza costi).
    """
    position_matrix = np.asarray(positions, dtype=float)
    return_matrix = np.asarray(asset_returns, dtype=float)
    weight_vector = np.asarray(weights, dtype=float)

    if position_matrix.shape != return_matrix.shape:
        raise ValueError("Posizioni e rendimenti hanno forme diverse.")

    if permutations < 1:
        raise ValueError("Serve almeno una permutazione.")

    observations, assets = position_matrix.shape

    if observations < 3:
        raise ValueError("Osservazioni insufficienti per il test di permutazione.")

    def statistics_of(matrix: np.ndarray) -> tuple[float, float]:
        series = (matrix * return_matrix) @ weight_vector
        std = series.std(ddof=1)
        sharpe = series.mean() / std * sqrt(TRADING_DAYS) if std > 0 else 0.0
        return float(series.mean() * TRADING_DAYS), float(sharpe)

    actual_mean, actual_sharpe = statistics_of(position_matrix)
    rng = np.random.default_rng(seed)
    means = np.empty(permutations)
    sharpes = np.empty(permutations)
    row_index = np.arange(observations)

    for index in range(permutations):
        offsets = rng.integers(1, observations, size=assets)
        shifted_rows = (row_index[:, None] - offsets[None, :]) % observations
        shifted = np.take_along_axis(position_matrix, shifted_rows, axis=0)
        means[index], sharpes[index] = statistics_of(shifted)

    def result(name: str, actual: float, sample: np.ndarray) -> PermutationTestResult:
        return PermutationTestResult(
            statistic=name,
            actual=actual,
            permutation_median=float(np.median(sample)),
            permutation_p95=float(np.quantile(sample, 0.95)),
            p_value=float((1 + (sample >= actual).sum()) / (permutations + 1)),
            permutations=permutations,
            seed=seed,
        )

    return {
        "gross_return_annual": result("gross_return_annual", actual_mean, means),
        "gross_sharpe": result("gross_sharpe", actual_sharpe, sharpes),
    }


# --------------------------------------------------------------------------
# PSR, DSR, MinTRL
# --------------------------------------------------------------------------


def _sharpe_moments(values: np.ndarray) -> tuple[float, float, float]:
    """Sharpe per periodo, asimmetria e curtosi (non in eccesso)."""
    std = values.std(ddof=1)

    # Soglia numerica: rendimenti costanti producono uno std ~1e-19, non 0.
    if std <= 1e-12:
        raise ValueError("Rendimenti a varianza nulla: Sharpe non definito.")

    centered = values - values.mean()
    m2 = float((centered**2).mean())
    skewness = float((centered**3).mean() / m2**1.5)
    kurtosis = float((centered**4).mean() / m2**2)

    return float(values.mean() / std), skewness, kurtosis


def probabilistic_sharpe_ratio(
    returns: pd.Series | np.ndarray | list[float],
    *,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Probabilità che lo Sharpe vero (per periodo) superi ``benchmark_sharpe``.

    ``benchmark_sharpe`` è espresso per periodo (giornaliero), non annuo.
    """
    values = _as_array(returns)
    sharpe, skewness, kurtosis = _sharpe_moments(values)
    denominator = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2

    if denominator <= 0:
        raise ValueError("Momenti dei rendimenti non validi per il PSR.")

    statistic = (sharpe - benchmark_sharpe) * sqrt(values.size - 1) / sqrt(denominator)
    return float(_NORMAL.cdf(statistic))


def expected_maximum_sharpe(
    *,
    trials: int,
    trials_sharpe_std: float,
    trials_sharpe_mean: float = 0.0,
) -> float:
    """Sharpe massimo atteso (per periodo) tra ``trials`` prove indipendenti."""
    if trials < 1:
        raise ValueError("Il numero di prove deve essere almeno 1.")

    if trials_sharpe_std < 0:
        raise ValueError("La deviazione standard degli Sharpe non può essere negativa.")

    if trials == 1:
        return float(trials_sharpe_mean)

    quantile = (1.0 - EULER_MASCHERONI) * _NORMAL.inv_cdf(1.0 - 1.0 / trials)
    quantile += EULER_MASCHERONI * _NORMAL.inv_cdf(1.0 - 1.0 / (trials * np.e))

    return float(trials_sharpe_mean + trials_sharpe_std * quantile)


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray | list[float],
    *,
    trials: int,
    trials_sharpe_std: float,
) -> float:
    """PSR contro lo Sharpe massimo atteso per puro caso tra ``trials`` prove."""
    threshold = expected_maximum_sharpe(
        trials=trials,
        trials_sharpe_std=trials_sharpe_std,
    )
    return probabilistic_sharpe_ratio(returns, benchmark_sharpe=threshold)


def minimum_track_record_length(
    returns: pd.Series | np.ndarray | list[float],
    *,
    benchmark_sharpe: float = 0.0,
    confidence: float = 0.95,
) -> float | None:
    """Osservazioni necessarie perché lo Sharpe superi ``benchmark_sharpe``.

    ``None`` quando lo Sharpe osservato non supera la soglia (nessuna
    lunghezza del track record basterebbe).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("La confidenza deve essere compresa tra 0 e 1.")

    values = _as_array(returns)
    sharpe, skewness, kurtosis = _sharpe_moments(values)

    if sharpe <= benchmark_sharpe:
        return None

    variance_term = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    z_value = _NORMAL.inv_cdf(confidence)

    return float(1.0 + variance_term * (z_value / (sharpe - benchmark_sharpe)) ** 2)


# --------------------------------------------------------------------------
# Report completo per un portafoglio walk-forward
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationReport:
    comparison: dict[str, dict[str, float | None]]
    decomposition: dict[str, float]
    capm: RegressionResult
    treynor_mazuy: RegressionResult
    on_off: pd.DataFrame
    permutation: dict[str, PermutationTestResult]
    by_window: pd.DataFrame
    probabilistic_sharpe: float | None
    deflated_sharpe: float | None
    min_track_record_years: float | None
    trials: int
    trials_sharpe_std: float | None

    def summary(self) -> dict[str, float | int | None]:
        strategy = self.comparison["strategy"]
        static = self.comparison["benchmark_static_exposure"]

        def difference(key: str) -> float | None:
            if strategy[key] is None or static[key] is None:
                return None
            return float(strategy[key] - static[key])

        return {
            "strategy_cagr": strategy["cagr"],
            "static_exposure_cagr": static["cagr"],
            "cagr_vs_static_exposure": difference("cagr"),
            "sharpe_vs_static_exposure": difference("annualized_sharpe"),
            "max_drawdown_vs_static_exposure": difference("max_drawdown"),
            "mean_exposure": static["exposure"],
            "capm_alpha_annual": self.capm.alpha_annual,
            "capm_alpha_t": self.capm.alpha_t,
            "capm_beta": self.capm.coefficients[1],
            "timing_gamma": self.treynor_mazuy.coefficients[2],
            "timing_gamma_t": self.treynor_mazuy.t_statistics[2],
            "permutation_p_value_return": self.permutation["gross_return_annual"].p_value,
            "permutation_p_value_sharpe": self.permutation["gross_sharpe"].p_value,
            "probabilistic_sharpe": self.probabilistic_sharpe,
            "deflated_sharpe": self.deflated_sharpe,
            "min_track_record_years": self.min_track_record_years,
            "trials": self.trials,
            **self.decomposition,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary(),
            "comparison": self.comparison,
            "decomposition": self.decomposition,
            "capm": asdict(self.capm),
            "treynor_mazuy": asdict(self.treynor_mazuy),
            "permutation": {
                name: asdict(result) for name, result in self.permutation.items()
            },
            "on_off": self.on_off.to_dict(orient="records"),
            "by_window": self.by_window.to_dict(orient="records"),
        }


def _by_window(
    curves: Mapping[str, pd.DataFrame],
    strategy: np.ndarray,
    benchmark: np.ndarray,
    exposure: np.ndarray,
) -> pd.DataFrame:
    first_curve = next(iter(curves.values()))

    if "window_number" not in first_curve.columns:
        return pd.DataFrame()

    windows = first_curve["window_number"].to_numpy()
    timestamps = pd.to_datetime(first_curve["timestamp"], utc=True)
    rows = []

    for window in pd.unique(windows):
        mask = windows == window
        mean_exposure = float(exposure[mask].mean())
        strategy_return = float(np.prod(1.0 + strategy[mask]) - 1.0)
        static_return = float(np.prod(1.0 + mean_exposure * benchmark[mask]) - 1.0)

        rows.append(
            {
                "window_number": int(window),
                "start": timestamps[mask].min().isoformat(),
                "end": timestamps[mask].max().isoformat(),
                "mean_exposure": mean_exposure,
                "strategy_return": strategy_return,
                "benchmark_return": float(np.prod(1.0 + benchmark[mask]) - 1.0),
                "static_exposure_return": static_return,
                "difference_vs_static": strategy_return - static_return,
            }
        )

    return pd.DataFrame(rows)


def evaluate_walk_forward_portfolio(
    portfolio_result,
    *,
    trials: int = 1,
    trials_sharpe_std: float | None = None,
    permutations: int = 5_000,
    seed: int = 0,
    hac_lags: int = 10,
) -> EvaluationReport:
    """Report di valutazione per un ``WalkForwardPortfolioResult``.

    ``trials`` e ``trials_sharpe_std`` (Sharpe giornalieri delle prove)
    servono al Deflated Sharpe Ratio: senza dispersione nota e con più di
    una prova il DSR non viene calcolato (``None``).
    """
    curves = portfolio_result.symbol_curves
    portfolio = portfolio_result.portfolio
    equity_curve = portfolio.equity_curve

    strategy = equity_curve["portfolio_return"].to_numpy(dtype=float)
    benchmark = equity_curve["benchmark_return"].to_numpy(dtype=float)
    exposure = equity_curve["gross_exposure"].to_numpy(dtype=float)

    weights = np.array([portfolio.weights[symbol] for symbol in curves], dtype=float)
    positions = _position_matrix(curves, "executed_position")
    asset_returns = _position_matrix(curves, "asset_return")

    capm = ols_newey_west(strategy, benchmark, lags=hac_lags)
    treynor_mazuy = ols_newey_west(
        strategy,
        np.column_stack([benchmark, benchmark**2]),
        lags=hac_lags,
    )

    try:
        psr: float | None = probabilistic_sharpe_ratio(strategy)
        min_trl = minimum_track_record_length(strategy)
    except ValueError:
        psr, min_trl = None, None

    if trials == 1:
        dsr = psr
    elif trials_sharpe_std is not None:
        try:
            dsr = deflated_sharpe_ratio(
                strategy,
                trials=trials,
                trials_sharpe_std=trials_sharpe_std,
            )
        except ValueError:
            dsr = None
    else:
        dsr = None

    return EvaluationReport(
        comparison=exposure_matched_comparison(
            strategy,
            benchmark,
            exposure,
            hac_lags=hac_lags,
        ),
        decomposition=decompose_gross_returns(positions, asset_returns, weights),
        capm=capm,
        treynor_mazuy=treynor_mazuy,
        on_off=on_off_returns(curves),
        permutation=timing_permutation_test(
            positions,
            asset_returns,
            weights,
            permutations=permutations,
            seed=seed,
        ),
        by_window=_by_window(curves, strategy, benchmark, exposure),
        probabilistic_sharpe=psr,
        deflated_sharpe=dsr,
        min_track_record_years=(
            min_trl / TRADING_DAYS if min_trl is not None else None
        ),
        trials=trials,
        trials_sharpe_std=trials_sharpe_std,
    )
