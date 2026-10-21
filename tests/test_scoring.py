import pytest

from project_alpha.scoring import (
    RobustScoreConfig,
    RobustScoreV4Config,
    RobustScoreV5Config,
    calculate_robust_score,
    calculate_robust_score_v4,
    calculate_robust_score_v5,
    calculate_robust_score_v5_quality,
)


def _base_metrics() -> dict[
    str,
    float | int | None,
]:
    return {
        "annualized_sharpe": 1.20,
        "total_return": 0.30,
        "benchmark_total_return": 0.20,
        "max_drawdown": -0.10,
        "completed_trades": 20,
    }


def _base_v5_metrics() -> dict[
    str,
    float | int | None,
]:
    return {
        "annualized_sharpe": 1.20,
        "cagr": 0.15,
        "max_drawdown": -0.10,
        "completed_trades": 20,
    }


def test_calculate_robust_score_uses_default_weights() -> None:
    result = calculate_robust_score(
        _base_metrics()
    )

    assert result.sharpe == pytest.approx(
        1.20
    )
    assert result.total_return == pytest.approx(
        0.30
    )
    assert (
        result.benchmark_total_return
        == pytest.approx(
            0.20
        )
    )
    assert result.excess_return == pytest.approx(
        0.10
    )
    assert result.max_drawdown == pytest.approx(
        -0.10
    )
    assert result.completed_trades == 20
    assert result.trade_shortfall == pytest.approx(
        0.0
    )
    assert result.score == pytest.approx(
        1.20
    )


def test_calculate_robust_score_penalizes_trade_shortfall() -> None:
    metrics = _base_metrics()
    metrics["completed_trades"] = 10

    result = calculate_robust_score(
        metrics
    )

    assert result.trade_shortfall == pytest.approx(
        0.5
    )
    assert result.score == pytest.approx(
        0.95
    )


def test_calculate_robust_score_uses_custom_weights() -> None:
    config = RobustScoreConfig(
        sharpe_weight=2.0,
        excess_return_weight=3.0,
        drawdown_weight=4.0,
        trade_shortfall_weight=5.0,
        target_completed_trades=20,
    )

    metrics = _base_metrics()
    metrics["completed_trades"] = 10

    result = calculate_robust_score(
        metrics,
        config,
    )

    expected_score = (
        2.0 * 1.20
        + 3.0 * 0.10
        + 4.0 * -0.10
        - 5.0 * 0.50
    )

    assert result.score == pytest.approx(
        expected_score
    )


def test_calculate_robust_score_rewards_excess_return() -> None:
    first_metrics = _base_metrics()
    second_metrics = _base_metrics()

    second_metrics["total_return"] = 0.40

    first = calculate_robust_score(
        first_metrics
    )
    second = calculate_robust_score(
        second_metrics
    )

    assert (
        second.excess_return
        > first.excess_return
    )
    assert second.score > first.score


def test_calculate_robust_score_penalizes_drawdown() -> None:
    first_metrics = _base_metrics()
    second_metrics = _base_metrics()

    second_metrics["max_drawdown"] = -0.30

    first = calculate_robust_score(
        first_metrics
    )
    second = calculate_robust_score(
        second_metrics
    )

    assert second.score < first.score


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        (
            "sharpe_weight",
            -1.0,
        ),
        (
            "excess_return_weight",
            -0.5,
        ),
        (
            "drawdown_weight",
            -2.0,
        ),
        (
            "trade_shortfall_weight",
            -0.1,
        ),
        (
            "sharpe_weight",
            float("inf"),
        ),
        (
            "drawdown_weight",
            float("nan"),
        ),
        (
            "sharpe_weight",
            True,
        ),
    ],
)
def test_robust_score_config_rejects_invalid_weights(
    field_name: str,
    invalid_value: object,
) -> None:
    kwargs = {
        "sharpe_weight": 1.0,
        "excess_return_weight": 1.0,
        "drawdown_weight": 1.0,
        "trade_shortfall_weight": 0.5,
        "target_completed_trades": 20,
    }

    kwargs[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match="finito non negativo",
    ):
        RobustScoreConfig(
            **kwargs
        )


@pytest.mark.parametrize(
    "target_completed_trades",
    [
        0,
        -1,
        True,
        10.5,
    ],
)
def test_robust_score_config_rejects_invalid_target_trades(
    target_completed_trades: object,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "target_completed_trades deve "
            "essere un intero positivo"
        ),
    ):
        RobustScoreConfig(
            target_completed_trades=(
                target_completed_trades
            )
        )


@pytest.mark.parametrize(
    "metric_name",
    [
        "annualized_sharpe",
        "total_return",
        "benchmark_total_return",
        "max_drawdown",
    ],
)
def test_calculate_robust_score_rejects_missing_metric(
    metric_name: str,
) -> None:
    metrics = _base_metrics()
    metrics.pop(metric_name)

    with pytest.raises(
        ValueError,
        match=metric_name,
    ):
        calculate_robust_score(
            metrics
        )


@pytest.mark.parametrize(
    ("metric_name", "invalid_value"),
    [
        (
            "annualized_sharpe",
            None,
        ),
        (
            "total_return",
            "bad",
        ),
        (
            "benchmark_total_return",
            True,
        ),
        (
            "max_drawdown",
            float("inf"),
        ),
    ],
)
def test_calculate_robust_score_rejects_invalid_metric_values(
    metric_name: str,
    invalid_value: object,
) -> None:
    metrics = _base_metrics()
    metrics[metric_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=metric_name,
    ):
        calculate_robust_score(
            metrics
        )


@pytest.mark.parametrize(
    "completed_trades",
    [
        None,
        "20",
        True,
        -1,
        10.5,
        float("inf"),
    ],
)
def test_calculate_robust_score_rejects_invalid_completed_trades(
    completed_trades: object,
) -> None:
    metrics = _base_metrics()
    metrics["completed_trades"] = (
        completed_trades
    )

    with pytest.raises(
        ValueError,
        match="completed_trades",
    ):
        calculate_robust_score(
            metrics
        )


def test_calculate_robust_score_v4_uses_worst_half_mean(
) -> None:
    result = calculate_robust_score_v4(
        base_score=1.0,
        temporal_fold_scores=[
            1.2,
            0.8,
            0.4,
            1.0,
        ],
    )

    assert result.base_score == pytest.approx(
        1.0
    )
    assert (
        result.temporal_downside_score
        == pytest.approx(
            0.6
        )
    )
    assert result.temporal_fold_scores == (
        1.2,
        0.8,
        0.4,
        1.0,
    )
    assert result.downside_fold_count == 2
    assert result.score == pytest.approx(
        0.8
    )


def test_calculate_robust_score_v4_uses_custom_weights(
) -> None:
    config = RobustScoreV4Config(
        base_score_weight=0.25,
        temporal_downside_weight=0.75,
    )

    result = calculate_robust_score_v4(
        base_score=1.2,
        temporal_fold_scores=[
            0.4,
            0.8,
            1.0,
            1.4,
        ],
        config=config,
    )

    expected_temporal_downside = (
        0.4 + 0.8
    ) / 2.0

    expected_score = (
        0.25 * 1.2
        + 0.75
        * expected_temporal_downside
    )

    assert (
        result.temporal_downside_score
        == pytest.approx(
            expected_temporal_downside
        )
    )
    assert result.score == pytest.approx(
        expected_score
    )


def test_calculate_robust_score_v4_rewards_temporal_stability(
) -> None:
    unstable_peak = (
        calculate_robust_score_v4(
            base_score=1.0,
            temporal_fold_scores=[
                1.4,
                1.3,
                0.1,
                0.2,
            ],
        )
    )

    stable_candidate = (
        calculate_robust_score_v4(
            base_score=0.9,
            temporal_fold_scores=[
                0.8,
                0.85,
                0.9,
                0.95,
            ],
        )
    )

    assert (
        unstable_peak.base_score
        > stable_candidate.base_score
    )
    assert (
        stable_candidate
        .temporal_downside_score
        > unstable_peak
        .temporal_downside_score
    )
    assert (
        stable_candidate.score
        > unstable_peak.score
    )


def test_calculate_robust_score_v4_supports_even_fold_count(
) -> None:
    config = RobustScoreV4Config(
        temporal_fold_count=6,
    )

    result = calculate_robust_score_v4(
        base_score=1.0,
        temporal_fold_scores=[
            1.2,
            0.1,
            0.4,
            1.1,
            0.7,
            0.9,
        ],
        config=config,
    )

    expected_temporal_downside = (
        0.1 + 0.4 + 0.7
    ) / 3.0

    assert result.downside_fold_count == 3
    assert (
        result.temporal_downside_score
        == pytest.approx(
            expected_temporal_downside
        )
    )
    assert result.score == pytest.approx(
        (
            1.0
            + expected_temporal_downside
        )
        / 2.0
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        (
            "base_score_weight",
            -1.0,
        ),
        (
            "temporal_downside_weight",
            -0.5,
        ),
        (
            "base_score_weight",
            float("inf"),
        ),
        (
            "temporal_downside_weight",
            float("nan"),
        ),
        (
            "base_score_weight",
            True,
        ),
    ],
)
def test_robust_score_v4_config_rejects_invalid_weights(
    field_name: str,
    invalid_value: object,
) -> None:
    kwargs = {
        "base_score_weight": 0.5,
        "temporal_downside_weight": 0.5,
        "temporal_fold_count": 4,
    }

    kwargs[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match="finito non negativo",
    ):
        RobustScoreV4Config(
            **kwargs
        )


def test_robust_score_v4_config_rejects_all_zero_weights(
) -> None:
    with pytest.raises(
        ValueError,
        match="Almeno uno dei pesi",
    ):
        RobustScoreV4Config(
            base_score_weight=0.0,
            temporal_downside_weight=0.0,
        )


@pytest.mark.parametrize(
    "temporal_fold_count",
    [
        0,
        1,
        3,
        -2,
        True,
        4.5,
    ],
)
def test_robust_score_v4_config_rejects_invalid_fold_count(
    temporal_fold_count: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="temporal_fold_count",
    ):
        RobustScoreV4Config(
            temporal_fold_count=(
                temporal_fold_count
            )
        )


def test_calculate_robust_score_v4_rejects_wrong_fold_count(
) -> None:
    with pytest.raises(
        ValueError,
        match="temporal_fold_count",
    ):
        calculate_robust_score_v4(
            base_score=1.0,
            temporal_fold_scores=[
                0.5,
                0.6,
                0.7,
            ],
        )


@pytest.mark.parametrize(
    "base_score",
    [
        True,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_calculate_robust_score_v4_rejects_invalid_base_score(
    base_score: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="base_score",
    ):
        calculate_robust_score_v4(
            base_score=base_score,
            temporal_fold_scores=[
                0.5,
                0.6,
                0.7,
                0.8,
            ],
        )


@pytest.mark.parametrize(
    "temporal_fold_score",
    [
        True,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_calculate_robust_score_v4_rejects_invalid_temporal_fold_score(
    temporal_fold_score: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="temporal_fold_score",
    ):
        calculate_robust_score_v4(
            base_score=1.0,
            temporal_fold_scores=[
                0.5,
                0.6,
                0.7,
                temporal_fold_score,
            ],
        )


def test_calculate_robust_score_v5_quality_uses_default_weights(
) -> None:
    result = calculate_robust_score_v5_quality(
        _base_v5_metrics()
    )

    expected_score = (
        1.20
        + 0.15
        - 0.10
    )

    assert result.sharpe == pytest.approx(
        1.20
    )
    assert result.cagr == pytest.approx(
        0.15
    )
    assert result.max_drawdown == pytest.approx(
        -0.10
    )
    assert result.completed_trades == 20
    assert result.trade_shortfall == pytest.approx(
        0.0
    )
    assert result.score == pytest.approx(
        expected_score
    )


def test_calculate_robust_score_v5_quality_ignores_benchmark(
) -> None:
    first_metrics = _base_v5_metrics()
    second_metrics = _base_v5_metrics()

    first_metrics["benchmark_total_return"] = (
        -0.50
    )
    second_metrics["benchmark_total_return"] = (
        5.00
    )

    first = calculate_robust_score_v5_quality(
        first_metrics
    )
    second = calculate_robust_score_v5_quality(
        second_metrics
    )

    assert first.score == pytest.approx(
        second.score
    )


@pytest.mark.parametrize(
    "sharpe_value",
    [
        None,
        float("nan"),
    ],
)
def test_calculate_robust_score_v5_quality_maps_missing_sharpe_to_zero(
    sharpe_value: float | None,
) -> None:
    metrics = _base_v5_metrics()
    metrics["annualized_sharpe"] = (
        sharpe_value
    )
    metrics["completed_trades"] = 0

    result = calculate_robust_score_v5_quality(
        metrics
    )

    assert result.sharpe == pytest.approx(
        0.0
    )
    assert result.trade_shortfall == pytest.approx(
        1.0
    )
    assert result.score == pytest.approx(
        0.0
        + 0.15
        - 0.10
        - 0.5
    )


def test_calculate_robust_score_v5_quality_scales_trade_target_via_config(
) -> None:
    metrics = _base_v5_metrics()
    metrics["completed_trades"] = 2

    result = calculate_robust_score_v5_quality(
        metrics,
        RobustScoreV5Config(
            target_completed_trades=5,
        ),
    )

    assert result.trade_shortfall == pytest.approx(
        0.6
    )
    assert result.score == pytest.approx(
        1.20
        + 0.15
        - 0.10
        - 0.5 * 0.6
    )


def test_calculate_robust_score_v5_quality_uses_custom_weights(
) -> None:
    metrics = _base_v5_metrics()
    metrics["completed_trades"] = 10

    config = RobustScoreV5Config(
        sharpe_weight=2.0,
        cagr_weight=3.0,
        drawdown_weight=4.0,
        trade_shortfall_weight=5.0,
        target_completed_trades=20,
    )

    result = calculate_robust_score_v5_quality(
        metrics,
        config,
    )

    expected_score = (
        2.0 * 1.20
        + 3.0 * 0.15
        + 4.0 * -0.10
        - 5.0 * 0.50
    )

    assert result.score == pytest.approx(
        expected_score
    )


@pytest.mark.parametrize(
    "metric_name",
    [
        "cagr",
        "max_drawdown",
    ],
)
def test_calculate_robust_score_v5_quality_rejects_missing_required_metric(
    metric_name: str,
) -> None:
    metrics = _base_v5_metrics()
    metrics.pop(metric_name)

    with pytest.raises(
        ValueError,
        match=metric_name,
    ):
        calculate_robust_score_v5_quality(
            metrics
        )


@pytest.mark.parametrize(
    ("metric_name", "invalid_value"),
    [
        (
            "annualized_sharpe",
            float("inf"),
        ),
        (
            "cagr",
            float("nan"),
        ),
        (
            "max_drawdown",
            float("-inf"),
        ),
    ],
)
def test_calculate_robust_score_v5_quality_rejects_invalid_metric(
    metric_name: str,
    invalid_value: object,
) -> None:
    metrics = _base_v5_metrics()
    metrics[metric_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=metric_name,
    ):
        calculate_robust_score_v5_quality(
            metrics
        )


def test_calculate_robust_score_v5_uses_worst_half_mean(
) -> None:
    result = calculate_robust_score_v5(
        base_score=1.0,
        temporal_fold_scores=[
            1.2,
            0.8,
            0.4,
            1.0,
        ],
    )

    assert result.base_score == pytest.approx(
        1.0
    )
    assert (
        result.temporal_downside_score
        == pytest.approx(
            0.6
        )
    )
    assert result.temporal_fold_scores == (
        1.2,
        0.8,
        0.4,
        1.0,
    )
    assert result.downside_fold_count == 2
    assert result.score == pytest.approx(
        0.8
    )


def test_calculate_robust_score_v5_rewards_temporal_stability(
) -> None:
    unstable = calculate_robust_score_v5(
        base_score=1.0,
        temporal_fold_scores=[
            1.4,
            1.3,
            0.1,
            0.2,
        ],
    )

    stable = calculate_robust_score_v5(
        base_score=0.9,
        temporal_fold_scores=[
            0.8,
            0.85,
            0.9,
            0.95,
        ],
    )

    assert unstable.base_score > stable.base_score
    assert (
        stable.temporal_downside_score
        > unstable.temporal_downside_score
    )
    assert stable.score > unstable.score


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("sharpe_weight", -1.0),
        ("cagr_weight", -1.0),
        ("drawdown_weight", -1.0),
        ("trade_shortfall_weight", -1.0),
        ("base_score_weight", -1.0),
        ("temporal_downside_weight", -1.0),
        ("sharpe_weight", float("inf")),
        ("cagr_weight", float("nan")),
        ("base_score_weight", True),
    ],
)
def test_robust_score_v5_config_rejects_invalid_weights(
    field_name: str,
    invalid_value: object,
) -> None:
    kwargs = {
        "sharpe_weight": 1.0,
        "cagr_weight": 1.0,
        "drawdown_weight": 1.0,
        "trade_shortfall_weight": 0.5,
        "target_completed_trades": 20,
        "base_score_weight": 0.5,
        "temporal_downside_weight": 0.5,
        "temporal_fold_count": 4,
    }

    kwargs[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match="finito non negativo",
    ):
        RobustScoreV5Config(
            **kwargs
        )


def test_robust_score_v5_config_rejects_zero_quality_weights(
) -> None:
    with pytest.raises(
        ValueError,
        match="pesi quality",
    ):
        RobustScoreV5Config(
            sharpe_weight=0.0,
            cagr_weight=0.0,
            drawdown_weight=0.0,
            trade_shortfall_weight=0.0,
        )


def test_robust_score_v5_config_rejects_zero_temporal_weights(
) -> None:
    with pytest.raises(
        ValueError,
        match="pesi temporali",
    ):
        RobustScoreV5Config(
            base_score_weight=0.0,
            temporal_downside_weight=0.0,
        )


@pytest.mark.parametrize(
    "target_completed_trades",
    [
        0,
        -1,
        True,
        10.5,
    ],
)
def test_robust_score_v5_config_rejects_invalid_target_trades(
    target_completed_trades: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="target_completed_trades",
    ):
        RobustScoreV5Config(
            target_completed_trades=(
                target_completed_trades
            )
        )


@pytest.mark.parametrize(
    "temporal_fold_count",
    [
        0,
        1,
        3,
        -2,
        True,
        4.5,
    ],
)
def test_robust_score_v5_config_rejects_invalid_fold_count(
    temporal_fold_count: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="temporal_fold_count",
    ):
        RobustScoreV5Config(
            temporal_fold_count=(
                temporal_fold_count
            )
        )


def test_calculate_robust_score_v5_rejects_wrong_fold_count(
) -> None:
    with pytest.raises(
        ValueError,
        match="temporal_fold_count",
    ):
        calculate_robust_score_v5(
            base_score=1.0,
            temporal_fold_scores=[
                0.5,
                0.6,
                0.7,
            ],
        )


@pytest.mark.parametrize(
    "invalid_score",
    [
        True,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_calculate_robust_score_v5_rejects_invalid_scores(
    invalid_score: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="base_score",
    ):
        calculate_robust_score_v5(
            base_score=invalid_score,
            temporal_fold_scores=[
                0.5,
                0.6,
                0.7,
                0.8,
            ],
        )

    with pytest.raises(
        ValueError,
        match="temporal_fold_score",
    ):
        calculate_robust_score_v5(
            base_score=1.0,
            temporal_fold_scores=[
                0.5,
                0.6,
                0.7,
                invalid_score,
            ],
        )
