import pandas as pd
import pytest

from project_alpha.validation import (
    generate_walk_forward_splits,
    split_by_date,
)


def test_split_by_date_separates_periods() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=6,
                freq="D",
                tz="UTC",
            ),
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
            ],
        }
    )

    result = split_by_date(
        frame,
        "2024-01-04",
    )

    assert len(result.in_sample) == 3
    assert len(result.out_of_sample) == 3

    assert (
        result.in_sample["timestamp"].max()
        < result.split_timestamp
    )

    assert (
        result.out_of_sample["timestamp"].min()
        >= result.split_timestamp
    )


def test_split_date_belongs_to_out_of_sample() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ],
            "close": [
                100.0,
                101.0,
                102.0,
            ],
        }
    )

    result = split_by_date(
        frame,
        "2024-01-02",
    )

    assert result.in_sample["close"].tolist() == [
        100.0,
    ]

    assert result.out_of_sample["close"].tolist() == [
        101.0,
        102.0,
    ]


def test_split_by_date_rejects_empty_frame() -> None:
    frame = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="dataset da dividere è vuoto",
    ):
        split_by_date(
            frame,
            "2024-01-01",
        )


def test_split_by_date_rejects_empty_partition() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=3,
                freq="D",
                tz="UTC",
            ),
            "close": [
                100.0,
                101.0,
                102.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="in-sample è vuota",
    ):
        split_by_date(
            frame,
            "2024-01-01",
        )

    with pytest.raises(
        ValueError,
        match="out-of-sample è vuota",
    ):
        split_by_date(
            frame,
            "2025-01-01",
        )


def test_generate_walk_forward_splits_creates_expanding_windows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=10,
                freq="D",
                tz="UTC",
            ),
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
            ],
        }
    )

    splits = generate_walk_forward_splits(
        frame=frame,
        train_size=4,
        test_size=2,
    )

    assert len(splits) == 3

    assert len(splits[0].in_sample) == 4
    assert len(splits[0].out_of_sample) == 2

    assert len(splits[1].in_sample) == 6
    assert len(splits[1].out_of_sample) == 2

    assert len(splits[2].in_sample) == 8
    assert len(splits[2].out_of_sample) == 2


def test_generate_walk_forward_splits_sets_window_metadata() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=6,
                freq="D",
                tz="UTC",
            ),
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
            ],
        }
    )

    splits = generate_walk_forward_splits(
        frame=frame,
        train_size=4,
        test_size=2,
    )

    first_window = splits[0]

    assert first_window.window_number == 1
    assert first_window.train_start == pd.Timestamp(
        "2024-01-01",
        tz="UTC",
    )
    assert first_window.train_end == pd.Timestamp(
        "2024-01-04",
        tz="UTC",
    )
    assert first_window.test_start == pd.Timestamp(
        "2024-01-05",
        tz="UTC",
    )
    assert first_window.test_end == pd.Timestamp(
        "2024-01-06",
        tz="UTC",
    )


def test_generate_walk_forward_splits_respects_step_size() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=9,
                freq="D",
                tz="UTC",
            ),
            "close": range(100, 109),
        }
    )

    splits = generate_walk_forward_splits(
        frame=frame,
        train_size=4,
        test_size=2,
        step_size=1,
    )

    assert len(splits) == 4

    assert splits[0].test_start == pd.Timestamp(
        "2024-01-05",
        tz="UTC",
    )
    assert splits[1].test_start == pd.Timestamp(
        "2024-01-06",
        tz="UTC",
    )


def test_generate_walk_forward_splits_rejects_invalid_sizes() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=5,
                freq="D",
                tz="UTC",
            ),
            "close": range(100, 105),
        }
    )

    with pytest.raises(
        ValueError,
        match="train_size deve essere positivo",
    ):
        generate_walk_forward_splits(
            frame=frame,
            train_size=0,
            test_size=2,
        )

    with pytest.raises(
        ValueError,
        match="test_size deve essere positivo",
    ):
        generate_walk_forward_splits(
            frame=frame,
            train_size=3,
            test_size=0,
        )

    with pytest.raises(
        ValueError,
        match="step_size deve essere positivo",
    ):
        generate_walk_forward_splits(
            frame=frame,
            train_size=3,
            test_size=2,
            step_size=0,
        )


def test_generate_walk_forward_splits_rejects_insufficient_data() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01",
                periods=5,
                freq="D",
                tz="UTC",
            ),
            "close": range(100, 105),
        }
    )

    with pytest.raises(
        ValueError,
        match="non contiene abbastanza righe",
    ):
        generate_walk_forward_splits(
            frame=frame,
            train_size=4,
            test_size=2,
        )