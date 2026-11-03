from pathlib import Path

import pandas as pd
import pytest

from app.data import COLUMN_NAMES, load_paysim, prepare, read_raw_csv, time_split


def make_frame(steps: list[int]) -> pd.DataFrame:
    fraud = [int(i % 7 == 0) for i in range(len(steps))]
    frame: pd.DataFrame = pd.DataFrame({"step": steps, "is_fraud": fraud})
    return frame


def even_steps(n_steps: int = 100, rows_per_step: int = 10) -> pd.DataFrame:
    return make_frame([s for s in range(1, n_steps + 1) for _ in range(rows_per_step)])


# --- time_split ---


def test_splits_are_in_time_order() -> None:
    split = time_split(even_steps())
    assert split.train["step"].max() < split.valid["step"].min()
    assert split.valid["step"].max() < split.test["step"].min()


def test_every_row_lands_in_exactly_one_split() -> None:
    df = even_steps()
    split = time_split(df)
    parts = [split.train.index, split.valid.index, split.test.index]
    assert sum(len(p) for p in parts) == len(df)
    assert set().union(*parts) == set(df.index)


def test_a_step_is_never_split_across_sets() -> None:
    # Step 5 holds 20 of 65 rows, and a naive cut at row 39 would land inside it
    df = make_frame([s for s in range(1, 11) for _ in range(20 if s == 5 else 5)])
    split = time_split(df, valid_frac=0.2, test_frac=0.2)
    steps = [set(p["step"]) for p in (split.train, split.valid, split.test)]
    assert steps[0].isdisjoint(steps[1])
    assert steps[1].isdisjoint(steps[2])
    assert steps[0].isdisjoint(steps[2])


def test_split_sizes_match_requested_fractions() -> None:
    df = even_steps()
    split = time_split(df, valid_frac=0.15, test_frac=0.15)
    assert len(split.train) / len(df) == pytest.approx(0.70, abs=0.02)
    assert len(split.valid) / len(df) == pytest.approx(0.15, abs=0.02)
    assert len(split.test) / len(df) == pytest.approx(0.15, abs=0.02)


def test_unsorted_input_is_fine() -> None:
    df = even_steps().sample(frac=1, random_state=0)
    split = time_split(df)
    assert split.train["step"].max() < split.test["step"].min()


@pytest.mark.parametrize(("valid_frac", "test_frac"), [(0, 0.2), (0.2, 0), (0.5, 0.5), (-0.1, 0.2)])
def test_rejects_bad_fractions(valid_frac: float, test_frac: float) -> None:
    with pytest.raises(ValueError, match="valid_frac"):
        time_split(even_steps(), valid_frac=valid_frac, test_frac=test_frac)


def test_rejects_too_few_steps() -> None:
    with pytest.raises(ValueError, match="Not enough"):
        time_split(make_frame([1] * 10))


def test_summary_reports_each_split() -> None:
    summary = time_split(even_steps()).summary()
    assert list(summary.index) == ["train", "valid", "test"]
    assert summary["rows"].sum() == 1000


# --- loading ---


def write_kaggle_csv(path: Path) -> None:
    row = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 181.0,
        "nameOrig": "C1305486145",
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C553264065",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "isFraud": 1,
        "isFlaggedFraud": 0,
    }
    pd.DataFrame([row, {**row, "step": 2, "isFraud": 0}]).to_csv(path, index=False)


def test_read_raw_csv_renames_and_types_columns(tmp_path: Path) -> None:
    csv = tmp_path / "paysim.csv"
    write_kaggle_csv(csv)
    df = read_raw_csv(csv)
    assert list(df.columns) == list(COLUMN_NAMES.values())
    assert df["type"].dtype == "category"
    assert df["is_fraud"].tolist() == [1, 0]


def test_read_raw_csv_rejects_wrong_file(tmp_path: Path) -> None:
    csv = tmp_path / "other.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="missing PaySim columns"):
        read_raw_csv(csv)


def test_prepare_then_load_round_trips(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_kaggle_csv(raw / "paysim.csv")
    out = prepare(raw, tmp_path / "processed" / "paysim.parquet")
    df = load_paysim(out)
    assert len(df) == 2
    assert "name_orig" in df.columns


def test_prepare_explains_missing_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="kaggle"):
        prepare(tmp_path, tmp_path / "out.parquet")
