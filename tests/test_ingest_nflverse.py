"""Tests for nflverse ingest helpers that don't require network access."""

import numpy as np
import pandas as pd
import pytest

from ingest_nflverse import _none_if_nan, _null_duplicate_source_ids, _sum_fumbles


def test_none_if_nan_passes_through_real_values():
    assert _none_if_nan("hello") == "hello"
    assert _none_if_nan(0) == 0
    assert _none_if_nan(0.0) == 0.0
    assert _none_if_nan(False) is False


def test_none_if_nan_collapses_nan_and_none():
    assert _none_if_nan(None) is None
    assert _none_if_nan(float("nan")) is None
    assert _none_if_nan(np.nan) is None
    assert _none_if_nan(pd.NA) is None


def test_none_if_nan_coerces_timestamp_to_isoformat():
    ts = pd.Timestamp("2024-03-04")
    assert _none_if_nan(ts) == "2024-03-04"


def test_none_if_nan_unwraps_numpy_scalars():
    assert _none_if_nan(np.int64(7)) == 7
    assert _none_if_nan(np.float64(1.5)) == 1.5
    # And the result is a plain Python scalar, not a numpy one.
    assert type(_none_if_nan(np.int64(7))) is int


def test_sum_fumbles_treats_missing_as_zero():
    row = pd.Series({"sack_fumbles_lost": 1, "rushing_fumbles_lost": np.nan})
    # receiving_fumbles_lost missing entirely
    assert _sum_fumbles(row) == 1


def test_sum_fumbles_adds_three_components():
    row = pd.Series(
        {
            "sack_fumbles_lost": 1,
            "rushing_fumbles_lost": 2,
            "receiving_fumbles_lost": 3,
        }
    )
    assert _sum_fumbles(row) == 6


def test_null_duplicate_source_ids_keeps_first_nulls_rest():
    rows = [
        {"gsis_id": "G1", "pfr_id": "DupePFR"},
        {"gsis_id": "G2", "pfr_id": "DupePFR"},
        {"gsis_id": "G3", "pfr_id": "UniquePFR"},
    ]
    cleaned = _null_duplicate_source_ids(rows, columns=("pfr_id",))
    assert cleaned[0]["pfr_id"] == "DupePFR"
    assert cleaned[1]["pfr_id"] is None
    assert cleaned[2]["pfr_id"] == "UniquePFR"


def test_null_duplicate_source_ids_treats_empty_string_as_null():
    rows = [
        {"gsis_id": "G1", "pfr_id": ""},
        {"gsis_id": "G2", "pfr_id": ""},
    ]
    cleaned = _null_duplicate_source_ids(rows, columns=("pfr_id",))
    # Empty strings are skipped from the dup-tracking entirely
    assert cleaned[0]["pfr_id"] == ""
    assert cleaned[1]["pfr_id"] == ""
