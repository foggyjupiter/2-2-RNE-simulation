"""Spec 10.1 Γc(N) reproduction (slow).

Pass criterion (spec): Γc increases with N; the value differences to the first study are
reported (``gransim gamma-c`` writes results/gamma_c/gamma_c_comparison.csv).
"""

import pytest

from gransim.analysis.gamma_c import (comparison_table, extract_gamma_c, gamma_c_cases,
                                      load_reference, trend_increasing)
from gransim.analysis.validation import run_many
from gransim.config import SimConfig


def test_reference_data_loads():
    ref = load_reference()
    assert ref[100][0] == pytest.approx(1.25) and ref[300][0] == pytest.approx(7.0)


@pytest.mark.slow
def test_gamma_c_increases_with_n():
    base = SimConfig(time={"settle_time": 0.5, "discard_time": 1.0, "record_time": 2.0})
    cases = gamma_c_cases(base, [100, 300], [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0])
    res = run_many(cases)
    gc = extract_gamma_c(res)
    for row in comparison_table(gc, load_reference()):
        print(row)
    assert trend_increasing(gc, "kurtosis_total_hp")
    assert trend_increasing(gc, "sustained_fraction")
