import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import analysis  # noqa: E402

pytestmark = pytest.mark.skipif(not analysis.RAW.exists(), reason="raw data not downloaded")


@pytest.fixture(scope="module")
def res():
    return analysis.main()


def test_sample_and_design(res):
    assert res["integrity"]["n"] == 50083
    assert res["integrity"]["srm_p"] > 0.05
    assert res["integrity"]["balance_failures_p_lt_05"] == 0


def test_reproduces_published_headline(res):
    # Karlan & List (2007), Table 2A: response 1.8% vs 2.2%; $0.813 vs $0.967 per letter
    p = res["pooled"]
    assert abs(p["control_conversion"] - 0.018) < 0.0005
    assert abs(p["treatment_conversion"] - 0.022) < 0.0005
    assert abs(p["control_revenue_per_letter"] - 0.813) < 0.001
    assert abs(p["treatment_revenue_per_letter"] - 0.967) < 0.001
    assert p["pooled_conversion_p"] < 0.01


def test_higher_ratios_not_better(res):
    assert res["pooled"]["ratio_joint_test_conversion_p"] > 0.1


def test_discount_efficiency_falls_with_depth(res):
    e = {r["arm"]: r["incremental_per_match_dollar"] for r in res["discount_efficiency"]}
    assert e["1:1"] > e["3:1"]


def test_elasticity_negative_with_ci_excluding_zero(res):
    lo, hi = res["elasticity"]["arc_elasticity_ci"]
    assert res["elasticity"]["arc_elasticity_participation"] < 0 and hi < 0


def test_targeting_spends_less(res):
    t = res["targeting"]
    assert t["match_cost_target"] < t["match_cost_all"]


def test_results_file_written(res):
    assert json.loads((ROOT / "outputs" / "results.json").read_text())["integrity"]["n"] == 50083
