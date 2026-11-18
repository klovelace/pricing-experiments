"""Pricing-experiment analysis of a randomized price test (Karlan & List 2007).

50,083 prior donors were randomized into a control letter or a matching-grant
letter. A 1:1, 2:1 or 3:1 match cuts the donor's *price* of delivering $1 to the
charity to $0.50, $0.33 or $0.25. Match threshold and the suggested-ask anchor
were cross-randomized. This is a multi-arm price test with conversion and
heavy-tailed, zero-inflated revenue, the same shape as a fee or discount test
on a consumer product.

Run:  python src/analysis.py
Outputs: outputs/results.json, outputs/tables/*.csv, outputs/figures/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "karlan_list_2007.dta"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

ARMS = ["Control", "1:1", "2:1", "3:1"]
PRICE = {"Control": 1.0, "1:1": 1 / 2, "2:1": 1 / 3, "3:1": 1 / 4}
COVARS = ["hpa", "freq", "years", "mrm2"]  # pre-treatment giving history
SEED = 20071201

# Chart palette (validated categorical slots; see README)
C_CTRL, C1, C2, C3 = "#898781", "#2a78d6", "#eb6834", "#1baf7a"
ARM_COLOR = {"Control": C_CTRL, "1:1": C1, "2:1": C2, "3:1": C3}
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
                     "grid.color": "#e1e0d9", "grid.linewidth": 0.8, "axes.axisbelow": True})


def load() -> pd.DataFrame:
    df = pd.read_stata(RAW, convert_categoricals=False)
    df["arm"] = np.select([df["control"] == 1, df["ratio"] == 1, df["ratio"] == 2, df["ratio"] == 3], ARMS, "?")
    df["price"] = df["arm"].map(PRICE)
    df["gave"] = df["gave"].astype(int)
    df["amount"] = df["amount"].astype(float)
    # Pre-period covariates: impute the few missing with medians and keep a flag.
    for c in COVARS:
        df[f"{c}_miss"] = df[c].isna().astype(int)
        df[c] = df[c].fillna(df[c].median())
    df["red"] = df["red0"].fillna(0).astype(int)
    df["state_known"] = df["red0"].notna().astype(int)
    return df


# ---------------------------------------------------------------- 1. integrity
def integrity(df: pd.DataFrame) -> dict:
    n = df["arm"].value_counts().reindex(ARMS)
    # Design: control 1/3, treatment 2/3 split evenly across three ratios.
    expected = np.array([1 / 3, 2 / 9, 2 / 9, 2 / 9]) * len(df)
    chi2, p_srm = stats.chisquare(n.values, expected)
    rows = []
    for c in ["hpa", "freq", "years", "mrm2", "female", "couple", "red0", "median_hhincome"]:
        x = df[c]
        t, p = stats.ttest_ind(x[df["treatment"] == 1].dropna(), x[df["treatment"] == 0].dropna(), equal_var=False)
        rows.append({"covariate": c, "control_mean": x[df["treatment"] == 0].mean(),
                     "treatment_mean": x[df["treatment"] == 1].mean(), "p_value": p})
    bal = pd.DataFrame(rows)
    bal.to_csv(TAB / "balance.csv", index=False)
    return {"n": int(len(df)), "arm_counts": n.to_dict(), "srm_p": float(p_srm),
            "balance_min_p": float(bal["p_value"].min()), "balance_failures_p_lt_05": int((bal["p_value"] < 0.05).sum())}


# ---------------------------------------------------------------- 2. primary effects
def arm_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Per-arm conversion, revenue per letter and conditional gift, vs control,
    with HC2 robust SEs and Holm correction across the three price arms."""
    rows = []
    for metric in ["gave", "amount"]:
        m = smf.ols(f"{metric} ~ C(arm, Treatment('Control'))", data=df).fit(cov_type="HC2")
        base = df.loc[df["arm"] == "Control", metric].mean()
        for arm in ARMS[1:]:
            k = f"C(arm, Treatment('Control'))[T.{arm}]"
            lo, hi = m.conf_int().loc[k]
            rows.append({"metric": metric, "arm": arm, "control_mean": base, "arm_mean": base + m.params[k],
                         "diff": m.params[k], "ci_lo": lo, "ci_hi": hi, "lift": m.params[k] / base, "p": m.pvalues[k]})
    res = pd.DataFrame(rows)
    res["p_holm"] = res.groupby("metric")["p"].transform(lambda p: multipletests(p, method="holm")[1])
    res.to_csv(TAB / "arm_effects.csv", index=False)
    return res


def heavy_tail_robustness(df: pd.DataFrame) -> dict:
    """Revenue per user is zero-inflated and heavy-tailed. Check the pooled
    revenue effect under winsorization and a percentile bootstrap."""
    out = {}
    givers = df.loc[df["gave"] == 1, "amount"]
    for q in (0.99, 0.995):
        cap = givers.quantile(q)
        d = df.assign(amt_w=df["amount"].clip(upper=cap))
        m = smf.ols("amt_w ~ treatment", data=d).fit(cov_type="HC2")
        out[f"winsorized_{q}"] = {"cap": float(cap), "diff": float(m.params["treatment"]),
                                  "ci": [float(x) for x in m.conf_int().loc["treatment"]], "p": float(m.pvalues["treatment"])}
    rng = np.random.default_rng(SEED)
    c, t = df.loc[df.treatment == 0, "amount"].values, df.loc[df.treatment == 1, "amount"].values
    b = [t[rng.integers(0, len(t), len(t))].mean() - c[rng.integers(0, len(c), len(c))].mean() for _ in range(4000)]
    out["bootstrap_ci"] = [float(x) for x in np.percentile(b, [2.5, 97.5])]
    out["top1pct_share_of_revenue"] = float(givers.sort_values(ascending=False).head(max(1, int(len(givers) * 0.01))).sum() / givers.sum())
    return out


def pooled_and_ratio_tests(df: pd.DataFrame) -> dict:
    t = df[df["treatment"] == 1]
    m_any = smf.ols("gave ~ treatment", data=df).fit(cov_type="HC2")
    m_amt = smf.ols("amount ~ treatment", data=df).fit(cov_type="HC2")
    # Do deeper discounts beat the 1:1 match? Joint test across ratios within treatment.
    m_r = smf.ols("gave ~ C(arm)", data=t).fit(cov_type="HC2")
    f_ratio = m_r.f_test("C(arm)[T.2:1] = 0, C(arm)[T.3:1] = 0")
    m_r_amt = smf.ols("amount ~ C(arm)", data=t).fit(cov_type="HC2")
    f_ratio_amt = m_r_amt.f_test("C(arm)[T.2:1] = 0, C(arm)[T.3:1] = 0")
    ctrl = df.loc[df["treatment"] == 0]
    return {
        "control_conversion": float(ctrl["gave"].mean()),
        "treatment_conversion": float(t["gave"].mean()),
        "pooled_conversion_diff": float(m_any.params["treatment"]),
        "pooled_conversion_ci": [float(x) for x in m_any.conf_int().loc["treatment"]],
        "pooled_conversion_p": float(m_any.pvalues["treatment"]),
        "pooled_conversion_lift": float(m_any.params["treatment"] / ctrl["gave"].mean()),
        "control_revenue_per_letter": float(ctrl["amount"].mean()),
        "treatment_revenue_per_letter": float(t["amount"].mean()),
        "pooled_revenue_diff": float(m_amt.params["treatment"]),
        "pooled_revenue_ci": [float(x) for x in m_amt.conf_int().loc["treatment"]],
        "pooled_revenue_p": float(m_amt.pvalues["treatment"]),
        "ratio_joint_test_conversion_p": float(f_ratio.pvalue),
        "ratio_joint_test_revenue_p": float(f_ratio_amt.pvalue),
    }


# ---------------------------------------------------------------- 3. demand curve & elasticity
def elasticity(df: pd.DataFrame) -> dict:
    g = df.groupby("arm").agg(conv=("gave", "mean"), rev=("amount", "mean"), n=("gave", "size")).reindex(ARMS)
    g["price"] = [PRICE[a] for a in ARMS]
    # Arc elasticity of participation, control -> pooled match (price 1 -> 0.5..0.25)
    t = df[df["treatment"] == 1]
    p0, p1 = 1.0, float(t["price"].mean())
    q0, q1 = g.loc["Control", "conv"], t["gave"].mean()
    arc = ((q1 - q0) / ((q1 + q0) / 2)) / ((p1 - p0) / ((p1 + p0) / 2))
    # Log-log slope across the three matched price points only (does price matter *within* discounts?)
    gg = g.loc[ARMS[1:]]
    slope = np.polyfit(np.log(gg["price"]), np.log(gg["conv"]), 1)[0]
    # Bootstrap CI for the arc elasticity
    rng = np.random.default_rng(SEED)
    c, tr = df.loc[df["treatment"] == 0, "gave"].values, t["gave"].values
    tp = t["price"].values
    boots = []
    for _ in range(2000):
        ci, ti = rng.integers(0, len(c), len(c)), rng.integers(0, len(tr), len(tr))
        a0, a1, pp = c[ci].mean(), tr[ti].mean(), tp[ti].mean()
        boots.append(((a1 - a0) / ((a1 + a0) / 2)) / ((pp - 1) / ((pp + 1) / 2)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    g.to_csv(TAB / "demand_by_price.csv")
    return {"arc_elasticity_participation": float(arc), "arc_elasticity_ci": [float(lo), float(hi)],
            "loglog_slope_within_matched_arms": float(slope)}


# ---------------------------------------------------------------- 4. discount efficiency
def discount_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """The 'price cut' is paid for by a matching donor. Cost of the discount per
    letter = match dollars consumed; benefit = incremental donor dollars."""
    ctrl_rev = df.loc[df["arm"] == "Control", "amount"].mean()
    rows = []
    for arm, r in zip(ARMS[1:], [1, 2, 3]):
        a = df.loc[df["arm"] == arm, "amount"]
        incr = a.mean() - ctrl_rev
        match_cost = (a * r).mean()
        rows.append({"arm": arm, "revenue_per_letter": a.mean(), "incremental_per_letter": incr,
                     "match_dollars_per_letter": match_cost, "incremental_per_match_dollar": incr / match_cost,
                     "total_to_charity_per_letter": (a * (1 + r)).mean()})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "discount_efficiency.csv", index=False)
    return out


# ---------------------------------------------------------------- 5. variance reduction (CUPED-style)
def variance_reduction(df: pd.DataFrame) -> dict:
    out = {}
    cov = " + ".join(COVARS + [f"{c}_miss" for c in COVARS])
    for metric in ["gave", "amount"]:
        raw = smf.ols(f"{metric} ~ treatment", data=df).fit(cov_type="HC2")
        # Lin (2013) adjustment: centered covariates interacted with treatment
        d = df.copy()
        for c in COVARS:
            d[c + "_c"] = d[c] - d[c].mean()
        inter = " + ".join(f"{c}_c + treatment:{c}_c" for c in COVARS)
        adj = smf.ols(f"{metric} ~ treatment + {inter}", data=d).fit(cov_type="HC2")
        se_raw, se_adj = raw.bse["treatment"], adj.bse["treatment"]
        out[metric] = {"se_raw": float(se_raw), "se_adjusted": float(se_adj),
                       "ci_width_reduction": float(1 - se_adj / se_raw),
                       "equivalent_sample_saving": float(1 - (se_adj / se_raw) ** 2),
                       "estimate_raw": float(raw.params["treatment"]), "estimate_adjusted": float(adj.params["treatment"])}
    return out


# ---------------------------------------------------------------- 6. heterogeneity
SEGMENTS = {
    "Red state": ("red", 1), "Blue state": ("red", 0),
    "Small prior donor (<$35)": ("ltmedmra", 1), "Large prior donor (>=$35)": ("ltmedmra", 0),
    "Gave in last 12 months": ("recent", 1), "Lapsed >12 months": ("recent", 0),
}


def heterogeneity(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = df[df["state_known"] == 1].copy()
    d["recent"] = (d["mrm2"] <= 12).astype(int)
    rows = []
    for name, (col, val) in SEGMENTS.items():
        s = d[d[col] == val]
        for metric in ["gave", "amount"]:
            m = smf.ols(f"{metric} ~ treatment", data=s).fit(cov_type="HC2")
            lo, hi = m.conf_int().loc["treatment"]
            rows.append({"segment": name, "metric": metric, "n": len(s), "control_mean": s.loc[s["treatment"] == 0, metric].mean(),
                         "effect": m.params["treatment"], "ci_lo": lo, "ci_hi": hi, "p": m.pvalues["treatment"]})
    seg = pd.DataFrame(rows)
    seg["p_holm"] = seg.groupby("metric")["p"].transform(lambda p: multipletests(p, method="holm")[1])
    seg.to_csv(TAB / "segment_effects.csv", index=False)
    inter = {}
    for col in ["red", "ltmedmra", "recent"]:
        m = smf.ols(f"amount ~ treatment * {col}", data=d).fit(cov_type="HC2")
        inter[col] = {"interaction": float(m.params[f"treatment:{col}"]), "p": float(m.pvalues[f"treatment:{col}"])}
    return seg, inter


def targeting_policy(df: pd.DataFrame) -> dict:
    """Should the discount go to everyone, or only to segments where it pays?
    Honest evaluation: learn the rule on one half, score it on the other, swap."""
    d = df[df["state_known"] == 1].copy()
    d["recent"] = (d["mrm2"] <= 12).astype(int)
    d["cell"] = d["red"].astype(str) + d["ltmedmra"].astype(str) + d["recent"].astype(str)
    rng = np.random.default_rng(SEED)
    d["fold"] = rng.integers(0, 2, len(d))
    results = []
    for f in (0, 1):
        tr, te = d[d["fold"] != f], d[d["fold"] == f]
        eff = tr.groupby("cell").apply(lambda s: s.loc[s.treatment == 1, "amount"].mean() * 1 - s.loc[s.treatment == 0, "amount"].mean(), include_groups=False)
        cost = tr[tr.treatment == 1].groupby("cell").apply(lambda s: (s["amount"] * s["ratio"]).mean(), include_groups=False)
        target = set(eff[eff > 0].index)
        # Value on held-out half, per letter: incremental donor $ in targeted cells
        te_eff = te.groupby("cell").apply(lambda s: s.loc[s.treatment == 1, "amount"].mean() - s.loc[s.treatment == 0, "amount"].mean(), include_groups=False)
        share = te["cell"].value_counts(normalize=True)
        v_target = float(sum(te_eff.get(c, 0) * share.get(c, 0) for c in target))
        v_all = float(sum(te_eff.get(c, 0) * share.get(c, 0) for c in te_eff.index))
        te_cost = te[te.treatment == 1].groupby("cell").apply(lambda s: (s["amount"] * s["ratio"]).mean(), include_groups=False)
        c_target = float(sum(te_cost.get(c, 0) * share.get(c, 0) for c in target))
        c_all = float(sum(te_cost.get(c, 0) * share.get(c, 0) for c in te_cost.index))
        results.append({"targeted_share": float(sum(share.get(c, 0) for c in target)), "v_target": v_target, "v_all": v_all,
                        "match_cost_target": c_target, "match_cost_all": c_all})
    r = pd.DataFrame(results).mean()
    return {k: float(v) for k, v in r.items()}


# ---------------------------------------------------------------- 7. anchor (price presentation) test
def anchor_test(df: pd.DataFrame) -> pd.DataFrame:
    t = df[df["treatment"] == 1].copy()
    t["anchor"] = np.select([t.askd1 == 1, t.askd2 == 1, t.askd3 == 1], ["1.00x", "1.25x", "1.50x"], "?")
    rows = []
    for metric in ["gave", "amount"]:
        m = smf.ols(f"{metric} ~ C(anchor)", data=t).fit(cov_type="HC2")
        base = t.loc[t["anchor"] == "1.00x", metric].mean()
        for a in ["1.25x", "1.50x"]:
            k = f"C(anchor)[T.{a}]"
            lo, hi = m.conf_int().loc[k]
            rows.append({"metric": metric, "anchor": a, "base_1x": base, "diff": m.params[k], "ci_lo": lo, "ci_hi": hi, "p": m.pvalues[k]})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "anchor_test.csv", index=False)
    return out


# ---------------------------------------------------------------- 8. design for the next test
def power_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Letters per arm needed to detect a relative lift in revenue per letter
    (two-sided alpha .05, power .80), from observed control variance,
    with and without covariate adjustment."""
    ctrl = df.loc[df["treatment"] == 0, "amount"]
    mu, sd = ctrl.mean(), ctrl.std()
    vr = variance_reduction(df)["amount"]["equivalent_sample_saving"]
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    rows = []
    for lift in [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]:
        n = 2 * (z * sd / (lift * mu)) ** 2
        rows.append({"relative_lift": lift, "n_per_arm": n, "n_per_arm_adjusted": n * (1 - vr)})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "power_curve.csv", index=False)
    return out


# ---------------------------------------------------------------- figures
def fig_demand(df, effects):
    g = df.groupby("arm").agg(conv=("gave", "mean"), rev=("amount", "mean")).reindex(ARMS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, col, fmt, title in [(axes[0], "gave", "conv", lambda v: f"{v*100:.2f}%", "Response rate"),
                                        (axes[1], "amount", "rev", lambda v: f"${v:.2f}", "Revenue per letter")]:
        xs = [PRICE[a] for a in ARMS]
        se = df.groupby("arm")[metric].agg(lambda s: s.std() / np.sqrt(len(s))).reindex(ARMS)
        offsets = {"Control": (10, -4), "1:1": (-12, -4), "2:1": (0, 30), "3:1": (10, -4)}
        for a, x in zip(ARMS, xs):
            v = g.loc[a, col]
            ax.errorbar(x, v, yerr=1.96 * se[a], fmt="o", color=ARM_COLOR[a], ms=8, mec="white", mew=2, capsize=0, lw=2)
            dx, dy = offsets[a]
            ax.annotate(f"{a}\n{fmt(v)}", (x, v), textcoords="offset points", xytext=(dx, dy), fontsize=9, color="#52514e",
                        ha="right" if dx < 0 else ("center" if dx == 0 else "left"), va="center")
        ax.set_ylim(0, g[col].max() * 1.45)
        ax.set_xlabel("Price of giving $1 to the charity")
        ax.set_xticks(xs, ["$1.00", "$0.50", "$0.33", "$0.25"])
        ax.set_xlim(1.12, 0.15)
        ax.set_title(title, loc="left", fontsize=11, fontweight="semibold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, f=fmt: f(v).replace(".00%", "%")))
    fig.suptitle("A discount lifts response; deeper discounts add nothing", x=0.01, ha="left", fontsize=13, fontweight="semibold")
    fig.text(0.01, 0.005, "Means with 95% CIs. N = 50,083 randomized letters (Karlan & List 2007).", fontsize=8.5, color="#898781")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(FIG / "fig1_demand_curve.png", dpi=180)
    plt.close(fig)


def fig_efficiency(eff):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ys = np.arange(len(eff))[::-1]
    ax.barh(ys, eff["incremental_per_match_dollar"], height=0.3, color=[ARM_COLOR[a] for a in eff["arm"]])
    for y, v, a in zip(ys, eff["incremental_per_match_dollar"], eff["arm"]):
        ax.text(v + 0.004, y, f"${v:.2f}", va="center", fontsize=10, color="#0b0b0b")
    ax.set_yticks(ys, [f"{a} match" for a in eff["arm"]])
    ax.set_xlabel("Incremental donor dollars per matching dollar spent")
    ax.set_title("The 1:1 match buys the most response per discount dollar", loc="left", fontsize=11, fontweight="semibold")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_discount_efficiency.png", dpi=180)
    plt.close(fig)


def fig_segments(seg):
    s = seg[seg["metric"] == "amount"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ys = np.arange(len(s))[::-1]
    for y, (_, r) in zip(ys, s.iterrows()):
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color="#86b6ef", lw=2, solid_capstyle="round")
        ax.plot(r.effect, y, "o", color=C1, ms=8, mec="white", mew=2)
        sign = "" if abs(r.effect) < 0.005 else ("+" if r.effect > 0 else "−")
        ax.text(max(r.ci_hi, 0) + 0.03, y, f"{sign}${abs(r.effect):.2f}" + (" *" if r.p_holm < 0.05 else ""), va="center", fontsize=9, color="#52514e")
    ax.axvline(0, color="#52514e", lw=1)
    ax.set_yticks(ys, s["segment"])
    ax.set_xlabel("Effect of the match on revenue per letter ($), 95% CI")
    ax.set_title("The effect is concentrated in red states and small prior donors", loc="left", fontsize=11, fontweight="semibold")
    ax.grid(axis="y", visible=False)
    fig.text(0.01, 0.01, "* significant after Holm correction across six segments", fontsize=8.5, color="#898781")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig3_segments.png", dpi=180)
    plt.close(fig)


def fig_power(pc):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(pc["relative_lift"] * 100, pc["n_per_arm"] / 1000, color=C1, lw=2)
    for _, r in pc.iterrows():
        ax.plot(r.relative_lift * 100, r.n_per_arm / 1000, "o", color=C1, ms=7, mec="white", mew=2)
    for lift in (0.05, 0.10, 0.20):
        n = pc.loc[np.isclose(pc.relative_lift, lift), "n_per_arm"].iloc[0] / 1000
        ax.annotate(f"{n:,.0f}K", (lift * 100, n), textcoords="offset points", xytext=(8, 4), fontsize=9, color="#52514e")
    ax.set_xlabel("Revenue-per-user lift to detect (%)")
    ax.set_ylabel("Users per arm")
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}K"))
    n10 = pc.loc[np.isclose(pc.relative_lift, 0.10), "n_per_arm"].iloc[0] / 1000
    ax.set_title(f"Detecting a 10% revenue lift needs ~{n10:.0f}K users per arm",
                 loc="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_power.png", dpi=180)
    plt.close(fig)


def main() -> dict:
    df = load()
    res = {"integrity": integrity(df)}
    eff = arm_effects(df)
    res["pooled"] = pooled_and_ratio_tests(df)
    res["heavy_tail"] = heavy_tail_robustness(df)
    res["arms"] = eff.to_dict(orient="records")
    res["elasticity"] = elasticity(df)
    de = discount_efficiency(df)
    res["discount_efficiency"] = de.to_dict(orient="records")
    res["variance_reduction"] = variance_reduction(df)
    seg, inter = heterogeneity(df)
    res["segments"] = seg.to_dict(orient="records")
    res["interactions"] = inter
    res["targeting"] = targeting_policy(df)
    res["anchor"] = anchor_test(df).to_dict(orient="records")
    pc = power_curve(df)
    res["power"] = pc.to_dict(orient="records")
    fig_demand(df, eff)
    fig_efficiency(de)
    fig_segments(seg)
    fig_power(pc)
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    r = main()
    p = r["pooled"]
    print(json.dumps({k: r[k] for k in ["heavy_tail", "targeting"]}, indent=1, default=float))
    print(pd.DataFrame(r["discount_efficiency"]).round(3).to_string())
    print(pd.DataFrame(r["segments"]).round(4).to_string())
    print(pd.DataFrame(r["anchor"]).round(4).to_string())
    print(pd.DataFrame(r["arms"]).round(4).to_string())
    print(pd.DataFrame(r["power"]).round(0).to_string())
