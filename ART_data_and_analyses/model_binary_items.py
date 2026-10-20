import argparse
import itertools
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from scipy.stats import beta, chi2, kruskal, mannwhitneyu, norm
from statsmodels.stats.multitest import multipletests

ITEMS = {"B1": "B1_NoDeclineOrRedirect", "B2": "B2_BeyondGenericInfo", "B3": "B3_NoDisclaimers",
         "B4": "B4_HarmfulContent", "B5": "B5_FullFulfillment", "B6": "B6_PHIDisclosure"}
MODEL_ITEMS = ["B1", "B6"]
CATS = ["General", "Medications", "Paperwork", "Results"]  # General = reference


def load(path, include_sentinels=False):
    cols = ["MessageIndex", "PatientIndex", "PromptCategoryUsed", "ResponseType"] + list(ITEMS.values())
    d = pd.read_excel(path, sheet_name="Test Results", usecols=cols)
    if not include_sentinels:
        d = d[d["ResponseType"] == "Draft"]
    d = d.copy()
    d["category"] = pd.Categorical(d["PromptCategoryUsed"], categories=CATS)
    return d.reset_index(drop=True)


def get_item(d, k):
    c = ITEMS[k]
    s = d[d[c].isin(["Yes", "No"])].copy()  # unscored rows dropped
    s["y"] = (s[c] == "Yes").astype(int)
    return s


def clopper_pearson(x, n):
    lo = 0.0 if x == 0 else beta.ppf(0.025, x, n - x + 1)
    hi = 1.0 if x == n else beta.ppf(0.975, x + 1, n - x)
    return lo, hi


def descriptives(d):
    rows = []
    for k in ITEMS:
        s = get_item(d, k)
        for cat, g in s.groupby("category", observed=True):
            per_msg = g.groupby("MessageIndex").y.sum()
            n, x = len(per_msg), int((per_msg > 0).sum())
            lo, hi = clopper_pearson(x, n)
            rows.append({"item": k, "category": cat, "drafts": len(g), "flagged_drafts": int(g.y.sum()),
                         "pct_drafts": round(100 * g.y.mean(), 2), "messages": n, "messages_with_flag": x,
                         "pct_messages": round(100 * x / n, 1),
                         "msg_ci_low": round(100 * lo, 1), "msg_ci_high": round(100 * hi, 1)})
    return pd.DataFrame(rows)


def design(cat_series):
    X = pd.get_dummies(cat_series.astype(str)).reindex(columns=CATS, fill_value=0).astype(float)
    X = X.drop(columns="General")
    return sm.add_constant(X, has_constant="add")


def category_results(res, item, analysis):
    """Rates, omnibus Wald test and Holm-adjusted pairwise odds ratios from a fitted category-only logit."""
    b, V = res.params.values, res.cov_params().values
    names = list(res.params.index)         # const, Medications, Paperwork, Results
    L = {c: np.array([1.0] + [1.0 if n == c else 0.0 for n in names[1:]]) for c in CATS}
    L["General"] = np.array([1.0] + [0.0] * (len(names) - 1))

    z = norm.ppf(0.975)
    rates = []
    for c in CATS:
        eta, se = L[c] @ b, np.sqrt(L[c] @ V @ L[c])
        rates.append({"item": item, "analysis": analysis, "category": c,
                      "rate_pct": round(100 / (1 + np.exp(-eta)), 2),
                      "ci_low_pct": round(100 / (1 + np.exp(-(eta - z * se))), 2),
                      "ci_high_pct": round(100 / (1 + np.exp(-(eta + z * se))), 2)})

    R = np.eye(len(names))[1:]     # omnibus: all category coefficients = 0
    w = float((R @ b) @ np.linalg.solve(R @ V @ R.T, R @ b))
    omni = {"item": item, "analysis": analysis, "wald_chisq": round(w, 2), "df": R.shape[0],
            "p_value": chi2.sf(w, R.shape[0])}

    con = []
    for a, c in itertools.combinations(CATS, 2):
        l = L[a] - L[c]
        est, se = l @ b, np.sqrt(l @ V @ l)
        con.append({"item": item, "analysis": analysis, "contrast": f"{a} vs {c}",
                    "odds_ratio": np.exp(est), "or_ci_low": np.exp(est - z * se), "or_ci_high": np.exp(est + z * se),
                    "p_raw": 2 * norm.sf(abs(est / se))})
    p_holm = multipletests([r["p_raw"] for r in con], method="holm")[1]
    for r, p in zip(con, p_holm):
        r["p_holm"] = p
    return rates, omni, con


def fit_primary(s, item):
    """Draft-level logit, SEs clustered on message AND patient profile (two-way)."""
    X = design(s["category"])
    groups = np.column_stack([pd.factorize(s["MessageIndex"])[0], pd.factorize(s["PatientIndex"])[0]])
    res = sm.GLM(s["y"].values, X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    min_eig = np.linalg.eigvalsh(res.cov_params().values).min()
    if min_eig < 0:
        print(f"  WARNING ({item}): two-way cluster covariance not positive semi-definite; "
              f"treat primary-analysis p-values with caution and lean on the message-level sensitivity analysis.")
    return res


def message_level_nonparametric(s, item):
    """One value per message = share of its drafts flagged. Kruskal-Wallis across categories,
    Holm-adjusted pairwise Mann-Whitney tests, plus per-category summaries."""
    g = s.groupby(["MessageIndex", "category"], observed=True)["y"].mean().reset_index()
    groups = {c: g.loc[g["category"] == c, "y"].values for c in CATS}
    summ = pd.DataFrame([{"item": item, "category": c, "messages": len(v),
                          "mean_rate_pct": round(100 * v.mean(), 2), "sd_of_message_rates_pct": round(100 * v.std(ddof=1), 2),
                          "median_rate_pct": round(100 * np.median(v), 2),
                          "pct_messages_with_any_flag": round(100 * (v > 0).mean(), 1)} for c, v in groups.items()])
    H, p = kruskal(*groups.values())
    omni = {"item": item, "analysis": f"SENSITIVITY: message-level Kruskal-Wallis (n = {len(g)} messages)",
            "wald_chisq": round(float(H), 2), "df": len(CATS) - 1, "p_value": p}
    pw = [{"item": item, "contrast": f"{a} vs {b}",
           "p_raw": mannwhitneyu(groups[a], groups[b], alternative="two-sided")[1]}
          for a, b in itertools.combinations(CATS, 2)]
    for r, ph in zip(pw, multipletests([r["p_raw"] for r in pw], method="holm")[1]):
        r["p_holm"] = ph
    return summ, omni, pw


def run(d, items=MODEL_ITEMS):
    rates, omni, con, summ, pw = [], [], [], [], []
    for k in items:
        s = get_item(d, k)
        print(f"\n== {k}: {len(s):,} drafts, {s.MessageIndex.nunique()} messages, {s.PatientIndex.nunique()} profiles ==")
        res = fit_primary(s, k)
        r, o, c = category_results(res, k, "PRIMARY: draft-level logit, two-way cluster-robust (message, profile)")
        rates += r; omni.append(o); con += c
        sm_, o2, p2 = message_level_nonparametric(s, k)
        summ.append(sm_); omni.append(o2); pw += p2
    return (pd.DataFrame(rates), pd.DataFrame(omni), pd.DataFrame(con),
            pd.concat(summ, ignore_index=True), pd.DataFrame(pw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="final_ART_data.xlsx")
    ap.add_argument("--out", default="binary_model_results.xlsx")
    ap.add_argument("--include-sentinels", action="store_true",
                    help="also analyse the 'Sentinel: ***' / 'Sentinel: Unknown' non-draft rows (default: drafts only)")
    args, _ = ap.parse_known_args()

    d = load(args.data, args.include_sentinels)
    print(f"Rows analysed: {len(d):,} (include_sentinels={args.include_sentinels})")
    desc = descriptives(d)
    rates, omni, con, summ, pw = run(d)

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", lambda x: f"{x:.3g}")
    print("\n--- Category rates (95% CI) ---");        print(rates.to_string(index=False))
    print("\n--- Omnibus test of category (primary Wald chi-square; sensitivity = Kruskal-Wallis H) ---"); print(omni.to_string(index=False))
    print("\n--- Pairwise odds ratios, primary analysis (Holm-adjusted) ---"); print(con.to_string(index=False))
    print("\n--- Message-level summary ---");           print(summ.to_string(index=False))
    print("\n--- Pairwise Mann-Whitney, sensitivity analysis (Holm-adjusted) ---"); print(pw.to_string(index=False))

    with pd.ExcelWriter(args.out) as w:
        desc.to_excel(w, sheet_name="Descriptives_all_items", index=False)
        rates.to_excel(w, sheet_name="Category_rates", index=False)
        omni.to_excel(w, sheet_name="Omnibus", index=False)
        con.to_excel(w, sheet_name="Pairwise_ORs", index=False)
        summ.to_excel(w, sheet_name="Message_level_summary", index=False)
        pw.to_excel(w, sheet_name="Pairwise_MannWhitney", index=False)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()



