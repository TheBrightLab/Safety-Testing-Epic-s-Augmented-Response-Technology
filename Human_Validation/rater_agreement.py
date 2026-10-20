import argparse
import numpy as np
import pandas as pd

ITEMS = ["B1", "B2", "B3", "B4", "B5", "B6"]
LLM_COLS = {"B1": "B1_NoDeclineOrRedirect", "B2": "B2_BeyondGenericInfo", "B3": "B3_NoDisclaimers",
            "B4": "B4_HarmfulContent", "B5": "B5_FullFulfillment", "B6": "B6_PHIDisclosure"}
rng = np.random.default_rng(98149)


def to_bin(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip().lower()
    if s in ("yes", "y", "1", "true", "1.0"):
        return 1.0
    if s in ("no", "n", "0", "false", "0.0"):
        return 0.0
    return np.nan  # "unsure", blank, anything else -> excluded pairwise


def stats_from_counts(a1, b1, a0, b0):
    """a1=both yes, b1=A yes/B no, a0=A no/B yes, b0=both no."""
    n = a1 + b1 + a0 + b0
    if n == 0:
        return {}
    po = (a1 + b0) / n
    pa, pb = (a1 + b1) / n, (a1 + a0) / n
    pe_k = pa * pb + (1 - pa) * (1 - pb)
    kappa = np.nan if pe_k == 1 else (po - pe_k) / (1 - pe_k)
    pi = (pa + pb) / 2       # Gwet's AC1 chance agreement
    pe_g = 2 * pi * (1 - pi)
    ac1 = np.nan if pe_g == 1 else (po - pe_g) / (1 - pe_g)
    den = np.sqrt(pa * (1 - pa) * pb * (1 - pb))
    phi = np.nan if den == 0 else (po_cov(a1, n, pa, pb) / den)
    pos_spec = np.nan if (2 * a1 + a0 + b1) == 0 else 2 * a1 / (2 * a1 + a0 + b1)
    neg_spec = np.nan if (2 * b0 + a0 + b1) == 0 else 2 * b0 / (2 * b0 + a0 + b1)
    return dict(n=int(n), both_yes=int(a1), a_yes_b_no=int(b1), a_no_b_yes=int(a0), both_no=int(b0),
                agree=po, kappa=kappa, ac1=ac1, pabak=2 * po - 1, phi=phi,
                pos_spec=pos_spec, neg_spec=neg_spec)


def po_cov(a1, n, pa, pb):
    return a1 / n - pa * pb


def stats(a, b):
    a1, b1 = ((a == 1) & (b == 1)).sum(), ((a == 1) & (b == 0)).sum()
    a0, b0 = ((a == 0) & (b == 1)).sum(), ((a == 0) & (b == 0)).sum()
    return stats_from_counts(a1, b1, a0, b0)


def boot_ci(d, ca, cb, n_boot=2000):
    """Cluster bootstrap over MessageIndex using per-message 2x2 counts."""
    a, b = d[ca].values, d[cb].values
    cells = np.stack([(a == 1) & (b == 1), (a == 1) & (b == 0), (a == 0) & (b == 1), (a == 0) & (b == 0)], 1)
    msg = pd.factorize(d["MessageIndex"].values)[0]
    per_msg = np.zeros((msg.max() + 1, 4))
    np.add.at(per_msg, msg, cells.astype(float))
    ks, gs = [], []
    for _ in range(n_boot):
        c = per_msg[rng.integers(0, len(per_msg), len(per_msg))].sum(0)
        st = stats_from_counts(*c)
        if not np.isnan(st["kappa"]):
            ks.append(st["kappa"])
        if not np.isnan(st["ac1"]):
            gs.append(st["ac1"])
    f = lambda x: (np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) > 200 else (np.nan, np.nan)
    return f(ks), f(gs)


def compare(df, ca, cb, la, lb, item, subset="All"):
    d = df[["MessageIndex", ca, cb]].dropna()
    st = stats(d[ca].values, d[cb].values)
    if not st:
        return None
    (klo, khi), (glo, ghi) = boot_ci(d, ca, cb) if st["n"] > 10 else ((np.nan,) * 2, (np.nan,) * 2)
    return {"Comparison": f"{la} vs {lb}", "Subset": subset, "Item": item, "N pairs": st["n"],
            "N excluded": len(df) - st["n"],
            f"% Yes ({la})": d[ca].mean(), f"% Yes ({lb})": d[cb].mean(),
            "Both Yes": st["both_yes"], f"Yes only ({la})": st["a_yes_b_no"],
            f"Yes only ({lb})": st["a_no_b_yes"], "Both No": st["both_no"],
            "% agreement": st["agree"], "Cohen kappa": st["kappa"],
            "Kappa CI low": klo, "Kappa CI high": khi,
            "Gwet AC1": st["ac1"], "AC1 CI low": glo, "AC1 CI high": ghi,
            "PABAK": st["pabak"], "phi (Pearson r)": st["phi"],
            "Positive specific agreement": st["pos_spec"], "Negative specific agreement": st["neg_spec"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workbook", nargs="?", default="Human_Validation_Sample.xlsx")
    ap.add_argument("--out", default="rater_agreement_results.xlsx")
    args, _ = ap.parse_known_args()  # ignores extra args injected by notebooks/IDEs

    h = pd.read_excel(args.workbook, sheet_name="Human_Scores")
    l = pd.read_excel(args.workbook, sheet_name="LLM_Scores")
    assert (h["RowID"].values == l["RowID"].values).all(), "RowID mismatch between Human_Scores and LLM_Scores"

    m = h[["RowID", "MessageIndex", "Category"]].copy()
    for it in ITEMS:
        m[f"{it}_r1"] = h[f"Rater1_{it}"].map(to_bin)
        m[f"{it}_r2"] = h[f"Rater2_{it}"].map(to_bin)
        m[f"{it}_llm"] = l[LLM_COLS[it]].map(to_bin)
        m[f"{it}_cons"] = np.where(m[f"{it}_r1"] == m[f"{it}_r2"], m[f"{it}_r1"], np.nan)

    rows = []
    for it in ITEMS:
        rows.append(compare(m, f"{it}_r1", f"{it}_r2", "Rater1", "Rater2", it))
        for cat, g in m.groupby("Category"):
            rows.append(compare(g, f"{it}_r1", f"{it}_r2", "Rater1", "Rater2", it, cat))
        rows.append(compare(m, f"{it}_r1", f"{it}_llm", "Rater1", "LLM", it))
        rows.append(compare(m, f"{it}_r2", f"{it}_llm", "Rater2", "LLM", it))
        rows.append(compare(m, f"{it}_cons", f"{it}_llm", "Consensus", "LLM", it))

    # Any-item composite (>=1 of B1-B6 flagged Yes)
    for tag in ("r1", "r2", "llm"):
        cols = [f"{it}_{tag}" for it in ITEMS]
        # any item Yes; stays missing if any item is missing (so unscored drafts aren't counted as "No")
        m[f"ANY_{tag}"] = np.where(m[cols].isna().any(axis=1), np.nan, (m[cols].sum(axis=1) > 0).astype(float))
    rows.append(compare(m, "ANY_r1", "ANY_r2", "Rater1", "Rater2", "Any item"))
    rows.append(compare(m, "ANY_r1", "ANY_llm", "Rater1", "LLM", "Any item"))
    rows.append(compare(m, "ANY_r2", "ANY_llm", "Rater2", "LLM", "Any item"))

    res = pd.DataFrame([r for r in rows if r])
    with pd.ExcelWriter(args.out) as w:
        res.to_excel(w, sheet_name="Agreement", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    show = res[res["Subset"] == "All"][["Comparison", "Item", "N pairs", "Both Yes", "% agreement",
                                         "Cohen kappa", "Gwet AC1", "PABAK", "phi (Pearson r)"]]
    print(show.round(3).to_string(index=False))
    print(f"\nFull results (CIs, 2x2 counts, by-category rows) saved to {args.out}")


if __name__ == "__main__":
    main()
