import sys
from pathlib import Path
import numpy as np

COLS = ["transaction_id", "day_index", "outlet_id", "product_code",
        "category_code", "quantity", "unit_price", "unit_cost",
        "discount_rate", "stock_before_units", "payment_code", "customer_rating"]
I = {name: n for n, name in enumerate(COLS)}
PRODUCTS = ["Orange Juice", "Mango Juice", "Apple Juice", "Watermelon Juice",
            "Banana Smoothie", "Mixed Fruit Smoothie", "Lemonade", "Iced Tea"]
PRODUCT_CAT = np.array([1, 1, 1, 1, 2, 2, 3, 4])
OUTLETS = ["Outlet A", "Outlet B", "Outlet C"]
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path("freshflow_sales.csv").resolve()
RESULTS = OUT.parent / "freshflow_output_final"
RESULTS.mkdir(exist_ok=True)


def save_csv(name, header, rows):
    np.savetxt(RESULTS / name, np.asarray(rows, dtype=float), delimiter=",",
               header=header, comments="", fmt="%.4f")


def divide(a, b):
    """Division that returns NaN rather than infinity when b is zero."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    shape = np.broadcast_shapes(a.shape, b.shape)
    ans = np.full(shape, np.nan)
    np.divide(a, b, out=ans, where=np.broadcast_to(b != 0, shape))
    return ans


def whole(x):
    x = np.asarray(x, float)
    return np.isfinite(x) & (np.abs(x - np.rint(x)) < 1e-9)


def load():
    if not OUT.exists():
        raise FileNotFoundError(f"CSV not found: {OUT}")
    with OUT.open(encoding="utf-8-sig") as f:
        header = f.readline().strip().split(",")
    if header != COLS:
        raise ValueError("CSV header does not match the 12 assignment columns.")
    x = np.genfromtxt(OUT, delimiter=",", skip_header=1, dtype=float,
                      filling_values=np.nan, ndmin=2)
    if x.shape != (5000, 12):
        raise ValueError(f"Expected 5000 rows and 12 columns; got {x.shape}.")
    return x


def audit(x):
    n = len(x)
    f = {}
    code = lambda c, allowed: (~whole(x[:, c])) | (~np.isin(x[:, c], allowed))
    ids, counts = np.unique(x[np.isfinite(x[:, I["transaction_id"]]), I["transaction_id"]],
                            return_counts=True)
    dup_ids = ids[counts > 1]
    f["duplicate_id"] = np.isin(x[:, I["transaction_id"]], dup_ids)
    f["missing_price"] = np.isnan(x[:, I["unit_price"]])
    f["missing_cost"] = np.isnan(x[:, I["unit_cost"]])
    f["missing_rating"] = np.isnan(x[:, I["customer_rating"]])
    f["bad_id"] = (~whole(x[:, I["transaction_id"]])) | (x[:, I["transaction_id"]] <= 0)
    f["bad_day"] = code(I["day_index"], np.arange(1, 31))
    f["bad_outlet"] = code(I["outlet_id"], [1, 2, 3])
    f["bad_product"] = code(I["product_code"], np.arange(1, 9))
    f["bad_category"] = code(I["category_code"], [1, 2, 3, 4])
    f["category_mismatch"] = np.zeros(n, bool)
    okp = whole(x[:, I["product_code"]]) & np.isin(x[:, I["product_code"]], np.arange(1, 9))
    r = np.where(okp)[0]
    f["category_mismatch"][r] = x[r, I["category_code"]] != PRODUCT_CAT[x[r, I["product_code"]].astype(int)-1]
    f["bad_quantity"] = (~whole(x[:, I["quantity"]])) | (x[:, I["quantity"]] <= 0)
    f["bad_price"] = (~np.isfinite(x[:, I["unit_price"]])) | (x[:, I["unit_price"]] <= 0)
    f["bad_cost"] = (~np.isfinite(x[:, I["unit_cost"]])) | (x[:, I["unit_cost"]] < 0)
    f["bad_discount"] = (~np.isfinite(x[:, I["discount_rate"]])) | (x[:, I["discount_rate"]] < 0) | (x[:, I["discount_rate"]] > 1)
    f["bad_stock"] = (~whole(x[:, I["stock_before_units"]])) | (x[:, I["stock_before_units"]] < 0)
    f["bad_payment"] = code(I["payment_code"], [1, 2, 3, 4])
    f["bad_rating"] = np.isfinite(x[:, I["customer_rating"]]) & ((x[:, I["customer_rating"]] < 1) | (x[:, I["customer_rating"]] > 5))
    f["missing_any"] = np.isnan(x).any(axis=1)
    return f


def main():
    x = load()
    print("\nFRESHFLOW JUICE COMPANY - NUMPY ANALYSIS")
    print("Shape:", x.shape, "| dtype:", x.dtype, "| bytes:", x.nbytes)
    print("\nA. DATA PROFILE")
    for j, name in enumerate(COLS):
        col = x[:, j]
        print(f"{name}: missing={np.isnan(col).sum()}, finite={np.isfinite(col).sum()}")
    flags = audit(x)

    print("\nB. DATA QUALITY AUDIT")
    for name, mask in flags.items():
        print(f"{name}: {int(mask.sum())}")
    all_flags = np.column_stack(list(flags.values())).any(axis=1)
    print("Rows with any flag:", int(all_flags.sum()))
    flag_matrix = np.column_stack(list(flags.values())).astype(int)
    save_csv("quality_flags.csv", ",".join(flags.keys()), flag_matrix)
    audit_text = ["DATA QUALITY AUDIT", f"Rows: {len(x)}",
                  f"Rows with at least one flag: {int(all_flags.sum())}"]
    audit_text += [f"{name}: {int(mask.sum())}" for name, mask in flags.items()]
    (RESULTS / "02_data_quality_audit.txt").write_text("\\n".join(audit_text), encoding="utf-8")

    # Strict financial rows: keep raw data untouched; exclude duplicate IDs and
    # records with invalid fields needed to calculate sales/profit reliably.
    required = ["bad_id", "bad_day", "bad_outlet", "bad_product", "bad_category",
                "category_mismatch", "duplicate_id", "bad_quantity", "bad_price",
                "bad_cost", "bad_discount", "bad_stock", "bad_payment"]
    valid = ~np.column_stack([flags[k] for k in required]).any(axis=1)
    clean = x[valid].copy()
    print("\nC. CLEANING")
    print("Raw rows:", len(x), "| included:", len(clean), "| excluded:", len(x)-len(clean))
    print("Missing ratings retained (not replaced):", int(np.isnan(clean[:, I["customer_rating"]]).sum()))
    print("Raw array unchanged:", x.shape == (5000, 12))
    clean_note = [
        "CLEANING STRATEGY",
        "The original NumPy array is kept unchanged.",
        "Financial analysis excludes duplicate IDs and rows with invalid identifiers,",
        "codes, category mapping, quantity, price, cost, discount, stock or payment code.",
        "Missing prices/costs are not imputed. Missing ratings remain missing.",
        "Ratings outside 1-5 are excluded only from rating statistics.",
        "Unusual but plausible prices/discounts are flagged for review, not deleted.",
        f"Rows included: {len(clean)}; rows excluded: {len(x)-len(clean)}."
    ]
    (RESULTS / "03_cleaning_strategy.txt").write_text("\\n".join(clean_note), encoding="utf-8")
    dictionary = [
        "DATA DICTIONARY",
        "transaction_id: unique positive transaction identifier",
        "day_index: observation day, 1-30",
        "outlet_id: 1 Outlet A, 2 Outlet B, 3 Outlet C",
        "product_code: 1 Orange, 2 Mango, 3 Apple, 4 Watermelon, 5 Banana Smoothie, 6 Mixed Fruit Smoothie, 7 Lemonade, 8 Iced Tea",
        "category_code: 1 Juice, 2 Smoothie, 3 Lemonade, 4 Tea",
        "quantity: positive integer units sold",
        "unit_price: selling price per unit (INR)",
        "unit_cost: direct cost per unit (INR)",
        "discount_rate: fraction from 0 to 1 (e.g. 0.15 = 15%)",
        "stock_before_units: recorded stock before sale",
        "payment_code: 1 Cash, 2 UPI, 3 Card, 4 Other digital payment",
        "customer_rating: 1-5; missing values allowed"
    ]
    (RESULTS / "01_data_dictionary.txt").write_text("\\n".join(dictionary), encoding="utf-8")

    q, price, cost, disc = (clean[:, I[k]] for k in
                            ["quantity", "unit_price", "unit_cost", "discount_rate"])
    gross = q * price
    discount = gross * disc
    revenue = gross - discount
    cogs = q * cost
    profit = revenue - cogs
    margin = divide(profit, revenue) * 100
    avg_price = divide(revenue, q)
    metrics = np.column_stack([gross, discount, revenue, cogs, profit, margin, avg_price])
    save_csv("derived_metrics.csv",
             "gross_sales,discount_amount,net_revenue,cost_of_goods_sold,gross_profit,gross_margin_pct,average_selling_price",
             metrics)
    print("\nD. METRICS")
    print(f"Gross sales: {gross.sum():,.2f}")
    print(f"Discount: {discount.sum():,.2f}")
    print(f"Net revenue: {revenue.sum():,.2f}")
    print(f"Cost of goods sold: {cogs.sum():,.2f}")
    print(f"Gross profit: {profit.sum():,.2f}")
    print(f"Gross margin: {divide(profit.sum(), revenue.sum())*100:.2f}%")
    print(f"Average selling price: {divide(revenue.sum(), q.sum()):.2f}")

    print("\nE. OUTLET AND PRODUCT COMPARISON")
    outlet_rows = []
    for code in (1, 2, 3):
        m = clean[:, I["outlet_id"]] == code
        outlet_rows.append([code, m.sum(), revenue[m].sum(), profit[m].sum(), q[m].sum()])
        print(f"{OUTLETS[code-1]}: revenue={revenue[m].sum():,.2f}, profit={profit[m].sum():,.2f}")
    save_csv("outlet_comparison.csv", "outlet_id,transactions,revenue,gross_profit,units", outlet_rows)
    product_rows = []
    for code in range(1, 9):
        m = clean[:, I["product_code"]] == code
        product_rows.append([code, m.sum(), q[m].sum(), revenue[m].sum(), profit[m].sum()])
        print(f"{PRODUCTS[code-1]}: units={q[m].sum():.0f}, revenue={revenue[m].sum():,.2f}, profit={profit[m].sum():,.2f}")
    save_csv("product_comparison.csv", "product_code,transactions,units,revenue,gross_profit", product_rows)

    print("\nF. DESCRIPTIVE STATISTICS (mean, median, population SD, Q1, Q3)")
    for label, values in [("Net revenue", revenue), ("Gross profit", profit),
                          ("Quantity", q), ("Discount rate", disc)]:
        print(label, np.round([np.mean(values), np.median(values), np.std(values),
                               np.min(values), np.max(values), np.ptp(values),
                               np.var(values), *np.percentile(values, [25, 75])], 2))
    hist, edges = np.histogram(revenue, bins=5)
    print("Revenue histogram counts:", hist.tolist(), "edges:", np.round(edges, 2).tolist())
    ratings = clean[:, I["customer_rating"]]
    ratings = ratings[np.isfinite(ratings) & (ratings >= 1) & (ratings <= 5)]
    print("Valid customer ratings:", len(ratings), "mean:", round(np.mean(ratings), 2),
          "median:", round(np.median(ratings), 2))
    print("Revenue skewness:", round(float(np.mean(((revenue-revenue.mean())/revenue.std())**3)), 3))

    print("\nG. DAILY TREND")
    days = np.arange(1, 31)
    daily = np.array([revenue[clean[:, I["day_index"]] == d].sum() for d in days])
    change = np.r_[np.nan, np.diff(daily)]
    pct = np.r_[np.nan, divide(np.diff(daily), daily[:-1]) * 100]
    moving = np.array([np.nan if i < 2 else daily[i-2:i+1].mean() for i in range(30)])
    threshold = np.nanstd(change[1:]) * 2
    unusual = np.abs(change) > threshold
    trend = np.column_stack([days, daily, change, pct, moving, unusual.astype(int)])
    save_csv("daily_trend.csv", "day,revenue,change,change_pct,three_day_average,unusual_day", trend)
    print("Highest revenue day:", int(days[np.argmax(daily)]),
          "| lowest:", int(days[np.argmin(daily)]), "| unusual days:", days[unusual].tolist())

    print("\nH. DISCOUNT ANALYSIS")
    bands = [(0, .05), (.05, .10), (.10, .20), (.20, .30), (.30, 1.000001)]
    band_rows = []
    for low, high in bands:
        m = (disc >= low) & (disc < high)
        band_rows.append([low, high, m.sum(), q[m].sum(), revenue[m].sum(),
                          profit[m].sum(), divide(profit[m].sum(), revenue[m].sum())*100])
    save_csv("discount_analysis.csv",
             "lower_rate,upper_rate,transactions,units,net_revenue,gross_profit,gross_margin_pct", band_rows)
    for row in band_rows:
        print(f"{row[0]:.0%}-{row[1]:.0%}: n={row[2]}, units={row[3]:.0f}, "
              f"revenue={row[4]:.2f}, margin={row[6]:.2f}%")
    print("These are group comparisons; they do not prove discounts caused sales changes.")

    print("\nI. ANOMALY REGISTER")
    # Flag review cases; outliers are not automatically removed.
    raw_price = x[:, I["unit_price"]]
    high_price = np.isfinite(raw_price) & (raw_price > np.nanpercentile(raw_price, 99))
    high_disc = x[:, I["discount_rate"]] > .30
    zero_stock = (x[:, I["stock_before_units"]] == 0) & (x[:, I["quantity"]] > 0)
    # IQR and z-score are review signals, not automatic deletion rules.
    raw_rev = x[:, I["quantity"]] * x[:, I["unit_price"]] * (1-x[:, I["discount_rate"]])
    finite_rev = raw_rev[np.isfinite(raw_rev)]
    q1, q3 = np.percentile(finite_rev, [25, 75])
    iqr = q3-q1
    iqr_review = np.isfinite(raw_rev) & ((raw_rev < q1-1.5*iqr) | (raw_rev > q3+1.5*iqr))
    z = divide(raw_rev-np.nanmean(finite_rev), np.nanstd(finite_rev))
    z_review = np.isfinite(z) & (np.abs(z) > 3)
    anomaly_masks = [("high_price_review", high_price), ("high_discount_review", high_disc),
                     ("zero_stock_before_sale", zero_stock), ("revenue_iqr_review", iqr_review),
                     ("revenue_zscore_review", z_review)]
    anomaly_rows = []
    for reason, mask in anomaly_masks:
        for row in x[mask]:
            rid = row[I["transaction_id"]]
            # Financial impact is shown only when all required values are valid.
            can_calc = (np.isfinite(row[I["quantity"]]) and row[I["quantity"]] > 0 and
                        np.isfinite(row[I["unit_price"]]) and row[I["unit_price"]] > 0 and
                        np.isfinite(row[I["unit_cost"]]) and row[I["unit_cost"]] >= 0 and
                        np.isfinite(row[I["discount_rate"]]) and 0 <= row[I["discount_rate"]] <= 1)
            impact = (row[I["quantity"]] * row[I["unit_price"]] *
                      (1-row[I["discount_rate"]]) - row[I["quantity"]] * row[I["unit_cost"]]) if can_calc else np.nan
            anomaly_rows.append([rid, reason == "high_price_review", reason == "high_discount_review",
                                 reason == "zero_stock_before_sale", reason == "revenue_iqr_review",
                                 reason == "revenue_zscore_review", impact])
    save_csv("anomaly_register.csv",
             "transaction_id,high_price,high_discount,zero_stock,revenue_iqr,revenue_zscore,estimated_gross_profit",
             anomaly_rows if anomaly_rows else [[np.nan]*7])
    print("High price:", int(high_price.sum()), "High discount:", int(high_disc.sum()),
          "Zero stock with sale:", int(zero_stock.sum()),
          "IQR review:", int(iqr_review.sum()), "Z-score review:", int(z_review.sum()))

    print("\nJ. DISCOUNT SCENARIOS (same valid rows; quantity and cost held constant)")
    scenarios = []
    for rate in (.05, .15):
        r = gross * (1-rate)
        p = r-cogs
        scenarios.append([rate, r.sum(), p.sum(), r.sum()-revenue.sum(), p.sum()-profit.sum()])
    save_csv("discount_scenarios.csv",
             "assumed_discount,revenue,gross_profit,revenue_change,profit_change", scenarios)
    for s in scenarios:
        print(f"Flat {s[0]:.0%}: revenue={s[1]:,.2f}, gross profit={s[2]:,.2f}")
    print("Scenario only, not a forecast; demand response is not modelled.")

    print("\nK. VALIDATION")
    # Independently recalculate the first five valid rows.
    for n, row in enumerate(clean[:5], 1):
        g = row[I["quantity"]] * row[I["unit_price"]]
        d = g * row[I["discount_rate"]]
        r = g-d
        p = r-row[I["quantity"]]*row[I["unit_cost"]]
        pos = n - 1
        np.testing.assert_allclose([g, d, r, p],
                                   [gross[pos], discount[pos], revenue[pos], profit[pos]],
                                   rtol=1e-8, atol=1e-8)
        print(f"Sample {n}: hand check passed")
    assert np.isnan(divide(1, 0))
    assert np.isnan(np.array([np.nan])[0])
    assert clean.ndim == 2 and clean.shape[1] == 12
    print("NaN, zero denominator and array-shape tests passed.")

    print("\nL. THREE FINDINGS AND ACTIONS")
    best_out = int(np.argmax([r[2] for r in outlet_rows])) + 1
    best_prod_units = int(np.argmax([r[2] for r in product_rows])) + 1
    best_prod_profit = int(np.argmax([r[4] for r in product_rows])) + 1
    findings = [
        f"{OUTLETS[best_out-1]} has the highest net revenue; compare its product mix with other outlets.",
        f"{PRODUCTS[best_prod_units-1]} sells the most units; monitor stock and availability.",
        f"{PRODUCTS[best_prod_profit-1]} has the highest gross profit; review its price and cost pattern."
    ]
    for k, item in enumerate(findings, 1):
        print(f"{k}. {item}")
    print("\nLimitations: 30 days only; gross profit excludes operating expenses; group differences")
    print("do not prove causation; scenarios hold quantity constant; no inventory waste claim is made.")
    report = [
        "FRESHFLOW JUICE COMPANY - SHORT REPORT",
        f"Dataset: {len(x)} rows, 30 days, 3 outlets and 8 products.",
        f"Financial records used: {len(clean)}.",
        f"Net revenue: INR {revenue.sum():.2f}; gross profit: INR {profit.sum():.2f};",
        f"gross margin: {divide(profit.sum(), revenue.sum())*100:.2f}%.",
        "Findings and recommended actions:",
        *[f"{i+1}. {v}" for i, v in enumerate(findings)],
        "Scenario: flat 5% and 15% discounts were compared on the same records.",
        "Quantity and unit cost were held constant; this is not a forecast.",
        "Limitations: only 30 days; gross profit excludes operating expenses;",
        "group differences do not establish causation; transaction records alone",
        "cannot establish inventory waste or stock flow.",
        "Appendix: see data dictionary, audit, cleaning strategy, CSV outputs and code."
    ]
    (RESULTS / "04_final_report.txt").write_text("\\n".join(report), encoding="utf-8")
    print("\nFinished. Output folder:", RESULTS)


if __name__ == "__main__":
    main()
