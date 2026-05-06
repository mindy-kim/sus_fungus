import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]

VARIANT_BENCHMARKS  = WORKSPACE / "solver_variant_benchmarks.json"
FM_BENCHMARKS       = WORKSPACE / "benchmark_fm_warmstart.json"
OUTPUT_TEX          = WORKSPACE / "results_table.tex"

# Which classical variant represents the "best LNS" arm
BEST_LNS_VARIANT = "lns_related_regret2"


def load_grouped(path: Path) -> dict[str, dict[str, dict]]:
    """Load a benchmark JSON and return {variant: {instance: row}}."""
    data = json.loads(path.read_text())
    grouped: dict[str, dict[str, dict]] = {}
    for row in data["rows"]:
        grouped.setdefault(row["variant"], {})[row["instance"]] = row
    return grouped


def pct_change(new_val: float, ref_val: float) -> float:
    """Signed % change relative to ref_val (negative = improvement)."""
    if ref_val == 0:
        return 0.0
    return 100.0 * (new_val - ref_val) / ref_val


def fmt_pct(val: float) -> str:
    return rf"{val:+.1f}\%"


def instance_label(filename: str) -> str:
    return filename.replace(".vrp", "").replace("_", r"\_")


def num_customers(filename: str) -> int:
    """101_11_2.vrp → 100 customers (nodes - 1)."""
    return int(filename.split("_")[0]) - 1


def build_table(instances: list[str], grouped: dict, fm_grouped: dict | None) -> str:
    has_fm = fm_grouped is not None

    if has_fm:
        col_spec = r"@{}l r r r r r r r r r r r r@{}"
    else:
        col_spec = r"@{}l r r r r r r r r r@{}"

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    header1_parts = [
        r"\multicolumn{2}{@{}l}{}",
        r"& \multicolumn{2}{c}{Baseline}",
        r"& \multicolumn{3}{c}{+~Local search}",
        r"& \multicolumn{3}{c}{+~LNS}",
    ]
    if has_fm:
        header1_parts.append(r"& \multicolumn{3}{c@{}}{+~FM warm start}")
    lines.append("  " + " ".join(header1_parts) + r" \\")

    cmidrule = r"\cmidrule(lr){3-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}"
    if has_fm:
        cmidrule += r" \cmidrule(l){11-13}"
    lines.append("  " + cmidrule)

    header2 = (
        r"  Instance & $n$ "
        r"& Obj. & $t$\,(s) "
        r"& Obj. & $\Delta$\,(\%) & $t$\,(s) "
        r"& Obj. & $\Delta$\,(\%) & $t$\,(s)"
    )
    if has_fm:
        header2 += r" & Obj. & $\Delta$\,(\%) & $t$\,(s)"
    header2 += r" \\"
    lines.append(header2)
    lines.append(r"  \midrule")

    baseline_data = grouped["baseline"]
    ls_data       = grouped["local_search"]
    lns_data      = grouped[BEST_LNS_VARIANT]

    col_totals = {"base_obj": 0.0, "base_t": 0.0,
                  "ls_obj": 0.0,   "ls_t": 0.0,
                  "lns_obj": 0.0,  "lns_t": 0.0}
    if has_fm:
        col_totals.update({"fm_obj": 0.0, "fm_t": 0.0})

    for inst in instances:
        n      = num_customers(inst)
        label  = instance_label(inst)

        b_obj  = baseline_data[inst]["result"]
        b_t    = baseline_data[inst]["time"]
        ls_obj = ls_data[inst]["result"]
        ls_t   = ls_data[inst]["time"]
        ln_obj = lns_data[inst]["result"]
        ln_t   = lns_data[inst]["time"]

        ls_delta  = pct_change(ls_obj,  b_obj)
        lns_delta = pct_change(ln_obj,  b_obj)

        col_totals["base_obj"] += b_obj;  col_totals["base_t"] += b_t
        col_totals["ls_obj"]   += ls_obj; col_totals["ls_t"]   += ls_t
        col_totals["lns_obj"]  += ln_obj; col_totals["lns_t"]  += ln_t

        row = (
            f"  {label} & {n} "
            f"& {b_obj:,.2f} & {b_t:.3f} "
            f"& {ls_obj:,.2f} & {ls_delta:+.1f} & {ls_t:.3f} "
            f"& {ln_obj:,.2f} & {lns_delta:+.1f} & {ln_t:.3f}"
        )

        if has_fm:
            fm_obj   = fm_grouped[inst]["fm_obj"]
            fm_t     = fm_grouped[inst]["fm_time"]
            fm_delta = pct_change(fm_obj, b_obj)
            col_totals["fm_obj"] += fm_obj
            col_totals["fm_t"]   += fm_t
            row += f" & {fm_obj:,.2f} & {fm_delta:+.1f} & {fm_t:.3f}"

        row += r" \\"
        lines.append(row)

    lines.append(r"  \midrule")
    b_tot  = col_totals["base_obj"]
    ls_tot = col_totals["ls_obj"]
    ln_tot = col_totals["lns_obj"]
    totals_row = (
        r"  \textbf{Total} & "
        f"& {b_tot:,.2f} & {col_totals['base_t']:.3f} "
        f"& {ls_tot:,.2f} & {pct_change(ls_tot, b_tot):+.1f} & {col_totals['ls_t']:.3f} "
        f"& {ln_tot:,.2f} & {pct_change(ln_tot, b_tot):+.1f} & {col_totals['lns_t']:.3f}"
    )
    if has_fm:
        fm_tot = col_totals["fm_obj"]
        totals_row += (
            f" & {fm_tot:,.2f} & {pct_change(fm_tot, b_tot):+.1f} & {col_totals['fm_t']:.3f}"
        )
    totals_row += r" \\"
    lines.append(totals_row)

    lns_label = BEST_LNS_VARIANT.replace("lns_", "").replace("_", " + ")
    fm_note   = r" FM warm start uses the trained \textsc{EdgeFlowMatchingModel}." if has_fm else ""
    caption = (
        r"Per-instance objective and solve time for each solver tier. "
        rf"$\Delta$\,(\%) is relative to the Baseline. "
        rf"LNS variant: \texttt{{{lns_label}}}.{fm_note}"
    )

    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        r"\label{tab:solver_results}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def main():
    if not VARIANT_BENCHMARKS.exists():
        raise FileNotFoundError(f"Missing {VARIANT_BENCHMARKS}")

    grouped    = load_grouped(VARIANT_BENCHMARKS)
    instances  = sorted(grouped["baseline"].keys())

    fm_grouped: dict | None = None
    if FM_BENCHMARKS.exists():
        fm_data    = json.loads(FM_BENCHMARKS.read_text())
        fm_grouped = {row["instance"]: row for row in fm_data["rows"]}
        instances = [i for i in instances if i in fm_grouped]
        print(f"FM benchmark found — including FM column ({len(instances)} instances).")
    else:
        print("No FM benchmark found — generating three-column table.")
        print(f"Add {FM_BENCHMARKS.name} to include the FM column.")

    tex = build_table(instances, grouped, fm_grouped)
    OUTPUT_TEX.write_text(tex, encoding="utf-8")
    print(f"Written → {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
