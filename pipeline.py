"""Global temperature anomaly pipeline (NASA GISTEMP v4).

Usage:
    python pipeline.py --download            # fetch latest data, then process
    python pipeline.py --input data/GLB.csv  # process a local copy

Outputs (in --outdir, default outputs/):
    annual.csv     year, annual mean anomaly (°C), months used
    decadal.csv    decade, mean anomaly (°C)
    summary.json   trend (°C/decade), warmest years, data checks
    anomaly.png    annual anomalies with 10-year rolling mean
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

GISTEMP_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MIN_MONTHS = 10  # a year needs at least this many valid months
VALID_RANGE = (-3.0, 3.0)  # plausible global anomaly bounds in °C


def download(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {GISTEMP_URL}")
    urllib.request.urlretrieve(GISTEMP_URL, dest)
    return dest


def load(path: Path) -> pd.DataFrame:
    """Read the GISTEMP table: first line is a title, '***' marks missing."""
    df = pd.read_csv(path, skiprows=1, na_values=["***", "****"])
    missing = [c for c in ["Year"] + MONTHS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")
    df = df[["Year"] + MONTHS].copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)
    for m in MONTHS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    return df.sort_values("Year").reset_index(drop=True)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Mask out-of-range values and drop duplicate years."""
    checks = {"duplicate_years_dropped": int(df["Year"].duplicated().sum())}
    df = df.drop_duplicates(subset="Year", keep="last")
    vals = df[MONTHS]
    bad = (vals < VALID_RANGE[0]) | (vals > VALID_RANGE[1])
    checks["out_of_range_values_masked"] = int(bad.sum().sum())
    df[MONTHS] = vals.mask(bad)
    checks["missing_monthly_values"] = int(df[MONTHS].isna().sum().sum())
    return df.reset_index(drop=True), checks


def annual_means(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "year": df["Year"],
        "months_used": df[MONTHS].notna().sum(axis=1),
        "anomaly_c": df[MONTHS].mean(axis=1).round(3),
    })
    return out[out["months_used"] >= MIN_MONTHS].reset_index(drop=True)


def decadal_means(annual: pd.DataFrame) -> pd.DataFrame:
    d = annual.assign(decade=(annual["year"] // 10) * 10)
    return (d.groupby("decade")["anomaly_c"].mean().round(3)
             .reset_index().rename(columns={"anomaly_c": "mean_anomaly_c"}))


def trend_per_decade(annual: pd.DataFrame, start: int | None = None) -> float:
    a = annual if start is None else annual[annual["year"] >= start]
    if len(a) < 2:
        raise ValueError("Need at least 2 years to fit a trend")
    slope, _ = np.polyfit(a["year"], a["anomaly_c"], 1)
    return round(float(slope) * 10, 4)


def plot(annual: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(annual["year"], annual["anomaly_c"],
           color=np.where(annual["anomaly_c"] >= 0, "#c0504d", "#4f81bd"),
           width=0.9, alpha=0.6, label="Annual mean")
    ax.plot(annual["year"], annual["anomaly_c"].rolling(10, center=True).mean(),
            color="black", lw=2, label="10-year rolling mean")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Anomaly vs 1951–1980 (°C)")
    ax.set_title("Global surface temperature anomaly (NASA GISTEMP v4)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(input_path: Path, outdir: Path, make_plot: bool = True) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    df, checks = clean(load(input_path))
    annual = annual_means(df)
    decadal = decadal_means(annual)

    annual.to_csv(outdir / "annual.csv", index=False)
    decadal.to_csv(outdir / "decadal.csv", index=False)

    warmest = annual.nlargest(5, "anomaly_c")
    summary = {
        "source": GISTEMP_URL,
        "years": [int(annual["year"].min()), int(annual["year"].max())],
        "complete_years": int(len(annual)),
        "trend_c_per_decade_full": trend_per_decade(annual),
        "trend_c_per_decade_since_1970": trend_per_decade(annual, 1970),
        "warmest_years": [{"year": int(r.year), "anomaly_c": float(r.anomaly_c)}
                          for r in warmest.itertuples()],
        "data_checks": checks,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    if make_plot:
        plot(annual, outdir / "anomaly.png")
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="local GISTEMP CSV")
    src.add_argument("--download", action="store_true", help="fetch latest GISTEMP CSV")
    p.add_argument("--outdir", type=Path, default=Path("outputs"))
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    path = download(Path("data/GLB.Ts+dSST.csv")) if args.download else args.input
    try:
        summary = run(path, args.outdir, make_plot=not args.no_plot)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
