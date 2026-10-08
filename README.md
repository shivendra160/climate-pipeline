# Global Temperature Anomaly Pipeline

A terminal-based Python pipeline that downloads NASA GISTEMP v4 global temperature data, cleans it, and produces verifiable outputs: annual and decadal means, the warming trend in °C per decade, and the warmest years.

## What it does

1. **Download**: fetches `GLB.Ts+dSST.csv` from NASA GISS.
2. **Validate**: checks the required columns, treats `***` as missing, masks values outside ±3 °C, and drops duplicate years.
3. **Process**: computes annual means (a year needs at least 10 valid months), decadal means, and a linear trend for the full record and since 1970.
4. **Output**: writes `annual.csv`, `decadal.csv`, `summary.json` and `anomaly.png` to `outputs/`.

## Run

```bash
pip install -r requirements.txt
python pipeline.py --download          # latest data
python pipeline.py --input my_copy.csv # local file
```

## Test

```bash
python -m unittest discover -s tests -v
```

The tests use a small synthetic file with known answers: an exact 0.2 °C/decade trend, a year with too few months, an out-of-range value, and a duplicate year. They check every cleaning rule and the end-to-end outputs.

## Data

NASA GISS Surface Temperature Analysis (GISTEMP v4). Anomalies are relative to 1951–1980.
