"""Tests on a small synthetic file in GISTEMP format. Run: python -m unittest -v"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline  # noqa: E402

HEADER = "Land-Ocean: Global Means\n"
COLS = "Year," + ",".join(pipeline.MONTHS) + ",J-D\n"


def row(year, vals):
    return f"{year}," + ",".join(vals) + ",***\n"


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        lines = [HEADER, COLS]
        # 2000..2019: every month = 0.02*(year-2000) -> exact trend 0.2 °C/decade
        for y in range(2000, 2020):
            v = f"{0.02 * (y - 2000):.2f}"
            lines.append(row(y, [v] * 12))
        # 2020: only 9 months valid -> must be excluded
        lines.append(row(2020, ["0.50"] * 9 + ["***"] * 3))
        # 2021: one absurd value (9.99) -> masked, year still kept (11 months)
        lines.append(row(2021, ["0.60"] * 11 + ["9.99"]))
        # duplicate 2019 row -> dropped, last kept
        lines.append(row(2019, ["0.38"] * 12))
        self.csv = self.tmp / "sample.csv"
        self.csv.write_text("".join(lines))

    def test_missing_months_excluded(self):
        df, _ = pipeline.clean(pipeline.load(self.csv))
        years = pipeline.annual_means(df)["year"].tolist()
        self.assertNotIn(2020, years)
        self.assertIn(2021, years)

    def test_out_of_range_masked(self):
        df, checks = pipeline.clean(pipeline.load(self.csv))
        self.assertEqual(checks["out_of_range_values_masked"], 1)
        a = pipeline.annual_means(df).set_index("year")
        self.assertAlmostEqual(a.loc[2021, "anomaly_c"], 0.60)
        self.assertEqual(a.loc[2021, "months_used"], 11)

    def test_duplicates_dropped(self):
        _, checks = pipeline.clean(pipeline.load(self.csv))
        self.assertEqual(checks["duplicate_years_dropped"], 1)

    def test_trend_known_value(self):
        df, _ = pipeline.clean(pipeline.load(self.csv))
        a = pipeline.annual_means(df)
        a = a[a["year"] < 2020]
        self.assertAlmostEqual(pipeline.trend_per_decade(a), 0.2, places=4)

    def test_decadal(self):
        df, _ = pipeline.clean(pipeline.load(self.csv))
        d = pipeline.decadal_means(pipeline.annual_means(df)).set_index("decade")
        self.assertAlmostEqual(d.loc[2000, "mean_anomaly_c"], 0.09, places=3)

    def test_end_to_end_outputs(self):
        out = self.tmp / "out"
        summary = pipeline.run(self.csv, out, make_plot=True)
        for f in ["annual.csv", "decadal.csv", "summary.json", "anomaly.png"]:
            self.assertTrue((out / f).exists(), f)
        self.assertEqual(json.loads((out / "summary.json").read_text()), summary)
        self.assertEqual(summary["warmest_years"][0]["year"], 2021)

    def test_bad_input_raises(self):
        bad = self.tmp / "bad.csv"
        bad.write_text("title\nfoo,bar\n1,2\n")
        with self.assertRaises(ValueError):
            pipeline.load(bad)


if __name__ == "__main__":
    unittest.main()
