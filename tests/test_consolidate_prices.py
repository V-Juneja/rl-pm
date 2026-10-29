import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rl_pm import consolidate_prices as merger


class ConsolidationTests(unittest.TestCase):
    def test_edge_extension_preserves_core_and_uses_matching_boundary_prices(self):
        chosen = {f"2000-01-{day:02d}": {"adjusted_close": str(100 + day)}
                  for day in range(11, 32)}
        rows = {f"2000-01-{day:02d}": {"adjusted_close": str((100 + day) * 2)}
                for day in range(1, 32)}
        rows["2000-01-31"]["adjusted_close"] = "393"
        before = {day: dict(row) for day, row in chosen.items()}
        extra, reports = merger.aligned_edge_extensions(rows, chosen)
        self.assertEqual(set(extra), {f"2000-01-{day:02d}" for day in range(1, 11)})
        self.assertEqual(float(extra["2000-01-10"]["adjusted_close"]), 110)
        self.assertEqual(chosen, before)
        self.assertEqual(reports, [("prefix_aligned", 0.5, 20)])

    def test_archive_adjustment_scale_does_not_create_false_return(self):
        rows = {f"2000-01-{day:02d}": {"adjusted_close": str(day * 2)}
                for day in range(1, 26)}
        chosen = {f"2000-01-{day:02d}": {"adjusted_close": str(day)}
                  for day in range(1, 25)}
        factor, status, overlap = merger.align_adjusted_prices(rows, chosen)
        self.assertEqual(status, "aligned")
        self.assertEqual(factor, 0.5)
        self.assertEqual(overlap, 24)
        self.assertAlmostEqual(float(rows["2000-01-25"]["adjusted_close"])
                               / float(chosen["2000-01-24"]["adjusted_close"]), 25 / 24)

    def test_inconsistent_archive_cannot_be_joined(self):
        rows = {f"2000-01-{day:02d}": {"adjusted_close": "10"}
                for day in range(1, 25)}
        chosen = {day: {"adjusted_close": "20"} for day in rows}
        chosen["2000-01-15"]["adjusted_close"] = "40"
        factor, status, overlap = merger.align_adjusted_prices(rows, chosen)
        self.assertIsNone(factor)
        self.assertEqual(status, "inconsistent_adjustments")
        self.assertEqual(rows["2000-01-15"]["adjusted_close"], "10")

    def test_renamed_ticker_history_stops_at_symbol_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data/raw"
            raw.mkdir(parents=True)
            (raw / "BALL_yahoo_daily.csv").write_text(
                "date,adjusted_close\n2022-05-09,70\n2022-05-10,71\n"
            )
            (raw / "BLL_yahoo_daily.csv").write_text(
                "date,adjusted_close\n2026-01-02,25\n"
            )
            output = raw / "combined"
            with patch.object(merger, "ROOT", root), patch.object(
                merger, "SOURCES", (("raw_yahoo", raw, 0, "_yahoo_daily.csv"),)
            ):
                merger.combine(output)
            with (output / "BLL_combined_daily.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["date"] for row in rows], ["2022-05-09"])
            self.assertEqual(rows[0]["source"], "data/raw/BALL_yahoo_daily.csv")

    def test_unions_dates_keeps_source_priority_and_reports_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data/raw"
            tiingo = root / "data/tiingo"
            recovery = root / "data/recovery"
            scraped = root / "data/scraped/stockanalysis"
            output = raw / "combined"
            for folder in (raw, tiingo, recovery, scraped):
                folder.mkdir(parents=True)

            raw_file = raw / "ABC_yahoo_daily.csv"
            raw_file.write_text(
                "date,open,high,low,close,volume,adjusted_close\n"
                "2000-01-03,10,11,9,10,100,9.5\n"
            )
            tiingo_file = tiingo / "ABC.csv"
            tiingo_file.write_text(
                "date,open,high,low,close,volume,adjusted_close\n"
                "2000-01-03,10,12,9,11,120,10.5\n"
                "2000-01-04,11,12,10,11,110,10.8\n"
            )

            original = (merger.ROOT, merger.SOURCES)
            try:
                merger.ROOT = root
                merger.SOURCES = (
                    ("raw_yahoo", raw, 0, "_yahoo_daily.csv"),
                    ("tiingo", tiingo, 1, None),
                    ("yahoo_recovery", recovery, 2, None),
                    ("stockanalysis", scraped, 3, None),
                )
                manifest, conflicts = merger.combine(output)
            finally:
                merger.ROOT, merger.SOURCES = original

            with (output / "ABC_combined_daily.csv").open() as stream:
                merged = list(csv.DictReader(stream))
            self.assertEqual([row["date"] for row in merged], ["2000-01-03", "2000-01-04"])
            self.assertEqual(merged[0]["adjusted_close"], "9.5")
            self.assertEqual(merged[1]["source"], "data/tiingo/ABC.csv")
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(manifest[0]["rows"], 2)


if __name__ == "__main__":
    unittest.main()
