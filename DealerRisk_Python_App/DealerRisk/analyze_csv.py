"""Optional command-line entry point; uses the same engine as the dashboard."""
import argparse
from dataclasses import fields
from pathlib import Path
import json

from risk_engine import Settings, analyze, safe_csv


def main():
    parser = argparse.ArgumentParser(description="Analyze a borrower-level loan CSV")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--settings", type=Path, help="Optional JSON object of Settings fields")
    parser.add_argument("--output", type=Path, default=Path("analysis_output"))
    args = parser.parse_args()
    try:
        options = json.loads(args.settings.read_text()) if args.settings else {}
        if not isinstance(options, dict) or set(options)-{f.name for f in fields(Settings)}:
            raise ValueError("Unknown settings; see README.md.")
        result = analyze(args.csv.read_text(encoding="utf-8-sig"), Settings(**options))
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "portfolio_analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (args.output / "borrower_losses.csv").write_text(safe_csv(result["accounts"]), encoding="utf-8")
    print(f"Saved analysis to {args.output.resolve()}")


if __name__ == "__main__":
    main()
