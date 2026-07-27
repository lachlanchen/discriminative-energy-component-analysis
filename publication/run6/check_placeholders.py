"""Audit unresolved Run 6 manuscript result placeholders."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "main.tex"
KEY_PATTERN = re.compile(r"\\RunSixPending\{([A-Z0-9_]+)\}")


def pending_keys() -> list[str]:
    keys = KEY_PATTERN.findall(SOURCE.read_text(encoding="utf-8"))
    return sorted(set(keys))


def pdf_has_pending_marker(path: Path) -> bool:
    completed = subprocess.run(
        ["pdftotext", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return "TBD(" in completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expect-pending",
        action="store_true",
        help="succeed when at least one pending key exists (pre-results audit)",
    )
    parser.add_argument("--pdf", type=Path, default=ROOT / "main.pdf")
    args = parser.parse_args()

    keys = pending_keys()
    visible = pdf_has_pending_marker(args.pdf)
    if args.expect_pending:
        if not visible:
            print("expected visible TBD(KEY) markers, but the PDF has none")
            return 1
        print(f"unresolved Run 6 result placeholders: {len(keys)}")
        for key in keys:
            print(f"  {key}")
        return 0

    if visible:
        print("compiled PDF still contains visible TBD(KEY) markers")
        return 1
    print("compiled PDF contains no TBD(KEY) markers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
