"""Run the M2 stub validation and write results/m2_stub_validation.json (task 3.5).

Usage: python scripts/stub_validation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aib.validation import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
